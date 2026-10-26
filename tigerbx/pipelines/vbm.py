import csv
import os
from types import SimpleNamespace
import time
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import nibabel as nib
import numpy as np
from tqdm import tqdm

from tigerbx.modules.reg import _parse_reg_plan, apply_warp, modulate, run_case, smooth
from tigerbx.modules.bx import run as bx_run
from tigerbx.core.io import get_template, load_nifti, remove_quietly, save_nib, resolve_inputs, write_fail_log
from tigerbx.core.cgw import tissue_pve
from tigerbx.core.qa import QUALITY_KEYS, block_mean, homogeneity, image_quality, quality_outliers

_logger = logging.getLogger('tigerbx')
_logger.addHandler(logging.NullHandler())


DEFAULT_FWHM = 7.065            # mm; = Gaussian sigma of 3 mm (3 voxels on the 1 mm template)
SUMMARY_NAME = 'vbm_summary.csv'
SUMMARY_COLUMNS = ('file', 'status', 'qc_flags', 'TIV_ml', 'GM_ml', 'WM_ml', 'CSF_ml') + QUALITY_KEYS + ('homog_z', 'error')
HOMOG_MIN_N = 10                # homogeneity z and quality outliers need a sample: batches of >= 10 successful inputs


def vbm(input=None, output=None, model=None, template=None, reg_plan='AD',
        gpu=False, gz=False, save_displacement=False, affine_type='numreg',
        verbose=0, *, fwhm=DEFAULT_FWHM, save_modulated=False):
    """DeepVBM: registration + GM segmentation -> modulated, smoothed GM in template space.

    Outputs per input (all in the output folder): ``_SmoothedGM`` and ``_norm`` (and ``_ModulatedGM`` with
    ``save_modulated=True``). A ``vbm_summary.csv`` with QC flags, TIV and native GM / WM / CSF volumes of every input
    (and the error of every failed one) is written to the output folder.

    Returns the result dict (images) for a single input, otherwise a list of file dicts (one per successful
    input), as tigerbx.run / tigerbx.reg do.
    """
    if not isinstance(input, list):
        input = [input]

    args = SimpleNamespace()
    args.input = input
    args.output = output
    args.model = model
    args.gpu = gpu
    args.gz = gz
    args.reg_plan = reg_plan or 'AD'
    args.vbm = True
    args.template = template
    args.save_displacement = save_displacement
    args.affine_type = affine_type
    args.fwhm = fwhm
    args.save_modulated = save_modulated
    args.verbose = verbose
    return run_args(args)


# result_dict key of the moved T1 written by each reg plan step
_STEP_IMAGE = {'R': 'rigid', 'A': 'affine', 'V': 'VXM', 'D': 'VFA'}


def _final_image_key(reg_plan, result_dict):
    """Key of the T1 after the last plan step, whose warp the GM map went through (for 'ADV' that is VMnet)."""
    return [_STEP_IMAGE[s] for s in _parse_reg_plan(reg_plan) if _STEP_IMAGE[s] in result_dict][-1]


def _check_fwhm(fwhm):
    """None -> DEFAULT_FWHM; otherwise a finite FWHM > 0 in mm (checked once, before any input is processed)."""
    if fwhm is None:
        return DEFAULT_FWHM
    try:
        fwhm = float(fwhm)
    except (TypeError, ValueError):
        raise ValueError(f'fwhm must be a number in mm (got {fwhm!r})') from None
    if not np.isfinite(fwhm) or fwhm <= 0:
        raise ValueError(f'fwhm must be a finite number > 0 mm (got {fwhm})')
    return fwhm


def _volume_ml(nib_img):
    """Sum of voxel values (mask or PVE) x voxel volume, in ml."""
    vox = float(np.prod(nib_img.header.get_zooms()[:3])) / 1000.0
    return float(np.asarray(nib_img.dataobj, dtype=np.float64).sum() * vox)


def _write_summary(path, rows):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in SUMMARY_COLUMNS})


def _save_summary(path, rows):
    """_write_summary that logs instead of raising (the outputs of the batch are already written)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _write_summary(path, rows)
    except OSError as exc:
        _logger.warning('could not write %s: %s', path, exc)


def _add_homogeneity(ok):
    """Set homog_z on the rows of successful inputs; NaN (and flagged) for an unusable (non-finite / flat) map."""
    maps = np.stack([r['_homog_map'] for r in ok])
    valid = np.isfinite(maps).all(axis=tuple(range(1, maps.ndim)))
    mask = maps[valid].mean(0) > 0.1 if valid.any() else None
    _, z = homogeneity(maps, mask)
    for r, zi in zip(ok, z):
        r['homog_z'] = round(float(zi), 2) if np.isfinite(zi) else ''
        if not zi >= -3:                         # z < -3 or an unusable map
            r['_flags'].append('homog')


def _add_quality_flags(ok):
    """Flag image_quality measures that are outliers in the batch (core.qa.quality_outliers)."""
    values = {k: np.array([r[k] if isinstance(r.get(k), (int, float)) else np.nan for r in ok], float)
              for k in QUALITY_KEYS}
    for r, fl in zip(ok, quality_outliers(values)):
        r['_flags'].extend(fl)


def _set_qc_flags(rows):
    """qc_flags of every row ('fail', quality outliers, 'homog'; ';'-separated, empty = nothing to check).
    Returns {flag: count}."""
    ok = [r for r in rows if r['status'] == 'ok' and '_homog_map' in r]
    for r in rows:
        r['_flags'] = [] if r['status'] == 'ok' else ['fail']
    if len(ok) >= HOMOG_MIN_N:
        for check in (_add_quality_flags, _add_homogeneity):
            try:
                check(ok)
            except Exception as exc:             # noqa: BLE001 - an advisory check must not lose the summary
                _logger.warning('VBM %s failed: %s', check.__name__.strip('_').replace('_', ' '), exc)
    counts = {}
    for r in rows:
        fl = r.pop('_flags')
        r['qc_flags'] = ';'.join(fl)
        for k in fl:
            counts[k] = counts.get(k, 0) + 1
        r.pop('_homog_map', None)
    return counts


def _fail_log_path(f, args, common_folder):
    """<output>/<stem>_vbmfail.log, next to the other outputs of *f*."""
    ftemplate, out_dir = get_template(f, args.output, args.gz, common_folder)
    d, base = os.path.split(ftemplate)
    return os.path.join(d, base.replace('.nii', '').replace('.gz', '').replace('@@@@', 'vbmfail.log'))


def _run_one(args, f, common_folder, fwhm, save_modulated, printer, _dbg):
    """VBM for one input: returns (result_dict, result_filedict, summary_row)."""
    # one bx call for the tBX mask + TSEG, reused by the registration stage; CSF / GM / WM maps from TSEG (core.cgw)
    bx_result = bx_run('mt' + ('g' if args.gpu else ''), f, output=None, verbose=0, save_outputs=False)
    img = load_nifti(f)
    pve = tissue_pve(np.squeeze(np.asarray(img.dataobj, np.float32)), bx_result['tseg'].dataobj,
                     np.asarray(bx_result['tbxmask'].dataobj) > 0, img.header.get_zooms()[:3])
    csf_pve, gm_pve_nib, wm_pve = (nib.Nifti1Image(p, img.affine) for p in pve)

    result_dict, result_filedict, reg_ctx_obj = run_case(
        args, f, common_folder=common_folder,
        save_reg_output=False, verbose=0, bx_result=bx_result)
    if reg_ctx_obj is None:
        raise RuntimeError('Missing RegistrationContext from tigerbx.reg for VBM pipeline.')
    ftemplate = reg_ctx_obj.ftemplate

    reg_gm = apply_warp(gm_pve_nib, reg_ctx_obj, interpolation='linear')
    result_dict['Reg_GM'] = reg_gm

    mod_gm = modulate(reg_gm, reg_ctx_obj)
    result_dict['Modulated_GM'] = mod_gm
    if save_modulated:
        result_filedict['Modulated_GM'] = save_nib(mod_gm, ftemplate, 'ModulatedGM')

    last = _final_image_key(args.reg_plan, result_dict)
    result_filedict['norm'] = save_nib(result_dict[last], ftemplate, 'norm')
    result_dict['norm'] = result_dict[last]

    smooth_gm = smooth(mod_gm, fwhm=fwhm)
    result_dict['Smoothed_GM'] = smooth_gm
    result_filedict['Smoothed_GM'] = save_nib(smooth_gm, ftemplate, 'SmoothedGM')
    _dbg(result_filedict['Smoothed_GM'])

    row = dict(file=f, status='ok',
               TIV_ml=round(_volume_ml(bx_result['tbxmask']), 2), GM_ml=round(_volume_ml(gm_pve_nib), 2),
               WM_ml=round(_volume_ml(wm_pve), 2), CSF_ml=round(_volume_ml(csf_pve), 2))
    try:
        quality = image_quality(img, bx_result['tseg'])
    except Exception as exc:                    # noqa: BLE001 - quality measures are advisory
        _logger.warning('image_quality failed for %s: %s', os.path.basename(f), exc)
        quality = {k: float('nan') for k in QUALITY_KEYS}
    row.update({k: round(v, 4) for k, v in quality.items()})
    row['_homog_map'] = block_mean(smooth_gm.get_fdata())
    for k in ('TIV_ml', 'GM_ml', 'WM_ml', 'CSF_ml') + QUALITY_KEYS:
        result_dict[k] = row[k]
    return result_dict, result_filedict, row


def run_args(args):
    verbose = getattr(args, 'verbose', 0)
    fwhm = _check_fwhm(getattr(args, 'fwhm', None))
    save_modulated = getattr(args, 'save_modulated', False)

    def printer(*msg):
        if verbose >= 1:
            _logger.info(' '.join(str(x) for x in msg))

    def _dbg(*msg):
        if verbose >= 2:
            _logger.debug(' '.join(str(x) for x in msg))

    run_d = vars(args)
    if not run_d.get('reg_plan'):
        run_d['reg_plan'] = 'AD'
    run_d['vbm'] = True
    run_d.setdefault('tseg', False)

    input_file_list, common_folder = resolve_inputs(args.input)
    printer('Reg plan:', run_d['reg_plan'])
    printer('Total nii files:', len(input_file_list))
    if not input_file_list:
        return []

    result_all = []
    rows = []
    is_single = len(input_file_list) == 1
    summary_dir = args.output or os.path.dirname(os.path.abspath(input_file_list[0]))
    summary_path = os.path.join(summary_dir, SUMMARY_NAME)
    _pbar = tqdm(input_file_list, desc='tigerbx-vbm', unit='file', disable=(verbose > 0))
    for count, f in enumerate(_pbar, 1):
        _pbar.set_postfix_str(os.path.basename(f))
        printer(f'{count} Preprocessing :', os.path.basename(f))
        t = time.time()
        try:
            fail_log = _fail_log_path(f, args, common_folder)
        except Exception:                        # noqa: BLE001 - e.g. an unusable output folder
            fail_log = os.path.join(summary_dir, os.path.basename(f) + '_vbmfail.log')
        try:
            result_dict, result_filedict, row = _run_one(args, f, common_folder, fwhm, save_modulated,
                                                         printer, _dbg)
        except Exception as exc:                 # noqa: BLE001 - one bad input must not stop a batch
            msg = f'{type(exc).__name__}: {exc}'
            write_fail_log(fail_log, f, msg)
            rows.append(dict(file=f, status='failed', error=msg))
            if is_single:
                _save_summary(summary_path, rows)
                raise
            _logger.warning('FAILED %s: %s', os.path.basename(f), msg)
            continue
        remove_quietly(fail_log)                 # an earlier failure of this input is now stale
        rows.append(row)
        printer('Processing time: %d seconds' % (time.time() - t))
        if is_single:
            result_all = result_dict
        else:
            result_all.append(result_filedict)

    counts = _set_qc_flags(rows)
    _save_summary(summary_path, rows)
    failed = counts.pop('fail', 0)
    flagged = sum(1 for r in rows if r['status'] == 'ok' and r['qc_flags'])
    if failed or flagged:
        detail = ', '.join(f'{k} {v}' for k, v in sorted(counts.items()))
        _logger.warning('VBM: %d failed, %d flagged%s of %d; see %s (qc_flags)', failed, flagged,
                        f' ({detail})' if detail else '', len(rows), summary_path)
    else:
        printer('Summary:', summary_path)
    return result_all
