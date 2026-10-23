"""VFA brain embedding: regional features of a T1w image for downstream models.

Per input: tBET + ASEG (one bx call) -> numreg affine to the bundled MNI-space template -> VFA encoder (levels 1-2,
the registration model's own encoder) -> mean feature of every ASEG structure (37 structures, the deprecated labels
are left out) -> ``<stem>_embed.npz``:

    roi        float32 [37, 96]  mean encoder feature per structure; columns = L1 (2 mm, 32 ch) then L2 (4 mm, 64 ch);
                                 a structure missing from the ASEG is a row of NaN
    labels     int      [37]     ASEG label of each row (doc/seglabel.md)
    columns    str      [96]     'L1_00' ... 'L2_63'
    volume_ml  float32  [37]     native ASEG volume of each structure (ml)
    tiv_ml     float             total intracranial volume = tBET mask volume (ml)
    quality    float [5]         image_quality(): noise, bias, contrast, edge, res (quality_keys)
    model, tigerbx_version, template

Whole-brain means are not stored: every encoder level ends in InstanceNorm, so they are nearly the same for every
image. Features are on the template grid after an affine only, so rows are comparable between subjects; standardise
per scanner / site before pooling data from several sources.
"""
import csv
import os
import time
import logging
from types import SimpleNamespace

import numpy as np
import nibabel as nib
from tqdm import tqdm

import tigerbx
from tigerbx import lib_tool
from tigerbx.bx import run as bx_run
from tigerbx.core import numreg
from tigerbx.core.io import get_template, remove_quietly, resolve_inputs, write_fail_log
from tigerbx.core.onnx import create_session
from tigerbx.core.qa import QUALITY_KEYS, image_quality

_logger = logging.getLogger('tigerbx')
_logger.addHandler(logging.NullHandler())

DEFAULT_MODEL = 'mprage_embed_v001_vfa.onnx'
# ASEG structures kept (aseg43 without the labels deprecated in 0.3.0: 30, 31, 62, 63, 77, 85)
LABELS = (2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52,
          53, 54, 58, 60, 251, 252, 253, 254, 255)
LEVELS = ('L1', 'L2')
SUMMARY_NAME = 'embed_summary.csv'
SUMMARY_COLUMNS = ('file', 'status', 'TIV_ml') + QUALITY_KEYS + ('error',)


def roi_means(features, seg, labels=LABELS):
    """Mean of each feature map inside each label.

    features: list of [C, h, w, d] maps on grids that evenly divide seg's [H, W, D] (factor H // h, exact).
    The label masks are block-averaged to each map's grid (partial-volume weights), then normalised.
    Returns [len(labels), sum C]; NaN rows for labels absent from seg.
    """
    seg = np.asarray(seg)
    if seg.ndim != 3:
        raise ValueError(f'seg must be 3D (got shape {seg.shape})')
    seg = np.rint(seg).astype(np.int32)
    feats = [np.asarray(f) for f in features]
    for f in feats:
        if f.ndim != 4 or min(f.shape[1:]) < 1 or any(S % n for S, n in zip(seg.shape, f.shape[1:])):
            raise ValueError(f'feature grid {f.shape[1:]} does not evenly divide the seg grid {seg.shape}')
    out = np.full((len(labels), sum(f.shape[0] for f in feats)), np.nan)
    for i, lab in enumerate(labels):
        m = seg == lab
        if not m.any():
            continue                                  # absent label -> NaN row
        c0 = 0
        for f in feats:
            C, h, w, d = f.shape
            fh, fw, fd = seg.shape[0] // h, seg.shape[1] // w, seg.shape[2] // d
            # voxel count of the label in every block = partial-volume weight of each feature voxel
            cnt = m.reshape(h, fh, w, fw, d, fd).sum((1, 3, 5), dtype=np.int32).ravel()
            idx = np.flatnonzero(cnt)
            out[i, c0:c0 + C] = f.reshape(C, -1)[:, idx].astype(np.float64) @ cnt[idx] / cnt[idx].sum()
            c0 += C
    return out.astype(np.float32)


def _embed_one(f, args, common_folder, session):
    bx_result = bx_run('ma' + ('g' if args.gpu else ''), f, output=None, verbose=0, save_outputs=False)
    aseg_nib, mask_nib = bx_result['aseg'], bx_result['tbetmask']

    reg_args = SimpleNamespace(plan='A', input=[f], output=args.output, model=None, gpu=args.gpu, gz=True,
                               template=None, save_displacement=False, affine_type='numreg', verbose=0)
    from tigerbx.reg import run_case        # tigerbx.reg is also the reg() function
    result_dict, _, ctx = run_case(reg_args, f, common_folder=common_folder, save_reg_output=False,
                                   verbose=0, bx_result=bx_result)
    image = np.asarray(result_dict['affine'].dataobj, np.float32)[None, None]
    M = ctx.displacement_dict['affine_matrix']['M']
    seg = numreg.apply(aseg_nib, ctx.template_nib, M, order=0)

    l1, l2 = session.run(None, {session.get_inputs()[0].name: image})
    roi = roi_means([l1[0], l2[0]], seg)

    aseg = np.asarray(aseg_nib.dataobj)
    vox_ml = float(np.prod(aseg_nib.header.get_zooms()[:3])) / 1000
    counts = np.bincount(aseg.ravel().astype(np.int64), minlength=256)
    volume = (counts[list(LABELS)] * vox_ml).astype(np.float32)
    tiv = float((np.asarray(mask_nib.dataobj) > 0).sum()
                * np.prod(mask_nib.header.get_zooms()[:3]) / 1000)
    try:
        quality = image_quality(nib.load(f), aseg_nib)
    except Exception as exc:                    # noqa: BLE001 - quality measures are advisory
        _logger.warning('image_quality failed for %s: %s', os.path.basename(f), exc)
        quality = {k: float('nan') for k in QUALITY_KEYS}
    columns = [f'{lv}_{c:02d}' for lv, n in zip(LEVELS, (l1.shape[1], l2.shape[1])) for c in range(n)]
    return dict(roi=roi, labels=np.array(LABELS), columns=np.array(columns), volume_ml=volume,
                tiv_ml=tiv, quality=np.array([quality[k] for k in QUALITY_KEYS]),
                quality_keys=np.array(QUALITY_KEYS), model=os.path.basename(args.model),
                tigerbx_version=tigerbx.__version__, template='mni_template.nii.gz')


def _out_paths(f, args, common_folder):
    ftemplate, _ = get_template(f, args.output, True, common_folder)
    d, base = os.path.split(ftemplate)
    stem = base.replace('.nii', '').replace('.gz', '')
    return (os.path.join(d, stem.replace('@@@@', 'embed.npz')),
            os.path.join(d, stem.replace('@@@@', 'embedfail.log')))


def _save_summary(summary_dir, rows):
    path = os.path.join(summary_dir, SUMMARY_NAME)
    try:
        os.makedirs(summary_dir, exist_ok=True)
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, '') for k in SUMMARY_COLUMNS})
    except OSError as exc:
        _logger.warning('could not write %s: %s', path, exc)


def embed(input=None, output=None, model=None, gpu=False, verbose=0):
    """VFA brain embedding of T1w images (see module docstring for the ``<stem>_embed.npz`` contents).

    A failing input is logged, gets ``<stem>_embedfail.log`` and is skipped (a single input raises);
    ``embed_summary.csv`` in the output folder lists status, TIV and image quality per input.
    Returns the embedding dict for a single input, otherwise a list of ``_embed.npz`` paths (successful inputs).
    """
    args = SimpleNamespace(input=input if isinstance(input, list) else [input], output=output,
                           model=model or DEFAULT_MODEL, gpu=gpu, verbose=verbose)
    return run_args(args)


def run_args(args):
    verbose = getattr(args, 'verbose', 0)
    args.model = getattr(args, 'model', None) or DEFAULT_MODEL
    input_file_list, common_folder = resolve_inputs(args.input)
    if not input_file_list:
        return []
    session = create_session(lib_tool.get_model(args.model), args.gpu)
    is_single = len(input_file_list) == 1
    summary_dir = args.output or os.path.dirname(os.path.abspath(input_file_list[0]))
    rows, result_all = [], []
    for f in tqdm(input_file_list, desc='tigerbx-embed', unit='file', disable=(verbose > 0)):
        t = time.time()
        fail_log = os.path.join(summary_dir, os.path.basename(f) + '_embedfail.log')
        try:
            out_npz, fail_log = _out_paths(f, args, common_folder)
            res = _embed_one(f, args, common_folder, session)
            os.makedirs(os.path.dirname(out_npz), exist_ok=True)
            np.savez(out_npz, **res)
        except Exception as exc:                 # noqa: BLE001 - one bad input must not stop a batch
            msg = f'{type(exc).__name__}: {exc}'
            write_fail_log(fail_log, f, msg)
            rows.append(dict(file=f, status='failed', error=msg))
            if is_single:
                _save_summary(summary_dir, rows)
                raise
            _logger.warning('FAILED %s: %s', os.path.basename(f), msg)
            continue
        remove_quietly(fail_log)                 # an earlier failure of this input is now stale
        rows.append(dict(file=f, status='ok', TIV_ml=round(res['tiv_ml'], 2),
                         **{k: round(float(v), 4) for k, v in zip(QUALITY_KEYS, res.get('quality', []))}))
        if verbose >= 1:
            _logger.info('%s: %.1f s', os.path.basename(f), time.time() - t)
        result_all = res if is_single else result_all + [out_npz]

    _save_summary(summary_dir, rows)
    failed = sum(r['status'] != 'ok' for r in rows)
    if failed:
        _logger.warning('embed: %d of %d failed; see %s', failed, len(rows), os.path.join(summary_dir, SUMMARY_NAME))
    return result_all
