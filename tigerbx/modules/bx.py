import os
from os.path import basename, join, isdir, dirname, relpath
import time
import logging
import numpy as np

import nibabel as nib
from tqdm import tqdm

from tigerbx import lib_tool
from tigerbx import lib_bx
import copy
from tigerbx.core.io import get_template, load_nifti, save_nib, resolve_inputs
from tigerbx.core.onnx import create_session, predict
from tigerbx.core.resample import resample_to_img, reorder_img, resample_voxel
from tigerbx.core.spatial import crop_cube
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

_logger = logging.getLogger('tigerbx')
_logger.addHandler(logging.NullHandler())


_NATIVE_111 = (0.99, 1.01)
_NATIVE_111_LOOSE = (0.9, 1.1)
# Padding (mm) of the tBX input crop around the non-zero voxels, and its minimum extent (mm) per axis
_TBX_CROP_MM = 16.0
_TBX_CROP_MIN_MM = 256.0
# Model-space input crop of the other bx models: tbx > 0 + 16 voxels, unless that box exceeds 256^3 voxels; then
# the tBX mask + 32 voxels
_TBX111_CROP_MAX_VOXELS = 256 ** 3
_TBX111_MASK_CROP_PADDING = 32


def _crop_nib(nib_img, xyz6):
    """Crop a nibabel image to xyz6 bounds and correct the affine origin."""
    x_min, x_max, y_min, y_max, z_min, z_max = xyz6
    data = nib_img.get_fdata()
    cropped = data[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
    new_affine = nib_img.affine.copy()
    new_affine[:3, 3] = nib_img.affine[:3, :3] @ np.array([x_min, y_min, z_min]) + nib_img.affine[:3, 3]
    return nib.Nifti1Image(cropped, new_affine)


def _crop_nonzero_native(img, pad_mm=None):
    """tBX input crop on the input grid: the bounding box of the voxels that are exactly != 0, `pad_mm`
    (default 16) wider and at least 256 mm per axis (centred on the box, slid back inside the image), clipped. No thresholds and no head
    detection; on the input grid because resampling to 1 mm spreads tiny non-zero values over a zero background.
    Images resampled onto a large grid (e.g. a PET field of view) are mostly zeros, and without the crop their 1 mm
    model input can exceed what cuDNN supports. The 256 mm minimum leaves usual fields of view whole: a tight crop
    of an already skull-stripped image shrinks the tBX mask (Dice vs the given brain 0.962 -> 0.950)."""
    arr = np.asanyarray(img.dataobj)
    if arr.ndim < 3:
        return img
    nz = arr != 0
    if nz.ndim > 3:
        nz = nz.reshape(nz.shape[:3] + (-1,)).any(-1)
    if not nz.any():
        return img
    pad_mm = _TBX_CROP_MM if pad_mm is None else pad_mm
    zoom = np.asarray(img.header.get_zooms()[:3], float)
    pad = np.ceil(pad_mm / zoom).astype(int)
    min_n = np.ceil(_TBX_CROP_MIN_MM / zoom).astype(int)
    sl = []
    for k in range(3):
        idx = np.flatnonzero(nz.any(tuple(a for a in range(3) if a != k)))
        lo, hi = idx[0] - pad[k], idx[-1] + 1 + pad[k]
        grow = max(min_n[k] - (hi - lo), 0)
        lo, hi = lo - grow // 2, hi + grow - grow // 2
        shift = (max(-lo, 0) - max(hi - nz.shape[k], 0)) if grow else 0   # slide a grown window back inside
        sl.append(slice(max(lo + shift, 0), min(hi + shift, nz.shape[k])))
    if all(s.start == 0 and s.stop == n for s, n in zip(sl, nz.shape)):
        return img
    return img.slicer[tuple(sl)]


def _prepare_tbx111_crop(tbx_nib, tbxmask_nib=None, native_111_range=_NATIVE_111):
    """Build the unified tBX-derived model input crop (`tbx111_crop`) for all non-tBX bx models: the tBX brain in
    model space, cropped to tbx > 0 + 16 voxels. When that box exceeds 256^3 voxels (the tiny positive values that
    reordering an oblique input leaves far outside the brain make it the whole grid, e.g. 34 M voxels, too large
    for the CUDA Concat of TSEG), the box of the tBX mask (nearest onto the same grid) + 32 voxels is used instead.
    seg18 rotated 26 deg, TSEG Dice 0.7700 -> 0.7998; inputs within 256^3 are unchanged."""
    zoom = tbx_nib.header.get_zooms()[:3]
    native_111_min, native_111_max = native_111_range
    is_native_111 = (max(zoom) <= native_111_max and min(zoom) >= native_111_min)

    if is_native_111:
        tbx111 = reorder_img(tbx_nib, resample='continuous')
    else:
        tbx111 = resample_voxel(tbx_nib, (1, 1, 1), interpolation='continuous')
        tbx111 = reorder_img(tbx111, resample='continuous')

    arr_111 = tbx111.get_fdata()
    _, xyz6_111 = crop_cube(arr_111, arr_111 > 0)
    size = np.prod([xyz6_111[2 * a + 1] - xyz6_111[2 * a] + 1 for a in range(3)])
    if tbxmask_nib is not None and size > _TBX111_CROP_MAX_VOXELS:
        keep = np.asarray(resample_to_img(tbxmask_nib, tbx111, interpolation='nearest').dataobj) > 0
        _, xyz6_111 = crop_cube(arr_111, keep, padding=_TBX111_MASK_CROP_PADDING)
    return _crop_nib(tbx111, xyz6_111)


def _infer_mask(model_ff, f, GPU, patch, brainmask_nib=None, tbx111=None, session=None):
    """Shared inference core: load, run model, resample, apply brainmask, cast dtype.

    Returns (output_nib, mask_nib_resp); mask_nib_resp is in model space (before resampling to input space).
    """
    input_nib = load_nifti(f)

    if tbx111 is None:
        input_nib_resp = lib_bx.read_tbx_input(f, input_nib=_crop_nonzero_native(input_nib))
    else:
        input_nib_resp = copy.deepcopy(tbx111)  # avoid modifying caller's copy

    mask_nib_resp = lib_bx.run(
        model_ff, input_nib_resp, GPU=GPU, patch=patch, session=session)

    mask_nib = resample_to_img(
        mask_nib_resp, input_nib, interpolation="nearest")

    if brainmask_nib is None:
        output = lib_tool.read_nib(mask_nib)
    else:
        output = lib_tool.read_nib(mask_nib) * lib_tool.read_nib(brainmask_nib)

    dtype = np.uint8 if np.max(output) <= 255 else np.int16
    output = output.astype(dtype)

    output_nib = nib.Nifti1Image(output, input_nib.affine, input_nib.header)
    output_nib.header.set_data_dtype(dtype)

    return output_nib, mask_nib_resp


def produce_tbxmask(model, f, GPU=False, patch=False, session=None):
    """Run the tBX model and return the brain mask (nibabel image on the input grid)."""
    model_ff = lib_tool.get_model(model)

    output_nib, mask_nib_resp = _infer_mask(
        model_ff, f, GPU, patch, session=session)

    if not np.any(lib_tool.read_nib(mask_nib_resp) > 0):
        raise RuntimeError(f'tBX produced an empty brain mask for {f!r}.')

    return output_nib


def produce_mask(model, f, GPU=False, brainmask_nib=None, tbx111=None,
                 patch=False, session=None):
    """Run a segmentation model and return output_nib.

    brainmask_nib: tBX mask applied to the output (zeros outside brain).
    tbx111:       Pre-computed tBX brain used as model input instead of raw f.
    """
    model_ff = lib_tool.get_model(model)

    output_nib, _ = _infer_mask(
        model_ff, f, GPU, patch, brainmask_nib, tbx111, session=session)

    return output_nib


def _all_outputs_exist(f, output_dir, run_d, gz, common_folder=None):
    """Return True if all requested outputs for file f already exist on disk."""
    ftemplate, _ = get_template(f, output_dir, gz, common_folder)
    checks = {
        'tbxmask': 'tbxmask',
        'tbx':     'tbx',
        'aseg':    'aseg',
        'tseg':    'tseg',
        'dgm':     'dgm',
        'ct':      'ct',
    }
    for flag, postfixes in checks.items():
        if run_d.get(flag):
            if not isinstance(postfixes, (tuple, list)):
                postfixes = (postfixes,)
            for postfix in postfixes:
                if not os.path.isfile(ftemplate.replace('@@@@', postfix)):
                    return False
    return True

# flags removed in this release -> what they did (an argstring with one of them raises)
REMOVED_FLAGS = {'q': 'tBX QC score; see tigerbx.core.qa.image_quality', 'S': 'SynthSeg-style segmentation',
                 'W': 'white-matter hypointensity mask',
                 'C': 'CGW model; CSF / GM / WM maps now come from TSEG: tigerbx.seg(save_tissue=True), core.cgw'}


def run(argstring, input=None, output=None, model=None, verbose=0,
        chunk_size=50, continue_=False, silent=False, save_outputs=True):

    if silent:
        warnings.warn(
            "silent= is deprecated, use verbose=0 (default).",
            DeprecationWarning, stacklevel=2)
        verbose = 0

    from types import SimpleNamespace as Namespace
    args = Namespace()
    if not isinstance(input, list):
        input = [input]
    args.input = input
    args.output = output
    args.model = model
    args.clean_onnx = 'clean_onnx' in argstring
    args.gpu = 'g' in argstring
    args.verbose = verbose
    args.chunk_size = chunk_size
    args.continue_ = continue_
    args.save_outputs = save_outputs

    if args.clean_onnx:
        argstring = ''
    removed = [c for c in REMOVED_FLAGS if c in argstring]
    if removed:
        raise ValueError('tigerbx.run: flag(s) ' + ', '.join(f"'{c}' ({REMOVED_FLAGS[c]})" for c in removed)
                         + ' have been removed.')
    args.tbxmask = 'm' in argstring
    args.aseg = 'a' in argstring      # kept for 0.2.x scripts: the TSEG model output without label 99 (_aseg)
    args.tseg = 't' in argstring
    args.tbx = 'b' in argstring
    args.ct = 'c' in argstring
    args.dgm = 'd' in argstring
    args.gz = 'z' in argstring
    args.patch = 'p' in argstring
    return run_args(args)


# ── per-model inference helpers ───────────────────────────────────────────────

def _fill_unlabeled_intracranial(tseg_nib, tbxmask_nib):
    """Label the tBX-mask voxels that tseg leaves at 0 as Unlabeled-Intracranial (99)."""
    tseg = np.asarray(tseg_nib.dataobj)
    mask = np.asarray(tbxmask_nib.dataobj) > 0
    if tseg.shape != mask.shape:
        return tseg_nib
    tseg = tseg.copy()
    tseg[mask & (tseg == 0)] = lib_bx.TSEG_UNLABELED_INTRACRANIAL
    return nib.Nifti1Image(tseg, tseg_nib.affine, tseg_nib.header)


def _run_seg(key, session, model_ff, f, cache, GPU, patch, save_outputs=True):
    result_nib = produce_mask(model_ff, f, GPU=GPU,
                              brainmask_nib=cache['tbxmask_nib'],
                              tbx111=cache['tbx111_crop'],
                              patch=patch, session=session)
    outs = {key: result_nib}
    if key == 'tseg':                  # one TSEG inference gives _aseg (model output) and / or _tseg (+ label 99)
        outs = {}
        if cache['want_aseg']:
            outs['aseg'] = result_nib
        if cache['want_tseg']:
            outs['tseg'] = _fill_unlabeled_intracranial(result_nib, cache['tbxmask_nib'])
    files = {}
    if save_outputs:
        for k, nib_k in outs.items():
            files[k] = save_nib(nib_k, cache['ftemplate'], k)
            _logger.debug('Writing output file: %s', files[k])
    return outs, files


def _run_ct(key, session, model_ff, f, cache, GPU, patch, save_outputs=True):
    input_nib = load_nifti(f)
    tbx111_crop = cache['tbx111_crop']
    tbx_img = lib_tool.read_nib(tbx111_crop)
    image = tbx_img[None, ...][None, ...]
    mx = np.max(image)
    if mx > 0:
        image = image / mx
    ct = predict(model_ff, image, GPU, session=session)[0, 0, ...]
    ct[ct < 0.2] = 0
    ct[ct > 5] = 5
    ct = ct * (tbx_img > 0).astype(int)
    ct_nib = nib.Nifti1Image(ct, tbx111_crop.affine)
    ct_nib = resample_to_img(ct_nib, input_nib, interpolation="nearest")
    ct_nib.header.set_data_dtype(float)
    if save_outputs:
        fn = save_nib(ct_nib, cache['ftemplate'], 'ct')
        _logger.debug('Writing output file: %s', fn)
        return {'ct': ct_nib}, {'ct': fn}
    return {'ct': ct_nib}, {}


_MODEL_RUNNERS = {
    'tseg': _run_seg,
    'dgm':  _run_seg,
    'ct':   _run_ct,
}

_SEG_ORDER = ['tseg', 'dgm', 'ct']


# ── main entry point ──────────────────────────────────────────────────────────

def run_args(args):
    run_d      = vars(args)
    verbose    = run_d.get('verbose', 0)
    chunk_size = run_d.get('chunk_size', 50)
    save_outputs = run_d.get('save_outputs', True)

    def printer(*msg):
        if verbose >= 1:
            _logger.info(' '.join(str(x) for x in msg))

    def _dbg(*msg):
        if verbose >= 2:
            _logger.debug(' '.join(str(x) for x in msg))

    def _warn(*msg):
        _logger.warning(' '.join(str(x) for x in msg))

    run_d.setdefault('aseg', False)
    if True not in [run_d['tbxmask'], run_d['aseg'], run_d['tseg'], run_d['tbx'], run_d['dgm'],
                    run_d['ct'], run_d['patch']]:
        run_d['tbx'] = True

    if run_d['clean_onnx']:
        lib_tool.clean_onnx()
        printer('Exiting...')
        return 1

    input_file_list, common_folder = resolve_inputs(args.input)

    output_dir = args.output
    omodel = {
        'tbx':  'mprage_tbx_v005_opt.onnx',
        'tseg': 'mprage_tseg_v0011_opt.onnx',
        'ct':   'mprage_mix_ct.onnx',
        'dgm':  'mprage_dgm12_v002_opt.onnx',
    }

    if isinstance(args.model, dict):
        for mm in args.model.keys():
            omodel[mm] = args.model[mm]
    elif isinstance(args.model, str):
        import ast
        model_dict = ast.literal_eval(args.model)
        for mm in model_dict.keys():
            omodel[mm] = model_dict[mm]

    # --continue: skip files whose outputs already exist
    if run_d.get('continue_') and save_outputs:
        original_n = len(input_file_list)
        input_file_list = [f for f in input_file_list
                           if not _all_outputs_exist(f, output_dir, run_d, args.gz, common_folder)]
        printer(f'Skipping {original_n - len(input_file_list)} already-processed files')

    printer('Total nii files:', len(input_file_list))

    if not input_file_list:
        return []

    _needs_tbx111_crop = any(run_d.get(k) for k in ['aseg', 'tseg', 'dgm', 'ct'])
    if run_d.get('ct'):
        _native_111_range = _NATIVE_111
    else:
        _native_111_range = _NATIVE_111_LOOSE
    active_models = [(k, _MODEL_RUNNERS[k]) for k in _SEG_ORDER
                     if run_d.get(k) or (k == 'tseg' and run_d.get('aseg'))]

    result_accum = {f: [{}, {}] for f in input_file_list}
    tbx_model_ff = lib_tool.get_model(omodel['tbx'])

    # ── Chunked loop ──────────────────────────────────────────────────────────
    for chunk_start in range(0, len(input_file_list), chunk_size):
        chunk = input_file_list[chunk_start : chunk_start + chunk_size]
        tbx_cache = {}

        # Phase 1: tBX (all files in chunk, one shared session)
        tbx_session = create_session(tbx_model_ff, args.gpu)
        _pbar = tqdm(chunk, desc='tBX', unit='file', disable=(len(input_file_list) <= 1 or verbose >= 2))
        for count, f in enumerate(_pbar, chunk_start + 1):
            _dbg(f'{count} Processing: {os.path.basename(f)}')

            ftemplate, _ = get_template(f, output_dir, args.gz, common_folder)

            tbxmask_nib = produce_tbxmask(
                omodel['tbx'], f, GPU=args.gpu, patch=run_d['patch'],
                session=tbx_session)

            input_nib = load_nifti(f)
            tbx_arr = lib_tool.read_nib(input_nib) * lib_tool.read_nib(tbxmask_nib)
            tbx_nib = nib.Nifti1Image(tbx_arr, input_nib.affine, input_nib.header)

            tbx111_crop = None
            if _needs_tbx111_crop:
                tbx111_crop = _prepare_tbx111_crop(
                    tbx_nib, tbxmask_nib, native_111_range=_native_111_range)

            _pbar.set_postfix_str(os.path.basename(f))
            rd  = result_accum[f][0]
            rfd = result_accum[f][1]

            if run_d['tbxmask']:
                rd['tbxmask']  = tbxmask_nib
                if save_outputs:
                    fn = save_nib(tbxmask_nib, ftemplate, 'tbxmask')
                    _dbg('Writing output file: ', fn)
                    rfd['tbxmask'] = fn

            if run_d['tbx']:
                imatbx = tbx_nib.get_fdata()
                if lib_tool.check_dtype(imatbx, input_nib.dataobj.dtype):
                    imatbx  = imatbx.astype(input_nib.dataobj.dtype)
                    tbx_nib = nib.Nifti1Image(imatbx, tbx_nib.affine, tbx_nib.header)
                rd['tbx']  = tbx_nib
                if save_outputs:
                    fn = save_nib(tbx_nib, ftemplate, 'tbx')
                    _dbg('Writing output file: ', fn)
                    rfd['tbx'] = fn

            tbx_cache[f] = {
                'tbxmask_nib':     tbxmask_nib,
                'tbx111_crop':     tbx111_crop,
                'ftemplate':        ftemplate,
                'want_aseg':        run_d['aseg'],
                'want_tseg':        run_d['tseg'],
            }
        del tbx_session

        # Phase 2+: one session per model, loop over all files in chunk
        for key, runner in active_models:
            model_ff = lib_tool.get_model(omodel[key])
            session  = create_session(model_ff, args.gpu)
            _seg_pbar = tqdm(chunk, desc=key, unit='file', disable=(len(input_file_list) <= 1 or verbose >= 2))
            for f in _seg_pbar:
                _seg_pbar.set_postfix_str(os.path.basename(f))
                rd_upd, rfd_upd = runner(
                    key, session, model_ff, f, tbx_cache[f], args.gpu, run_d['patch'], save_outputs)
                result_accum[f][0].update(rd_upd)
                result_accum[f][1].update(rfd_upd)
            del session

        tbx_cache.clear()

    # ── return (format unchanged) ─────────────────────────────────────────────
    if len(input_file_list) == 1:
        return result_accum[input_file_list[0]][0]   # result_dict with nib objects
    return [result_accum[f][1] for f in input_file_list]  # list of result_filedict
