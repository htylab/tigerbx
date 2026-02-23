import sys
import os
from os.path import basename, join, isdir, dirname, commonpath, relpath
import time
import logging
import numpy as np

import glob
import nibabel as nib
from tqdm import tqdm

from tigerbx import lib_tool
from tigerbx import lib_bx
from tigerbx.lib_crop import crop_cube
import copy
from nilearn.image import resample_to_img, reorder_img
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

_logger = logging.getLogger('tigerbx')
_logger.addHandler(logging.NullHandler())


# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(os.path.abspath(__file__))

# Calibrated QC threshold: qc_raw at which BET Dice ≈ 0.95 (from qc_stat calibration).
# qc_raw values at or above this level are considered "good" and displayed as 100.
_QC_RAW_GOOD = 0.7581


def _crop_nib(nib_img, xyz6):
    """Crop a nibabel image to xyz6 bounds and correct the affine origin."""
    x_min, x_max, y_min, y_max, z_min, z_max = xyz6
    data = nib_img.get_fdata()
    cropped = data[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
    new_affine = nib_img.affine.copy()
    new_affine[:3, 3] = nib_img.affine[:3, :3] @ np.array([x_min, y_min, z_min]) + nib_img.affine[:3, 3]
    return nib.Nifti1Image(cropped, new_affine)


def _infer_mask(model_ff, f, GPU, patch, brainmask_nib=None, tbet111=None, session=None):
    """Shared inference core: load, run model, resample, apply brainmask, cast dtype.

    Returns (output_nib, mask_nib_resp, prob_resp).
    mask_nib_resp and prob_resp are in model space (before resampling to input space).
    """
    input_nib = nib.load(f)

    if tbet111 is None:
        input_nib_resp = lib_bx.read_file(model_ff, f)
    else:
        input_nib_resp = copy.deepcopy(tbet111)  # avoid modifying caller's copy

    mask_nib_resp, prob_resp = lib_bx.run(
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

    return output_nib, mask_nib_resp, prob_resp


def produce_betmask(model, f, GPU=False, patch=False, session=None):
    """Run BET model and return (output_nib, qc_score, qc_raw).

    QC is always computed: qc_raw in [0, 1] based on mean binary entropy
    across all predicted brain voxels.

    qc_score is rescaled so that qc_raw >= _QC_RAW_GOOD (calibrated Dice >= 0.95
    boundary) displays as 100:
        qc_score = int(clip(qc_raw / _QC_RAW_GOOD * 100, 0, 100))
    """
    model_ff = lib_tool.get_model(model)

    output_nib, mask_nib_resp, prob_resp = _infer_mask(
        model_ff, f, GPU, patch, session=session)

    probmax     = np.max(prob_resp, axis=0)
    within_mask = lib_tool.read_nib(mask_nib_resp) > 0
    p           = probmax.clip(1e-7, 1 - 1e-7)
    entropy     = -(p * np.log(p) + (1 - p) * np.log(1 - p))  # binary entropy [0, ln2]
    qc_raw   = 1.0 - float(np.mean(entropy[within_mask])) / np.log(2)
    qc_score = int(np.clip(qc_raw / _QC_RAW_GOOD * 100, 0, 100))

    return output_nib, qc_score, qc_raw


def produce_mask(model, f, GPU=False, brainmask_nib=None, tbet111=None,
                 patch=False, session=None):
    """Run a segmentation model and return output_nib.

    brainmask_nib: BET mask applied to the output (zeros outside brain).
    tbet111:       Pre-computed BET brain used as model input instead of raw f.
    """
    model_ff = lib_tool.get_model(model)

    output_nib, _, _ = _infer_mask(
        model_ff, f, GPU, patch, brainmask_nib, tbet111, session=session)

    return output_nib


def _all_outputs_exist(f, output_dir, run_d, gz):
    """Return True if all requested outputs for file f already exist on disk."""
    ftemplate, _ = get_template(f, output_dir, gz)
    checks = {
        'betmask': 'tbetmask',
        'bet':     'tbet',
        'aseg':    'aseg',
        'dgm':     'dgm',
        'wmh':     'wmh',
        'syn':     'syn',
        'cgw':     'cgw_pve0',
        'ct':      'ct',
    }
    for flag, postfix in checks.items():
        if run_d.get(flag):
            if not os.path.isfile(ftemplate.replace('@@@@', postfix)):
                return False
    return True

def save_nib(data_nib, ftemplate, postfix):
    output_file = ftemplate.replace('@@@@', postfix)
    nib.save(data_nib, output_file)
    
    return output_file

def get_template(f, output_dir, get_z, common_folder=None):
    f_output_dir = output_dir
    ftemplate = basename(f).replace('.nii', f'_@@@@.nii').replace('.npz', f'_@@@@.nii.gz')

    if f_output_dir is None: #save the results in the same dir of T1_raw.nii.gz
        f_output_dir = os.path.dirname(os.path.abspath(f))
        
    else:
        os.makedirs(f_output_dir, exist_ok=True)
        #ftemplate = basename(f).replace('.nii', f'_@@@@.nii')
        # When we save results in the same directory, sometimes the result
        # filenames will all be the same, e.g., aseg.nii.gz, aseg.nii.gz.
        # In this case, the program tries to add a header to it.
        # For example, IXI001_aseg.nii.gz.
        if common_folder is not None:
            header = relpath(dirname(f), common_folder).replace(os.sep, '_')
            ftemplate = header + '_' + ftemplate
    
    if get_z and '.gz' not in ftemplate:
        ftemplate += '.gz'
    ftemplate = join(f_output_dir, ftemplate)

    return ftemplate, f_output_dir


def run(argstring, input=None, output=None, model=None, verbose=0,
        chunk_size=50, continue_=False, silent=False):

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

    if args.clean_onnx:
        argstring = ''
    args.betmask = 'm' in argstring
    args.aseg = 'a' in argstring
    args.bet = 'b' in argstring
    args.ct = 'c' in argstring
    args.cgw = 'C' in argstring
    args.dgm = 'd' in argstring
    args.wmh = 'W' in argstring
    args.syn = 'S' in argstring
    args.qc = 'q' in argstring
    args.gz = 'z' in argstring
    args.patch = 'p' in argstring
    return run_args(args)


# ── per-model inference helpers ───────────────────────────────────────────────

def _run_seg(key, session, model_ff, f, cache, GPU, patch):
    result_nib = produce_mask(model_ff, f, GPU=GPU,
                              brainmask_nib=cache['tbetmask_nib'],
                              tbet111=cache['tbet_seg_crop'],
                              patch=patch, session=session)
    fn = save_nib(result_nib, cache['ftemplate'], key)
    _logger.debug('Writing output file: %s', fn)
    return {key: result_nib}, {key: fn}


def _run_cgw(key, session, model_ff, f, cache, GPU, patch):
    input_nib = nib.load(f)
    tbet_nib111_crop = cache['tbet_nib111_crop']
    normalize_factor = np.max(input_nib.get_fdata())
    bet_img = lib_tool.read_nib(tbet_nib111_crop)
    image = bet_img[None, ...][None, ...] / normalize_factor
    cgw = lib_tool.predict(model_ff, image, GPU, session=session)[0]
    rd, rfd = {'cgw': []}, {'cgw': []}
    for kk in [1, 2, 3]:
        pve = cgw[kk] * (bet_img > 0)
        pve_nib = nib.Nifti1Image(pve, tbet_nib111_crop.affine)
        pve_nib = resample_to_img(pve_nib, input_nib, interpolation="linear")
        pve_nib.header.set_data_dtype(float)
        fn = save_nib(pve_nib, cache['ftemplate'], f'cgw_pve{kk-1}')
        _logger.debug('Writing output file: %s', fn)
        rfd['cgw'].append(fn)
        rd['cgw'].append(pve_nib)
    return rd, rfd


def _run_ct(key, session, model_ff, f, cache, GPU, patch):
    input_nib = nib.load(f)
    tbet_nib111_crop = cache['tbet_nib111_crop']
    bet_img = lib_tool.read_nib(tbet_nib111_crop)
    image = bet_img[None, ...][None, ...]
    mx = np.max(image)
    if mx > 0:
        image = image / mx
    ct = lib_tool.predict(model_ff, image, GPU, session=session)[0, 0, ...]
    ct[ct < 0.2] = 0
    ct[ct > 5] = 5
    ct = ct * (bet_img > 0).astype(int)
    ct_nib = nib.Nifti1Image(ct, tbet_nib111_crop.affine)
    ct_nib = resample_to_img(ct_nib, input_nib, interpolation="nearest")
    ct_nib.header.set_data_dtype(float)
    fn = save_nib(ct_nib, cache['ftemplate'], 'ct')
    _logger.debug('Writing output file: %s', fn)
    return {'ct': ct_nib}, {'ct': fn}


_MODEL_RUNNERS = {
    'aseg': _run_seg,
    'dgm':  _run_seg,
    'wmh':  _run_seg,
    'syn':  _run_seg,
    'cgw':  _run_cgw,
    'ct':   _run_ct,
}

_SEG_ORDER = ['aseg', 'dgm', 'wmh', 'syn', 'cgw', 'ct']


# ── main entry point ──────────────────────────────────────────────────────────

def run_args(args):

    run_d      = vars(args)
    verbose    = run_d.get('verbose', 0)
    chunk_size = run_d.get('chunk_size', 50)

    def printer(*msg):
        if verbose >= 1:
            _logger.info(' '.join(str(x) for x in msg))

    def _dbg(*msg):
        if verbose >= 2:
            _logger.debug(' '.join(str(x) for x in msg))

    def _warn(*msg):
        _logger.warning(' '.join(str(x) for x in msg))

    if True not in [run_d['betmask'], run_d['aseg'], run_d['bet'], run_d['dgm'],
                    run_d['ct'], run_d['qc'], run_d['wmh'],
                    run_d['cgw'], run_d['syn'], run_d['patch']]:
        run_d['bet'] = True

    if run_d['clean_onnx']:
        lib_tool.clean_onnx()
        printer('Exiting...')
        return 1

    input_file_list = args.input
    if os.path.isdir(args.input[0]):
        input_file_list = glob.glob(join(args.input[0], '*.nii'))
        input_file_list += glob.glob(join(args.input[0], '*.nii.gz'))
    elif '*' in args.input[0]:
        input_file_list = glob.glob(args.input[0])

    output_dir = args.output
    omodel = {
        'bet':  'mprage_bet_v005_mixsynthv4.onnx',
        'aseg': 'mprage_aseg43_v007_16ksynth.onnx',
        'ct':   'mprage_mix_ct.onnx',
        'dgm':  'mprage_dgm12_v002_mix6.onnx',
        'wmh':  'mprage_wmh_v002_betr111.onnx',
        'cgw':  'mprage_cgw_v001_r111.onnx',
        'syn':  'mprage_synthseg_v003_r111.onnx',
    }

    if isinstance(args.model, dict):
        for mm in args.model.keys():
            omodel[mm] = args.model[mm]
    elif isinstance(args.model, str):
        import ast
        model_dict = ast.literal_eval(args.model)
        for mm in model_dict.keys():
            omodel[mm] = model_dict[mm]

    # check duplicate basename
    base_ffs = [basename(f) for f in input_file_list]
    common_folder = None
    if len(base_ffs) != len(set(base_ffs)):
        common_folder = commonpath(input_file_list)

    # --continue: skip files whose outputs already exist
    if run_d.get('continue_'):
        original_n = len(input_file_list)
        input_file_list = [f for f in input_file_list
                           if not _all_outputs_exist(f, output_dir, run_d, args.gz)]
        printer(f'Skipping {original_n - len(input_file_list)} already-processed files')

    printer('Total nii files:', len(input_file_list))

    if not input_file_list:
        return []

    _needs_111 = any(run_d.get(k) for k in ['aseg', 'dgm', 'wmh', 'syn', 'cgw', 'ct'])
    active_models = [(k, _MODEL_RUNNERS[k]) for k in _SEG_ORDER if run_d.get(k)]

    result_accum = {f: [{}, {}] for f in input_file_list}
    bet_model_ff = lib_tool.get_model(omodel['bet'])

    # ── Chunked loop ──────────────────────────────────────────────────────────
    for chunk_start in range(0, len(input_file_list), chunk_size):
        chunk = input_file_list[chunk_start : chunk_start + chunk_size]
        bet_cache = {}

        # Phase 1: BET (all files in chunk, one shared session)
        bet_session = lib_tool.create_session(bet_model_ff, args.gpu)
        _pbar = tqdm(chunk, desc='tigerbx BET', unit='file', disable=(verbose > 0))
        for count, f in enumerate(_pbar, chunk_start + 1):
            _pbar.set_postfix_str(os.path.basename(f))
            printer(f'{count} Processing: {os.path.basename(f)}')

            ftemplate, _ = get_template(f, output_dir, args.gz, common_folder)

            tbetmask_nib, qc_score, qc_raw = produce_betmask(
                omodel['bet'], f, GPU=args.gpu, patch=run_d['patch'],
                session=bet_session)

            input_nib = nib.load(f)
            tbet_arr = lib_tool.read_nib(input_nib) * lib_tool.read_nib(tbetmask_nib)
            tbet_nib = nib.Nifti1Image(tbet_arr, input_nib.affine, input_nib.header)

            tbet_nib111_crop = tbet_seg_crop = None
            if _needs_111:
                tbet_nib111 = lib_tool.resample_voxel(tbet_nib, (1, 1, 1),
                                                       interpolation='continuous')
                tbet_nib111 = reorder_img(tbet_nib111, resample='continuous')
                zoom = tbet_nib.header.get_zooms()
                if max(zoom) > 1.1 or min(zoom) < 0.9:
                    tbet_seg = tbet_nib111
                else:
                    tbet_seg = reorder_img(tbet_nib, resample='continuous')
                arr_111 = tbet_nib111.get_fdata()
                _, xyz6_111 = crop_cube(arr_111, arr_111 > 0)
                tbet_nib111_crop = _crop_nib(tbet_nib111, xyz6_111)
                if tbet_seg is tbet_nib111:
                    tbet_seg_crop = tbet_nib111_crop
                else:
                    arr_seg = tbet_seg.get_fdata()
                    _, xyz6_seg = crop_cube(arr_seg, arr_seg > 0)
                    tbet_seg_crop = _crop_nib(tbet_seg, xyz6_seg)

            printer('QC score:', qc_score)
            rd  = result_accum[f][0]
            rfd = result_accum[f][1]
            rd['QC'] = qc_score
            rd['QC_raw'] = qc_raw
            rfd['QC'] = qc_score

            if qc_score < 50:
                _warn(f'Low QC score ({qc_score}) for {os.path.basename(f)}'
                      ' — check result carefully.')
            if run_d['qc'] or qc_score < 50:
                qcfile = ftemplate.replace('.nii', '').replace('.gz', '')
                qcfile = qcfile.replace('@@@@', f'qc-{qc_score}.log')
                with open(qcfile, 'a') as the_file:
                    the_file.write(f'QC: {qc_score} \n')
                _dbg('Writing output file: ', qcfile)

            if run_d['betmask']:
                fn = save_nib(tbetmask_nib, ftemplate, 'tbetmask')
                _dbg('Writing output file: ', fn)
                rd['tbetmask']  = tbetmask_nib
                rfd['tbetmask'] = fn

            if run_d['bet']:
                imabet = tbet_nib.get_fdata()
                if lib_tool.check_dtype(imabet, input_nib.dataobj.dtype):
                    imabet  = imabet.astype(input_nib.dataobj.dtype)
                    tbet_nib = nib.Nifti1Image(imabet, tbet_nib.affine, tbet_nib.header)
                fn = save_nib(tbet_nib, ftemplate, 'tbet')
                _dbg('Writing output file: ', fn)
                rd['tbet']  = tbet_nib
                rfd['tbet'] = fn

            bet_cache[f] = {
                'tbetmask_nib':     tbetmask_nib,
                'tbet_seg_crop':    tbet_seg_crop,
                'tbet_nib111_crop': tbet_nib111_crop,
                'ftemplate':        ftemplate,
            }
        del bet_session

        # Phase 2+: one session per model, loop over all files in chunk
        for key, runner in active_models:
            model_ff = lib_tool.get_model(omodel[key])
            session  = lib_tool.create_session(model_ff, args.gpu)
            printer(f'Running {key} for {len(chunk)} files...')
            for f in chunk:
                rd_upd, rfd_upd = runner(
                    key, session, model_ff, f, bet_cache[f], args.gpu, run_d['patch'])
                result_accum[f][0].update(rd_upd)
                result_accum[f][1].update(rfd_upd)
            del session

        bet_cache.clear()

    # ── return (format unchanged) ─────────────────────────────────────────────
    if len(input_file_list) == 1:
        return result_accum[input_file_list[0]][0]   # result_dict with nib objects
    return [result_accum[f][1] for f in input_file_list]  # list of result_filedict


