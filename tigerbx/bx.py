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


def _infer_mask(model_ff_list, f, GPU, patch, brainmask_nib=None, tbet111=None):
    """Shared inference core: load, run model, resample, apply brainmask, cast dtype.

    Returns (output_nib, mask_nib_resp, prob_resp).
    mask_nib_resp and prob_resp are in model space (before resampling to input space).
    """
    input_nib = nib.load(f)

    if tbet111 is None:
        input_nib_resp = lib_bx.read_file(model_ff_list[0], f)
    else:
        input_nib_resp = copy.deepcopy(tbet111)  # avoid modifying caller's copy

    mask_nib_resp, prob_resp = lib_bx.run(
        model_ff_list, input_nib_resp, GPU=GPU, patch=patch)

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


def produce_betmask(model, f, GPU=False, patch=False):
    """Run BET model and return (output_nib, qc_score, qc_raw).

    QC is always computed: qc_raw in [0, 1] based on mean binary entropy
    across all predicted brain voxels.

    qc_score is rescaled so that qc_raw >= _QC_RAW_GOOD (calibrated Dice >= 0.95
    boundary) displays as 100:
        qc_score = int(clip(qc_raw / _QC_RAW_GOOD * 100, 0, 100))
    """
    if not isinstance(model, list):
        model = [model]
    model_ff_list = [lib_tool.get_model(mm) for mm in model]

    output_nib, mask_nib_resp, prob_resp = _infer_mask(
        model_ff_list, f, GPU, patch)

    probmax     = np.max(prob_resp, axis=0)
    within_mask = lib_tool.read_nib(mask_nib_resp) > 0
    p           = probmax.clip(1e-7, 1 - 1e-7)
    entropy     = -(p * np.log(p) + (1 - p) * np.log(1 - p))  # binary entropy [0, ln2]
    # qc_raw: confidence score in [0, 1]
    # 1.0 = perfectly confident everywhere, 0.0 = maximum uncertainty (p=0.5 everywhere)
    # mean entropy across all brain voxels, normalised by ln2
    # use mean (not percentile) so interior voxels dominate over boundary voxels
    qc_raw   = 1.0 - float(np.mean(entropy[within_mask])) / np.log(2)
    qc_score = int(np.clip(qc_raw / _QC_RAW_GOOD * 100, 0, 100))

    return output_nib, qc_score, qc_raw


def produce_mask(model, f, GPU=False, brainmask_nib=None, tbet111=None, patch=False):
    """Run a segmentation model and return output_nib.

    brainmask_nib: BET mask applied to the output (zeros outside brain).
    tbet111:       Pre-computed BET brain used as model input instead of raw f.
    """
    if not isinstance(model, list):
        model = [model]
    model_ff_list = [lib_tool.get_model(mm) for mm in model]

    output_nib, _, _ = _infer_mask(
        model_ff_list, f, GPU, patch, brainmask_nib, tbet111)

    return output_nib

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


def run(argstring, input=None, output=None, model=None, verbose=0, silent=False):

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


def run_args(args):

    run_d = vars(args) #store all arg in dict

    verbose = run_d.get('verbose', 0)

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
        # Producing extracted brain by default
        
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
    omodel = dict()
    omodel['bet'] = 'mprage_bet_v005_mixsynthv4.onnx'
    omodel['aseg'] = 'mprage_aseg43_v007_16ksynth.onnx'
    omodel['ct'] = 'mprage_mix_ct.onnx'
    omodel['dgm'] = 'mprage_dgm12_v002_mix6.onnx'
    omodel['wmh'] = 'mprage_wmh_v002_betr111.onnx'
    omodel['cgw'] = 'mprage_cgw_v001_r111.onnx'
    omodel['syn'] = 'mprage_synthseg_v003_r111.onnx'

 
    # if you want to use other models
    if isinstance(args.model, dict):
        for mm in args.model.keys():
            omodel[mm] = args.model[mm]
    elif isinstance(args.model, str):
        import ast
        model_dict = ast.literal_eval(args.model)
        for mm in model_dict.keys():
            omodel[mm] = model_dict[mm]


    printer('Total nii files:', len(input_file_list))

    #check duplicate basename
    #for detail, check get_template
    base_ffs = [basename(f) for f in input_file_list]
    common_folder = None
    if len(base_ffs) != len(set(base_ffs)):
        common_folder = commonpath(input_file_list)
        
    result_all = []
    _pbar = tqdm(input_file_list, desc='tigerbx', unit='file', disable=(verbose > 0))
    for count, f in enumerate(_pbar, 1):
        result_dict = dict()
        result_filedict = dict()

        _pbar.set_postfix_str(os.path.basename(f))
        printer(f'{count} Processing :', os.path.basename(f))
        t = time.time()

        ftemplate, f_output_dir = get_template(f, output_dir, args.gz, common_folder)       

        tbetmask_nib, qc_score, qc_raw = produce_betmask(omodel['bet'], f, GPU=args.gpu)
        input_nib = nib.load(f)
        tbet_nib = lib_tool.read_nib(input_nib) * lib_tool.read_nib(tbetmask_nib)

        tbet_nib = nib.Nifti1Image(tbet_nib, input_nib.affine, input_nib.header)

        _needs_111 = any(run_d[k] for k in
                         ['aseg', 'dgm', 'wmh', 'syn', 'cgw', 'ct'])
        tbet_nib111 = None
        tbet_seg = None
        tbet_nib111_crop = None
        tbet_seg_crop = None
        if _needs_111:
            tbet_nib111 = lib_tool.resample_voxel(tbet_nib, (1, 1, 1), interpolation='continuous')
            tbet_nib111 = reorder_img(tbet_nib111, resample='continuous')
            zoom = tbet_nib.header.get_zooms()
            if max(zoom) > 1.1 or min(zoom) < 0.9:
                tbet_seg = tbet_nib111
            else:
                tbet_seg = reorder_img(tbet_nib, resample='continuous')

            # Crop to brain ROI to save memory and speed up inference
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

        result_dict['QC'] = qc_score
        result_dict['QC_raw'] = qc_raw
        result_filedict['QC'] = qc_score
        if qc_score < 50:
            _warn(f'Low QC score ({qc_score}) for {os.path.basename(f)} — check result carefully.')
        if run_d['qc'] or qc_score < 50:
            qcfile = ftemplate.replace('.nii','').replace('.gz', '')
            qcfile = qcfile.replace('@@@@', f'qc-{qc_score}.log')
            with open(qcfile, 'a') as the_file:
                the_file.write(f'QC: {qc_score} \n')
            _dbg('Writing output file: ', qcfile)

        if run_d['betmask']:
            fn = save_nib(tbetmask_nib, ftemplate, 'tbetmask')
            _dbg('Writing output file: ', fn)
            result_dict['tbetmask'] = tbetmask_nib
            result_filedict['tbetmask'] = fn

        if run_d['bet']:

            imabet = tbet_nib.get_fdata()
            if lib_tool.check_dtype(imabet, input_nib.dataobj.dtype):
                imabet = imabet.astype(input_nib.dataobj.dtype)
                tbet_nib = nib.Nifti1Image(imabet, tbet_nib.affine, tbet_nib.header)

            fn = save_nib(tbet_nib, ftemplate, 'tbet')
            _dbg('Writing output file: ', fn)
            result_dict['tbet'] = tbet_nib
            result_filedict['tbet'] = fn
        
        for seg_str in ['aseg', 'dgm', 'wmh', 'syn']:
            if run_d[seg_str]:
                result_nib = produce_mask(omodel[seg_str], f, GPU=args.gpu,
                                         brainmask_nib=tbetmask_nib, tbet111=tbet_seg_crop, patch=run_d['patch'])
                
                fn = save_nib(result_nib, ftemplate, seg_str)
                _dbg('Writing output file: ', fn)
                result_filedict[seg_str] = fn
                result_dict[seg_str] = result_nib

        if run_d['cgw']: # FSL style segmentation of CSF, GM, WM
            model_ff = lib_tool.get_model(omodel['cgw'])
            normalize_factor = np.max(input_nib.get_fdata())
            bet_img = lib_tool.read_nib(tbet_nib111_crop)

            image = bet_img[None, ...][None, ...]
            image = image/normalize_factor
            cgw = lib_tool.predict(model_ff, image, args.gpu)[0]

            result_dict['cgw'] = []
            result_filedict['cgw'] = []
            for kk in [1, 2, 3]:
                pve = cgw[kk]
                pve = pve * (bet_img > 0)

                pve_nib = nib.Nifti1Image(pve, tbet_nib111_crop.affine)
                pve_nib = resample_to_img(
                    pve_nib, input_nib, interpolation="linear")

                pve_nib.header.set_data_dtype(float)

                fn = save_nib(pve_nib, ftemplate, f'cgw_pve{kk-1}')
                _dbg('Writing output file: ', fn)
                result_filedict['cgw'].append(fn)
                result_dict['cgw'].append(pve_nib)

        if run_d['ct']:
            model_ff = lib_tool.get_model(omodel['ct'])
            bet_img = lib_tool.read_nib(tbet_nib111_crop)
            image = bet_img[None, ...][None, ...]
            mx = np.max(image)
            if mx > 0:
                image = image / mx
            ct = lib_tool.predict(model_ff, image, args.gpu)[0, 0, ...]

            ct[ct < 0.2] = 0
            ct[ct > 5] = 5
            ct = ct * (bet_img > 0).astype(int)

            ct_nib = nib.Nifti1Image(ct, tbet_nib111_crop.affine)
            ct_nib = resample_to_img(
                ct_nib, input_nib, interpolation="nearest")

            ct_nib.header.set_data_dtype(float)
            
            fn = save_nib(ct_nib, ftemplate, 'ct')
            _dbg('Writing output file: ', fn)
            result_dict['ct'] = ct_nib
            result_filedict['ct'] = fn
            
               
        printer('Processing time: %d seconds' %  (time.time() - t))
        if len(input_file_list) == 1:
            result_all = result_dict
        else:
            result_all.append(result_filedict) #storing output filenames
    return result_all


