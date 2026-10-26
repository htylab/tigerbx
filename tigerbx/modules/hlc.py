import os
from os.path import basename, join, isdir, dirname
import time
import logging
import numpy as np

import platform
import nibabel as nib
from tqdm import tqdm

from tigerbx import lib_tool
from tigerbx.modules.bx import produce_tbxmask
from tigerbx.core.io import get_template, load_nifti, save_nib, resolve_inputs
from tigerbx.core.onnx import create_session, predict
from tigerbx.core.resample import resample_to_img, reorder_img, resample_voxel
from tigerbx.core.spatial import crop_cube, restore_result

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

_logger = logging.getLogger('tigerbx')

def hlc(input=None, output=None, model=None, save='h', GPU=False, gz=True, patch=False, verbose=0):
    
    from types import SimpleNamespace as Namespace
    args = Namespace()
    if not isinstance(input, list):
        input = [input]
    args.input = input
    args.output = output
    args.model = model
    args.gpu = GPU
    args.save = save
    args.patch = patch
    args.gz = gz
    args.verbose = verbose
    return run_args(args)

# Outputs of an HLC model optimised with tigerbx.dev.onnx_optimize --heads
# parc:0:57:argmax,lr:57:60:argmax,dw:60:63:argmax,ct:63:64:slice,cgw:64:67:slice. Without them (or with patch
# inference, which blends logits) the 67-channel logits are used. The ct and cgw heads are not used (thickness:
# tigerbx.run('c'); CSF / GM / WM maps from TSEG: core.cgw).
HLC_HEADS = ('parc', 'lr', 'dw', 'ct', 'cgw')


def _hlc_outputs(session, model_ff, image, gpu, patch, save):
    """The HLC results ``save`` needs: parc / lr / dw labels [X, Y, Z]."""
    want = []
    if 'h' in save:
        want += ['parc', 'lr', 'dw']
    if not want:
        return {}
    if not patch and set(HLC_HEADS) <= {o.name for o in session.get_outputs()}:
        out = session.run(want, {session.get_inputs()[0].name: image.astype(np.float32)})
        return {k: v[0] for k, v in zip(want, out)}
    logits = predict(model_ff, image, gpu, mode='patch' if patch else None, session=session)
    res = {}
    if 'parc' in want:
        res.update(parc=get_argmax(logits, 0, 57), lr=get_argmax(logits, 57, 60), dw=get_argmax(logits, 60, 63))
    return res


def get_argmax(logits, start, end):
    """Label index of channels [start, end) per voxel (softmax does not change the argmax, so it is skipped)."""
    import numpy as np
    return np.argmax(logits[0, start:end], axis=0)

#import numpy as np


TSEG_NLR = np.array([14, 15, 16, 24, 251, 252, 253, 254, 255], dtype=np.int32)
TSEG_LEFT = np.array([4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28], dtype=np.int32)
TSEG_RIGHT = np.array([43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60], dtype=np.int32)

base_index = np.array([ii for ii in range(1, 36) if ii !=4], dtype=np.int32)

DKT_LEFT = base_index + 1000
DKT_RIGHT = base_index + 2000
WMP_LEFT = base_index + 3000
WMP_RIGHT = base_index + 4000

WMP_LEFT = np.append(WMP_LEFT, 5001)
WMP_RIGHT = np.append(WMP_RIGHT, 5002)


DKT = np.concatenate([DKT_LEFT, DKT_RIGHT])
#WMP_DKT = np.concatenate([DKT_LEFT + 2000, DKT_RIGHT + 2000])
WMP = np.concatenate([WMP_LEFT, WMP_RIGHT])
#WMP_ONLY = np.array([ 5001, 5002], dtype=np.int32)

# Explicitly define LEFT and RIGHT
LEFT = np.concatenate([TSEG_LEFT, DKT_LEFT, WMP_LEFT])
RIGHT = np.concatenate([TSEG_RIGHT, DKT_RIGHT, WMP_RIGHT])

# Encoding lookup table
import numpy as np

# Encoding lookup table
ENCODE= {}
DECODE = {}

# 1. TSEG_NLR mapping
for idx, value in enumerate(TSEG_NLR, start=1):
    ENCODE[value] = idx
    DECODE[idx] = value
    

# 2. TSEG left/right mapping
max_value = max(ENCODE.values())  # current maximum value
for idx, (left, right) in enumerate(zip(TSEG_LEFT, TSEG_RIGHT), start=max_value + 1):
    ENCODE[left] = idx
    ENCODE[right] = idx
    DECODE[idx] = left

# 3. DKT and WMP mapping
max_value = max(ENCODE.values())  # update max value
for ii in base_index:
    max_value += 1
    ENCODE[ii + 1000] = max_value  # DKT_LEFT
    ENCODE[ii + 2000] = max_value  # DKT_RIGHT
    ENCODE[ii + 3000] = max_value  # WMP_LEFT
    ENCODE[ii + 4000] = max_value  # WMP_RIGHT
    DECODE[max_value] = ii + 1000

max_value += 1
# 4. Final mapping pair
ENCODE[5001] = max_value
ENCODE[5002] = max_value

DECODE[max_value] = 5001

DESIRED_LABELS = set(ENCODE.keys())

def HLC_encoder(mask):
    # Ensure input data type is int32
    mask = mask.astype(np.int32)
    
    # Initialize output arrays
    out = np.zeros_like(mask, dtype=np.int32)
    lrseg = np.zeros_like(mask, dtype=np.int8)
    dwseg = np.zeros_like(mask, dtype=np.int8)
    
    # Build encoding lookup table
    max_label = max(ENCODE.keys()) + 1
    encode_lookup = np.zeros(max_label, dtype=np.int32)
    for key, value in ENCODE.items():
        encode_lookup[key] = value
    
    # Apply encoding
    out = encode_lookup[mask]
    
    # Generate lrseg (left/right hemisphere)
    lrseg[np.isin(mask, LEFT)] = 1
    lrseg[np.isin(mask, RIGHT)] = 2
    
    # Generate dwseg (gray/white matter)
    dwseg[np.isin(mask, DKT)] = 1
    dwseg[np.isin(mask, WMP)] = 2
    #dwseg[np.isin(mask, WMP_ONLY)] = 2
    
    return out, lrseg, dwseg

def HLC_decoder(out, lrseg, dwseg):

    # Ensure input data types
    out = out.astype(np.int32)
    lrseg = lrseg.astype(np.int32)
    dwseg = dwseg.astype(np.int32)
    gray_mask = (dwseg == 1)
    white_mask = (dwseg == 2)
    left_mask = (lrseg==1)
    right_mask = (lrseg ==2)

    output = np.zeros_like(lrseg)

    
    # Build decoding lookup table
    lookup = np.zeros(max(DECODE.keys()) + 1, dtype=np.int32)
    for key, value in DECODE.items():
        lookup[key] = value
    
    # Apply basic decoding
    reversed_out = lookup[out]

    for idx in TSEG_NLR:
        output[reversed_out == idx] = idx
    
    # Handle hemisphere and tissue conditions
    for idx, (left_idx, right_idx) in enumerate(zip(TSEG_LEFT, TSEG_RIGHT)):
        mask_temp = (reversed_out == left_idx) & left_mask
        output[mask_temp] = left_idx
        mask_temp = (reversed_out == left_idx) & right_mask
        output[mask_temp] = right_idx

    dkt_mask = (reversed_out > 1000) & (reversed_out < 5000)
    dkt_value = reversed_out * dkt_mask
    output += dkt_value * gray_mask * left_mask
    output += (dkt_value + 1000) * gray_mask * right_mask * dkt_mask
    output += (dkt_value + 2000) * white_mask * left_mask * dkt_mask
    output += (dkt_value + 3000) * white_mask * right_mask * dkt_mask
    mask_temp = (reversed_out == 5001) & left_mask
    output[mask_temp] = 5001
    mask_temp = (reversed_out == 5001) & right_mask
    output[mask_temp] = 5002
    return output

def run_args(args):

    if args.save == 'all': args.save = 'mbh'
    if any(c in args.save for c in 'cgw'):
        raise ValueError("tigerbx.hlc: the CSF / GM / WM outputs (save 'c', 'g', 'w') have been removed; "
                         "use tigerbx.seg(save_tissue=True) (CSF / GM / WM maps from TSEG, core.cgw).")
    if 't' in args.save:
        raise ValueError("tigerbx.hlc: the cortical thickness output (save 't') has been removed; "
                         "use tigerbx.run('c') (the CT model).")

    verbose = getattr(args, 'verbose', 0)

    def printer(*msg):
        if verbose >= 1:
            _logger.info(' '.join(str(x) for x in msg))

    def _dbg(*msg):
        if verbose >= 2:
            _logger.debug(' '.join(str(x) for x in msg))

    input_file_list, common_folder = resolve_inputs(args.input)

    output_dir = args.output
    omodel = dict()
    omodel['tbx'] = 'mprage_tbx_v005_opt.onnx'
    omodel['HLC'] = 'mprage_hlc_v004_opt.onnx'

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

    # one session per model for the whole batch
    tbx_session = create_session(lib_tool.get_model(omodel['tbx']), args.gpu)
    hlc_model_ff = None
    hlc_session = None
    if 'h' in args.save:
        hlc_model_ff = lib_tool.get_model(omodel['HLC'])
        hlc_session = create_session(hlc_model_ff, args.gpu)

    result_all = []
    result_filedict = dict()
    _pbar = tqdm(input_file_list, desc='tigerbx-hlc', unit='file', disable=(verbose > 0))
    for count, f in enumerate(_pbar, 1):
        result_dict = dict()
        result_filedict = dict()

        _pbar.set_postfix_str(os.path.basename(f))
        printer(f'{count} Preprocessing :', os.path.basename(f))
        t = time.time()
        ftemplate, f_output_dir = get_template(f, output_dir, args.gz, common_folder)


        tbxmask_nib = produce_tbxmask(omodel['tbx'], f, GPU=args.gpu, session=tbx_session)
        input_nib = load_nifti(f)
        tbx_nib = lib_tool.read_nib(input_nib) * lib_tool.read_nib(tbxmask_nib)

        tbx_nib = nib.Nifti1Image(tbx_nib, input_nib.affine, input_nib.header)


        zoom = tbx_nib.header.get_zooms() 

        if max(zoom) > 1.1 or min(zoom) < 0.9:
            tbx_nib111 = resample_voxel(tbx_nib, (1, 1, 1),interpolation='continuous')
            tbx_nib111 = reorder_img(tbx_nib111, resample='continuous')
        else:
            tbx_nib111 = reorder_img(tbx_nib, resample='continuous')

        tbxmask_nib111 =  resample_to_img(tbxmask_nib, tbx_nib111, interpolation="nearest")

        #print(tbx_nib111.shape, tbx_nib111.get_fdata().dtype, tbx_nib.get_fdata().dtype)

        tbx_image = lib_tool.read_nib(tbx_nib111)
        tbxmask_image = lib_tool.read_nib(tbxmask_nib111)
            
        image_orig = tbx_image

        image, xyz6 = crop_cube(image_orig, tbxmask_image, min_size=(160, 160, 160) if args.patch else None)

        mx = np.max(image)
        if mx > 0:
            image = image / mx
        image = image[None, ...][None, ...]
        printer('Perform HLC model....')
        heads = _hlc_outputs(hlc_session, hlc_model_ff, image, args.gpu, args.patch, args.save) if hlc_session else {}


        if 'm' in args.save:            
            fn = save_nib(tbxmask_nib, ftemplate, 'tbxmask')
            result_dict['tbxmask'] = tbxmask_nib
            result_filedict['tbxmask'] = fn
        if 'b' in args.save:
            imatbx = tbx_nib.get_fdata()
            if lib_tool.check_dtype(imatbx, input_nib.dataobj.dtype):
                imatbx = imatbx.astype(input_nib.dataobj.dtype)

            tbx_nib = nib.Nifti1Image(imatbx, input_nib.affine, input_nib.header)
        
            fn = save_nib(tbx_nib, ftemplate, 'tbx')
            result_dict['tbx'] = tbx_nib
            result_filedict['tbx'] = fn
        if 'h' in args.save:
            HLCparc = HLC_decoder(heads['parc'], heads['lr'], heads['dw'])

            HLCparc = restore_result(image_orig.shape, HLCparc, xyz6).astype(int)

            hlc_nib = nib.Nifti1Image(HLCparc, tbx_nib111.affine, tbx_nib111.header)
            hlc_nib = resample_to_img(hlc_nib,
            input_nib, interpolation="nearest")
            # labels go up to 5002: never inherit a uint8 input dtype (nibabel would then scale and merge labels)
            hlc_nib = nib.Nifti1Image(np.asarray(hlc_nib.dataobj).astype(np.int16), hlc_nib.affine, hlc_nib.header)
            hlc_nib.header.set_data_dtype(np.int16)
            fn = save_nib(hlc_nib, ftemplate, 'hlc')
            result_dict['hlc'] = hlc_nib
            result_filedict['hlc'] = fn

        printer('Processing time: %d seconds' %  (time.time() - t))
        if len(input_file_list) == 1:
            result_all = result_dict
        else:
            result_all.append(result_filedict) #storing output filenames
    return result_all

'''
if __name__ == "__main__":
    main()
    if platform.system() == 'Windows':
        os.system('pause')
'''
