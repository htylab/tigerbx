import os
from os.path import basename, join, isdir, dirname, commonpath
import time
import logging
import numpy as np

import glob
import platform
import nibabel as nib
from tqdm import tqdm

from tigerbx import lib_tool
from tigerbx.bx import produce_betmask, save_nib, get_template
from nilearn.image import resample_to_img, reorder_img

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

from tigerbx.lib_crop import crop_cube, restore_result



def get_argmax(logits, start, end):
    import numpy as np
    from scipy.special import softmax
    logits_slice = logits[:, start:end, ...].squeeze(0)  # [C', D, H, W]
    
    softmax_output = softmax(logits_slice, axis=0)
    
    argmax_output = np.argmax(softmax_output, axis=0)
    
    return argmax_output

#import numpy as np


ASEG_NLR = np.array([14, 15, 16, 24, 251, 252, 253, 254, 255], dtype=np.int32)
ASEG_LEFT = np.array([4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28], dtype=np.int32)
ASEG_RIGHT = np.array([43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60], dtype=np.int32)

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
LEFT = np.concatenate([ASEG_LEFT, DKT_LEFT, WMP_LEFT])
RIGHT = np.concatenate([ASEG_RIGHT, DKT_RIGHT, WMP_RIGHT])

# Encoding lookup table
import numpy as np

# Encoding lookup table
ENCODE= {}
DECODE = {}

# 1. ASEG_NLR mapping
for idx, value in enumerate(ASEG_NLR, start=1):
    ENCODE[value] = idx
    DECODE[idx] = value
    

# 2. ASEG left/right mapping
max_value = max(ENCODE.values())  # current maximum value
for idx, (left, right) in enumerate(zip(ASEG_LEFT, ASEG_RIGHT), start=max_value + 1):
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

    for idx in ASEG_NLR:
        output[reversed_out == idx] = idx
    
    # Handle hemisphere and tissue conditions
    for idx, (left_idx, right_idx) in enumerate(zip(ASEG_LEFT, ASEG_RIGHT)):
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

    if args.save == 'all': args.save = 'mbhtcgw'

    verbose = getattr(args, 'verbose', 0)

    def printer(*msg):
        if verbose >= 1:
            _logger.info(' '.join(str(x) for x in msg))

    def _dbg(*msg):
        if verbose >= 2:
            _logger.debug(' '.join(str(x) for x in msg))

    input_file_list = args.input
    if os.path.isdir(args.input[0]):
        input_file_list = glob.glob(join(args.input[0], '*.nii'))
        input_file_list += glob.glob(join(args.input[0], '*.nii.gz'))

    elif '*' in args.input[0]:
        input_file_list = glob.glob(args.input[0])

    output_dir = args.output
    omodel = dict()
    omodel['bet'] = 'mprage_bet_v005_mixsynthv4.onnx'
    omodel['HLC'] = 'mprage_hlc_v004_resunetplusT2.onnx'
 
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
    result_filedict = dict()
    _pbar = tqdm(input_file_list, desc='tigerbx-hlc', unit='file', disable=(verbose > 0))
    for count, f in enumerate(_pbar, 1):
        result_dict = dict()
        result_filedict = dict()

        _pbar.set_postfix_str(os.path.basename(f))
        printer(f'{count} Preprocessing :', os.path.basename(f))
        t = time.time()
        ftemplate, f_output_dir = get_template(f, output_dir, args.gz, common_folder)


        tbetmask_nib, qc_score, _ = produce_betmask(omodel['bet'], f, GPU=args.gpu)
        input_nib = nib.load(f)
        tbet_nib = lib_tool.read_nib(input_nib) * lib_tool.read_nib(tbetmask_nib)

        tbet_nib = nib.Nifti1Image(tbet_nib, input_nib.affine, input_nib.header)


        zoom = tbet_nib.header.get_zooms() 

        if max(zoom) > 1.1 or min(zoom) < 0.9:
            tbet_nib111 = lib_tool.resample_voxel(tbet_nib, (1, 1, 1),interpolation='continuous')
            tbet_nib111 = reorder_img(tbet_nib111, resample='continuous')
        else:
            tbet_nib111 = reorder_img(tbet_nib, resample='continuous')

        tbetmask_nib111 =  resample_to_img(tbetmask_nib, tbet_nib111, interpolation="nearest")

        #print(tbet_nib111.shape, tbet_nib111.get_fdata().dtype, tbet_nib.get_fdata().dtype)

        tbet_image = lib_tool.read_nib(tbet_nib111)
        tbetmask_image = lib_tool.read_nib(tbetmask_nib111)
            
        image_orig = tbet_image

        image, xyz6 = crop_cube(image_orig, tbetmask_image, min_size=(160, 160, 160) if args.patch else None)

        mx = np.max(image)
        if mx > 0:
            image = image / mx
        image = image[None, ...][None, ...]
        model_ff = lib_tool.get_model(omodel['HLC'])
        printer('Perform HLC model....')
        if args.patch:
            logits = lib_tool.predict(model_ff, image, args.gpu, mode='patch')
        else:
            logits = lib_tool.predict(model_ff, image, args.gpu)


        if 'm' in args.save:            
            fn = save_nib(tbetmask_nib, ftemplate, 'tbetmask')
            result_dict['tbetmask'] = tbetmask_nib
            result_filedict['tbetmask'] = fn
        if 'b' in args.save:
            imabet = tbet_nib.get_fdata()
            if lib_tool.check_dtype(imabet, input_nib.dataobj.dtype):
                imabet = imabet.astype(input_nib.dataobj.dtype)

            tbet_nib = nib.Nifti1Image(imabet, input_nib.affine, input_nib.header)
        
            fn = save_nib(tbet_nib, ftemplate, 'tbet')
            result_dict['tbet'] = tbet_nib
            result_filedict['tbet'] = fn
        if 'h' in args.save:
            all_arg = get_argmax(logits, 0, 57)
            lr_arg = get_argmax(logits, 57, 60)
            dw_arg = get_argmax(logits, 60, 63)
            HLCparc = HLC_decoder(all_arg, lr_arg, dw_arg)

            HLCparc = restore_result(image_orig.shape, HLCparc, xyz6).astype(int)

            hlc_nib = nib.Nifti1Image(HLCparc, tbet_nib111.affine, tbet_nib111.header)
            hlc_nib = resample_to_img(hlc_nib,
            input_nib, interpolation="nearest")
            fn = save_nib(hlc_nib, ftemplate, 'hlc')
            result_dict['hlc'] = hlc_nib
            result_filedict['hlc'] = fn

        if 't' in args.save: #thickness
            ct = logits[0,63,...].squeeze()
            ct[ct < 0.2] = 0
            ct[ct > 5] = 5

            ct = restore_result(image_orig.shape, ct, xyz6)

            ct = ct * (tbet_image > 0)

            
            ct_nib = nib.Nifti1Image(ct, tbet_nib111.affine, tbet_nib111.header)
            ct_nib = resample_to_img(
                ct_nib, input_nib, interpolation="nearest")

            ct_nib.header.set_data_dtype(float)
            
            fn = save_nib(ct_nib, ftemplate, 'ct')
            result_dict['ct'] = ct_nib
            result_filedict['ct'] = fn

        
        cgw = logits[0,64:67,...].squeeze()
        cgwnames = ['CSF', 'GM', 'WM']
        cgw_short = ['c', 'g', 'w']
        for kk in range(3):
            if cgw_short[kk] not in args.save: continue
            pve = cgw[kk]
            pve = restore_result(image_orig.shape, pve, xyz6)
            pve[pve<0.05] = 0
            pve = np.clip(pve* (tbet_image>0), 0, 1)


            pve_nib = nib.Nifti1Image(pve, tbet_nib111.affine, tbet_nib111.header)
            pve_nib = resample_to_img(
                pve_nib, input_nib, interpolation="linear")

            pve_nib.header.set_data_dtype(float)
            fn = save_nib(pve_nib, ftemplate, f'{cgwnames[kk]}')
            result_dict[f'{cgwnames[kk]}'] = pve_nib
            result_filedict[f'{cgwnames[kk]}'] = fn
    
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
