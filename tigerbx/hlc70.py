import sys
import os
from os.path import basename, join, isdir, dirname, commonpath, relpath
import argparse
import time
import numpy as np

import glob
import platform
import nibabel as nib

from tigerbx import lib_tool
from tigerbx import lib_bx
from tigerbx.bx import produce_mask, save_nib, get_template
import copy
from nilearn.image import resample_to_img, reorder_img, resample_img
from itertools import product
import concurrent.futures
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    parser = argparse.ArgumentParser()
    setup_parser(parser)
    args = parser.parse_args()
    run_args(args)

def setup_parser(parser):
    #parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, nargs='+', help='Path to the input image(s); can be a folder containing images in the specific format (nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output segmentation (default: the directory of input files)')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
    parser.add_argument('--model', default=None, type=str, help='Specifying the model name')
    parser.add_argument('--save', default='all', type=str, help='Specifying the model name')
    parser.add_argument('-z', '--gz', action='store_true', help='Forcing storing in nii.gz format')
    parser.add_argument('-p', '--patch', action='store_true', help='patch inference')

    #args = parser.parse_args()
    #run_args(args)


def hlc(input=None, output=None, model=None, save_str='all', GPU=False, gz=True, patch=False):
    from argparse import Namespace
    args = Namespace()
    if not isinstance(input, list):
        input = [input]
    
    args.input = input
    args.output = output
    args.model = model
    args.gpu = GPU
    args.gz = gz
    args.save = save_str
    args.patch = patch
    return run_args(args)

def get_argmax(logits, start, end):
    import numpy as np
    from scipy.special import softmax
    logits_slice = logits[:, start:end, ...].squeeze(0)  # [C', D, H, W]
    
    softmax_output = softmax(logits_slice, axis=0)
    
    argmax_output = np.argmax(softmax_output, axis=0)
    
    return argmax_output

def HLC_decoder(out, lrseg, dwseg):
    # Label transformation dictionary with integer values
    reverse_transform = {
        0: 0, 11: 43, 12: 44, 13: 46, 14: 47, 15: 49,
        16: 50, 17: 51, 18: 52, 1: 14, 2: 15, 3: 16,
        19: 53, 20: 54, 4: 24, 21: 58, 22: 60, 23: 62,
        24: 63, 5: 85, 6: 251, 7: 252, 8: 253, 9: 254,
        10: 255, 52: 4002, 53: 4003, 25: 4005, 26: 4006,
        27: 4007, 28: 4008, 29: 4009, 30: 4010, 31: 4011,
        32: 4012, 33: 4013, 34: 4014, 35: 4015, 36: 4016,
        37: 4017, 38: 4018, 39: 4019, 40: 4020, 41: 4021,
        42: 4022, 43: 4023, 44: 4024, 45: 4025, 46: 4026,
        47: 4027, 48: 4028, 49: 4029, 50: 4030, 51: 4031,
        54: 4034, 55: 4035, 58: 4001, 56: 4032, 57: 4033
    }

    # Define index lists
    aseg_left_indexs = [4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 30, 31]
    aseg_right_indexs = [43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63]
    dkt_left_indexs = list(range(1005, 1032)) + [1002, 1003, 1034, 1035]
    wmp_left_indexs = list(range(3005, 3036)) + [3001, 3002, 3003]
    
    dkt_right_indexs = [i + 1000 for i in dkt_left_indexs]
    wmp_right_indexs = [i + 1000 for i in wmp_left_indexs]
    wmp_dkt_right_indexs = [i + 2000 for i in dkt_right_indexs]
    wmp_dkt_left_indexs = [i + 2000 for i in dkt_left_indexs]
    
    left_indexs = aseg_left_indexs + dkt_left_indexs + wmp_left_indexs
    right_indexs = aseg_right_indexs + dkt_right_indexs + wmp_right_indexs

    # Initialize output array
    reversed_out = np.zeros_like(out)

    # Apply basic transformation
    for key, value in reverse_transform.items():
        reversed_out[out == key] = value

    # Apply conditional transformations based on hemisphere and gray/white matter
    for label in np.unique(reversed_out):
        if label in right_indexs:
            idx = np.where(right_indexs == label)[0][0]
            mask = (reversed_out == label) & (lrseg == 1) & (dwseg == 0)
            reversed_out[mask] = left_indexs[idx]

        if label in wmp_dkt_right_indexs:
            idx = np.where(wmp_dkt_right_indexs == label)[0][0]
            # Left hemisphere, gray matter
            mask1 = (reversed_out == label) & (lrseg == 1) & (dwseg == 1)
            reversed_out[mask1] = dkt_left_indexs[idx]
            # Right hemisphere, gray matter
            mask2 = (reversed_out == label) & (lrseg == 2) & (dwseg == 1)
            reversed_out[mask2] = dkt_right_indexs[idx]
            # Left hemisphere, white matter parcellation
            mask3 = (reversed_out == label) & (lrseg == 1) & (dwseg == 2)
            reversed_out[mask3] = wmp_dkt_left_indexs[idx]

    return reversed_out


def run_args(args):

    run_d = vars(args) #store all arg in dict

    if args.save == 'all': args.save = 'cCbmh'
 
    input_file_list = args.input
    if os.path.isdir(args.input[0]):
        input_file_list = glob.glob(join(args.input[0], '*.nii'))
        input_file_list += glob.glob(join(args.input[0], '*.nii.gz'))

    elif '*' in args.input[0]:
        input_file_list = glob.glob(args.input[0])

    output_dir = args.output
    omodel = dict()
    omodel['bet'] = 'mprage_bet_v005_mixsynthv4.onnx'
    omodel['HLC'] = 'mprage_hlc_v001_init.onnx'
 
    # if you want to use other models
    if isinstance(args.model, dict):
        for mm in args.model.keys():
            omodel[mm] = args.model[mm]
    elif isinstance(args.model, str):
        import ast
        model_dict = ast.literal_eval(args.model)
        for mm in model_dict.keys():
            omodel[mm] = model_dict[mm]


    print('Total nii files:', len(input_file_list))

    #check duplicate basename
    #for detail, check get_template
    base_ffs = [basename(f) for f in input_file_list]
    common_folder = None
    if len(base_ffs) != len(set(base_ffs)):
        common_folder = commonpath(input_file_list)
        
    count = 0
    result_all = []
    result_filedict = dict()
    for f in input_file_list:
        count += 1
        result_dict = dict()
        result_filedict = dict()

        print(f'{count} Processing (2~5 minutes per MPRAGE):', os.path.basename(f))
        t = time.time()
        ftemplate, f_output_dir = get_template(f, output_dir, args.gz, common_folder)


        tbetmask_nib, qc_score = produce_mask(omodel['bet'], f, GPU=args.gpu, QC=True)
        input_nib = nib.load(f)
        tbet_nib = lib_bx.read_nib(input_nib) * lib_bx.read_nib(tbetmask_nib)

        if lib_tool.check_dtype(tbet_nib, input_nib.dataobj.dtype):
            tbet_nib = tbet_nib.astype(input_nib.dataobj.dtype)

        tbet_nib = nib.Nifti1Image(tbet_nib, input_nib.affine, input_nib.header)


        zoom = tbet_nib.header.get_zooms() 

        if max(zoom) > 1.1 or min(zoom) < 0.9:
            tbet_nib111 = lib_bx.resample_voxel(tbet_nib, (1, 1, 1),interpolation='continuous')
            tbet_nib111 = reorder_img(tbet_nib111, resample='continuous')
        else:
            tbet_nib111 = reorder_img(tbet_nib, resample='continuous')

        tbet_image = lib_bx.read_nib(tbet_nib111)
            
        image = tbet_image[None, ...][None, ...]
        image = image/np.max(image)
        model_ff = lib_tool.get_model(omodel['HLC'])
        if args.patch:
            logits = lib_tool.predict(model_ff, image, args.gpu, mode='patch')
        else:
            logits = lib_tool.predict(model_ff, image, args.gpu)


        if 'm' in args.save:            
            fn = save_nib(tbetmask_nib, ftemplate, 'tbetmask')
            result_dict['tbetmask'] = tbetmask_nib
            result_filedict['tbetmask'] = fn
        if 'b' in args.save:
            fn = save_nib(tbet_nib, ftemplate, 'tbet')
            result_dict['tbet'] = tbet_nib
            result_filedict['tbet'] = fn
        if 'h' in args.save:
            all_arg = get_argmax(logits, 0, 60)
            lr_arg = get_argmax(logits, 60, 63)
            dw_arg = get_argmax(logits, 63, 66)
            HLCparc = HLC_decoder(all_arg, lr_arg, dw_arg)
            hlc_nib = nib.Nifti1Image(HLCparc, tbet_nib111.affine, tbet_nib111.header)
            hlc_nib = resample_to_img(hlc_nib,
            input_nib, interpolation="nearest")
            fn = save_nib(hlc_nib, ftemplate, 'hlc')
            result_dict['hlc'] = hlc_nib
            result_filedict['hlc'] = fn

        if 'c' in args.save:
            ct = logits[0,66,...].squeeze()
            ct[ct < 0.2] = 0
            ct[ct > 5] = 5
            ct = ct * (tbet_image > 0)

            ct_nib = nib.Nifti1Image(ct, tbet_nib111.affine, tbet_nib111.header)
            ct_nib = resample_to_img(
                ct_nib, input_nib, interpolation="nearest")

            ct_nib.header.set_data_dtype(float)
            
            fn = save_nib(ct_nib, ftemplate, 'ct')
            result_dict['ct'] = ct_nib
            result_filedict['ct'] = fn

        if 'C' in args.save:
            cgw = logits[0,67:70,...].squeeze()
            for kk in range(3):
                pve = cgw[kk]
                pve = pve* (tbet_image>0)

                pve_nib = nib.Nifti1Image(pve, tbet_nib111.affine, tbet_nib111.header)
                pve_nib = resample_to_img(
                    pve_nib, input_nib, interpolation="linear")

                pve_nib.header.set_data_dtype(float)
                fn = save_nib(pve_nib, ftemplate, f'cgw{kk+1}')
                result_dict[f'cgw{kk+1}'] = pve_nib
                result_filedict[f'cgw{kk+1}'] = fn
     
        print('Processing time: %d seconds' %  (time.time() - t))
        if len(input_file_list) == 1:
            result_all = result_dict
        else:
            result_all.append(result_filedict) #storing output filenames
    return result_all


if __name__ == "__main__":
    main()
    if platform.system() == 'Windows':
        os.system('pause')

