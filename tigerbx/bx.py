import sys
import os
from os.path import basename, join, isdir
import argparse
import time
import numpy as np

import glob
import platform
import nibabel as nib

from tigerbx import lib_tool
from tigerbx import lib_bx

from nilearn.image import resample_to_img, reorder_img

def produce_mask(model, f, GPU=False, brainmask_nib=None):

    model_ff = lib_tool.get_model(model)
    input_nib = nib.load(f)
    input_nib_resp = lib_bx.read_file(model_ff, f)
    mask_nib_resp, prob_resp = lib_bx.run(
        model_ff, input_nib_resp,  GPU=GPU)

    mask_nib = resample_to_img(
        mask_nib_resp, input_nib, interpolation="nearest")

    if brainmask_nib is None:
        output = mask_nib.get_fdata()
    else:
        output = mask_nib.get_fdata() * brainmask_nib.get_fdata()
    output = output.astype(int)

    output_nib = nib.Nifti1Image(output, input_nib.affine, input_nib.header)

    return output_nib

def save_nib(data_nib, ftemplate, postfix):
    output_file = ftemplate.replace('@@@@', postfix)
    nib.save(data_nib, output_file)
    print('Writing output file: ', output_file)


def main():
      
    parser = argparse.ArgumentParser()
    parser.add_argument('input',  type=str, nargs='+', help='Path to the input image, can be a folder for the specific format(nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output segmentation, default: the directory of input files')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
    parser.add_argument('-m', '--betmask', action='store_true', help='Producing bet mask')
    parser.add_argument('-a', '--aseg', action='store_true', help='Producing aseg mask')
    parser.add_argument('-b', '--bet', action='store_true', help='Producing bet images')
    parser.add_argument('-d', '--deepgm', action='store_true',
                        help='Producing deepgm mask')
    parser.add_argument('-k', '--dkt', action='store_true',
                        help='Producing dkt mask')
    parser.add_argument('-c', '--ct', action='store_true',
                        help='Producing cortical thickness map')
    parser.add_argument('-w', '--wmp', action='store_true',
                        help='Producing white matter parcellation')
    parser.add_argument('-f', '--fast', action='store_true', help='Fast processing with low-resolution model')
    parser.add_argument('--model', default=None, type=str, help='Specifies the modelname')
    #parser.add_argument('--report',default='True',type = strtobool, help='Produce additional reports')
    args = parser.parse_args()
    run_args(args)

def run(argstring, input, output=None, model=None):

    from argparse import Namespace
    args = Namespace()

    args.betmask = 'm' in argstring
    args.aseg = 'a' in argstring
    args.bet = 'b' in argstring
    args.deepgm = 'd' in argstring
    args.dkt = 'k' in argstring
    args.fast = 'f' in argstring
    args.gpu = 'g' in argstring
    args.ct = 'c' in argstring
    args.wmp = 'w' in argstring

    if not isinstance(input, list):
        input = [input]
    args.input = input
    args.output = output
    args.model = model
    run_args(args)   


def run_args(args):

    get_m = args.betmask
    get_a = args.aseg
    get_b = args.bet
    get_d = args.deepgm
    get_k = args.dkt
    get_c = args.ct
    get_w = args.wmp

    if True not in [get_m, get_a, get_b, get_d, get_k, get_c, get_w]:
        get_b = True
        # Producing extracted brain by default 

    input_file_list = args.input
    if os.path.isdir(args.input[0]):
        input_file_list = glob.glob(join(args.input[0], '*.nii'))
        input_file_list += glob.glob(join(args.input[0], '*.nii.gz'))

    elif '*' in args.input[0]:
        input_file_list = glob.glob(args.input[0])

    output_dir = args.output

    #model_name = args.model
    #model_aseg = 'mprage_v0006_aseg43_full.onnx'

    default_model = dict()
    default_model['bet128'] = 'mprage_bet_v001_kuor128.onnx'
    default_model['aseg128'] = 'mprage_aseg43_v001_MXRWr128.onnx'

    default_model['bet'] = 'mprage_bet_v002_full.onnx'
    default_model['aseg'] = 'mprage_aseg43_v005_crop.onnx'
    #default_model['dkt'] = 'mprage_dkt_v001_f16r256.onnx'
    default_model['dkt'] = 'mprage_dkt_v002_train.onnx'
    default_model['dktc'] = 'mprage_dktc_v004_3k.onnx'
    
    #default_model['dgm'] = 'mprage_aseg43_v005_crop.onnx'
    default_model['dgm'] = 'mprage_dgm12_v002_mix6.onnx'
    default_model['wmp'] = 'mprage_wmp_v002_14k.onnx'


    # if you want to use other models
    if isinstance(args.model, dict):
        for mm in args.model.keys():
            default_model[mm] = args.model[mm]
    elif isinstance(args.model, str):
        import ast
        model_dict = ast.literal_eval(args.model)
        for mm in model_dict.keys():
            default_model[mm] = model_dict[mm]

    if args.fast:
        model_bet = default_model['bet128']
        model_aseg = default_model['aseg128']
        
    else:
        model_bet = default_model['bet']
        #model_aseg = 'mprage_v0006_aseg43_full.onnx'
        #model_aseg = 'mprage_aseg43_v002_WangM1r256.onnx'
        model_aseg = default_model['aseg']
    model_dkt = default_model['dkt']
    model_dgm = default_model['dgm']
    model_dktc = default_model['dktc']
    model_wmp = default_model['wmp']



    print('Total nii files:', len(input_file_list))

    for f in input_file_list:

        print('Processing :', os.path.basename(f))
        t = time.time()

        f_output_dir = output_dir

        if f_output_dir is None:
            f_output_dir = os.path.dirname(os.path.abspath(f))
        else:
            os.makedirs(f_output_dir, exist_ok=True)


        ftemplate = basename(f).replace('.nii', f'_@@@@.nii')
        ftemplate = join(f_output_dir, ftemplate)

        
        tbetmask_nib = produce_mask(model_bet, f, GPU=args.gpu)
        if get_m:
            save_nib(tbetmask_nib, ftemplate, 'tbetmask')

        if get_b:
            input_nib = nib.load(f)
            bet = input_nib.get_fdata() * tbetmask_nib.get_fdata()
            bet = bet.astype(input_nib.dataobj.dtype)


            bet = nib.Nifti1Image(bet, input_nib.affine,
                                  input_nib.header)

            save_nib(bet, ftemplate, 'tbet')
        

        if get_a:
            aseg_nib = produce_mask(model_aseg, f, GPU=args.gpu,
                                    brainmask_nib=tbetmask_nib)
            save_nib(aseg_nib, ftemplate, 'aseg')


        if get_d:
            if 'aseg' in model_dgm:
                if 'aseg_nib' not in locals():
                    aseg_nib = produce_mask(model_aseg, f, GPU=args.gpu,
                        brainmask_nib=tbetmask_nib)
                aseg = aseg_nib.get_fdata()
                deepgm = aseg * 0
                count = 0
                for ii in [10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54]:
                    count += 1
                    deepgm[aseg==ii] = count


                dgm_nib = nib.Nifti1Image(deepgm.astype(int),
                                            input_nib.affine, input_nib.header)
            else:

                dgm_nib = produce_mask(model_dgm, f, GPU=args.gpu,
                        brainmask_nib=tbetmask_nib)


            save_nib(dgm_nib, ftemplate, 'dgm')

        if get_k:
            dkt_nib = produce_mask(model_dkt, f, GPU=args.gpu,
                                    brainmask_nib=tbetmask_nib)
 
            save_nib(dkt_nib, ftemplate, 'dkt')
        
        if get_w:
            wmp_nib = produce_mask(model_wmp, f, GPU=args.gpu,
                                    brainmask_nib=tbetmask_nib)
 
            save_nib(wmp_nib, ftemplate, 'wmp')

        if get_c:

            brain_mask = tbetmask_nib.get_fdata()
            input_nib = nib.load(f)
            
            model_ff = lib_tool.get_model(model_dktc)
            vol_nib = lib_bx.resample_voxel(input_nib, (1, 1, 1))
            vol_nib = reorder_img(vol_nib, resample='continuous')

            data = vol_nib.get_fdata()
            image = data[None, ...][None, ...]
            image = image/np.max(image)

            logits = lib_tool.predict(model_ff, image, args.gpu)[0, ...]

            #mask_dkt = np.argmax(logits[:-1, ...], axis=0)
            ct = logits[-1, ...]
            

            ct_nib = nib.Nifti1Image(ct, vol_nib.affine, vol_nib.header)
            ct_nib = resample_to_img(
                ct_nib, input_nib, interpolation="continuous")

            ct = ct_nib.get_fdata() * brain_mask
            ct[ct < 0] = 0

            ct_nib = nib.Nifti1Image(ct,
                                     ct_nib.affine, ct_nib.header)
            
            save_nib(ct_nib, ftemplate, 'ct')

        print('Processing time: %d seconds' %  (time.time() - t))


if __name__ == "__main__":
    main()
    if platform.system() == 'Windows':
        os.system('pause')

