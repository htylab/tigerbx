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

def produce_mask(model, f, GPU=False, brainmask_nib=None, QC=False):

    model_ff = lib_tool.get_model(model)
    input_nib = nib.load(f)
    input_nib_resp = lib_bx.read_file(model_ff, f)
    mask_nib_resp, prob_resp = lib_bx.run(
        model_ff, input_nib_resp,  GPU=GPU)

    mask_nib = resample_to_img(
        mask_nib_resp, input_nib, interpolation="nearest")

    if brainmask_nib is None:

        output = lib_bx.read_nib(mask_nib)
    else:
        output = lib_bx.read_nib(mask_nib) * lib_bx.read_nib(brainmask_nib)
    output = output.astype(int)

    output_nib = nib.Nifti1Image(output, input_nib.affine, input_nib.header)
    output_nib.header.set_data_dtype(int)

    if QC:
        probmax = np.max(prob_resp, axis=0)
        qc_score = np.percentile(
            probmax[lib_bx.read_nib(mask_nib_resp) > 0], 1) - 0.5
        #qc = np.percentile(probmax, 1) - 0.5
        qc_score = int(qc_score * 1000)
        return output_nib, qc_score
    else:

        return output_nib

def save_nib(data_nib, ftemplate, postfix):
    output_file = ftemplate.replace('@@@@', postfix)
    nib.save(data_nib, output_file)
    print('Writing output file: ', output_file)
    return output_file


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
    parser.add_argument('-s', '--seg3', action='store_true',
                        help='Producing GM, WM, CSF segmentation (working in progress)')
    parser.add_argument('-q', '--qc', action='store_true',
                        help='Save QC score. Pay attention to the results with QC scores less than 50.')
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
    args.seg3 = 's' in argstring
    args.qc = 'q' in argstring

    if not isinstance(input, list):
        input = [input]
    args.input = input
    args.output = output
    args.model = model
    return run_args(args)   


def run_args(args):

    get_m = args.betmask
    get_a = args.aseg
    get_b = args.bet
    get_d = args.deepgm
    get_k = args.dkt
    get_c = args.ct
    get_w = args.wmp
    get_s = args.seg3
    get_q = args.qc

    if True not in [get_m, get_a, get_b, get_d,
                    get_k, get_c, get_w, get_s, get_q]:
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

    #default_model['bet'] = 'mprage_bet_v002_full.onnx'
    default_model['bet'] = 'mprage_bet_v004_anisofocal.onnx'
    default_model['aseg'] = 'mprage_aseg43_v005_crop.onnx'
    #default_model['dkt'] = 'mprage_dkt_v001_f16r256.onnx'
    default_model['dkt'] = 'mprage_dkt_v002_train.onnx'
    #default_model['dktc'] = 'mprage_dktc_v004_3k.onnx'
    default_model['ct'] = 'mprage_ct_v003_14k.onnx'

       
    #default_model['dgm'] = 'mprage_aseg43_v005_crop.onnx'
    default_model['dgm'] = 'mprage_dgm12_v002_mix6.onnx'
    default_model['wmp'] = 'mprage_wmp_v003_14k8.onnx'
    default_model['seg3'] = 'mprage_seg3_v001_qc2r128.onnx'


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
    #model_dktc = default_model['dktc']
    model_ct = default_model['ct']
    model_wmp = default_model['wmp']
    model_seg3 = default_model['seg3']



    print('Total nii files:', len(input_file_list))
    count = 0
    result_all = []
    result_filedict = dict()
    for f in input_file_list:
        count += 1

        result_dict = dict()
        result_filedict = dict()

        print(f'{count} Processing :', os.path.basename(f))
        t = time.time()

        f_output_dir = output_dir

        if f_output_dir is None:
            f_output_dir = os.path.dirname(os.path.abspath(f))
        else:
            os.makedirs(f_output_dir, exist_ok=True)


        ftemplate = basename(f).replace('.nii', f'_@@@@.nii')
        ftemplate = join(f_output_dir, ftemplate)

        
        tbetmask_nib, qc_score = produce_mask(model_bet, f, GPU=args.gpu, QC=True)
        print('QC score:', qc_score)

        result_dict['QC'] = qc_score
        result_filedict['QC'] = qc_score
        if qc_score < 50:
            print('Pay attention to the result with QC < 50. ')
        if get_q or qc_score < 50:
            qcfile = basename(f).replace('.nii','').replace('.gz', '') + f'-qc-{qc_score}.log'
            qcfile = join(f_output_dir, qcfile)
            with open(qcfile, 'a') as the_file:
                the_file.write(f'QC: {qc_score} \n')

        if get_m:
            fn = save_nib(tbetmask_nib, ftemplate, 'tbetmask')
            result_dict['tbetmask'] = tbetmask_nib
            result_filedict['tbetmask'] = fn

        if get_b:
            input_nib = nib.load(f)

            bet = lib_bx.read_nib(input_nib) * lib_bx.read_nib(tbetmask_nib)
            bet = bet.astype(input_nib.dataobj.dtype)


            bet = nib.Nifti1Image(bet, input_nib.affine,
                                  input_nib.header)

            fn = save_nib(bet, ftemplate, 'tbet')
            result_dict['tbet'] = bet
            result_filedict['tbet'] = fn
        

        if get_a:
            aseg_nib = produce_mask(model_aseg, f, GPU=args.gpu,
                                    brainmask_nib=tbetmask_nib)
            fn = save_nib(aseg_nib, ftemplate, 'aseg')
            result_dict['aseg'] = aseg_nib
            result_filedict['aseg'] = fn


        if get_d:
            if 'aseg' in model_dgm:
                if 'aseg_nib' not in locals():
                    aseg_nib = produce_mask(model_aseg, f, GPU=args.gpu,
                        brainmask_nib=tbetmask_nib)
                aseg = lib_bx.read_nib(aseg_nib)
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


            fn = save_nib(dgm_nib, ftemplate, 'dgm')
            result_dict['dgm'] = dgm_nib
            result_filedict['dgm'] = fn

        if get_k:
            dkt_nib = produce_mask(model_dkt, f, GPU=args.gpu,
                                    brainmask_nib=tbetmask_nib)
 
            fn = save_nib(dkt_nib, ftemplate, 'dkt')
            result_dict['dkt'] = dkt_nib
            result_filedict['dkt'] = fn
        
        if get_w:
            wmp_nib = produce_mask(model_wmp, f, GPU=args.gpu,
                                    brainmask_nib=tbetmask_nib)
 
            fn = save_nib(wmp_nib, ftemplate, 'wmp')
            result_dict['wmp'] = wmp_nib
            result_filedict['wmp'] = fn

        if get_s:
            seg3_nib = produce_mask(model_seg3, f, GPU=args.gpu,
                                   brainmask_nib=tbetmask_nib)

            fn = save_nib(seg3_nib, ftemplate, 'seg3')
            result_dict['seg3'] = seg3_nib
            result_filedict['seg3'] = fn

        if get_c:

            brain_mask = lib_bx.read_nib(tbetmask_nib)
            input_nib = nib.load(f)
            
            model_ff = lib_tool.get_model(model_ct)
            vol_nib = lib_bx.resample_voxel(input_nib, (1, 1, 1))
            vol_nib = reorder_img(vol_nib, resample='continuous')

            data = lib_bx.read_nib(vol_nib)
            image = data[None, ...][None, ...]
            image = image/np.max(image)

            #logits = lib_tool.predict(model_ff, image, args.gpu)[0, ...]
            #ct = logits[-1, ...]

            ct = lib_tool.predict(model_ff, image, args.gpu)[0, 0, ...]

            #mask_dkt = np.argmax(logits[:-1, ...], axis=0)
            
            

            ct_nib = nib.Nifti1Image(ct, vol_nib.affine, vol_nib.header)
            ct_nib = resample_to_img(
                ct_nib, input_nib, interpolation="continuous")

            ct = lib_bx.read_nib(ct_nib) * brain_mask
            ct[ct < 0] = 0
            ct[ct > 5] = 5

            ct_nib = nib.Nifti1Image(ct,
                                     ct_nib.affine, ct_nib.header)
            ct_nib.header.set_data_dtype(float)
            
            fn = save_nib(ct_nib, ftemplate, 'ct')
            result_dict['ct'] = ct_nib
            result_filedict['ct'] = fn

        print('Processing time: %d seconds' %  (time.time() - t))
        if len(input_file_list) == 1:
            result_all = result_dict
        elif len(input_file_list) < 20: 
            #maximum length of result_all set to 20 to reduce memory consumption

            result_all.append(result_dict)
        else:
            result_all.append(result_filedict) #storing output filenames
    return result_all


if __name__ == "__main__":
    main()
    if platform.system() == 'Windows':
        os.system('pause')

