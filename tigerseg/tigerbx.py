import sys
import os
from os.path import basename, join, isdir
import argparse
import time

import glob
import platform
import nibabel as nib

from tigerseg import lib_tool
from tigerseg import lib_bx

from nilearn.image import resample_to_img

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
    parser.add_argument('-f', '--fast', action='store_true', help='Fast processing with low-resolution model')
    #parser.add_argument('--model', default=None, type=str, help='Specifies the modelname')
    #parser.add_argument('--report',default='True',type = strtobool, help='Produce additional reports')
    args = parser.parse_args()

    get_m = args.betmask
    get_a = args.aseg
    get_b = args.bet
    get_d = args.deepgm
    get_k = args.dkt

    if True not in [get_m, get_a, get_b, get_d, get_k]:
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

    if args.fast:
        model_bet = 'mprage_bet_v001_kuor128.onnx'
        model_aseg = 'mprage_aseg43_v001_MXRWr128.onnx'
        
    else:
        model_bet = 'mprage_bet_v002_full.onnx'
        #model_aseg = 'mprage_v0006_aseg43_full.onnx'
        #model_aseg = 'mprage_aseg43_v002_WangM1r256.onnx'
        model_aseg = 'mprage_aseg43_v005_crop.onnx'
    model_dkt = 'mprage_dkt_v001_f16r256.onnx'
    model_dgm = 'mprage_dgm12_v001_wangM1V2.onnx'



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
            save_nib(tbetmask_nib, ftemplate,'tbetmask')

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
            aseg = aseg_nib.get_fdata()
            deepgm = aseg * 0
            count = 0
            for ii in [10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54]:
                count += 1
                deepgm[aseg==ii] = count


            dgm_nib = nib.Nifti1Image(deepgm.astype(int),
                                         input_nib.affine, input_nib.header)

            save_nib(dgm_nib, ftemplate, 'dgm')

        if get_k:
            dkt_nib = produce_mask(model_dkt, f, GPU=args.gpu,
                                    brainmask_nib=tbetmask_nib)
 
            save_nib(dkt_nib, ftemplate, 'dkt')



        print('Processing time: %d seconds' %  (time.time() - t))


if __name__ == "__main__":
    main()
    if platform.system() == 'Windows':
        os.system('pause')

