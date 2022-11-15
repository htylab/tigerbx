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


def produce_mask(model, f, f_output_dir, args,
                postfix, brainmask_nib=None, write=True):

    seg_mode = basename(model).split('_')[1]
    input_nib = nib.load(f)
    model_ff = lib_tool.get_model(model)
    input_data = lib_bx.read_file(model_ff, f)
    mask, _ = lib_bx.run(
        model_ff, input_data,  GPU=args.gpu)
    output_file, pred_nib = lib_bx.write_file(model_ff, f, f_output_dir,
                                              mask, postfix=postfix,
                                              inmem=True)

    if brainmask_nib is None:
        output = pred_nib.get_fdata()
    else:
        output = pred_nib.get_fdata() * brainmask_nib.get_fdata()
    output = output.astype(int)

    output_nib = nib.Nifti1Image(output, input_nib.affine, input_nib.header)
    if write:
        nib.save(output_nib, output_file)
    print('Writing output file: ', output_file)

    return pred_nib

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
        model_aseg = 'mprage_aseg43_v002_WangM1r256.onnx'
    model_dkt = 'mprage_dkt_v001_f16r256.onnx'
    model_dgm = 'mprage_dgm12_v001_wangM1V2.onnx'

    #if args.model is None:

    #else:
    #    model_aseg, model_name = args.model.split('*')
    

    print('Total nii files:', len(input_file_list))

    for f in input_file_list:

        print('Processing :', os.path.basename(f))
        t = time.time()

        f_output_dir = output_dir

        if f_output_dir is None:
            f_output_dir = os.path.dirname(os.path.abspath(f))
        else:
            os.makedirs(f_output_dir, exist_ok=True)
        
        tbetmask_nib = produce_mask(model_bet, f, f_output_dir, args,
                                    'tbetmask', write=get_m)
        if get_b:
            input_nib = nib.load(f)
            bet = input_nib.get_fdata() * tbetmask_nib.get_fdata()
            bet = bet.astype(input_nib.dataobj.dtype)

            bet = nib.Nifti1Image(bet, input_nib.affine, input_nib.header)

            output_file = basename(f).replace('.nii', f'_tbet.nii')
            output_file = join(f_output_dir, output_file)
            nib.save(bet, output_file)
            print('Writing output file: ', output_file)
            
        if get_a:
            produce_mask(model_aseg, f, f_output_dir,
             args, 'aseg', brainmask_nib=tbetmask_nib)
        if get_d:
            produce_mask(model_dgm, f, f_output_dir,
                         args, 'dgm', brainmask_nib=tbetmask_nib)
        if get_k:
            produce_mask(model_dkt, f, f_output_dir,
                         args, 'dkt', brainmask_nib=tbetmask_nib)

        print('Processing time: %d seconds' %  (time.time() - t))


if __name__ == "__main__":
    main()
    if platform.system() == 'Windows':
        os.system('pause')
