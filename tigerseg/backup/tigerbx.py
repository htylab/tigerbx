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
        #model_aseg = 'mprage_aseg43_v002_WangM1r220.onnx'

    model_dkt = 'mprage_dkt_v001_f16r256.onnx'

    #if args.model is None:

    #else:
    #    model_aseg, model_name = args.model.split('*')
    

    print('Total nii files:', len(input_file_list))

    for f in input_file_list:

        print('Processing :', os.path.basename(f))
        t = time.time()         

        model_bet_ff = lib_tool.get_model(model_bet)

        input_data = lib_bx.read_file(model_bet_ff, f)
        mask, _ = lib_bx.run(
            model_bet_ff, input_data, GPU=args.gpu)

        f_output_dir = output_dir

        if f_output_dir is None:
            f_output_dir = os.path.dirname(os.path.abspath(f))
        else:
            os.makedirs(f_output_dir, exist_ok=True)

        mask_file, mask_niimem = lib_bx.write_file(model_bet_ff,
                                            f, f_output_dir, 
                                            mask, postfix='tbetmask', inmem=True)


        if get_b:
            input_nib = nib.load(f)
            bet = input_nib.get_fdata() * mask_niimem.get_fdata()
            bet = bet.astype(
                input_nib.dataobj.dtype)

            bet = nib.Nifti1Image(bet, input_nib.affine, input_nib.header)

            output_file = basename(f).replace(
                '.nii', f'_tbet.nii')
            output_file = join(f_output_dir, output_file)
            nib.save(bet, output_file)
            print('Writing output file: ', output_file)
            
        if get_m:
            nib.save(mask_niimem, mask_file)
            print('Writing output file: ', mask_file)

        if get_a or get_d:
            model_aseg = lib_tool.get_model(model_aseg)
            input_nib = nib.load(f)
            input_data = lib_bx.read_file(model_aseg, f)
            asegmask, _ = lib_bx.run(model_aseg, input_data, GPU=args.gpu)
            aseg_file, aseg_niimem = lib_bx.write_file(model_aseg, f, f_output_dir,
                                                                        asegmask, postfix='aseg', inmem=True)
            aseg = aseg_niimem.get_fdata() * mask_niimem.get_fdata()
            aseg = aseg.astype(int)

            if get_a:
                asegnii = nib.Nifti1Image(aseg, input_nib.affine, input_nib.header)
                nib.save(asegnii, aseg_file)
                print('Writing output file: ', aseg_file)

            if get_d:
                deepgm = aseg * 0
                count = 0
                for ii in [10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54]:
                    count += 1
                    deepgm[aseg==ii] = count

                deepgm = deepgm.astype(int)

                output_file = basename(f).replace(
                    '.nii', f'_deegm.nii')
                output_file = join(f_output_dir, output_file)
                deepgm = nib.Nifti1Image(
                    deepgm, input_nib.affine, input_nib.header)
                nib.save(deepgm, output_file)
                print('Writing output file: ', output_file)

        if get_k:
            input_nib = nib.load(f)
            model_dkt = lib_tool.get_model(model_dkt)
            input_data = lib_bx.read_file(model_dkt, f)
            dktmask, _ = lib_bx.run(
                model_dkt, input_data,  GPU=args.gpu)
            dkt_file, dkt_niimem = lib_bx.write_file(model_dkt, f, f_output_dir,
                                                    dktmask, postfix='dkt', inmem=True)
            dkt = dkt_niimem.get_fdata() * mask_niimem.get_fdata()
            dkt = dkt.astype(int)

            dktnii = nib.Nifti1Image(dkt, input_nib.affine, input_nib.header)
            nib.save(dktnii, dkt_file)
            print('Writing output file: ', dkt_file)

        print('Processing time: %d seconds' %  (time.time() - t))


if __name__ == "__main__":
    main()
    if platform.system() == 'Windows':
        os.system('pause')
