import sys
import os
from os.path import basename, join, isdir
import argparse
import time
import tigerseg.segment
import tigerseg.methods.mprage
import glob
import platform
import nibabel as nib



def main():

    default_model = 'mprage_v0004_bet_full.onnx'    
    parser = argparse.ArgumentParser()
    parser.add_argument('input',  type=str, nargs='+', help='Path to the input image, can be a folder for the specific format(nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output segmentation, default: the directory of input files')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
    parser.add_argument('-m', '--betmask', action='store_true', help='Producing bet mask')
    parser.add_argument('-a', '--aseg', action='store_true', help='Producing aseg mask')
    parser.add_argument('-b', '--bet', action='store_true', help='Producing bet mask')
    parser.add_argument('-d', '--deepgm', action='store_true',
                        help='Producing deepgm mask')
    parser.add_argument('-f', '--fast', action='store_true', help='Fast processing with low-resolution model')
    parser.add_argument('--model', default=default_model, type=str, help='Specifies the modelname')
    #parser.add_argument('--report',default='True',type = strtobool, help='Produce additional reports')
    args = parser.parse_args()

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
        model_name = 'mprage_v0002_bet_kuor128.onnx'
        model_aseg = 'mprage_v0001_aseg43_MXRWr128.onnx'
    else:
        model_name = args.model
        model_aseg = 'mprage_v0005_aseg43_full.onnx'
    

    print('Total nii files:', len(input_file_list))

    for f in input_file_list:

        print('Processing :', os.path.basename(f))
        t = time.time()
            
        input_data = tigerseg.methods.mprage.read_file(model_name, f)

        mask = tigerseg.segment.apply(model_name, input_data,  GPU=args.gpu)

        f_output_dir = output_dir

        if f_output_dir is None:
            f_output_dir = os.path.dirname(os.path.abspath(f))
        else:
            os.makedirs(f_output_dir, exist_ok=True)

        mask_file, mask_niimem = tigerseg.methods.mprage.write_file(model_name,
                                            f, f_output_dir, 
                                            mask, postfix='tbetmask', inmem=True)


        
        if args.bet:
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
            
             
        if args.betmask:
            nib.save(mask_niimem, mask_file)
            print('Writing output file: ', mask_file)

        if args.aseg or args.deepgm:
            input_nib = nib.load(f)
            input_data = tigerseg.methods.mprage.read_file(model_aseg, f)
            asegmask = tigerseg.segment.apply(
                model_aseg, input_data,  GPU=args.gpu)
            aseg_file, aseg_niimem = tigerseg.methods.mprage.write_file(model_aseg,
                                                                        f, f_output_dir,
                                                                        asegmask, postfix='aseg', inmem=True)
            aseg = aseg_niimem.get_fdata() * mask_niimem.get_fdata()
            aseg = aseg.astype(int)

            if args.aseg:
                asegnii = nib.Nifti1Image(aseg, input_nib.affine, input_nib.header)
                nib.save(asegnii, aseg_file)
                print('Writing output file: ', aseg_file)

            if args.deepgm:
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

            
            
            





        print('Processing time: %d seconds' %  (time.time() - t))




if __name__ == "__main__":
    main()
    if platform.system() == 'Windows':
        os.system('pause')
