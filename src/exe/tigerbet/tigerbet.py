import sys
import os
from os.path import basename, join, isdir
import argparse
import time
import tigerseg.segment
#import tigerseg.methods.mprage
from tigerseg.methods.mprage import read_file, write_file
from distutils.util import strtobool
import glob
import platform


def main():

    default_model = 'mprage_v0004_bet_full.onnx'    
    parser = argparse.ArgumentParser()
    parser.add_argument('input',  type=str, nargs='+', help='Path to the input image, can be a folder for the specific format(nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output segmentation, default: the directory of input files')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
    parser.add_argument('-m', '--mask', action='store_true', help='Produced mask')
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

    model_name = args.model    

    print('Total nii files:', len(input_file_list))

    for f in input_file_list:

        print('Processing :', os.path.basename(f))
        t = time.time()
            
        input_data = read_file(model_name, f)

        mask = tigerseg.segment.apply(model_name, input_data,  GPU=args.gpu)

        f_output_dir = output_dir

        if f_output_dir is None:
            f_output_dir = os.path.dirname(os.path.abspath(f))
        else:
            os.makedirs(f_output_dir, exist_ok=True)

        write_file(model_name, f, f_output_dir, input_data*mask,
            postfix='bet', dtype='orig')
        
        if args.mask:
            write_file(model_name, f, f_output_dir, mask, postfix='betmask')



        print('Processing time: %d seconds' %  (time.time() - t))




if __name__ == "__main__":
    main()
    if platform.system() == 'Windows':
        os.system('pause')