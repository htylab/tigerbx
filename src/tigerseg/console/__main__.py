import sys
import os
import argparse
from .. import segment
from distutils.util import strtobool
import glob

def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', metavar='input_file',type=str, help='Path to the input image, can be a folder for the specific format(nii.gz)')
    parser.add_argument('output_dir', metavar='output_dir', type=path, help='Filepath for output segmentation')
    parser.add_argument('--model', default='', type=str, help='specifies the modelname')
    parser.add_argument('--GPU',default='False',type = strtobool, help='True: GPU, False: CPU, default: False, CPU')
    parser.add_argument('--report',default='True',type = strtobool, help='Produce additional reports')
    args = parser.parse_args()

    input_file_list = glob.glob(args.input_file)
    segment.apply_files(model_name=args.model,
                        input_file_list=input_file_list,
                        output_dir=args.output_dir,                        
                        GPU=args.GPU,
                        report=args.report)



if __name__ == "__main__":
    main()