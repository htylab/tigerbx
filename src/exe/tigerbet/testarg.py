import sys
import os
import argparse
import time
#import tigerseg.segment
#import tigerseg.methods.mprage
from distutils.util import strtobool
#import glob

def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')


def main():

    default_model = 'mprage_v0004_bet_full.onnx'    
    parser = argparse.ArgumentParser()
    parser.add_argument('input',  type=str, help='Path to the input image, can be a folder for the specific format(nii.gz)')
    parser.add_argument('output', type=str, help='Filepath for output segmentation, default: the directory of input files')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
    parser.add_argument('-m', '--mask', action='store_true', help='Produced mask')
    parser.add_argument('--model', default=default_model, type=str, help='Specifies the modelname')
    #parser.add_argument('--report',default='True',type = strtobool, help='Produce additional reports')
    args = parser.parse_args()
    

    print(args)


if __name__ == "__main__":
    main()
    os.system('pause')