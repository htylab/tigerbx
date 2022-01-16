import sys
import os
import argparse
# import logging
from tigerseg import segment
# import SimpleITK as sitk
import pkg_resources
import numpy as np
from distutils.util import strtobool


def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')


def main():
    version = pkg_resources.require("tigerseg")[0].version

    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='input',type=path, help='Path to the input image, can be a folder for the specific format(nii.gz)')
    parser.add_argument('output', metavar='output', type=str, help='Filepath for output segmentation')
    parser.add_argument('--modelpath', default=os.getcwd(), type=str, help='spcifies the path to the trained model')
    parser.add_argument('--CPU',default='False',type = strtobool, help='Only use CPU or not')
    parser.add_argument('--onnx',default='True',type = strtobool, help='Use onnxruntime or tensorflow')
    parser.add_argument('--version', help="Shows the current version", action='version', version=version)
    args = parser.parse_args()

    if args.onnx == True:
        segment.onnx_apply(input=args.input,output=args.output,modelpath=args.modelpath,only_CPU=args.CPU)
    else:
        segment.apply(input=args.input,output=args.output,modelpath=args.modelpath,only_CPU=args.CPU)



if __name__ == "__main__":
    main()