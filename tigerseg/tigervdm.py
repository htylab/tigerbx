import sys
import os
from os.path import join
import argparse
from distutils.util import strtobool
import glob
from scipy.io import savemat
import nibabel as nib
import numpy as np
import time
from . import lib_vdm as vdm
from . import lib_tigertool as tigertool


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input',  type=str, nargs='+', help='Path to the input image, can be a folder for the specific format(nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output segmentation, default: the directory of input files')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')

    args = parser.parse_args() 

    input_file_list = args.input
    if os.path.isdir(args.input[0]):
        input_file_list = glob.glob(join(args.input[0], '*.nii'))
        input_file_list += glob.glob(join(args.input[0], '*.nii.gz'))

    elif '*' in args.input[0]:
        input_file_list = glob.glob(args.input[0])

    output_dir = args.output

    print('Total nii files:', len(input_file_list))

    model_name = tigertool.get_model('vdm_3dunet_v0001_orig')

    for f in input_file_list:

        print('Predicting:', f)
        t = time.time()
        input_data = vdm.read_file(model_name, f)
        vdmi, _ = vdm.run(
            model_name, input_data, GPU=args.gpu)
        output_file, _ = vdm.write_file(
            model_name, f, output_dir, vdmi)

        print('Processing time: %d seconds' % (time.time() - t))    

if __name__ == "__main__":
    main()
