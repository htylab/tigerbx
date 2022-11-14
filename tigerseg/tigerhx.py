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
import methods.cine4d as cine4d
import methods.tigertool as tigertool

def get_report(input_file, output_file):
    
    temp = nib.load(output_file)
    mask4d = temp.get_fdata()
    voxel_size = temp.header.get_zooms()
    
    LV_vol = np.sum(mask4d==1, axis=(0, 1, 2))* np.prod(voxel_size[0:3]) / 1000.
    LVM_vol = np.sum(mask4d==2, axis=(0, 1, 2))* np.prod(voxel_size[0:3]) / 1000.
    RV_vol = np.sum(mask4d==3, axis=(0, 1, 2))* np.prod(voxel_size[0:3]) / 1000.
    
    dict1 = {"input":np.asanyarray(nib.load(input_file).dataobj),
             'LV': (mask4d==1)*1,
             'LVM':(mask4d==2)*1,
             'RV': (mask4d==3)*1,
             'LV_vol': LV_vol,
             'LVM_vol': LVM_vol,
             'RV_vol': RV_vol}

    savemat(output_file.replace('.nii.gz', '.mat'), dict1, do_compression=True)


def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input',  type=str, nargs='+', help='Path to the input image, can be a folder for the specific format(nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output segmentation, default: the directory of input files')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
    parser.add_argument('-r', '--report', action='store_true',
                        help='Produce additional reports')

    args = parser.parse_args() 

    input_file_list = args.input
    if os.path.isdir(args.input[0]):
        input_file_list = glob.glob(join(args.input[0], '*.nii'))
        input_file_list += glob.glob(join(args.input[0], '*.nii.gz'))

    elif '*' in args.input[0]:
        input_file_list = glob.glob(args.input[0])

    output_dir = args.output

    print('Total nii files:', len(input_file_list))

    model_name = tigertool.get_model('cine4d_xyz_v002_m12ac')

    for f in input_file_list:

        print('Predicting:', f)
        t = time.time()
        input_data = cine4d.read_file(model_name, f)
        mask_pred, _ = cine4d.run(
            model_name, input_data, GPU=args.gpu)
        mask_pred = cine4d.post(mask_pred)
        output_file, _ = cine4d.write_file(
            model_name, f, output_dir, mask_pred, postfix='hseg3')

        if args.report:
            get_report(f, output_file)

        print('Processing time: %d seconds' % (time.time() - t))    

if __name__ == "__main__":
    main()
