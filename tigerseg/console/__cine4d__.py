import sys
import os
import argparse
from .. import segment
from distutils.util import strtobool
import glob
from scipy.io import savemat
import nibabel as nib
import numpy as np


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
    parser.add_argument('-i', '--input', metavar='INPUT_FILE', required=True, type=str, help='Path to the input image, can be a folder for the specific format(nii.gz)', nargs='+')
    parser.add_argument('-o', '--output', metavar='OUTPUT_DIR', default=None, type=path, help='Filepath for output segmentation, default: the directory of input files')
    parser.add_argument('--model', default='cine4d_xyz_v002_m12ac', type=str, help='specifies the modelname')
    parser.add_argument('--GPU',default='False',type = strtobool, help='True: GPU, False: CPU, default: False, CPU')
    parser.add_argument('--report',default='True',type = strtobool, help='Produce additional reports')
    args = parser.parse_args()
    
    input_file_list = []
    for arg in args.input:
        input_file_list += glob.glob(arg)

    output_file_list = segment.apply_files(model_name=args.model,
                        input_file_list=input_file_list,
                        output_dir=args.output,                        
                        GPU=args.GPU)

    if args.report:
        for ii in range(len(input_file_list)):
            input_file = input_file_list[ii]
            output_file = output_file_list[ii]
            get_report(input_file, output_file)



if __name__ == "__main__":
    main()
