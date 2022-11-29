# -*- coding: utf-8 -*-
import shutil
from math import factorial
import os
import warnings
import urllib.request
from os.path import join, isdir, basename, isfile, dirname
import time
import importlib
import nibabel as nib
import numpy as np
import sys
from tigerseg import lib_tool


warnings.filterwarnings("ignore", category=UserWarning)
nib.Nifti1Header.quaternion_threshold = -100

#model_server = 'https://github.com/htylab/tigerseg/releases/download/modelhub/'
model_server = 'https://data.mrilab.org/onnxmodel/releasev1/'

import sys


# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(os.path.abspath(__file__))

model_path = join(application_path, 'models')
# print(model_path)
os.makedirs(model_path, exist_ok=True)

def apply_files(model_name, input_file_list, output_dir=None, GPU=False):

    model_name = lib_tool.get_model(model_name)

    seg_method = basename(model_name).split('_')[0]
    if seg_method == 'mprage':
        from tigerseg import lib_bx as seg_module
    elif seg_method == 'cine4d':
        from tigerseg import lib_hx as seg_module
    elif seg_method == 'vdm':
        from tigerseg import lib_vdm as seg_module

    
    print('Total nii files:', len(input_file_list))

    output_file_list = []

    for f in input_file_list:

        print('Predicting:', f)
        t = time.time()
          
        input_data = seg_module.read_file(model_name, f)
        
        mask, _ = seg_module.run(model_name, input_data, GPU=GPU)

        if output_dir is not None:
            output_file, _ = seg_module.write_file(model_name, f, output_dir, mask)
            output_file_list.append(output_file)

        else:
            output_file, _ = seg_module.write_file(model_name, f, dirname(os.path.abspath(f)), mask)
            output_file_list.append(output_file)


        print('Processing time: %d seconds' %  (time.time() - t))


    return output_file_list
