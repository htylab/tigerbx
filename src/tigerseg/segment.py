# -*- coding: utf-8 -*-
from math import factorial
import os
import warnings
import urllib.request
from os.path import join, isdir, basename, isfile, dirname
import time
import importlib
import nibabel as nib
import numpy as np


warnings.filterwarnings("ignore", category=UserWarning)
nib.Nifti1Header.quaternion_threshold = -100

#model_server = 'https://github.com/htylab/tigerseg/releases/download/modelhub/'
model_server = 'https://data.mrilab.org/onnxmodel/'
model_path = join(os.path.dirname(os.path.abspath(__file__)), 'models')
# print(model_path)
os.makedirs(model_path, exist_ok=True)

def apply_files(model_name, input_file_list, output_dir=None, GPU=False, model_path=model_path):

    seg_method = basename(model_name).split('_')[0]
    seg_module = importlib.import_module('tigerseg.methods.' + seg_method)
    
    print('Total nii files:', len(input_file_list))

    output_file_list = []

    for f in input_file_list:

        print('Predicting:', f)
        t = time.time()
          
        input_data = seg_module.read_file(model_name, f)
        
        mask = apply(model_name, input_data, GPU=GPU, model_path=model_path)

        if output_dir is not None:
            output_file, _ = seg_module.write_file(model_name, f, output_dir, mask)
            output_file_list.append(output_file)

        else:
            output_file, _ = seg_module.write_file(model_name, f, dirname(os.path.abspath(f)), mask)
            output_file_list.append(output_file)


        print('Processing time: %d seconds' %  (time.time() - t))


    return output_file_list


def apply(model_name, input_data, GPU=False, model_path=model_path):

    #download model files
    model_ffs = []
    for f in model_name.replace('@', '#').replace('*', '#').split('#'):

        if isfile(f):
            model_ffs.append(f)
            continue
        if '.onnx' in f:
            fn = f
        else:
            fn = f + '.onnx'

        model_url = model_server + fn
        model_file = join(model_path, fn)
        model_ffs.append(model_file)
        if not os.path.exists(model_file):
            print(f'Downloading model files....')
            print(model_url, model_file)
            urllib.request.urlretrieve(model_url, model_file)
        
    #todo: verify model files

        #apply 只處理 numpy array --> model --> mask output
    #有三種模式
    #

    seg_method = basename(model_ffs[0]).split('_')[0]
    seg_module = importlib.import_module('tigerseg.methods.' + seg_method)
    run_SingleModel = getattr(seg_module, 'run_SingleModel')

    
    if '#' in model_name: #model ensemble by softmax summation
        mask_softmax_sum = 0

        for model_ff in model_ffs:
            mask_pred, mask_softmax = run_SingleModel(model_ff, input_data, GPU)
            mask_softmax_sum += mask_softmax

        mask_pred = np.argmax(mask_softmax_sum, axis=0)

    elif '*' in model_name: #mask segmentation
        print('Intersection of masks')

        mask1, mask_softmax = run_SingleModel(model_ffs[0], input_data, GPU)
        mask2, mask_softmax = run_SingleModel(model_ffs[1], input_data, GPU)

        mask_pred = mask1 * (mask2 > 0)
        

    elif '@' in model_name: #todo two-stage segmenation
        print('Two-stage model')
        model1_ff, model2_ff = model_ffs[0], model_ffs[1]
        mask1, mask_softmax = run_SingleModel(model1_ff, input_data, GPU)
        mask2, mask_softmax = run_SingleModel(model2_ff, input_data, GPU)

        mask_pred = mask1 * mask2
        

    else: #single model mode
        model_ff = model_ffs[0]
        mask_pred, mask_softmax = run_SingleModel(model_ff, input_data, GPU)
    
    if 'post' in dir(seg_module):
        mask_pred = seg_module.post(mask_pred)

    return mask_pred


