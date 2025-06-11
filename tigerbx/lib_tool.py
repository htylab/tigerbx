# -*- coding: utf-8 -*-

# +
import os
import re
import subprocess
import onnxruntime as ort
import shutil
import warnings
from os.path import join, isdir, basename, isfile, dirname
import nibabel as nib
import numpy as np
import sys
from os.path import isfile, join
from tigerbx import lib_bx
from nilearn.image import resample_img
from typing import Union, Tuple, List
from scipy.ndimage import gaussian_filter
warnings.filterwarnings("ignore", category=UserWarning)
ort.set_default_logger_severity(3)
nib.Nifti1Header.quaternion_threshold = -100

model_servers = ['https://github.com/htylab/tigerbx/releases/download/modelhub/',
                    'https://data.mrilab.org/onnxmodel/dev/']

# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(os.path.abspath(__file__))

model_path = join(application_path, 'models')
# print(model_path)
os.makedirs(model_path, exist_ok=True)


def download(url, file_name):
    import urllib.request
    import certifi
    import shutil
    import ssl
    context = ssl.create_default_context(cafile=certifi.where())
    #urllib.request.urlopen(url, cafile=certifi.where())
    with urllib.request.urlopen(url,
                                context=context) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
        

def get_model(f):
    from os.path import join, isfile
    import os


    if isfile(f):
        return f

    if '.onnx' in f:
        fn = f
    else:
        fn = f + '.onnx'
    
    model_file = join(model_path, fn)
    
    if not os.path.exists(model_file):
        
        for server in model_servers:
            try:
                print(f'Downloading model files....')
                model_url = server + fn
                print(model_url, model_file)
                download(model_url, model_file)
                download_ok = True
                print('Download finished...')
                break
            except:
                download_ok = False

        if not download_ok:
            raise ValueError('Server error. Please check the model name or internet connection.')
                
    return model_file



def cpu_count():
    """ Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program"""

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        pass

    # Python 2.6+
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # https://github.com/giampaolo/psutil
    try:
        import psutil
        return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
    except (ImportError, AttributeError):
        pass

    # POSIX
    try:
        res = int(os.sysconf('SC_NPROCESSORS_ONLN'))

        if res > 0:
            return res
    except (AttributeError, ValueError):
        pass

    # Windows
    try:
        res = int(os.environ['NUMBER_OF_PROCESSORS'])

        if res > 0:
            return res
    except (KeyError, ValueError):
        pass

    # jython
    try:
        from java.lang import Runtime
        runtime = Runtime.getRuntime()
        res = runtime.availableProcessors()
        if res > 0:
            return res
    except ImportError:
        pass

    # BSD
    try:
        sysctl = subprocess.Popen(['sysctl', '-n', 'hw.ncpu'],
                                  stdout=subprocess.PIPE)
        scStdout = sysctl.communicate()[0]
        res = int(scStdout)

        if res > 0:
            return res
    except (OSError, ValueError):
        pass

    # Linux
    try:
        res = open('/proc/cpuinfo').read().count('processor\t:')

        if res > 0:
            return res
    except IOError:
        pass

    # Solaris
    try:
        pseudoDevices = os.listdir('/devices/pseudo/')
        res = 0
        for pd in pseudoDevices:
            if re.match(r'^cpuid@[0-9]+$', pd):
                res += 1

        if res > 0:
            return res
    except OSError:
        pass

    # Other UNIXes (heuristic)
    try:
        try:
            dmesg = open('/var/run/dmesg.boot').read()
        except IOError:
            dmesgProcess = subprocess.Popen(['dmesg'], stdout=subprocess.PIPE)
            dmesg = dmesgProcess.communicate()[0]

        res = 0
        while '\ncpu' + str(res) + ':' in dmesg:
            res += 1

        if res > 0:
            return res
    except OSError:
        pass

    raise Exception('Can not determine number of CPUs on this system')


def predict(model, data, GPU, mode=None):
    #from .tool import cpu_count
    #will reload model file every time

    so = ort.SessionOptions()
    cpu = max(int(cpu_count()*0.7), 1)
    so.intra_op_num_threads = cpu
    so.inter_op_num_threads = cpu
    so.log_severity_level = 3

    if GPU and (ort.get_device() == "GPU"):

        session = ort.InferenceSession(model,
                                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                                       sess_options=so)
    else:
        session = ort.InferenceSession(model,
                                       providers=['CPUExecutionProvider'],
                                       sess_options=so)

    data_type = 'float64'
    if session.get_inputs()[0].type == 'tensor(float)':
        data_type = 'float32'
    if mode == 'reg':
        input_names = [input.name for input in session.get_inputs()]
        inputs = {input_names[0]: data[0], input_names[1]: data[1]}
        return session.run(None, inputs)
    if mode == 'affine_transform':
        input_names = [input.name for input in session.get_inputs()]
        inputs = {input_names[0]: data[0], input_names[1]: data[1], input_names[2]: data[2]}
        return session.run(None, inputs)
    if mode == 'encode':
        mu, sigma = session.run(None, {session.get_inputs()[0].name: data.astype(data_type)}, )
        return mu, sigma
    
    if mode == 'decode':
        result = session.run(None, {session.get_inputs()[0].name: data.astype(data_type)}, )
        return result[0]
        
    if mode == 'patch':
        logits = patch_inference_3d_lite(session, data.astype(data_type), patch_size = (160,)*3, gaussian = True)
        # print(data.shape)
        # logits = session.run(None, {session.get_inputs()[0].name: data.astype(data_type)}, )[0]
        # print('logits type', type(logits))
        
        return logits
        
    return session.run(None, {session.get_inputs()[0].name: data.astype(data_type)}, )[0]
def patch_inference_3d_lite(session, 
                       vol_d: np.ndarray, 
                       patch_size : Tuple[int, ...] = (128,)*3, 
                       tile_step_size: float = 0.5, 
                       gaussian = True ):
    patches, point_list = img_to_patches(vol_d, patch_size, tile_step_size)#patches.shape = (patch_num, 1, 1, 128, 128, 128)  
    gaussian_map = compute_gaussian(patch_size)
    patch_logits_shape = session.run(None, {session.get_inputs()[0].name: patches[0]}, )[0].shape
    prob_tensor = np.zeros(((patch_logits_shape[1],) + vol_d.shape[-3:]))
    weight_tensor = np.zeros(vol_d.shape[-3:])
    if gaussian:
        weight_patch = gaussian_map
    else:
        weight_patch = torch.ones(patch_size, device=vol_d.device)
    for p in point_list:
        weight_tensor[p[0]:p[0]+patch_size[0], p[1]:p[1]+patch_size[1], p[2]:p[2]+patch_size[2]] += weight_patch
    for patch, p in zip(patches, point_list):
        logits = session.run(None, {session.get_inputs()[0].name: patch}, )[0]#logits.shape = (1, c, 128, 128, 128)      
        if gaussian:    
            output_patch = logits.squeeze(0)*gaussian_map
        prob_tensor[: , p[0] : p[0]+patch_size[0],  p[1] :  p[ 1]+patch_size[1],  p[2] :  p[2]+patch_size[2]] += output_patch
    prob_tensor= prob_tensor/weight_tensor
    return prob_tensor[np.newaxis, :]


    
def patch_inference_3d(session, 
                       vol_d: np.ndarray, 
                       patch_size : Tuple[int, ...] = (128,)*3, 
                       tile_step_size: float = 0.5, 
                       gaussian = False ):
    patches, point_list = img_to_patches(vol_d, patch_size, tile_step_size)#patches.shape = (patch_num, 1, 1, 128, 128, 128)  
    output_patch_list = []
    for patch in patches:
        logits = session.run(None, {session.get_inputs()[0].name: patch}, )[0]#logits.shape = (1, 1, 128, 128, 128)             
        output_patch_list.append(logits.squeeze(0))
    output_patches = np.concatenate([s[np.newaxis, ...] for s in output_patch_list], axis=0)#shape = (patch_num, 1, 128, 128, 128)  
    if gaussian:    
        gaussian_map = compute_gaussian(patch_size)
        output_patches = output_patches*gaussian_map
    # print(output_patches.shape) # (patch_num, channel, w, h, d)
    mean_prob = patches_to_img(output_patches, vol_d.shape[-3:], point_list)
    return mean_prob
def compute_steps_for_sliding_window(image_size: Tuple[int, ...], 
                                     tile_size: Tuple[int, ...], 
                                     tile_step_size: float) ->  List[List[int]]:
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)
    return steps


def compute_gaussian(tile_size: Union[Tuple[int, ...], List[int]], 
                     sigma_scale: float = 1. / 8,
                     value_scaling_factor: float = 1, 
                     dtype=np.float16) -> np.ndarray:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map /= (np.max(gaussian_importance_map) / value_scaling_factor)
    gaussian_importance_map = gaussian_importance_map.astype(dtype)
    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    mask = gaussian_importance_map == 0
    gaussian_importance_map[mask] = np.min(gaussian_importance_map[~mask])
    return gaussian_importance_map

def img_to_patches(vol_d: np.ndarray, patch_size: Tuple[int, ...], tile_step_size: float):
    steps = compute_steps_for_sliding_window(vol_d.shape[-3:], patch_size, tile_step_size)
    slice_list = []
    point_list = [[i, j, k] for i in steps[0] for j in steps[1] for k in steps[2]]
    for p in point_list:            
        slice_input = vol_d[:, :, p[0] : p[0]+patch_size[0], p[1] : p[1]+patch_size[1], p[2] : p[2]+patch_size[2]]
        slice_list.append(slice_input)
    return np.concatenate([s[np.newaxis, ...] for s in slice_list], axis=0), point_list

def patches_to_img(patches: np.ndarray, vol_d_size: Tuple[int, ...], point_list: List[List[int]]):
    '''
    patches shape = (patch_num, channel, w, h, d)
    '''
    patch_size = patches.shape[-3:]
    prob_tensor = np.zeros(((patches.shape[1],) + vol_d_size))
    
    for patch_dim, p in zip(range(patches.shape[0]), point_list):
        none_zero_mask1 = prob_tensor[:, p[0] : p[0]+patch_size[0],  p[1] :  p[ 1]+patch_size[1],  p[2] :  p[2]+patch_size[2]]!= 0 
        none_zero_mask2 = patches[patch_dim, : ,...]!= 0
        none_zero_num = np.clip(none_zero_mask1 + none_zero_mask2, a_min=1, a_max=None)
        prob_tensor[: , p[0] : p[0]+patch_size[0],  p[1] :  p[ 1]+patch_size[1],  p[2] :  p[2]+patch_size[2]] += patches[patch_dim, : ,...]
        prob_tensor[: , p[0] : p[0]+patch_size[0],  p[1] :  p[ 1]+patch_size[1],  p[2] :  p[2]+patch_size[2]] /= none_zero_num
    return prob_tensor[np.newaxis, :]

def clean_onnx():
    import glob
    ffs = glob.glob(join(model_path, '*.onnx'))
    for f in ffs:
        print('Removing ', f)
        os.remove(f)


def check_dtype(data, dtype):
    value_range = [data.min(), data.max() ]
    
    # Get the min and max allowable values for the specified data type
    if dtype == np.int8:
        min_val, max_val = np.iinfo(np.int8).min, np.iinfo(np.int8).max
    elif dtype == np.int16:
        min_val, max_val = np.iinfo(np.int16).min, np.iinfo(np.int16).max
    elif dtype == np.int32:
        min_val, max_val = np.iinfo(np.int32).min, np.iinfo(np.int32).max
    elif dtype == np.uint8:
        min_val, max_val = np.iinfo(np.uint8).min, np.iinfo(np.uint8).max
    elif dtype == np.uint16:
        min_val, max_val = np.iinfo(np.uint16).min, np.iinfo(np.uint16).max
    elif dtype == np.uint32:
        min_val, max_val = np.iinfo(np.uint32).min, np.iinfo(np.uint32).max
    else:
        #raise ValueError(f"Unsupported data type: {dtype}")
        return True

    # Check if the value range is within the allowable range for the data type
    return min_val <= value_range[0] <= max_val and min_val <= value_range[1] <= max_val
