# -*- coding: utf-8 -*-

# +
import os
import re
import subprocess
import onnxruntime as ort
import shutil
import tempfile
import warnings
from os.path import join, isdir, basename, isfile, dirname
import nibabel as nib
import numpy as np
import sys
from os.path import isfile, join
from nilearn.image import resample_img
from typing import Union, Tuple, List
from scipy.ndimage import gaussian_filter
warnings.filterwarnings("ignore", category=UserWarning)
ort.set_default_logger_severity(3)
nib.Nifti1Header.quaternion_threshold = -100

model_servers = ['https://github.com/htylab/tigerbx/releases/download/modelhub/',
	                    'https://data.mrilab.org/onnxmodel/dev/']

MODEL_DIR_ENV = 'TIGERBX_MODEL_DIR'
MODEL_DOWNLOAD_TIMEOUT_S = 60
MODEL_LOCK_TIMEOUT_S = 60 * 60
PATCH_SIZE_ENV = 'TIGERBX_PATCH_SIZE'
DEFAULT_PATCH_SIZE = (128, 128, 128)
MIN_CROP_SIZE = (160, 160, 160)


def _parse_patch_size(value: str):
    value = value.strip()
    if not value:
        raise ValueError("empty patch size")

    if value.isdigit():
        size = int(value)
        return (size, size, size)

    parts = [p for p in re.split(r"[x, ]+", value) if p]
    if len(parts) != 3:
        raise ValueError(f"patch size must be like '128' or '128,128,128' (got {value!r})")
    return tuple(int(p) for p in parts)


def _resolve_patch_size(patch_size):
    if patch_size is None:
        env = os.environ.get(PATCH_SIZE_ENV)
        if env:
            patch_size = _parse_patch_size(env)
        else:
            patch_size = DEFAULT_PATCH_SIZE
    elif isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)

    if len(patch_size) != 3:
        raise ValueError("patch_size must be an int or a 3-tuple like (128, 128, 128)")

    patch_size = tuple(int(s) for s in patch_size)
    if not all(s > 0 for s in patch_size):
        raise ValueError(f"patch_size must be > 0 (got {patch_size})")
    if not all(s < MIN_CROP_SIZE[i] for i, s in enumerate(patch_size)):
        raise ValueError(
            f"patch_size must be < {MIN_CROP_SIZE} to match MIN_CROP (got {patch_size})."
        )
    return patch_size


def _bundled_models_dir():
    if getattr(sys, 'frozen', False):
        return join(dirname(sys.executable), 'models')
    return join(dirname(os.path.abspath(__file__)), 'models')


def get_model_dir():
    model_dir = os.environ.get(MODEL_DIR_ENV)
    if model_dir:
        return model_dir
    from platformdirs import user_cache_dir
    return join(user_cache_dir('tigerbx'), 'models')


model_path = get_model_dir()


def download(url, file_name, timeout_s=MODEL_DOWNLOAD_TIMEOUT_S):
    import urllib.request
    import ssl
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        context = ssl.create_default_context()
    #urllib.request.urlopen(url, cafile=certifi.where())
    with urllib.request.urlopen(url,
                                context=context,
                                timeout=timeout_s) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
        

def _atomic_download(url, dst_path, timeout_s=MODEL_DOWNLOAD_TIMEOUT_S):
    dst_dir = dirname(dst_path)
    fd, tmp_path = tempfile.mkstemp(prefix=basename(dst_path) + '.', suffix='.tmp', dir=dst_dir)
    os.close(fd)
    try:
        download(url, tmp_path, timeout_s=timeout_s)
        os.replace(tmp_path, dst_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def get_model(f):
    from filelock import FileLock, Timeout

    if isfile(f):
        return f

    fn = f if f.endswith('.onnx') else f + '.onnx'
    
    override_dir = os.environ.get(MODEL_DIR_ENV)
    model_dir = override_dir or get_model_dir()
    model_file = join(model_dir, fn)

    if os.path.exists(model_file):
        return model_file

    if not override_dir:
        bundled_model_file = join(_bundled_models_dir(), fn)
        if os.path.exists(bundled_model_file):
            return bundled_model_file
    
    os.makedirs(model_dir, exist_ok=True)

    lock = FileLock(model_file + '.lock')
    try:
        with lock.acquire(timeout=MODEL_LOCK_TIMEOUT_S):
            if os.path.exists(model_file):
                return model_file

            errors = []
            for server in model_servers:
                model_url = server.rstrip('/') + '/' + fn
                try:
                    print('Downloading model file....')
                    print(model_url, model_file)
                    _atomic_download(model_url, model_file)
                    print('Download finished...')
                    return model_file
                except Exception as e:
                    errors.append(f'{model_url}: {e}')

            error_detail = '\n'.join(errors) if errors else '(no servers configured)'
            raise ValueError(
                'Server error. Please check the model name or internet connection.\n'
                f'{error_detail}'
            )
    except Timeout:
        raise TimeoutError(f'Timeout waiting for download lock: {model_file}.lock')
                
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


def predict(model, data, GPU, mode=None, patch_size=None, tile_step_size=0.5, gaussian=True):
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
        patch_size = _resolve_patch_size(patch_size)
        input_shape = session.get_inputs()[0].shape
        if input_shape is not None and len(input_shape) >= 5:
            expected_spatial = input_shape[-3:]
            if all(isinstance(s, int) for s in expected_spatial):
                if tuple(expected_spatial) != tuple(patch_size):
                    raise ValueError(
                        f"Model expects fixed spatial dims {tuple(expected_spatial)}, "
                        f"but patch_size is {tuple(patch_size)}. "
                        f"Set {PATCH_SIZE_ENV} or pass patch_size=({expected_spatial[0]},{expected_spatial[1]},{expected_spatial[2]})"
                    )

        try:
            logits = patch_inference_3d_lite(
                session,
                data.astype(data_type),
                patch_size=patch_size,
                tile_step_size=tile_step_size,
                gaussian=gaussian,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Patch inference failed (patch_size={patch_size}, tile_step_size={tile_step_size}, gaussian={gaussian})."
            ) from exc

        return logits
        
    return session.run(None, {session.get_inputs()[0].name: data.astype(data_type)}, )[0]
def patch_inference_3d_lite(session, 
                       vol_d: np.ndarray, 
                       patch_size : Tuple[int, ...] = (128,)*3, 
                       tile_step_size: float = 0.5, 
                       gaussian = True ):
    patches, point_list = img_to_patches(vol_d, patch_size, tile_step_size)#patches.shape = (patch_num, 1, 1, 128, 128, 128)  
    gaussian_map = compute_gaussian(patch_size) if gaussian else None
    patch_logits_shape = session.run(None, {session.get_inputs()[0].name: patches[0]}, )[0].shape
    prob_tensor = np.zeros(((patch_logits_shape[1],) + vol_d.shape[-3:]))
    weight_tensor = np.zeros(vol_d.shape[-3:])
    if gaussian:
        weight_patch = gaussian_map
    else:
        weight_patch = np.ones(patch_size, dtype=weight_tensor.dtype)
    for p in point_list:
        weight_tensor[p[0]:p[0]+patch_size[0], p[1]:p[1]+patch_size[1], p[2]:p[2]+patch_size[2]] += weight_patch
    for patch, p in zip(patches, point_list):
        logits = session.run(None, {session.get_inputs()[0].name: patch}, )[0]#logits.shape = (1, c, 128, 128, 128)      
        if gaussian:    
            output_patch = logits.squeeze(0) * gaussian_map
        else:
            output_patch = logits.squeeze(0)
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
    assert all(i >= j for i, j in zip(image_size, tile_size)), "image size must be as large or larger than patch_size"
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

def read_nib(input_nib):
    # in adni dataset, the 3D mprage is stored as a 4D array
    return np.squeeze(input_nib.get_fdata())


def resample_voxel(data_nib, voxelsize, target_shape=None, interpolation='continuous'):
    affine = data_nib.affine
    target_affine = affine.copy()
    factor = np.zeros(3)
    for i in range(3):
        factor[i] = voxelsize[i] / \
            np.sqrt(affine[0, i]**2 + affine[1, i]**2 + affine[2, i]**2)
        target_affine[:3, i] = target_affine[:3, i] * factor[i]
    new_nib = resample_img(data_nib, target_affine=target_affine,
                           target_shape=target_shape, interpolation=interpolation,
                           force_resample=True)
    return new_nib


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
