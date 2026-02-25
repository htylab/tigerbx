# -*- coding: utf-8 -*-

# +
import os
import shutil
import tempfile
import warnings
from os.path import join, isdir, basename, isfile, dirname
import nibabel as nib
import numpy as np
import sys
from os.path import isfile, join
warnings.filterwarnings("ignore", category=UserWarning)
nib.Nifti1Header.quaternion_threshold = -100

model_servers = ['https://github.com/htylab/tigerbx/releases/download/modelhub/',
	                    'https://data.mrilab.org/onnxmodel/dev/']

MODEL_DIR_ENV = 'TIGERBX_MODEL_DIR'
MODEL_DOWNLOAD_TIMEOUT_S = 60
MODEL_LOCK_TIMEOUT_S = 60 * 60


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



def read_nib(input_nib):
    # in adni dataset, the 3D mprage is stored as a 4D array
    return np.squeeze(input_nib.get_fdata())


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
