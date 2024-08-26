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


warnings.filterwarnings("ignore", category=UserWarning)
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

def get_template(template_ff):
    mni_template = nib.load(join(application_path, 'template', 'MNI152_T1_1mm_brain.nii.gz'))
    mni_affine = mni_template.affine
    
    if template_ff:
        full_path = join(application_path, 'template', template_ff)
        if isfile(template_ff):
            full_path = template_ff
        
        if isfile(full_path):
            user_template_nib = nib.load(full_path)
            #resampled_template = lib_bx.resample_voxel(user_template_nib, (1, 1, 1), (256, 256, 256))
            resampled_template = resample_img(user_template_nib, target_affine=mni_affine, target_shape=[160, 224, 192])
            return resampled_template
        else:
            raise FileNotFoundError("Template file does not exist.")
    else:
        template_nib = lib_bx.resample_voxel(mni_template, (1, 1, 1), (160, 224, 192))
        return template_nib


def get_template_seg(template_ff):
    mni_template = nib.load(join(application_path, 'template', 'MNI152_T1_1mm_brain_aseg.nii.gz'))
    mni_affine = mni_template.affine

    if template_ff:
        template_seg_ff = template_ff.replace('.nii', '_aseg.nii')
        full_path = join(application_path, 'template', template_seg_ff)
        if isfile(template_seg_ff):
            full_path = template_seg_ff
        
        if isfile(full_path):
            user_template_nib = nib.load(full_path)
            resampled_template = resample_img(user_template_nib, target_affine=mni_affine, target_shape=[160, 224, 192], interpolation='nearest')
            return resampled_template
        else:
            raise FileNotFoundError("Template file does not exist.")
    else:
        template_nib = lib_bx.resample_voxel(mni_template, (1, 1, 1), (160, 224, 192), interpolation='nearest')
        return template_nib
        

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
                                       providers=['CUDAExecutionProvider'],
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

    return session.run(None, {session.get_inputs()[0].name: data.astype(data_type)}, )[0]


def clean_onnx():
    import glob
    ffs = glob.glob(join(model_path, '*.onnx'))
    for f in ffs:
        print('Removing ', f)
        os.remove(f)
