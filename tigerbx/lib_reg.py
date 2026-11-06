import os
import warnings
from os.path import join, isfile
import nibabel as nib
import numpy as np
import sys
from types import SimpleNamespace
from tigerbx.core.io import get_template as build_output_template, save_nib
from tigerbx.core.resample import reorder_img, resample_voxel
from scipy.ndimage import gaussian_filter
from tigerbx.core import numreg
from tigerbx.core.warpfile import load_warp
warnings.filterwarnings("ignore", category=UserWarning)


# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(os.path.abspath(__file__))
    
    
def get_template(template_ff):
    mni_template = nib.load(join(application_path, 'template', 'mni_template.nii.gz'))
    
    if template_ff:
        full_path = join(application_path, 'template', template_ff)
        if isfile(template_ff):
            full_path = template_ff
        
        if isfile(full_path):
            # its own grid (size, voxels, position): the outputs are on it; the fixed-size models (tReg, VMnet)
            # run in a box on it (core.templatereg.model_box)
            return nib.load(full_path)
        else:
            raise FileNotFoundError("Template file does not exist.")
    else:
        template_nib = resample_voxel(mni_template, (1, 1, 1), (160, 224, 192))
        return template_nib


def get_template_seg(template_ff):
    mni_template = nib.load(join(application_path, 'template', 'mni_template_tseg.nii.gz'))

    if template_ff:
        template_seg_ff = template_ff.replace('.nii', '_tseg.nii')
        full_path = join(application_path, 'template', template_seg_ff)
        if isfile(template_seg_ff):
            full_path = template_seg_ff
        
        if isfile(full_path):
            return nib.load(full_path)                       # its own grid, as get_template
        else:
            raise FileNotFoundError("Template file does not exist.")
    else:
        template_nib = resample_voxel(mni_template, (1, 1, 1), (160, 224, 192), interpolation='nearest')
        return template_nib
    

def min_max_norm(img):
    max = np.max(img)
    min = np.min(img)
    if max == min:
        return np.zeros_like(img, dtype=float)
    norm_img = (img - min) / (max - min)

    return norm_img


def apply_gaussian_smoothing(image, fwhm):
    """Apply Gaussian smoothing with FWHM."""
    sigma = fwhm / np.sqrt(8 * np.log(2))
    return gaussian_filter(image, sigma=sigma)

def transform(image_path, warp_path, output_dir=None, GPU=False, interpolation='nearest'):
    """Apply a registration file (_warp.npz, core.warpfile) to an image on the input grid: writes _rigid /
    _affine / _treg (the steps the file holds) in template space."""
    w = load_warp(warp_path)
    order = 0 if interpolation in ('nearest', 'nearestNeighbor') else 1
    ref = SimpleNamespace(affine=w['template_affine'], shape=tuple(int(n) for n in w['template_shape']))
    if 'affine_M' not in w and 'rigid_M' not in w:
        raise ValueError(f'{warp_path} holds no rigid or affine registration.')
    if 'treg_warp' in w and 'affine_M' not in w:
        raise ValueError(f'{warp_path}: a dense warp needs the affine registration, which is missing.')

    ftemplate, _ = build_output_template(image_path, output_dir, True)   # ..._@@@@.nii.gz
    template_nib = reorder_img(get_template(None), resample='continuous')
    native_nib = nib.load(image_path)

    def save(arr, name):
        save_nib(nib.Nifti1Image(arr, ref.affine, template_nib.header), ftemplate, name)

    # each output sampled once from the image's own grid (the affine + warp composed)
    data = np.squeeze(np.asarray(native_nib.dataobj, dtype=np.float32))
    if 'rigid_M' in w:
        save(numreg.resample(native_nib, ref, w['rigid_M'], order=order, data=data), 'rigid')
    if 'affine_M' in w:
        save(numreg.resample(native_nib, ref, w['affine_M'], order=order, data=data), 'affine')
        if 'treg_warp' in w:
            save(numreg.resample(native_nib, ref, w['affine_M'], warp=w['treg_warp'], order=order, data=data), 'treg')
