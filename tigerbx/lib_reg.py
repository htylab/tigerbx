import os
import warnings
from os.path import join, isfile
import nibabel as nib
import numpy as np
import sys
from types import SimpleNamespace
from tigerbx.core.io import get_template as build_output_template, save_nib
from tigerbx.core.spatial import crop_image, pad_to_shape, remove_padding
from tigerbx.core.resample import resample_img, reorder_img, resample_voxel
from scipy.ndimage import gaussian_filter
from tigerbx.core import numreg
from tigerbx.core.warp import affine_transform, dense_transform
from tigerbx.core.warpfile import load_warp
warnings.filterwarnings("ignore", category=UserWarning)
nib.Nifti1Header.quaternion_threshold = -100



# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(os.path.abspath(__file__))
    
    
def get_template(template_ff):
    mni_template = nib.load(join(application_path, 'template', 'mni_template.nii.gz'))
    mni_affine = mni_template.affine
    
    if template_ff:
        full_path = join(application_path, 'template', template_ff)
        if isfile(template_ff):
            full_path = template_ff
        
        if isfile(full_path):
            user_template_nib = nib.load(full_path)
            #resampled_template = resample_voxel(user_template_nib, (1, 1, 1), (256, 256, 256))
            resampled_template = resample_img(user_template_nib, target_affine=mni_affine, target_shape=[160, 224, 192])
            return resampled_template
        else:
            raise FileNotFoundError("Template file does not exist.")
    else:
        template_nib = resample_voxel(mni_template, (1, 1, 1), (160, 224, 192))
        return template_nib


def get_template_seg(template_ff):
    mni_template = nib.load(join(application_path, 'template', 'mni_template_tseg.nii.gz'))
    mni_affine = mni_template.affine

    if template_ff:
        template_seg_ff = template_ff.replace('.nii', '_tseg.nii')
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
    _affine / _VXM / _VFA (the steps the file holds) in template space."""
    w = load_warp(warp_path)
    order = 0 if interpolation in ('nearest', 'nearestNeighbor') else 1
    ref = SimpleNamespace(affine=w['template_affine'], shape=tuple(int(n) for n in w['template_shape']))
    if 'affine_M' not in w and 'c2fvit_matrix' not in w and 'rigid_M' not in w:
        raise ValueError(f'{warp_path} holds no rigid or affine registration.')
    if ('vfa_warp' in w or 'vxm_warp' in w) and 'affine_M' not in w and 'c2fvit_matrix' not in w:
        raise ValueError(f'{warp_path}: a dense warp needs the affine registration, which is missing.')

    ftemplate, _ = build_output_template(image_path, output_dir, True)   # ..._@@@@.nii.gz
    template_nib = reorder_img(get_template(None), resample='continuous')
    native_nib = nib.load(image_path)

    def save(arr, name):
        save_nib(nib.Nifti1Image(arr, ref.affine, template_nib.header), ftemplate, name)

    if 'c2fvit_matrix' not in w:
        # numreg: each output sampled once from the image's own grid (the affine + warp composed)
        data = np.squeeze(np.asarray(native_nib.dataobj, dtype=np.float32))
        if 'rigid_M' in w:
            save(numreg.resample(native_nib, ref, w['rigid_M'], order=order, data=data), 'rigid')
        if 'affine_M' in w:
            save(numreg.resample(native_nib, ref, w['affine_M'], order=order, data=data), 'affine')
            for key, name in (('vxm_warp', 'VXM'), ('vfa_warp', 'VFA')):
                if key in w:
                    save(numreg.resample(native_nib, ref, w['affine_M'], warp=w[key], order=order, data=data), name)
        return

    input_nib = reorder_img(native_nib, resample=interpolation)
    input_data = input_nib.get_fdata().astype(np.float32)
    if 'rigid_M' in w:
        save(numreg.apply(input_nib, ref, w['rigid_M'], data=input_data, order=order), 'rigid')
    _, pad_width = pad_to_shape(template_nib.get_fdata(), (256, 256, 256))
    vol, _ = pad_to_shape(input_data, (256, 256, 256))
    vol, _ = crop_image(vol, target_shape=(256, 256, 256))
    affined = remove_padding(affine_transform(vol, w['c2fvit_init_shift'], w['c2fvit_matrix'], interpolation),
                             pad_width)
    save(affined, 'affine')
    for key, name in (('vxm_warp', 'VXM'), ('vfa_warp', 'VFA')):
        if key in w:
            save(dense_transform(affined, w[key], interpolation), name)
