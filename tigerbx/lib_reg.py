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
    mni_template = nib.load(join(application_path, 'template', 'mni_template_aseg.nii.gz'))
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
    
    data = np.load(warp_path, allow_pickle=True)
    displacement_dict = {
        key: data[key].item() if isinstance(data[key], np.ndarray) and data[key].dtype == object else data[key]
        for key in data.files
        }
    
    init_flow = displacement_dict['init_flow']
    rigid_matrix = displacement_dict['rigid_matrix']
    affine_matrix = displacement_dict['affine_matrix']
    reference_info = displacement_dict['reference_info']
    dense_warp = displacement_dict['dense_warp']
    Fuse_dense_warp = displacement_dict.get('Fuse_dense_warp')   # warp.npz from releases that had FuseMorph
    VFA_dense_warp = displacement_dict.get('VFA_dense_warp')      # absent in warp.npz files before 'D'

    if (init_flow is None or init_flow.shape == ()) and (reference_info is None or (isinstance(reference_info, dict) and not reference_info)):
        raise ValueError("Both init_flow and reference_info are None or empty. The program cannot proceed.")

    if (dense_warp is not None or Fuse_dense_warp is not None or VFA_dense_warp is not None) and (affine_matrix is None or (isinstance(affine_matrix, np.ndarray) and not np.any(affine_matrix))):
        raise ValueError("affine_matrix is None or empty, and at least one of dense_warp or Fuse_dense_warp is not None.")
    
    method_check = {'affine':'C2FViT', 'nonlinear':'VMnet'}
    if init_flow is None or init_flow.shape == ():
        method_check['affine'] = 'numreg'      # also old ANTs-affine warp.npz files (converted below)
    
    ftemplate, f_output_dir = build_output_template(image_path, output_dir, True)   # ..._@@@@.nii.gz
    
    template_nib =get_template(None)
    template_nib = reorder_img(template_nib, resample='continuous')
    template_data = template_nib.get_fdata()
    template_data, pad_width = pad_to_shape(template_data, (256, 256, 256))
    
    input_nib = nib.load(image_path)
    input_nib = reorder_img(input_nib, resample=interpolation)
    input_data = input_nib.get_fdata().astype(np.float32)
    
    if method_check['affine'] == 'C2FViT':
        input_data, _ = pad_to_shape(input_data, (256, 256, 256))
        input_data, _ = crop_image(input_data, target_shape=(256, 256, 256))
    #Rigid: numreg (dict with M) or, in warp.npz from before 0.3, the C2FViT-style rigid model (init_flow + matrix)
    if isinstance(rigid_matrix, dict) and 'M' in rigid_matrix:
        ref = SimpleNamespace(affine=np.asarray(reference_info['affine']), shape=tuple(reference_info['shape']))
        raw = reorder_img(nib.load(image_path), resample=interpolation).get_fdata().astype(np.float32)
        rigid = numreg.apply(input_nib, ref, rigid_matrix['M'], data=raw,
                             order=0 if interpolation in ('nearest', 'nearestNeighbor') else 1)
        save_nib(nib.Nifti1Image(rigid, template_nib.affine, template_nib.header), ftemplate, 'rigid')
    elif rigid_matrix is not None and rigid_matrix.shape != ():
        rigid = remove_padding(affine_transform(input_data, init_flow, rigid_matrix, interpolation), pad_width)
        rigid_nib = nib.Nifti1Image(rigid,
                                      template_nib.affine, template_nib.header)
        fn = save_nib(rigid_nib, ftemplate, 'rigid')
    #Affine
    affined = None
    if method_check['affine'] == 'C2FViT':
        if affine_matrix is not None and affine_matrix.shape != ():
            affined = remove_padding(affine_transform(input_data, init_flow, affine_matrix, interpolation), pad_width)
            affined_nib = nib.Nifti1Image(affined,
                                          template_nib.affine, template_nib.header)
            fn = save_nib(affined_nib, ftemplate, 'affine')
    elif affine_matrix is not None:
        if 'M' in affine_matrix:
            M = affine_matrix['M']
            ref = SimpleNamespace(affine=np.asarray(reference_info['affine']), shape=tuple(reference_info['shape']))
        else:                                   # warp.npz from the removed ANTs affine
            M, ref = legacy_ants_affine(affine_matrix, reference_info)
        affined = numreg.apply(input_nib, ref, M, data=input_data,
                               order=0 if interpolation in ('nearest', 'nearestNeighbor') else 1)
        affined_nib = nib.Nifti1Image(affined, template_nib.affine, template_nib.header)
        fn = save_nib(affined_nib, ftemplate, 'affine')
    #Nonlinear(SyN, SyNCC): removed; their warp.npz entries were paths to temporary ANTs files
    for key in ('SyN_dense_warp', 'SyNCC_dense_warp'):
        v = displacement_dict.get(key)
        if v is not None and np.asarray(v).shape != ():
            warnings.warn(f"{key} in {warp_path} is no longer supported (SyN / SyNCC were removed); skipped.")
    #Nonlinear(VMnet)
    if dense_warp is not None and dense_warp.shape != ():
        if affined is None:
            raise ValueError("dense_warp requires affine registration output, but affined is None.")
        reged = dense_transform(affined, dense_warp, interpolation)
        reged_nib = nib.Nifti1Image(reged,
                                    template_nib.affine, template_nib.header)
        fn = save_nib(reged_nib, ftemplate, 'VXM')
    #Nonlinear(FuseMorph; old warp.npz only)
    if Fuse_dense_warp is not None and Fuse_dense_warp.shape != ():
        if affined is None:
            raise ValueError("Fuse_dense_warp requires affine registration output, but affined is None.")
        Fused = dense_transform(affined, Fuse_dense_warp, interpolation)
        Fused_nib = nib.Nifti1Image(Fused,
                                    template_nib.affine, template_nib.header)
        fn = save_nib(Fused_nib, ftemplate, 'Fuse')
    #Nonlinear(VFA)
    if VFA_dense_warp is not None and VFA_dense_warp.shape != ():
        if affined is None:
            raise ValueError("VFA_dense_warp requires affine registration output, but affined is None.")
        vfa_moved = dense_transform(affined, VFA_dense_warp, interpolation)
        save_nib(nib.Nifti1Image(vfa_moved, template_nib.affine, template_nib.header), ftemplate, 'VFA')

def legacy_ants_affine(affine_matrix, reference_info):
    """warp.npz written by the removed ANTs affine (tigerbx <= 0.2.x) -> (M, reference grid) for numreg.apply.
    tigerbx passed RAS world coordinates to ITK as-is (no LPS flip), so the ITK matrix is already RAS -> RAS."""
    p = np.asarray(affine_matrix['parameters'], dtype=np.float64).ravel()
    c = np.asarray(affine_matrix['fixed_parameters'], dtype=np.float64).ravel()
    A, t = p[:9].reshape(3, 3), p[9:12]
    M = np.eye(4)
    M[:3, :3] = A
    M[:3, 3] = t + c - A @ c
    ref = np.eye(4)
    ref[:3, :3] = np.asarray(reference_info['direction'], dtype=np.float64).reshape(3, 3) * np.asarray(reference_info['spacing'])
    ref[:3, 3] = reference_info['origin']
    return M, SimpleNamespace(affine=ref, shape=tuple(reference_info['shape']))
