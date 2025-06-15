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
from tigerbx import lib_tool
from tigerbx import bx
from nilearn.image import resample_img, reorder_img
from typing import Union, Tuple, List
from scipy.ndimage import gaussian_filter
import optuna
import ants
warnings.filterwarnings("ignore", category=UserWarning)
ort.set_default_logger_severity(3)
nib.Nifti1Header.quaternion_threshold = -100



# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(os.path.abspath(__file__))
    
    
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
    

def pad_to_shape(img, target_shape):
    """
    Pads the input image with zeros to match the target shape.
    """
    padding = [(max(0, t - s)) for s, t in zip(img.shape, target_shape)]
    pad_width = [(p // 2, p - (p // 2)) for p in padding]
    padded_img = np.pad(img, pad_width, mode='constant', constant_values=0)
    return padded_img, pad_width


def crop_image(image, target_shape):
    """Crops the image to the target shape."""
    current_shape = image.shape
    crop_slices = []

    for i in range(len(target_shape)):
        start = (current_shape[i] - target_shape[i]) // 2
        end = start + target_shape[i]
        crop_slices.append(slice(start, end))

    cropped_image = image[tuple(crop_slices)]
    return cropped_image, crop_slices


def min_max_norm(img):
    max = np.max(img)
    min = np.min(img)

    norm_img = (img - min) / (max - min)

    return norm_img


def remove_padding(padded_img, pad_width):
    """
    Removes the padding from the input image based on the pad_width.
    """
    slices = [slice(p[0], -p[1] if p[1] != 0 else None) for p in pad_width]
    cropped_img = padded_img[tuple(slices)]
    return cropped_img


def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims], 
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = np.meshgrid(*[np.arange(s) for s in volshape], indexing='ij')
    grid = np.stack(grid_lst, axis=-1)

    # compute gradients
    J = np.gradient(disp + grid, axis=tuple(range(nb_dims)))

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2
        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def fwhm_to_sigma(fwhm):
    """
    Convert FWHM to sigma for Gaussian kernel.
    
    Parameters:
        fwhm (float): Full Width at Half Maximum.
        
    Returns:
        float: Corresponding sigma value.
    """
    return fwhm / np.sqrt(8 * np.log(2))


def apply_gaussian_smoothing(image, fwhm):
    """
    Apply Gaussian smoothing to a given image using FWHM.
    
    Parameters:
        image (numpy.ndarray): Input image to smooth.
        fwhm (float): Full Width at Half Maximum for Gaussian kernel.
        
    Returns:
        numpy.ndarray: Smoothed image.
    """
    sigma = fwhm_to_sigma(fwhm)
    smoothed_image = gaussian_filter(image, sigma=sigma)
    return smoothed_image

def dice(array1, array2, labels=None, include_zero=False):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.

    Parameters:
        array1: Input array 1.
        array2: Input array 2.
        labels: List of labels to compute dice on. If None, all labels will be used.
        include_zero: Include label 0 in label list. Default is False.
    """
    if labels is None:
        labels = np.concatenate([np.unique(a) for a in [array1, array2]])
        labels = np.sort(np.unique(labels))
    if not include_zero:
        labels = np.delete(labels, np.argwhere(labels == 0)) 

    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem

def FuseMorph_evaluate_params(params, warps, moving_seg, model_transform, fixed_seg_image, gpu):
    x, y, z = params
    warp = warps[0]*x + warps[1]*y + warps[2]*z
    output = lib_tool.predict(model_transform, [moving_seg, warp], GPU=gpu, mode='reg')
    #dice_output = lib_tool.predict(model_dice, [output[0], fixed_seg_image], GPU=gpu, mode='reg')
    #dice_score = np.mean(dice_output[0])
    output_np = output[0]
    fixed_seg_image_np = fixed_seg_image
    dice_scores = dice(output_np, fixed_seg_image_np)
    #return (x, y, z, dice_score, warp)
    return (x, y, z, np.mean(dice_scores), warp)

def optimize_fusemorph(warps, moving_seg, model_transform, fixed_seg_image, args):

    def objective(trial):
        x = trial.suggest_float("x", 0.9, 1.0)
        y = trial.suggest_float("y", 0.1, 1.0)
        z = trial.suggest_float("z", 0.1, 1.0)
        params = (x, y, z)
        x, y, z, dice_score, warp = FuseMorph_evaluate_params(
            params, warps, moving_seg, model_transform, fixed_seg_image, args.gpu
        )
        nonlocal best_dice, best_warp
        if dice_score > best_dice:
            best_dice = dice_score
            best_warp = warp

        return dice_score
    best_dice = float("-inf")
    best_warp = None
    moving_seg = np.expand_dims(np.expand_dims(moving_seg, axis=0), axis=1)
    sampler = optuna.samplers.TPESampler(seed=42) 
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=100)  # Number of trials can be adjusted
    best_params = study.best_params
    best_dice = study.best_value
    return best_params, best_dice, best_warp

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
    SyN_dense_warp = displacement_dict['SyN_dense_warp']
    SyNCC_dense_warp = displacement_dict['SyNCC_dense_warp']
    Fuse_dense_warp = displacement_dict['Fuse_dense_warp']

    if (init_flow is None or (isinstance(init_flow, np.ndarray) and not np.any(init_flow))) and (reference_info is None or (isinstance(reference_info, dict) and not reference_info)):
        raise ValueError("Both init_flow and reference_info are None or empty. The program cannot proceed.")

    if (dense_warp is not None or Fuse_dense_warp is not None) and (affine_matrix is None or (isinstance(affine_matrix, np.ndarray) and not np.any(affine_matrix))):
        raise ValueError("affine_matrix is None or empty, and at least one of dense_warp or Fuse_dense_warp is not None.")
    
    method_check = {'affine':'C2FViT', 'nonlinear':'VMnet'}
    if init_flow is None or init_flow.shape == ():
        method_check['affine'] = 'ants'
    
    ftemplate, f_output_dir = bx.get_template(image_path, output_dir, None)
    
    model_transform = lib_tool.get_model('mprage_transform_v002_near.onnx')
    model_transform_bili = lib_tool.get_model('mprage_transform_v002_bili.onnx')
    model_affine_transform = lib_tool.get_model('mprage_affinetransform_v002_near.onnx')
    model_affine_transform_bili = lib_tool.get_model('mprage_affinetransform_v002_bili.onnx')
        
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
        input_data = np.expand_dims(np.expand_dims(input_data, axis=0), axis=1)
        init_flow = init_flow.astype(np.float32)
    #Rigid    
    if rigid_matrix is not None and rigid_matrix.shape != ():
        rigid_matrix = np.expand_dims(rigid_matrix.astype(np.float32), axis=0)
        model = model_affine_transform if interpolation == 'nearest' else model_affine_transform_bili
        output = lib_tool.predict(model, [input_data, init_flow, rigid_matrix], GPU=GPU, mode='affine_transform')
        rigid = remove_padding(np.squeeze(output[0]), pad_width)
        rigid_nib = nib.Nifti1Image(rigid,
                                      template_nib.affine, template_nib.header)
        fn = bx.save_nib(rigid_nib, ftemplate, 'rigid')
    #Affine
    if method_check['affine'] == 'C2FViT':
        if affine_matrix is not None and affine_matrix.shape != ():
            affine_matrix = np.expand_dims(affine_matrix.astype(np.float32), axis=0)
            model = model_affine_transform if interpolation == 'nearest' else model_affine_transform_bili
            output = lib_tool.predict(model, [input_data, init_flow, affine_matrix], GPU=GPU, mode='affine_transform')
            affined = remove_padding(np.squeeze(output[0]), pad_width)
            affined_nib = nib.Nifti1Image(affined,
                                          template_nib.affine, template_nib.header)
            fn = bx.save_nib(affined_nib, ftemplate, 'Af')
    elif method_check['affine'] == 'ants':
        if isinstance(affine_matrix["parameters"], np.ndarray) and affine_matrix["parameters"].shape != ():
            ants_input, _ = get_ants_info(input_data, input_nib.affine)
            affined = ants_transform(ants_input, displacement_dict, interpolation=interpolation, mode='affine')
            
            affined_nib = nib.Nifti1Image(affined,
                                          template_nib.affine, template_nib.header)
            fn = bx.save_nib(affined_nib, ftemplate, 'Af')
    #Nonlinear(SyN, SyNCC)
    for ants_reg_str in ['SyN_dense_warp', 'SyNCC_dense_warp']:
        if displacement_dict[ants_reg_str] is not None and displacement_dict[ants_reg_str].shape != ():
            input_data = input_nib.get_fdata().astype(np.float32)
            ants_input, _ = get_ants_info(input_data, input_nib.affine)
            ants_output = ants_transform(ants_input, displacement_dict, interpolation=interpolation, mode=ants_reg_str)
            
            ants_output_nib = nib.Nifti1Image(ants_output,
                                          template_nib.affine, template_nib.header)
            fn = bx.save_nib(ants_output_nib, ftemplate, ants_reg_str.split("_")[0])
    #Nonlinear(VMnet)
    if dense_warp is not None and dense_warp.shape != ():
        affined_exp = np.expand_dims(np.expand_dims(affined, axis=0), axis=1)
        dense_warp = np.expand_dims(dense_warp.astype(np.float32), axis=0)
        model = model_transform if interpolation == 'nearest' else model_transform_bili
        output = lib_tool.predict(model, [affined_exp, dense_warp], GPU=GPU, mode='reg')
        reged = np.squeeze(output[0])
        reged_nib = nib.Nifti1Image(reged,
                                    template_nib.affine, template_nib.header)
        fn = bx.save_nib(reged_nib, ftemplate, 'reg')
    #Nonlinear(FuseMorph)
    if Fuse_dense_warp is not None and Fuse_dense_warp.shape != ():
        affined_exp = np.expand_dims(np.expand_dims(affined, axis=0), axis=1)
        Fuse_dense_warp = np.expand_dims(Fuse_dense_warp.astype(np.float32), axis=0)
        model = model_transform if interpolation == 'nearest' else model_transform_bili
        output = lib_tool.predict(model, [affined_exp, Fuse_dense_warp], GPU=GPU, mode='reg')
        Fused = np.squeeze(output[0])
        Fused_nib = nib.Nifti1Image(Fused,
                                    template_nib.affine, template_nib.header)
        fn = bx.save_nib(Fused_nib, ftemplate, 'Fuse')

def get_ants_info(image, affine):
    ants_img = ants.from_numpy(image)
    
    rot = affine[:3, :3]
    spacing = np.linalg.norm(rot, axis=0)
    direction_matrix = rot / spacing[:, np.newaxis]
    
    ants_img.set_origin(affine[:3, 3].tolist())
    ants_img.set_spacing(spacing.tolist())
    ants_img.set_direction(direction_matrix.tolist())
    
    ants_dict = {
        "reference_info": {
            "origin": ants_img.origin,
            "spacing": ants_img.spacing,
            "direction": ants_img.direction,
            "shape": ants_img.shape
            }
        }    
    return ants_img, ants_dict

def apply_ANTs_reg(ants_moving, ants_fixed, mode):
    registration = ants.registration(fixed=ants_fixed, moving=ants_moving, type_of_transform=mode)  
    transformed_moving_image = registration['warpedmovout']    
    transformed_array = transformed_moving_image.numpy()
    
    if mode=='Affine':
        aff_tx = ants.read_transform(registration['fwdtransforms'][0])
        ants_dict = {
            "affine_matrix": {
                "parameters": aff_tx.parameters,
                "fixed_parameters": aff_tx.fixed_parameters
                }
            }
    else:
        ants_dict = {mode + "_dense_warp": registration['fwdtransforms']}
    return transformed_array, ants_dict

def ants_transform(ants_moving, displacement_dict, interpolation='nearestNeighbor', mode='affine'):
    if interpolation == 'nearest':
        interpolation = 'nearestNeighbor'

    ref_info = displacement_dict["reference_info"]
    dummy_array = np.zeros(ref_info["shape"], dtype=np.float32)
    
    reference = ants.from_numpy(dummy_array)
    reference.set_origin(list(ref_info["origin"]))
    reference.set_spacing(list(ref_info["spacing"]))
    reference.set_direction(list(ref_info["direction"]))    
    
    if mode == 'affine':
        aff_tx = ants.create_ants_transform(dimension=3, transform_type="AffineTransform")
        aff_tx.set_parameters(displacement_dict["affine_matrix"]["parameters"])
        aff_tx.set_fixed_parameters(displacement_dict["affine_matrix"]["fixed_parameters"])
    
        resampled_img = ants.apply_ants_transform_to_image(
            transform=aff_tx,
            image=ants_moving,
            reference=reference,
            interpolation=interpolation  #linear
        )
    else:
        resampled_img = ants.apply_transforms(
        fixed=reference,
        moving=ants_moving,
        transformlist=displacement_dict[mode].tolist(),
        interpolator=interpolation  # segmentation 請務必使用 nearest
        )
        
    resampled_array = resampled_img.numpy()
    return resampled_array