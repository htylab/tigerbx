import glob
from os.path import join, basename, isdir
import os
import numpy as np
import nibabel as nib
from scipy.special import softmax
from nilearn.image import reorder_img, resample_to_img, resample_img
from scipy.ndimage import gaussian_filter

from tigerbx import lib_tool
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

label_all = dict()
label_all['aseg43'] = (2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,30,31,41,42,43,
                44,46,47,49,50,51,52,53,54,58,60,62,63,77,85,251,252,253,254,255)
label_all['dkt'] = ( 1002, 1003,
               1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015,
               1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026,
               1027, 1028, 1029, 1030, 1031, 1034, 1035, 2002, 2003, 2005, 2006,
               2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
               2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028,
               2029, 2030, 2031, 2034, 2035)

label_all['wmp'] = (  251,  252,  253,  254,  255, 3001, 3002, 3003, 3005, 3006, 3007,
                     3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018,
                     3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029,
                     3030, 3031, 3032, 3033, 3034, 3035, 4001, 4002, 4003, 4005, 4006,
                     4007, 4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017,
                     4018, 4019, 4020, 4021, 4022, 4023, 4024, 4025, 4026, 4027, 4028,
                     4029, 4030, 4031, 4032, 4033, 4034, 4035)

label_all['synthseg'] = ( 2,  3,  4,  5,  7,  8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24,
                         26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60)
#nib.Nifti1Header.quaternion_threshold = -100


def get_mode(model_ff):
    seg_mode, version, model_str = basename(model_ff).split('_')[1:4]  # aseg43, bet

    #print(seg_mode, version , model_str)

    return seg_mode, version, model_str

def getLarea(input_mask):
    from scipy import ndimage
    labeled_mask, cc_num = ndimage.label(input_mask)
    if cc_num > 0:
        labeled_mask, cc_num = ndimage.label(input_mask)
        mask = (labeled_mask == (np.bincount(
            labeled_mask.flat)[1:].argmax() + 1))
    else:
        mask = input_mask
    return mask

def get_affine(mat_size):

    target_shape = np.array((mat_size, mat_size, mat_size))
    new_resolution = [256/mat_size, ]*3
    new_affine = np.zeros((4, 4))
    new_affine[:3, :3] = np.diag(new_resolution)
    # putting point 0,0,0 in the middle of the new volume - this could be refined in the future
    new_affine[:3, 3] = target_shape * new_resolution/2.*-1
    new_affine[3, 3] = 1.
    #print(model_ff, target_shape)
    #print(new_affine, target_shape)
    return new_affine, target_shape


def get_mat_size(model_ff):
    import re
    tmp = re.compile('r\d{2,4}.onnx').findall(basename(model_ff))
    mat_size = -1
    if len(tmp) > 0:
        mat_size = int(tmp[0].replace('.onnx', '')[1:])
    #print(model_ff, mat_size)
    return mat_size

def read_nib(input_nib):

    # in adni dataset, the 3D mprage is stored as a 4D array

    return np.squeeze(input_nib.get_fdata())

def reorient(nii, orientation="RAS"):
    orig_ornt = nib.orientations.io_orientation(nii.affine)
    targ_ornt = nib.orientations.axcodes2ornt(orientation)
    transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
    reoriented_nii = nii.as_reoriented(transform)
    return reoriented_nii

def logit_to_prob(logits, seg_mode):
    label_num = dict()
    label_num['bet'] = 2
    label_num['aseg43'] = 44
    label_num['dkt'] = 63
    label_num['dgm12'] = 13
    label_num['wmp'] = 74
    label_num['seg3'] = 4
    label_num['wmh'] = 2
    label_num['tumor'] = 2
    label_num['synthseg'] = 33
    #so far we only use sigmoid in tBET
    if label_num[seg_mode] > logits.shape[0]:
        #sigmoid
        th = 0.5
        from scipy.special import expit
        prob = expit(logits)
    else:
        #softmax mode
        #print(logits.shape)        
        prob = softmax(logits, axis=0)
    return prob

def run(model_ff_list, input_nib, GPU, patch=False):

    if not isinstance(model_ff_list, list):
        model_ff_list = [model_ff_list]


    seg_mode, _ , model_str = get_mode(model_ff_list[0]) 

    #data = input_nib.get_fdata()
    data = read_nib(input_nib)

    image = data[None, ...][None, ...]
    if seg_mode == 'synthseg':
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
    else:
        image = image/np.max(image)

    prob = 0
    count = 0
    for model_ff in model_ff_list:
        count += 1
        if patch:
            logits = lib_tool.predict(model_ff, image, GPU, mode='patch')[0, ...]
        else:
            logits = lib_tool.predict(model_ff, image, GPU)[0, ...]
        prob += logit_to_prob(logits, seg_mode)
    prob = prob/count # average the prob

    if seg_mode =='bet': #sigmoid 1 channel
        th = 0.5
        mask_pred = np.ones(prob[0, ...].shape)
        mask_pred[prob[0, ...] < th] = 0
        mask_pred = getLarea(mask_pred)
    else:    
        mask_pred = np.argmax(prob, axis=0)
    

    if seg_mode in ['aseg43', 'dkt', 'wmp', 'synthseg']:
        labels = label_all[seg_mode]
        mask_pred_relabel = mask_pred * 0
        for ii in range(len(labels)):
            mask_pred_relabel[mask_pred == (ii + 1)] = labels[ii]
            #print((ii+1), labels[ii])
        mask_pred = mask_pred_relabel



    mask_pred = mask_pred.astype(int)


    output_nib = nib.Nifti1Image(
        mask_pred, input_nib.affine, input_nib.header)
    
    return output_nib, prob



def read_file(model_ff, input_file):

    mat_size = get_mat_size(model_ff)
    input_nib = nib.load(input_file)
    zoom = input_nib.header.get_zooms()

    if mat_size == -1 or mat_size == 111:

        if max(zoom) > 1.1 or min(zoom) < 0.9 or mat_size == 111:

            vol_nib = resample_voxel(input_nib, (1, 1, 1), interpolation='continuous')
            vol_nib = reorder_img(vol_nib, resample='continuous')
        else:
            vol_nib = reorder_img(input_nib, resample='continuous')
    else:
        affine, shape = get_affine(mat_size)
        vol_nib = resample_img(input_nib,
                           target_affine=affine, target_shape=shape)

    return vol_nib




def write_file(model_ff, input_file, output_dir,
               mask, postfix=None, dtype='mask', inmem=False):
    seg_mode, _ , model_str = get_mode(model_ff)

    mask_dtype = mask.dtype

    if not isdir(output_dir):
        print('Output dir does not exist.')
        return 0

    if postfix is None:
        postfix = seg_mode
    
    output_file = basename(input_file).replace('.nii', f'_{postfix}.nii')    

    output_file = join(output_dir, output_file)
    
    input_nib = nib.load(input_file)
    input_affine = input_nib.affine
    zoom = input_nib.header.get_zooms()

    mat_size = get_mat_size(model_ff)

    if mat_size == -1:
        target_affine = reorder_img(nib.load(input_file),
                                     resample='linear').affine
    else:
        target_affine, _ = get_affine(mat_size)

    if dtype == 'orig':
        result = nib.Nifti1Image(mask.astype(input_nib.dataobj.dtype), target_affine)
    else:
        result = nib.Nifti1Image(mask.astype(mask_dtype), target_affine)
    result = resample_to_img(result, input_nib, interpolation="nearest")
    result.header.set_zooms(zoom)

    if not inmem:        
        nib.save(result, output_file)
        print('Writing output file: ', output_file)

    return output_file, result


def resample_voxel(data_nib, voxelsize,
                     target_shape=None, interpolation='continuous'):

    affine = data_nib.affine
    target_affine = affine.copy()

    factor = np.zeros(3)
    for i in range(3):
        factor[i] = voxelsize[i] / \
            np.sqrt(affine[0, i]**2 + affine[1, i]**2 + affine[2, i]**2)
        target_affine[:3, i] = target_affine[:3, i]*factor[i]

    new_nib = resample_img(data_nib, target_affine=target_affine,
                           target_shape=target_shape, interpolation=interpolation,
                           force_resample=True)

    return new_nib


def pad_to_shape(img, target_shape):
    """
    Pads the input image with zeros to match the target shape.
    """
    padding = [(max(0, t - s)) for s, t in zip(img.shape, target_shape)]
    pad_width = [(p // 2, p - (p // 2)) for p in padding]
    padded_img = np.pad(img, pad_width, mode='constant', constant_values=0)
    return padded_img, pad_width


def min_max_norm(img):
    max = np.max(img)
    min = np.min(img)

    norm_img = (img - min) / (max - min)

    return norm_img


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