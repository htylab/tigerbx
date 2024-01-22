import glob
from os.path import join, basename, isdir
import os
import numpy as np
import nibabel as nib
from scipy.special import softmax
from nilearn.image import reorder_img, resample_to_img, resample_img

from tigerbx import lib_tool

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
nib.Nifti1Header.quaternion_threshold = -100


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

def run(model_ff, input_nib, GPU):


    seg_mode, _ , model_str = get_mode(model_ff)
     


    #data = input_nib.get_fdata()
    data = read_nib(input_nib)


    image = data[None, ...][None, ...]
    image = image/np.max(image)


    logits = lib_tool.predict(model_ff, image, GPU)[0, ...]


    label_num = dict()
    label_num['bet'] = 2
    label_num['aseg43'] = 44
    label_num['dkt'] = 63
    label_num['dgm12'] = 13
    label_num['wmp'] = 74
    label_num['seg3'] = 4
    label_num['wmh'] = 2
    #so far we only use sigmoid in tBET
    if label_num[seg_mode] > logits.shape[0]:
        #sigmoid
        th = 0.5
        from scipy.special import expit
        logits = expit(logits)
        th = 0.5
        mask_pred = np.argmax(logits, axis=0) + 1
        mask_pred[np.max(logits, axis=0) <= th] = 0
        mask_pred = getLarea(mask_pred)
        prob = logits

    else:
        #softmax mode
        #print(logits.shape)
        mask_pred = np.argmax(logits, axis=0)
        prob = softmax(logits, axis=0)
    
    #qc1 = np.mean(probmax[mask_pred > 0])
    #qc2 = np.mean(probmax[(mask_pred > 0) & (probmax < 0.95)] ** 2)
    #sum1 = np.sum(mask_pred>0)
    #sum2 = sum1 - np.sum((probmax[mask_pred > 0] ** 10)< 0.99)

    if seg_mode in ['aseg43', 'dkt', 'wmp']:
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

        if max(zoom) > 1.2 or min(zoom) < 0.8 or mat_size == 111:

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
                           target_shape=target_shape, interpolation=interpolation)

    return new_nib





