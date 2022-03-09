import pickle
import os
import collections
import logging
import sys
import nibabel as nib
import numpy as np
import glob as glob
from nilearn.image import reorder_img, new_img_like
from nilearn.image.image import _crop_img_to as crop_img_to
from .sitk_utils import resample_to_spacing, calculate_origin_offset


def preprocessing(data, mode=None):

    if mode is None:
        return data

    elif mode==0:
        from ..normalization import max_normalize
        return max_normalize(data)

    else:
        return data


def postprocessing(data, config, report=None, index=None):

    if config['postprocessing_mode'] is None:
        return

    elif config['postprocessing_mode']==0:
        
        mask_pred = data.get_fdata()
        factor = data.header.get_zooms()[0]*data.header.get_zooms()[1]*data.header.get_zooms()[2]
        for i in config['labels']:
            report.loc[index, f'{i}: '+config['labels_name'][i] if config['labels_name'] else i] = np.sum(mask_pred==i) * factor

        return

    elif config['postprocessing_mode']==1:

        mask_pred = np.array(data.get_fdata())
            
        if np.sum(mask_pred) == 0:
            type_pred = 0
        else:
            try:
                type_pred = np.bincount(mask_pred.astype('uint8').flatten())[1:].argmax() + 1
            except:
                type_pred = 0

        if type_pred==0:
            report.loc[index, 'High-Low Grade'] = 'No'
        else:
            report.loc[index, 'High-Low Grade'] = 'High' if type_pred>=3 else 'Low'

        return

    else:
        return


def get_affine(in_file):
    return read_image(in_file).affine


def read_image_files(image_files, image_shape=None, crop=None, label_indices=None):
    """
    :param image_files:
    :param image_shape:
    :param crop:
    :param use_nearest_for_last_file: If True, will use nearest neighbor interpolation for the last file. This is used
    because the last file may be the labels file. Using linear interpolation here would mess up the labels.
    :return:
    """

    if label_indices is None:
        label_indices = []
    elif not isinstance(label_indices, collections.Iterable) or isinstance(label_indices, str):
        label_indices = [label_indices]
    image_list = list()
    for index, image_file in enumerate(image_files):
        if (label_indices is None and (index + 1) == len(image_files)) \
                or (label_indices is not None and index in label_indices):
            interpolation = "nearest"
        else:
            interpolation = "linear"

        image_list.append(read_image(image_file, image_shape=image_shape, crop=crop, interpolation=interpolation))

    return image_list



def read_image(in_file, image_shape=None, preprocessing_mode=None, interpolation='linear', crop=None):
    image = nib.load(os.path.abspath(in_file))
    image_np = preprocessing(image.get_fdata(), mode=preprocessing_mode)
    image = nib.Nifti1Image(image_np,image.affine)
    image = fix_shape(image)

    if crop:
        image = crop_img_to(image, crop, copy=True)

    if image_shape:

        return resize(image, new_shape=image_shape, interpolation=interpolation) #for training

    else:
        return image


def read_image_by_mri_type(image_dir, image_shape=None, preprocessing_mode=None, crop=None, mri_types='fc12', interpolation='linear'):

    image_types = []
    for mri_type in mri_types.lower():
        if mri_type=='1':
            image_types.append('t1')
        elif  mri_type=='2':
            image_types.append('t2')
        elif  mri_type=='c':
            image_types.append('t1ce')
        elif  mri_type=='f':
            image_types.append('flair')
        else:
            raise ValueError(f'Get wrong MRI image type: {mri_type}')
    
    image_list = list()
    for image_type in image_types:
        image_file = glob.glob(os.path.join(image_dir, f'*{image_type}*.nii.gz'))[0]
        image_list.append(read_image(image_file, image_shape=image_shape, preprocessing_mode=preprocessing_mode, crop=crop, interpolation="linear"))

    return image_list



def fix_shape(image):
    if image.shape[-1] == 1:
        return image.__class__(dataobj=np.squeeze(image.get_data()), affine=image.affine)
    return image


def resize(image, new_shape, interpolation="linear"):
    image = reorder_img(image, resample=interpolation)
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)
    new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing,
                                   interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    image = new_img_like(image, new_data, affine=new_affine)

    return new_img_like(image, new_data, affine=new_affine)

def get_input_image(input):
    if os.path.isfile(input):
        logging.info(f'Read input: {input}')
        input_image = [input]

    else:
        logging.info(f'Looking for nii.gz in {input}')
        input_image = glob.glob(os.path.join(input,'*.nii.gz'))
        if len(input_image) < 1:
            sys.exit('No files found!')
    return input_image


def walk_input_dir(input):
    dirs=[]
    for root, _, files in os.walk(input):
        if np.char.array(files).count('.nii.gz').sum()>0:
            dirs.append(root)
    return dirs


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)