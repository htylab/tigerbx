import glob
from os.path import join, basename, isdir
import os
import numpy as np
from scipy.io import savemat
import nibabel as nib
import onnxruntime as ort
from scipy.special import softmax
from skimage import transform
from nilearn.image import reorder_img, resample_to_img

labels = (2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,30,31,41,42,43,44,46,47,49,50,51,52,53,54,58,60,62,63,77,85,251,252,253,254,255)
nib.Nifti1Header.quaternion_threshold = -100

def run_SingleModel(model_ff, input_data, GPU):

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 4
    '''
    try:
        if os.cpu_count() is None:
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
        else:
            so.intra_op_num_threads = os.cpu_count()
            so.inter_op_num_threads = os.cpu_count()
    except:
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        
    '''
    if GPU and (ort.get_device() == "GPU"):
        #ort.InferenceSession(model_file, providers=['CPUExecutionProvider'])
        session = ort.InferenceSession(model_ff,
                                       providers=['CUDAExecutionProvider'],
                                       sess_options=so)
    else:
        session = ort.InferenceSession(model_ff,
                                       providers=['CPUExecutionProvider'],
                                       sess_options=so)


    seg_mode = basename(model_ff).split('_')[2] #aseg43, bet  

    orig_data = input_data.copy()

    input_shape = session.get_inputs()[0].shape
    do_resize = False


    if 'MXRW' in model_ff:
        do_resize = True
        data = transform.resize(orig_data, (128, 128, 128),
                                 preserve_range=True)
    elif 'Wang' in model_ff:
        xx, yy, zz = orig_data.shape
        data = orig_data
        if xx*yy*zz > 220**3 or np.max([xx, yy, zz]) > 300:
            do_resize = True
            data = transform.resize(orig_data, (220, 220, 220),
                                 preserve_range=True)
    elif 'x' not in input_shape:
        #fix input size
        do_resize = True
        infer_size = input_shape[2:]
        data = transform.resize(orig_data, infer_size,
                                 preserve_range=True)
    else:
        data = orig_data
    '''
    if 'MX_RW' in model_ff:
        data = transform.resize(orig_data, (128, 128, 128),
                                 preserve_range=True)
    else:
        data = orig_data
    '''

    image = data[None, ...][None, ...]
    image = image/np.max(image)

    #sigmoid = session.run(None, {"modelInput": image.astype(np.float64)})[0]

    logits = predict(session, image)[0, ...]

    label_num = dict()
    label_num['bet'] = 2
    label_num['aseg43'] = 44

    if label_num[seg_mode] > logits.shape[0]:
        #sigmoid mode
        mask_pred = np.argmax(logits, axis=0) + 1
        mask_pred[np.max(logits, axis=0) < 0.5] = 0
        prob = logits
    else:
        #softmax mode
        mask_pred = np.argmax(logits, axis=0)
        prob = softmax(logits, axis=0)

    if seg_mode == 'aseg43':

        mask_pred_relabel = mask_pred * 0
        for ii in range(len(labels)):
            mask_pred_relabel[mask_pred == (ii + 1)] = labels[ii]

        mask_pred = mask_pred_relabel

    if do_resize:
        mask_pred = transform.resize(mask_pred, orig_data.shape,
                                     order=0, preserve_range=True)

    return mask_pred.astype(np.uint8), prob


def read_file(model_ff, input_file):
    #reorder_img : reorient image to RAS 

    return reorder_img(nib.load(input_file), resample='linear').get_fdata()


def write_file(model_ff, input_file, output_dir, mask):

    if not isdir(output_dir):
        print('Output dir does not exist.')
        return 0

    output_file = basename(input_file).replace('.nii.gz', '').replace('.nii', '') 
    output_file = output_file + '_pred.nii.gz'
    output_file = join(output_dir, output_file)
    print('Writing output file: ', output_file)

    input_nib = nib.load(input_file)
    affine = input_nib.affine
    zoom = input_nib.header.get_zooms()   
    result = nib.Nifti1Image(mask.astype(np.uint8), reorder_img(input_nib, resample='linear').affine)
    
    result = resample_to_img(result, input_nib, interpolation="nearest")
    result.header.set_zooms(zoom)

    nib.save(result, output_file)

    return output_file


def predict(model, data):
    if model.get_inputs()[0].type=='tensor(float)':
        return model.run(None, {model.get_inputs()[0].name: data.astype('float32')}, )[0]
    else:
        return model.run(None, {model.get_inputs()[0].name: data.astype('float64')}, )[0]
