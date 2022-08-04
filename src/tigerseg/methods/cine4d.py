import glob
from os.path import join, basename, isdir
import os
import numpy as np
import nibabel as nib
import onnxruntime as ort
from scipy.special import softmax
nib.Nifti1Header.quaternion_threshold = -100


def run_SingleModel(model_ff, input_data, GPU):

 

    so = ort.SessionOptions()

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
    
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 4
    print('************', so.intra_op_num_threads)
    '''
    if GPU and (ort.get_device() == "GPU"):
        print('Using GPU...')
        #ort.InferenceSession(model_file, providers=['CPUExecutionProvider'])
        session = ort.InferenceSession(model_ff,
                                       providers=['CUDAExecutionProvider'],
                                       sess_options=so)
    else:
        session = ort.InferenceSession(model_ff,
                                       providers=['CPUExecutionProvider'],
                                       sess_options=so)


    xyzt_mode=basename(model_ff).split('_')[2]
    

    data = input_data.copy()

    #affine = temp.affine
    #zoom = temp.header.get_zooms()

    xx, yy, zz, tt = data.shape
    mask_pred4d = data * 0
    mask_softmax4d = np.zeros((4, xx, yy, zz, tt))
    for tti in range(data.shape[-1]):

        image_raw = data[..., tti]
        image = image_raw[None, ...][None, ...]
        if np.max(image) == 0:
            continue
        image = image/np.max(image)

        logits = session.run(None, {"modelInput": image.astype(np.float32)})[0]

        mask_pred = post(np.argmax(logits[0, ...], axis=0))
        mask_softmax = softmax(logits[0, ...], axis=0)

        mask_pred4d[..., tti] = mask_pred
        mask_softmax4d[..., tti] = mask_softmax


    return mask_pred4d, mask_softmax4d


def read_file(model_ff, input_file):

    return nib.load(input_file).get_fdata()

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
    result = nib.Nifti1Image(mask.astype(np.uint8), affine)
    result.header.set_zooms(zoom)

    #if 'mprage' in model_name:
    #result = resample_to_img(result, f, interpolation="nearest")

    nib.save(result, output_file)
    return output_file



def post(mask):

    def getLarea(input_mask):
        from scipy import ndimage
        try:
            labeled_mask, cc_num = ndimage.label(input_mask)
            mask = (labeled_mask == (np.bincount(labeled_mask.flat)[1:].argmax() + 1))
        except:
            mask = input_mask
        return mask

    masknew = mask * 0
    for jj in range(1, int(mask.max()) + 1):
        masknew[getLarea(mask == jj)] = jj

    return masknew
