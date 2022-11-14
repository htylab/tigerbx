import glob
from os.path import join, basename, isdir
import os
import numpy as np
import nibabel as nib
import onnxruntime as ort
from scipy.special import softmax
from .tigertool import cpu_count

nib.Nifti1Header.quaternion_threshold = -100


def run(model_ff, input_data, GPU):

    cpu = max(int(cpu_count()*0.8), 1)

    so = ort.SessionOptions()
    so.intra_op_num_threads = cpu
    so.inter_op_num_threads = cpu
    so.log_severity_level = 3


    if GPU and (ort.get_device() == "GPU"):
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
    xx, yy, zz, tt = data.shape

    if xyzt_mode == 'xyt':
        data = np.transpose(data, [0, 1, 3, 2])

    if xyzt_mode == 'xy':
        xx, yy, zz, tt = data.shape
        data = np.reshape(data, [xx, yy, zz*tt])

    #affine = temp.affine
    #zoom = temp.header.get_zooms()

    
    mask_pred4d = data * 0
    mask_softmax4d = np.zeros(np.insert(data.shape, 0, 4))

    
    for tti in range(data.shape[-1]):

        image_raw = data[..., tti]
        image = image_raw[None, ...][None, ...]
        if np.max(image) == 0:
            continue
        image = image/np.max(image)
   
        logits = session.run(None, {"modelInput": image.astype(np.float32)})[0]

        mask_pred = post(np.argmax(logits[0, ...], axis=0))
        mask_softmax = softmax(logits[0, ...], axis=0)

        #print(xyzt_mode, tti, image.max(), mask_pred.max(), image.shape)

        mask_pred4d[..., tti] = mask_pred
        mask_softmax4d[..., tti] = mask_softmax


    if xyzt_mode == 'xyt':
        mask_pred4d = np.transpose(mask_pred4d, [0, 1, 3, 2])
        mask_softmax4d = np.transpose(mask_softmax4d, [0, 1, 2, 4, 3])

    if xyzt_mode == 'xy':
        mask_pred4d = np.reshape(mask_pred4d, [xx, yy, zz, tt])
        mask_softmax4d = np.reshape(mask_softmax4d, [4, xx, yy, zz, tt])

    return mask_pred4d, mask_softmax4d


def read_file(model_ff, input_file):

    return nib.load(input_file).get_fdata()

def write_file(model_ff, input_file, output_dir, mask, inmem=False, postfix='pred'):

    if not isdir(output_dir):
        print('Output dir does not exist.')
        return 0

    output_file = basename(input_file).replace('.nii.gz', '').replace('.nii', '') 
    output_file = output_file + f'_{postfix}.nii.gz'
    output_file = join(output_dir, output_file)
    print('Writing output file: ', output_file)

    input_nib = nib.load(input_file)
    affine = input_nib.affine
    zoom = input_nib.header.get_zooms()   
    result = nib.Nifti1Image(mask.astype(np.uint8), affine)
    result.header.set_zooms(zoom)

    if not inmem:
        nib.save(result, output_file)

    return output_file, result



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
