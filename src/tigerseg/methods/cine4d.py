import glob
from os.path import join, basename, isdir
import os
import numpy as np
from scipy.io import savemat
import nibabel as nib
import onnxruntime as ort
from scipy.special import softmax
nib.Nifti1Header.quaternion_threshold = -100


def run_SingleModel(model_ff, input_data, GPU):

    if GPU and (ort.get_device() == "GPU"):
        #ort.InferenceSession(model_file, providers=['CPUExecutionProvider'])
        session = ort.InferenceSession(model_ff, providers=['CUDAExecutionProvider'])
    else:
        session = ort.InferenceSession(model_ff, providers=['CPUExecutionProvider'])


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

def write_file(model_ff, input_file, output_dir, mask, report):

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

    if report:
        get_report(input_file, output_file)

    return output_file




def get_report(input_file, output_file):
    
    temp = nib.load(output_file)
    mask4d = temp.get_fdata()
    voxel_size = temp.header.get_zooms()
    
    LV_vol = np.sum(mask4d==1, axis=(0, 1, 2))* np.prod(voxel_size[0:3]) / 1000.
    LVM_vol = np.sum(mask4d==2, axis=(0, 1, 2))* np.prod(voxel_size[0:3]) / 1000.
    RV_vol = np.sum(mask4d==3, axis=(0, 1, 2))* np.prod(voxel_size[0:3]) / 1000.
    
    dict1 = {"input":np.asanyarray(nib.load(input_file).dataobj),
             'LV': (mask4d==1)*1,
             'LVM':(mask4d==2)*1,
             'RV': (mask4d==3)*1,
             'LV_vol': LV_vol,
             'LVM_vol': LVM_vol,
             'RV_vol': RV_vol}

    savemat(output_file.replace('.nii.gz', '.mat'), dict1, do_compression=True)


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
