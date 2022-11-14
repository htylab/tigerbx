from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt2d
from os.path import basename, join
import SimpleITK as sitk
from os.path import join, basename, isdir
import numpy as np
import nibabel as nib
import onnxruntime as ort
from scipy.special import softmax



nib.Nifti1Header.quaternion_threshold = -100

def run_SingleModel(model_ff, input_data, GPU):

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 4

    if GPU and (ort.get_device() == "GPU"):
        #ort.InferenceSession(model_file, providers=['CPUExecutionProvider'])
        session = ort.InferenceSession(model_ff,
                                       providers=['CUDAExecutionProvider'],
                                       sess_options=so)
    else:
        session = ort.InferenceSession(model_ff,
                                       providers=['CPUExecutionProvider'],
                                       sess_options=so)


    seg_mode = basename(model_ff).split('_')[2] #3dunet

    orig_data = input_data.copy()   

    if len(orig_data.shape) == 3:
        
        output_vol, vdm_pred = correct_3dvol(session, orig_data)
    
    elif len(orig_data.shape) == 4:
        output_vol = orig_data * 0
        vdm_pred = orig_data * 0.0
        for nn in range(orig_data.shape[-1]):
            orig_data3d = orig_data[..., nn]
            if nn == 0:
                output_vol[..., nn], vdm_pred = correct_3dvol(
                session, orig_data3d)
            else:
                output_vol[..., nn] = apply_vdm_3d(orig_data3d, vdm_pred)



    return output_vol.astype(np.int16), vdm_pred



    return output_vol, mask_softmax


def read_file(model_ff, input_file):
    #reorder_img : reorient image to RAS 

    return nib.load(input_file).get_fdata()


def write_file(model_ff, input_file, output_dir, vol_out, inmem=False):

    if not isdir(output_dir):
        print('Output dir does not exist.')
        return 0

    output_file = basename(input_file).replace('.nii.gz', '').replace('.nii', '') 
    output_file = output_file + '_vdm.nii.gz'
    output_file = join(output_dir, output_file)
    print('Writing output file: ', output_file)

    input_nib = nib.load(input_file)
    affine = input_nib.affine
    zoom = input_nib.header.get_zooms()
    result = nib.Nifti1Image(vol_out.astype(np.int16), affine)

    result.header.set_zooms(zoom)

    if not inmem:
        nib.save(result, output_file)

    return output_file, result


def predict(model, data):
    if model.get_inputs()[0].type == 'tensor(float)':
        return model.run(None, {model.get_inputs()[0].name: data.astype('float32')}, )[0]
    else:
        return model.run(None, {model.get_inputs()[0].name: data.astype('float64')}, )[0]


def apply_vdm_2d(ima, vdm, readout=1):

    arr = np.stack([vdm*readout, vdm*0], axis=-1)
    displacement_image = sitk.GetImageFromArray(arr, isVector=True)

    jac = sitk.DisplacementFieldJacobianDeterminant(displacement_image)
    tx = sitk.DisplacementFieldTransform(displacement_image)
    ref = sitk.GetImageFromArray(ima*0)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    # sitkNearestNeighbor, sitk.sitkLinear
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(tx)

    new_ima = resampler.Execute(sitk.GetImageFromArray(ima))
    new_ima = sitk.GetArrayFromImage(new_ima)
    jac_np = sitk.GetArrayFromImage(jac)
    return new_ima*jac_np


def apply_vdm_3d(vol, vdm_pred):
    output_vol = vol * 0

    for nslice in range(vol.shape[-1]):
        ima_org = vol[..., nslice].astype(np.float)
        vdm_slice = vdm_pred[..., nslice]
        ima_transform = apply_vdm_2d(ima_org, vdm_slice)

        output_vol[..., nslice] = ima_transform

    return output_vol


def correct_3dvol(session, orig_data):
    image = orig_data[None, ...][None, ...]

    image = image/np.max(image)

    #sigmoid = session.run(None, {"modelInput": image.astype(np.float64)})[0]

    logits = predict(session, image)

    mask_softmax = softmax(logits[0, ...], axis=0)

    softmax_all = 0
    for ii in range(101):
        softmax_all += mask_softmax[ii, ...] * ii

    vdm_pred = gaussian_filter(softmax_all*0.4 - 20, 0.5).astype(np.float)

    output_vol = apply_vdm_3d(orig_data, vdm_pred)

    return output_vol, vdm_pred
