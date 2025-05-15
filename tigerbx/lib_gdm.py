from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt2d
from os.path import basename, join
import SimpleITK as sitk
from os.path import join, basename, isdir
import numpy as np
import nibabel as nib
# import onnxruntime as ort
from scipy.special import softmax
from scipy.ndimage import median_filter
from scipy.ndimage import binary_dilation
from nilearn.image import resample_img


nib.Nifti1Header.quaternion_threshold = -100

def run(model_ff, input_data, b0_index, GPU, resample=True):
    import onnxruntime as ort
    
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

    vdm_mode = 'gan'
        
    orig_data = input_data  
    
    
    vdm_pred = gernerate_vdm(vdm_mode, session, orig_data, b0_index, resample=resample)

    output_vol = np.zeros(orig_data.shape)
    orig_data3d = orig_data.get_fdata()
    if len(orig_data.shape)==4:
        
        for bslice in range(orig_data.shape[3]):
            output_vol[...,bslice] = apply_vdm_3d(orig_data3d[...,bslice], vdm_pred, AP_RL='AP')
            
    else:
        
        output_vol = apply_vdm_3d(orig_data3d, vdm_pred, AP_RL='AP')

    

    return output_vol, vdm_pred


def read_file(model_ff, input_file):
    return nib.load(input_file)


def write_file(model_ff, input_file, output_dir, vol_out, inmem=False, postfix='gdmi'):

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
    

    if postfix=='vdm':
        result = nib.Nifti1Image(vol_out.astype('float32'), affine)
    else:
        overflow = 0
        if np.issubdtype(input_nib.get_data_dtype(), np.integer):
            if np.max(vol_out) > np.iinfo(input_nib.get_data_dtype()).max:
                overflow = 1
        elif np.issubdtype(input_nib.get_data_dtype(), np.floating):
            if np.max(vol_out) > np.finfo(input_nib.get_data_dtype()).max:
                overflow = 1

        result = nib.Nifti1Image(vol_out.astype(input_nib.get_data_dtype()), affine) if not overflow else nib.Nifti1Image(vol_out, affine)
        result.header.set_zooms(zoom)


    if not inmem:
        nib.save(result, output_file)

    return output_file, result


def predict(model, data):
    import onnxruntime as ort
    if model.get_inputs()[0].type == 'tensor(float)':
        return model.run(None, {model.get_inputs()[0].name: data.astype('float32')}, )[0]
    else:
        return model.run(None, {model.get_inputs()[0].name: data.astype('float64')}, )[0]

    
def get_b0_slice(ff):
    with open(ff) as f:
        bvals = f.readlines()[0].replace('\n', '').split(' ')
    bvals = [int(bval) for bval in bvals]
    return np.argmin(bvals)

    
def resample_to_new_resolution(data_nii, target_resolution, target_shape=None, interpolation='continuous'):
    affine = data_nii.affine
    target_affine = affine.copy()
    factor = np.zeros(3)
    for i in range(3):
        factor[i] = target_resolution[i] / np.sqrt(affine[0, i]**2 + affine[1, i]**2 + affine[2, i]**2)
        target_affine[:3, i] = target_affine[:3, i]*factor[i]
        
    new_nii = resample_img(data_nii, target_affine=target_affine,
                            target_shape=target_shape, interpolation=interpolation,
                            force_resample=True, copy_header=True)
    return new_nii


def apply_vdm_2d(ima, vdm, readout=1, AP_RL='AP'):

    if AP_RL == 'AP':
        arr = np.stack([vdm*readout, vdm*0], axis=-1)
    else:
        arr = np.stack([vdm*0, vdm*readout], axis=-1)
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


def apply_vdm_3d(ima, vdm, readout=1, AP_RL='AP'):

    if AP_RL == 'AP':
        arr = np.stack([vdm*0, vdm*readout, vdm*0], axis=-1)
    else:
        arr = np.stack([vdm*0, vdm*0, vdm*readout], axis=-1)
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



def gernerate_vdm(vdm_mode, session, orig_data, b0_index, resample=True):

    zoom = orig_data.header.get_zooms()[0:3]
    if len(orig_data.shape)>3:
        vol = orig_data.get_fdata()[...,b0_index]
    else:
        vol = orig_data.get_fdata()
    vol[vol<0] = 0
    
    if resample:
        resample_nii = resample_to_new_resolution(nib.Nifti1Image(vol, orig_data.affine), target_resolution=(1.7, 1.7, 1.7), target_shape=None, interpolation='continuous')
        vol_resize = resample_nii.get_fdata()
        vol_resize = ResizeWithPadOrCrop(vol_resize, (150, 150, 120))
        head_mask = get_head_mask(vol_resize, htype=1)!=0
        vol_resize = vol_resize / np.max(vol_resize)
    else:
        head_mask = get_head_mask(vol, htype=1)!=0
        vol_resize = vol / np.max(vol)
    
    # vol_resize_n = HistogramNormalize(vol_resize, mask=head_mask, num_bins=256, minv=0, maxv=1)
    # image = np.stack([vol_resize, vol_resize_n], axis=0)[None, ...]
    image = np.stack([vol_resize], axis=0)[None, ...]
    
    #sigmoid = session.run(None, {"modelInput": image.astype(np.float64)})[0]

    logits = predict(session, image)

    if resample:
        df_map = ResizeWithPadOrCrop(logits[0, 0, ...], resample_nii.shape)
        df_map = resample_to_new_resolution(nib.Nifti1Image(df_map, resample_nii.affine), target_resolution=zoom, target_shape=vol.shape, interpolation='linear').get_fdata() * 1.7 / zoom[1]
    else:
        df_map = logits[0, 0, ...]

    df_map_f = np.array(df_map*0, dtype='float64')
    for nslice in np.arange(df_map.shape[2]):
        df_map_slice = gaussian_filter(df_map[..., nslice], sigma=1.5).astype('float64')
        df_map_f[..., nslice] = df_map_slice
    vdm_pred = df_map_f

    

    return vdm_pred

def get_head_mask(ima, htype=1):
    if htype == 0:
        head_mask = ima * 0 + 1
    elif htype == 1:
        head_mask = median_filter(binary_dilation(ima > 15), size=3)        

    return head_mask


def HistogramNormalize( img: np.ndarray, 
                        mask: np.ndarray = None, 
                        num_bins: int = 256,
                        minv: int = 0,
                        maxv: int = 255,):
    orig_shape = img.shape
    hist_img = img[np.array(mask, dtype=bool)] if mask is not None else img
    
    hist, bins = np.histogram(hist_img.flatten(), num_bins)
    bins = (bins[:-1] + bins[1:]) / 2

    arr = hist.cumsum()
    # normalize the cumulative result
    
    mina = arr.min()
    maxa = arr.max()

    if mina == maxa:
        cum = arr * minv if minv is not None else arr

    else:
        norm = (arr - mina) / (maxa - mina)  # normalize the array first
        if (minv is None) or (maxv is None):
            cum = norm
        else:
            cum = (norm * (maxv - minv)) + minv  # rescale by minv and maxv, which is the normalized array by default

    # apply linear interpolation
    img = np.interp(img.flatten(), bins, cum)
    ret = img.reshape(orig_shape)
    
    out = ret.astype(img.dtype)
    
    return out

def ResizeWithPadOrCrop(image, image_size):

    slices = tuple()
    padding = []
    for i, size in enumerate(image_size):
        if image.shape[i] > size:
            d = image.shape[i] - size
            slices += (slice(int(d//2), int(image.shape[i]) - int(d/2 + 0.5)), )
            padding.append((0, 0))
            
        elif image.shape[i] < size:
            d = size - image.shape[i]
            slices += (slice(0, image.shape[i]), )
            padding.append((int(d//2), int(d/2 + 0.5)))
            
        else:
            slices += (slice(0, image.shape[i]), )
            padding.append((0, 0))
            
            
    image_resize = image.copy()
    image_resize = image_resize[slices]
    image_resize = np.pad(image_resize, 
                           padding, 
                           'constant', 
                           constant_values=np.zeros((len(image_size), 2)))
            
    return image_resize
