import os
from os.path import join
import glob
import nibabel as nib
import numpy as np
import time
import logging
import warnings
from scipy.ndimage import gaussian_filter
warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm

from tigerbx import lib_tool
from tigerbx.core.io import write_gdm_nifti_like_input
from tigerbx.core.deform import apply_vdm_3d
from tigerbx.core.onnx import predict_single_output as core_predict_single_output
from tigerbx.core.resample import resample_to_new_resolution as core_resample_to_new_resolution
from tigerbx.core.spatial import resize_with_pad_or_crop

nib.Nifti1Header.quaternion_threshold = -100

_logger = logging.getLogger('tigerbx')
_logger.addHandler(logging.NullHandler())

def gdm(input, output=None, b0_index=0, dmap=False, no_resample=False, GPU=False, verbose=0):

    from types import SimpleNamespace as Namespace
    args = Namespace()

    args.b0_index = str(b0_index)
    args.dmap = dmap
    args.no_resample = no_resample
    args.gpu = GPU
    args.verbose = verbose

    if not isinstance(input, list):
        input = [input]
    args.input = input
    args.output = output

    return run_args(args)


def run_args(args):

    verbose = getattr(args, 'verbose', 0)

    def printer(*msg):
        if verbose >= 1:
            _logger.info(' '.join(str(x) for x in msg))

    input_file_list = args.input
    if os.path.isdir(args.input[0]):
        input_file_list = glob.glob(join(args.input[0], '*.nii'))
        input_file_list += glob.glob(join(args.input[0], '*.nii.gz'))

    elif '*' in args.input[0]:
        input_file_list = glob.glob(args.input[0])

    output_dir = args.output

    if args.b0_index is None:
        b0_index = 0
    elif os.path.exists(args.b0_index.replace('.bval', '') + '.bval'):
        b0_index = get_b0_slice(args.b0_index.replace('.bval', '') + '.bval')
    else:
        b0_index = int(args.b0_index)

    resample = (not args.no_resample)

    printer('Total nii files:', len(input_file_list))

    model_name = lib_tool.get_model('vdm_unet3d_v002')

    result_all = []
    _pbar = tqdm(input_file_list, desc='tigerbx-gdm', unit='file', disable=(verbose > 0))
    for f in _pbar:

        _pbar.set_postfix_str(os.path.basename(f))
        printer('Predicting:', f)
        t = time.time()
        input_data = read_file(model_name, f)
        gdmi, gdmap = run(model_name, input_data, b0_index, GPU=args.gpu, resample=resample)

        if output_dir is None:
            f_output_dir = os.path.dirname(os.path.abspath(f))
        else:
            f_output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

        result_dict = {}
        fn, _ = write_gdm_nifti_like_input(f, f_output_dir, gdmi)
        result_dict['gdmi'] = fn

        if args.dmap:
            fn, _ = write_gdm_nifti_like_input(f, f_output_dir, gdmap, postfix='gdm')
            result_dict['gdm'] = fn

        printer('Processing time: %d seconds' % (time.time() - t))
        result_all.append(result_dict)

    if len(input_file_list) == 1:
        return result_all[0]
    return result_all


def run(model_ff, input_data, b0_index, GPU, resample=True):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 4

    if GPU and (ort.get_device() == "GPU"):
        session = ort.InferenceSession(
            model_ff,
            providers=['CUDAExecutionProvider'],
            sess_options=so,
        )
    else:
        session = ort.InferenceSession(
            model_ff,
            providers=['CPUExecutionProvider'],
            sess_options=so,
        )

    orig_data = input_data
    vdm_pred = gernerate_vdm(session, orig_data, b0_index, resample=resample)

    output_vol = np.zeros(orig_data.shape)
    orig_data3d = orig_data.get_fdata()
    if len(orig_data.shape) == 4:
        for bslice in range(orig_data.shape[3]):
            output_vol[..., bslice] = apply_vdm_3d(orig_data3d[..., bslice], vdm_pred, AP_RL='AP')
    else:
        output_vol = apply_vdm_3d(orig_data3d, vdm_pred, AP_RL='AP')

    return output_vol, vdm_pred


def read_file(_model_ff, input_file):
    return nib.load(input_file)


def predict(model, data):
    return core_predict_single_output(model, data)


def get_b0_slice(ff):
    with open(ff) as f:
        bvals = f.readlines()[0].replace('\n', '').split(' ')
    bvals = [int(bval) for bval in bvals]
    return np.argmin(bvals)


def resample_to_new_resolution(data_nii, target_resolution, target_shape=None, interpolation='continuous'):
    return core_resample_to_new_resolution(
        data_nii,
        target_resolution,
        target_shape=target_shape,
        interpolation=interpolation,
    )


def gernerate_vdm(session, orig_data, b0_index, resample=True):
    zoom = orig_data.header.get_zooms()[0:3]
    if len(orig_data.shape) > 3:
        vol = orig_data.get_fdata()[..., b0_index]
    else:
        vol = orig_data.get_fdata()
    vol[vol < 0] = 0

    if resample:
        resample_nii = resample_to_new_resolution(
            nib.Nifti1Image(vol, orig_data.affine),
            target_resolution=(1.7, 1.7, 1.7),
            target_shape=None,
            interpolation='continuous',
        )
        vol_resize = resample_nii.get_fdata()
        vol_resize = ResizeWithPadOrCrop(vol_resize, (150, 150, 120))
        mx = np.max(vol_resize)
        if mx > 0:
            vol_resize = vol_resize / mx
    else:
        mx = np.max(vol)
        if mx > 0:
            vol_resize = vol / mx
        else:
            vol_resize = vol

    image = np.stack([vol_resize], axis=0)[None, ...]
    logits = predict(session, image)

    if resample:
        df_map = ResizeWithPadOrCrop(logits[0, 0, ...], resample_nii.shape)
        df_map = resample_to_new_resolution(
            nib.Nifti1Image(df_map, resample_nii.affine),
            target_resolution=zoom,
            target_shape=vol.shape,
            interpolation='linear',
        ).get_fdata() * 1.7 / zoom[1]
    else:
        df_map = logits[0, 0, ...]

    df_map_f = np.array(df_map * 0, dtype='float64')
    for nslice in np.arange(df_map.shape[2]):
        df_map_slice = gaussian_filter(df_map[..., nslice], sigma=1.5).astype('float64')
        df_map_f[..., nslice] = df_map_slice
    vdm_pred = df_map_f

    return vdm_pred

def ResizeWithPadOrCrop(image, image_size):
    return resize_with_pad_or_crop(image, image_size)
