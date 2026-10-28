"""GDM: generative displacement mapping, a referenceless distortion correction of spin-echo EPI (b0 / DWI).

Per input: the b0 volume -> 1.7 mm, normalised, centre 150 x 150 x 120 -> the GDM network predicts a voxel
displacement map (VDM) along the phase-encoding axis -> back to the input grid (input voxels), smoothed in plane ->
every volume is resampled along that axis and modulated by the Jacobian. The network assumes the phase encoding
along the second array axis (A-P in an axial acquisition). Outputs: `_gdmi` (the corrected image, every volume of a
4D series) and, with `dmap`, `_gdm` (the VDM in input voxels, float32).
"""
import logging
import os
import warnings
from types import SimpleNamespace

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from tigerbx import lib_tool
from tigerbx.core import models
from tigerbx.core.io import load_nifti, save_nib
from tigerbx.core.deform import apply_vdm_3d
from tigerbx.core.onnx import create_session, predict_single_output
from tigerbx.core.resample import resample_to_new_resolution
from tigerbx.core.spatial import resize_with_pad_or_crop

nib.Nifti1Header.quaternion_threshold = -100

_logger = logging.getLogger('tigerbx')
_logger.addHandler(logging.NullHandler())

DEFAULT_MODEL = models.GDM
VOXEL_MM = 1.7
MODEL_SHAPE = (150, 150, 120)


def gdm(input=None, output=None, b0_index=0, dmap=False, no_resample=False, gpu=False, verbose=0, gz=True,
        model=None, GPU=None):
    """Distortion correction of spin-echo EPI (see the module docstring and doc/gdm.md).

    b0_index: the b0 volume of a 4D input (an index, or the .bval file of the series: its first b = 0 volume).
    dmap: also write `_gdm`, the displacement map. no_resample: give the network the image as it is (it must
    already be about 1.7 mm and 150 x 150 x 120). A failing input of a batch is logged and skipped (a warning lists
    them at the end); a single input raises. Returns the images dict for a single input, otherwise a list of dicts
    of the written files.
    """
    if GPU is not None:
        warnings.warn("tigerbx.gdm(GPU=...) is deprecated: use gpu=...", DeprecationWarning, stacklevel=2)
        gpu = GPU
    args = SimpleNamespace(input=input, output=output, b0_index=b0_index, dmap=dmap, no_resample=no_resample,
                           gpu=gpu, verbose=verbose, gz=gz, model=model)
    return run_args(args)


def run_args(args):
    from tigerbx.pipelines._common import batch
    state = {}

    def one(f, ftemplate):
        if 'session' not in state:
            state['session'] = create_session(lib_tool.get_model(getattr(args, 'model', None) or DEFAULT_MODEL),
                                              args.gpu)
        img = load_nifti(f)
        corrected, vdm = correct(state['session'], img, b0_volume(img, args.b0_index, f),
                                 resample=not args.no_resample)
        images = {'gdmi': _like(corrected, img)}
        if args.dmap:
            images['gdm'] = _like(vdm, img, np.float32)
        return images, {k: save_nib(v, ftemplate, k) for k, v in images.items()}

    return batch('gdm', args.input, args.output, getattr(args, 'gz', True), getattr(args, 'verbose', 0), one)


def b0_volume(img, b0_index, f=''):
    """Index of the b0 volume: `b0_index` an int, a digit string or a .bval path (the first b = 0 volume, else the
    smallest b). 0 for a 3D image."""
    if img.ndim < 4:
        return 0
    n = img.shape[3]
    if isinstance(b0_index, str) and not b0_index.strip().lstrip('-').isdigit():
        fb = b0_index if b0_index.endswith('.bval') else b0_index + '.bval'
        with open(fb) as fh:
            bvals = [float(b) for b in fh.read().split()]
        if len(bvals) != n:
            raise ValueError(f'{os.path.basename(fb)} has {len(bvals)} b-values for {n} volumes')
        index = int(np.argmin(bvals))
    else:
        index = int(b0_index)
    if not -n <= index < n:
        raise ValueError(f'b0_index {index} is out of range for {n} volumes of {os.path.basename(str(f))}')
    return index % n


def predict_vdm(session, vol, zooms, affine, resample=True):
    """The VDM of a 3D b0 volume on its own grid, in input voxels along the second axis."""
    vol = np.maximum(np.asarray(vol, np.float64), 0)
    if resample:
        small = resample_to_new_resolution(nib.Nifti1Image(vol, affine), target_resolution=(VOXEL_MM,) * 3,
                                           target_shape=None, interpolation='continuous')
        x = resize_with_pad_or_crop(small.get_fdata(), MODEL_SHAPE)
    else:
        x = vol
    mx = np.max(x)
    if mx > 0:
        x = x / mx
    out = predict_single_output(session, x[None, None])[0, 0]
    if resample:
        out = resize_with_pad_or_crop(out, small.shape)
        out = resample_to_new_resolution(nib.Nifti1Image(out, small.affine), target_resolution=zooms,
                                         target_shape=vol.shape, interpolation='linear').get_fdata()
        out = out * VOXEL_MM / zooms[1]
    vdm = np.zeros(out.shape)
    for k in range(out.shape[2]):
        vdm[..., k] = gaussian_filter(out[..., k].astype(np.float64), sigma=1.5)
    return vdm


def correct(session, img, b0=0, resample=True):
    """(corrected image array, VDM) of a 3D / 4D EPI image; the VDM comes from volume `b0`."""
    _check_phase_axis(img)
    data = np.asarray(img.dataobj, dtype=np.float64)
    if data.ndim > 4:
        raise ValueError(f'expected a 3D or 4D image, got shape {data.shape}')
    zooms = img.header.get_zooms()[:3]
    vdm = predict_vdm(session, data[..., b0] if data.ndim == 4 else data, zooms, img.affine, resample=resample)
    if data.ndim == 4:
        out = np.stack([apply_vdm_3d(data[..., v], vdm, AP_RL='AP') for v in range(data.shape[3])], -1)
    else:
        out = apply_vdm_3d(data, vdm, AP_RL='AP')
    return out, vdm


def _check_phase_axis(img):
    """Warn when the header says the phase encoding is not along the second axis, or that axis is not A-P."""
    phase = img.header.get_dim_info()[1] if hasattr(img.header, 'get_dim_info') else None
    axis = nib.aff2axcodes(img.affine)[1]
    if phase is not None and phase != 1:
        warnings.warn(f'the header gives the phase encoding along axis {phase}; GDM corrects along axis 1 '
                      '(A-P of an axial acquisition)', RuntimeWarning, stacklevel=4)
    elif axis not in ('A', 'P'):
        warnings.warn(f'the second image axis runs {axis}, not A-P; GDM corrects along that axis (A-P phase '
                      'encoding of an axial acquisition)', RuntimeWarning, stacklevel=4)


def _like(data, ref, dtype=None):
    """A NIfTI of `data` on the grid of `ref`: float32 for `dtype` float32, else the input's data type when the values
    fit (integers rounded), else float32."""
    dt = np.dtype(dtype or ref.get_data_dtype())
    if dt.kind in 'iu':
        info = np.iinfo(dt)
        if data.min() < info.min or data.max() > info.max:
            dt = np.dtype(np.float32)
        else:
            data = np.rint(data)
    elif dt != np.float64:
        dt = np.dtype(np.float32)
    out = nib.Nifti1Image(data.astype(dt), ref.affine)
    out.header.set_zooms(ref.header.get_zooms()[:data.ndim])
    out.header.set_xyzt_units(*ref.header.get_xyzt_units())
    return out
