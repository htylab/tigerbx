import math, os
from pathlib import Path

import numpy as np
import nibabel as nib
import onnx
import onnxruntime as ort
from nilearn.image import reorder_img, resample_img
from os.path import basename, join, isdir, dirname, commonpath, relpath

# ------------------------------------------------------------
# 1. tools
# ------------------------------------------------------------
def _resample_voxel(nib_obj, voxelsize=(1, 1, 1),
                    target_shape=None, interpolation="linear"):
    nib_obj = reorder_img(nib_obj, resample=interpolation)
    affine = nib_obj.affine.copy()
    affine[:3, :3] *= voxelsize / np.linalg.norm(affine[:3, :3], axis=0)
    return resample_img(nib_obj, target_affine=affine,
                        target_shape=target_shape, interpolation=interpolation)

def _make_patch(aseg_nib, tbet_nib, roi_label,
                patch_size=(64, 64, 64)):
    aseg = aseg_nib.get_fdata()
    tbet = tbet_nib.get_fdata()
    mask = (aseg == roi_label)
    if not mask.any():
        raise ValueError(f"label {roi_label} not found.")

    nz = np.nonzero(mask)
    center = [int((np.min(a) + np.max(a)) // 2) for a in nz]
    starts = [max(0, c - s // 2) for c, s in zip(center, patch_size)]
    ends   = [min(d, st + s) for d, st, s in zip(tbet.shape, starts, patch_size)]
    crop = tbet[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
    return nib.Nifti1Image(crop, np.eye(4))

def _intensity_rescale(img):
    d = img.get_fdata()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return nib.Nifti1Image(d, img.affine)

def nerve_preprocess(aseg_path, tbet_path):
    aseg = nib.load(aseg_path)
    tbet = nib.load(tbet_path)
    aseg = _resample_voxel(aseg, interpolation="nearest")
    tbet = _resample_voxel(tbet, interpolation="linear")

    roi_map = {"LH": 17, "LA": 53, "RA": 54, "RH": 18}
    patches = {}
    for tag, lbl in roi_map.items():
        patch = _make_patch(aseg, tbet, lbl)
        patches[tag] = _intensity_rescale(patch)
    return patches

def nerve_preprocess_nib(aseg, tbet):
    aseg = _resample_voxel(aseg, interpolation="nearest")
    tbet = _resample_voxel(tbet, interpolation="linear")

    roi_map = {"LH": 17, "LA": 53, "RA": 54, "RH": 18}
    patches = {}
    for tag, lbl in roi_map.items():
        patch = _make_patch(aseg, tbet, lbl)
        patches[tag] = _intensity_rescale(patch)
    return patches

# ------------------------------------------------------------
# 2. Encode / Decode
# ------------------------------------------------------------
def onnx_encode(enc_sess, patch_img):
    vol = patch_img.get_fdata().astype(np.float32)[None, None]
    latent = enc_sess.run(None, {enc_sess.get_inputs()[0].name: vol})[0]
    return latent

def onnx_decode(dec_sess, latent, affine=None):
    recon = dec_sess.run(None, {dec_sess.get_inputs()[0].name: latent})[0]
    recon_vol = recon.squeeze()
    return recon_vol if affine is None else nib.Nifti1Image(recon_vol, affine)

def encode_npy(enc_sess, patch_img, out_npy):
    latent = onnx_encode(enc_sess, patch_img)
    np.save(out_npy, latent)
    return latent

def decode_npy(dec_sess, latent_npy, affine=None, out_nii=None):
    latent = np.load(latent_npy)
    recon = onnx_decode(dec_sess, latent, affine)
    if out_nii and isinstance(recon, nib.Nifti1Image):
        nib.save(recon, out_nii)
    return recon


# ------------------------------------------------------------------
# Utility metrics
# ------------------------------------------------------------------
def _mae(a, b):                   # Mean Absolute Error
    return np.mean(np.abs(a - b))


def _mse(a, b):                   # Mean Squared Error
    return np.mean((a - b) ** 2)


def _psnr(a, b):                  # Peak Signal-to-Noise Ratio
    mse = _mse(a, b)
    return float("inf") if mse == 0 else 20 * math.log10(a.max() / math.sqrt(mse))

def get_ftemplate(f, output_dir, common_folder=None):
    f_output_dir = output_dir
    ftemplate = basename(f)

    if f_output_dir is None: #save the results in the same dir of T1_raw.nii.gz
        f_output_dir = os.path.dirname(os.path.abspath(f))
        
    else:
        os.makedirs(f_output_dir, exist_ok=True)
        #ftemplate = basename(f).replace('.nii', f'_@@@@.nii')
        # When we save results in the same directory, sometimes the result
        # filenames will all be the same, e.g., aseg.nii.gz, aseg.nii.gz.
        # In this case, the program tries to add a header to it.
        # For example, IXI001_aseg.nii.gz.
        if common_folder is not None:
            header = relpath(dirname(f), common_folder).replace(os.sep, '_')
            ftemplate = header + '_' + ftemplate
    
    return ftemplate, f_output_dir