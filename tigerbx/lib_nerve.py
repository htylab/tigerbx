import os
from pathlib import Path

import numpy as np
import nibabel as nib
from tigerbx.core.onnx import decode_latent as core_decode_latent, encode_latent as core_encode_latent
from tigerbx.core.resample import reorient_and_resample_voxel
from os.path import basename, join, isdir, dirname, relpath
from math import log10
from tigerbx.eval import ssim as ssim_metric

# ------------------------------------------------------------
# 1. tools
# ------------------------------------------------------------
def _resample_voxel(nib_obj, voxelsize=(1, 1, 1),
                    target_shape=None, interpolation="linear"):
    return reorient_and_resample_voxel(
        nib_obj,
        voxelsize=voxelsize,
        target_shape=target_shape,
        interpolation=interpolation,
    )

def _make_patch(aseg_nib, tbet_nib, roi_label,
                patch_size=(64, 64, 64)):
    aseg = aseg_nib.get_fdata()
    tbet = tbet_nib.get_fdata()
    mask = (aseg == roi_label)
    if not mask.any():
        print(f"label {roi_label} not found.")
        mask = mask * 0 + 1
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

    roi_map = {"LH": 17, "LA": 18, "RH": 53, "RA": 54}
    patches = {}
    for tag, lbl in roi_map.items():
        patch = _make_patch(aseg, tbet, lbl)
        patches[tag] = _intensity_rescale(patch)
    return patches

def nerve_preprocess_nib(aseg, tbet):
    aseg = _resample_voxel(aseg, interpolation="nearest")
    tbet = _resample_voxel(tbet, interpolation="linear")

    roi_map = {"LH": 17, "LA": 18, "RH": 53, "RA": 54}
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
    z_mu, z_sigma = core_encode_latent(enc_sess, vol)
    return z_mu, z_sigma

def onnx_decode(dec_sess, latent, affine=None):
    recon = core_decode_latent(dec_sess, latent)
    recon_vol = recon.squeeze()
    return recon_vol if affine is None else nib.Nifti1Image(recon_vol, affine)



# ------------------------------------------------------------------
# Utility metrics
# ------------------------------------------------------------------

def psnr(mse, peak):
    return float("inf") if mse == 0 else 20 * log10(peak) - 10 * log10(mse)


def compute_metrics(gt_nii, pred_nii, max_value=None):
    """
    Compute MAE, MSE, PSNR, SSIM for a pair of ground truth and predicted volumes.
    """
    gt = nib.load(gt_nii).get_fdata()
    pred = nib.load(pred_nii).get_fdata()
    diff = pred - gt
    mae = np.mean(np.abs(diff))
    mse = np.mean(diff**2)
    peak = np.max(gt) if max_value is None else max_value
    p = psnr(mse, peak)

    try:
        if gt.ndim == 4 and gt.shape[-1] <= 10:  # assume last dim is channels
            ssim_list = []
            for c in range(gt.shape[-1]):
                ssim_val_c = ssim_metric(gt[..., c], pred[..., c], data_range=peak)
                ssim_list.append(ssim_val_c)
            ssim_val = np.mean(ssim_list)
        elif gt.ndim in [2, 3]:
            ssim_val = ssim_metric(gt, pred, data_range=peak)
        else:
            ssim_val = np.nan
    except Exception as e:
        print(f"[WARN] SSIM computation failed: {e}")
        ssim_val = np.nan

    return mae, mse, p, ssim_val
