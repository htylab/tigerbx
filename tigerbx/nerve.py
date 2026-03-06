#!/usr/bin/env python3
# coding: utf-8
# ------------------------------------------------------------
# NERVE – ONNX Encode / Decode Pipeline (npy interface version)
# ------------------------------------------------------------
from math import log10
import os
from os.path import basename, join

import numpy as np
import nibabel as nib
import tempfile
import tigerbx

import time
import glob
import logging
from tigerbx import lib_tool
from tigerbx.core.io import get_template, detect_common_folder
from tigerbx.core.onnx import decode_latent as core_decode_latent, encode_latent as core_encode_latent
from tigerbx.core.resample import reorient_and_resample_voxel
from tigerbx.core.metrics import ssim as ssim_metric

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm

_logger = logging.getLogger('tigerbx')
_logger.addHandler(logging.NullHandler())


def _resample_voxel(nib_obj, voxelsize=(1, 1, 1), target_shape=None, interpolation="linear"):
    return reorient_and_resample_voxel(
        nib_obj,
        voxelsize=voxelsize,
        target_shape=target_shape,
        interpolation=interpolation,
    )


def _make_patch(aseg_nib, tbet_nib, roi_label, patch_size=(64, 64, 64)):
    aseg = aseg_nib.get_fdata()
    tbet = tbet_nib.get_fdata()
    mask = aseg == roi_label
    if not mask.any():
        raise ValueError(f"label {roi_label} not found.")

    nz = np.nonzero(mask)
    center = [int((np.min(a) + np.max(a)) // 2) for a in nz]
    starts = [max(0, c - s // 2) for c, s in zip(center, patch_size)]
    ends = [min(d, st + s) for d, st, s in zip(tbet.shape, starts, patch_size)]
    crop = tbet[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
    return nib.Nifti1Image(crop, np.eye(4))


def _intensity_rescale(img):
    data = img.get_fdata()
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    return nib.Nifti1Image(data, img.affine)


def nerve_preprocess_nib(aseg, tbet):
    aseg = _resample_voxel(aseg, interpolation="nearest")
    tbet = _resample_voxel(tbet, interpolation="linear")

    roi_map = {"LH": 17, "LA": 18, "RH": 53, "RA": 54}
    patches = {}
    for tag, lbl in roi_map.items():
        patch = _make_patch(aseg, tbet, lbl)
        patches[tag] = _intensity_rescale(patch)
    return patches


def onnx_encode(enc_sess, patch_img):
    vol = patch_img.get_fdata().astype(np.float32)[None, None]
    z_mu, z_sigma = core_encode_latent(enc_sess, vol)
    return z_mu, z_sigma


def onnx_decode(dec_sess, latent):
    recon = core_decode_latent(dec_sess, latent)
    return recon.squeeze()


def psnr(mse, peak):
    return float("inf") if mse == 0 else 20 * log10(peak) - 10 * log10(mse)


def compute_metrics(gt_nii, pred_nii, max_value=None):
    gt = nib.load(gt_nii).get_fdata()
    pred = nib.load(pred_nii).get_fdata()
    diff = pred - gt
    mae = np.mean(np.abs(diff))
    mse = np.mean(diff ** 2)
    peak = np.max(gt) if max_value is None else max_value
    p = psnr(mse, peak)

    try:
        if gt.ndim == 4 and gt.shape[-1] <= 10:
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
        _logger.warning("[WARN] SSIM computation failed: %s", e)
        ssim_val = np.nan

    return mae, mse, p, ssim_val

# ------------------------------------------------------------
# Encode: NIfTI → latent .npz
# ------------------------------------------------------------
def encode_nii(
    raw_path,
    encoder,
    output_dir="NERVE_latent",
    GPU=False,
    save_patch=False,
    f_template=None,
):
    """
    Encode a 3-D NIfTI into latent vectors (.npz).
    """
    # Select execution providers
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if GPU else ["CPUExecutionProvider"]
    )
    bx_string = "bag" if GPU else "ba"

    # Prepare I/O
    os.makedirs(output_dir, exist_ok=True)
    if f_template is None:
        f_template = basename(raw_path)

    # Build encoder session
    import onnxruntime as ort
    enc_sess = ort.InferenceSession(encoder, providers=providers)

    # Temporary workspace for tigerbx preprocessing
    with tempfile.TemporaryDirectory() as tmpdir:
        result = tigerbx.run(bx_string, raw_path, tmpdir, verbose=0)

    # Patch extraction
    patches = nerve_preprocess_nib(result["aseg"], result["tbet"])

    latent_dict, patch_arrays, first_affine = {}, [], None
    for tag, img in patches.items():
        z_mu, z_sigma = onnx_encode(enc_sess, img)
        latent_dict[tag] = z_mu[0]
        latent_dict[tag + '_sigma'] = z_sigma[0]
        _logger.debug(f"{tag} > latent shape {z_mu.shape}")

        patch_arrays.append(img.get_fdata())
        if first_affine is None:
            first_affine = img.affine.copy()

    # Save latents
    # Store only the model filename so outputs are stable across model cache locations.
    latent_dict["encoder"] = np.array(basename(encoder), dtype=object)
    latent_dict["affine"] = first_affine


    npz_ff = os.path.join(output_dir, f"{f_template}_nerve.npz")
    np.savez_compressed(npz_ff, **latent_dict)
    _logger.info(f"[Encode] Latents saved → {npz_ff}")

    # Optional: save original patches
    patch_ff = ''
    if save_patch:
        patch_ff = os.path.join(output_dir, f"{f_template}_nerve_patch.nii.gz")
        merged = np.stack(patch_arrays, axis=-1).astype(np.float32)
        nib.save(
            nib.Nifti1Image(merged, first_affine),
            patch_ff,
        )
        _logger.info("[Encode] Patch stack saved.")

    return npz_ff, patch_ff

# ------------------------------------------------------------
# Decode: latent .npz → reconstruction patches
# ------------------------------------------------------------
def decode_npz(
    npz_path,
    decoder,
    output_dir="NERVE_recon",
    GPU=False,
    f_template=False,
    eps=0
):
    """
    Decode latent NPZ back to NIfTI patches or a merged volume.
    """
    # Build decoder session
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if GPU else ["CPUExecutionProvider"]
    )
    import onnxruntime as ort
    dec_sess = ort.InferenceSession(decoder, providers=providers)

    os.makedirs(output_dir, exist_ok=True)

    if not f_template: f_template = npz_path

    # Load latent dict
    data = np.load(npz_path, allow_pickle=True)
    affine = data["affine"]
    tags = [k for k in data.files if ('sigma' not in k) and (k not in ("encoder", "affine"))]
    recon_arrays = []

    for tag in tags:
        mu = data[tag]
        sigma = data[tag + '_sigma']
        latent = mu + eps*np.random.randn(*mu.shape).astype(np.float32) * sigma
        recon = onnx_decode(dec_sess, latent[None, ...])
        recon_arrays.append(recon)

    merged = np.stack(recon_arrays, axis=-1).astype(np.float32)
    recon_ff = os.path.join(output_dir, f"{f_template}_nerve_recon.nii.gz")
    nib.save(nib.Nifti1Image(merged, affine), recon_ff)
    _logger.info(f"[Decode] Merged reconstruction saved → {recon_ff}")

    return recon_ff


def nerve(argstring, input, output=None, model=None, method='NERVE', verbose=0):
    from types import SimpleNamespace as Namespace
    args = Namespace()
    if not isinstance(input, list):
        input = [input]
    args.input = input
    args.output = output
    args.model = model
    args.method = method
    args.save_patch = 'p' in argstring
    args.encode = 'e' in argstring
    args.decode = 'd' in argstring
    args.gpu = 'g' in argstring
    args.evaluate = 'v' in argstring
    args.sigma = 's' in argstring
    args.verbose = verbose
    return run_args(args)




def run_args(args):

    verbose = getattr(args, 'verbose', 0)

    def printer(*msg):
        if verbose >= 1:
            _logger.info(' '.join(str(x) for x in msg))

    def _warn(*msg):
        _logger.warning(' '.join(str(x) for x in msg))

    #run_d = vars(args) #store all arg in dict

    if args.evaluate:
        printer('[Evaluation mode] Saving patches for encoding and decoding')
        args.encode = True
        args.decode = True
        args.save_patch = True


    input_file_list = args.input
    if os.path.isdir(args.input[0]):
        if args.decode and (not args.evaluate):
            input_file_list = glob.glob(join(args.input[0], '*.npz'))
        else:
            input_file_list = glob.glob(join(args.input[0], '*.nii'))
            input_file_list += glob.glob(join(args.input[0], '*.nii.gz'))

    elif '*' in args.input[0]:
        input_file_list = glob.glob(args.input[0])

    output_dir = args.output
    omodel = dict()


    if args.method == 'NERVE':
        omodel['encode'] = 'nerve_lp4_encoderv2.onnx'
        omodel['decode'] = 'nerve_lp4_decoder.onnx'
    else:
        omodel['encode'] = 'nerve_lp4_encoderv2.onnx' #NERME ONNX not yet implemented
        omodel['decode'] = 'nerve_lp4_decoder.onnx' #NERME ONNX not yet implemented
        #omodel['decoder'] = 'nerve_lp4_decoder.onnx'

    # if you want to use other models
    if isinstance(args.model, dict):
        for mm in args.model.keys():
            omodel[mm] = args.model[mm]
    elif isinstance(args.model, str):
        import ast
        model_dict = ast.literal_eval(args.model)
        for mm in model_dict.keys():
            omodel[mm] = model_dict[mm]


    printer('Total files:', len(input_file_list))

    common_folder = detect_common_folder(input_file_list)

    ftemplate, f_output_dir = get_template(input_file_list[0], output_dir, common_folder=common_folder)

    os.makedirs(f_output_dir, exist_ok=True)

    fcount = len(input_file_list)
    recon_pairs = []
    max_value = 0 # for calculating PSNR
    results = []
    t = time.time()
    _pbar = tqdm(input_file_list, desc='tigerbx-nerve', unit='file', disable=(verbose > 0))
    for count, f in enumerate(_pbar, 1):
        ftemplate, f_output_dir = get_template(f, output_dir, common_folder=common_folder)

        _pbar.set_postfix_str(os.path.basename(f))
        printer(f'Preprocessing {count}/{fcount}:', os.path.basename(f))

        encode_ok = True
        decode_ok = True

        if args.encode:
            try:
                npz_ff, patch_ff = encode_nii(f,
                    encoder=lib_tool.get_model(omodel['encode']),
                    output_dir=f_output_dir,
                    GPU=args.gpu,
                    save_patch=args.save_patch,
                    f_template=os.path.basename(ftemplate))
                results.append(npz_ff)
                results.append(patch_ff)
            except Exception as e:
                encode_ok = False
                _warn('Encoding error:', e)
        if args.decode:
            try:
                if args.evaluate: f = npz_ff
                recon_ff = decode_npz(f,
                        decoder=lib_tool.get_model(omodel['decode']),
                        output_dir=f_output_dir,
                        GPU=args.gpu,
                        f_template=os.path.basename(ftemplate),
                        eps=args.sigma)
                results.append(recon_ff)
            except Exception as e:
                decode_ok = False
                _warn('Decoding error:', e)


        if args.evaluate and encode_ok and decode_ok: # for calculating PSNR
            recon_pairs.append((f, patch_ff, recon_ff))
            max_e = nib.load(patch_ff).get_fdata().max()
            max_value = max(max_e, max_value)

    if args.evaluate:
        import csv
        records = []
        for orig_f, patch_ff, recon_ff in recon_pairs:
            mae, mse, p, ssim_val = compute_metrics(patch_ff, recon_ff, max_value)
            records.append(dict(ID=orig_f, MAE=mae, MSE=mse, PSNR=p, SSIM=ssim_val))

        if records:
            n = len(records)
            avg = {
                "ID": "Average",
                "MAE": sum(r["MAE"] for r in records) / n,
                "MSE": sum(r["MSE"] for r in records) / n,
                "PSNR": sum(r["PSNR"] for r in records) / n,
                "SSIM": sum(r["SSIM"] for r in records) / n,
            }
            records.append(avg)

            csv_ff = join(f_output_dir, f'{omodel["encode"]}_eval.csv')
            with open(csv_ff, 'w', newline='') as fh:
                writer = csv.DictWriter(fh, fieldnames=["ID", "MAE", "MSE", "PSNR", "SSIM"])
                writer.writeheader()
                writer.writerows(records)
            printer(f'[Evaluation] Saving {csv_ff} report')

            printer(f"[Average] MAE = {avg['MAE']:.6f}, "
                  f"MSE = {avg['MSE']:.6f}, "
                  f"PSNR = {avg['PSNR']:.2f}, "
                  f"SSIM = {avg['SSIM']:.4f}")
            results = records

        printer('Processing time: %d seconds' % (time.time() - t))
    return results
