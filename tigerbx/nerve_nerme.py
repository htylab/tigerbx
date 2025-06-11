#!/usr/bin/env python3
# coding: utf-8
# ------------------------------------------------------------
# NERVE – ONNX Encode / Decode Pipeline (npy interface version)
# ------------------------------------------------------------
import math
import os
import sys
from os.path import basename, join, isdir, dirname, commonpath, relpath
from pathlib import Path
from os.path import basename

import numpy as np
import nibabel as nib
import onnx
import onnxruntime as ort
import tempfile
import tigerbx

import argparse
import time
import glob
import platform
import nibabel as nib

from tigerbx import lib_tool
from tigerbx import lib_bx
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tigerbx.lib_nerve import nerve_preprocess_nib, encode_npy, decode_npy, _mae, _mse, _psnr, get_ftemplate


def main():
    parser = argparse.ArgumentParser()
    setup_parser(parser)
    args = parser.parse_args()
    run_args(args)

def setup_parser(parser):
    #parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, nargs='+', help='Path to the input image(s); can be a folder containing images in the specific format (nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output segmentation (default: the directory of input files)')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
    parser.add_argument('--model', default=None, type=str, help='Specifying the model name')
    parser.add_argument('--method', default='NERVE', type=str, help='Specifying the model name')
    parser.add_argument('-p', '--save_patch', action='store_true', help='Saving patches')

# ------------------------------------------------------------------
# 1. End-to-end evaluation (encode → decode → QA metrics)
# ------------------------------------------------------------------
def evaluate(
    t1_ff,
    encoder = "nerve_lp4_encoder.onnx",
    decoder = "nerve_lp4_decoder.onnx",
    outdir = "NERVE_output",
    provider = "CPUExecutionProvider",
):
    base = basename(t1_ff).replace(".nii.gz", "").replace(".nii", "")
    out_root = Path(outdir)
    out_root.mkdir(exist_ok=True)

    # --- Load ONNX models
    onnx.checker.check_model(onnx.load(encoder))
    onnx.checker.check_model(onnx.load(decoder))
    enc_sess = ort.InferenceSession(encoder, providers=[provider])
    dec_sess = ort.InferenceSession(decoder, providers=[provider])

    # --- Pre-processing
    with tempfile.TemporaryDirectory() as tmpdir:
        result = tigerbx.run("ba", t1_ff, tmpdir)   # BA pipeline
    patches = nerve_preprocess_nib(result["aseg"], result["tbet"])

    # --- Encode / Decode / Metrics
    stats = {}
    for tag, img in patches.items():
        patch_file = out_root / f"{base}_{tag}_patch.nii.gz"
        latent_file = out_root / f"{base}_{tag}_latent.npy"
        recon_file = out_root / f"{base}_{tag}_reconstruction.nii.gz"

        nib.save(img, patch_file)
        encode_npy(enc_sess, img, latent_file)

        recon_img = decode_npy(dec_sess, latent_file, img.affine, recon_file)
        vol = img.get_fdata().astype(np.float32)
        recon = recon_img.get_fdata().astype(np.float32)
       #stats[tag] = (_mae(vol, recon), _mse(vol, recon), _psnr(vol, recon))
        stats[tag] = dict(MAE=_mae(vol, recon), MSE= _mse(vol, recon), PSNR=_psnr(vol, recon))
        print(
            f"{tag} ▸ MAE:{stats[tag]['MAE']:.4e}  "
            f"MSE:{stats[tag]['MSE']:.4e}  "
            f"PSNR:{stats[tag]['PSNR']:.2f} dB"
        )

    # --- Aggregate metrics
    mae_avg = np.mean([v['MAE'] for v in stats.values()])
    mse_avg = np.mean([v['MSE'] for v in stats.values()])
    psnr_avg = np.mean([v['PSNR'] for v in stats.values()])
    print("--------------------------------------------------")
    print(f"Average ▸ MAE:{mae_avg:.4e}  MSE:{mse_avg:.4e}  PSNR:{psnr_avg:.2f} dB")
    print(f"All files written to: {out_root.resolve()}")

    # --- Return dictionary for later use
    return {
        "source":   t1_ff,
        "encoder":  encoder,
        "decoder":  decoder,
        "provider": provider,
        "outdir":   str(out_root),
        "patch_stats": stats,               # 每個 tag 的 MAE/MSE/PSNR
        "MAE_avg":  mae_avg,
        "MSE_avg":  mse_avg,
        "PSNR_avg": psnr_avg,
    }


def evaluate_dir(T1w_dir, output_dir='nerve_eval_output'):
    import pandas as pd
    import glob
    from os.path import join
    ffs = glob.glob(join(T1w_dir, '*.nii.gz'))
    records = []
    for vol in ffs:
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = evaluate(vol,
             encoder="nerve_lp4_encoder.onnx",
             decoder="nerve_lp4_decoder.onnx",
             outdir=tmpdir)                  # 設 False 可只存 latent
        
        for tag, vals in rec["patch_stats"].items():
            records.append({
                "Volume":  rec["source"],
                "Tag":     tag,
                **vals     # 展開 MAE / MSE / PSNR
            })
    
    df = pd.DataFrame(records)
    df.to_excel(join(output_dir, "nerve_metrics.xlsx"), index=False)


# ------------------------------------------------------------------
# 2. Encode-only helper (save all latents into one NPZ; optional patch merge)
# ------------------------------------------------------------------
def encode_nii(
    raw_path,
    encoder=None,
    output_dir="NERVE_latent",
    GPU=False,
    save_patch= False,
    f_template=None
):
    if GPU:
        provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        provider = ["CPUExecutionProvider"]
    
    f_output_dir = output_dir


    # --- Output folder
    if f_template==None: f_template = basename(raw_path)
    

    # --- Build encoder session
    onnx.checker.check_model(onnx.load(encoder))
    enc_sess = ort.InferenceSession(encoder, providers=provider)
    input_name = enc_sess.get_inputs()[0].name
    output_name = enc_sess.get_outputs()[0].name

    # --- Temporary workspace (auto-deleted)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = tigerbx.run("ba", raw_path, tmpdir, silent=True, GPU=GPU)
    print('Cleaning up temporary files...')

    patches = nerve_preprocess_nib(result["aseg"], result["tbet"])

    # --- Encode each patch
    latent_dict = {}
    patch_arrays = []
    first_affine = None

    for tag, img in patches.items():
        vol = img.get_fdata().astype(np.float32)[None, None]     # → [1,1,D,H,W]
        latent = enc_sess.run([output_name], {input_name: vol})[0]
        latent_dict[tag] = latent
        print(f"{tag} ▸ latent shape {latent.shape}")

        patch_arrays.append(img.get_fdata())
        if first_affine is None:
            first_affine = img.affine.copy()

    # --- Save latents
    latent_dict["encoder"] = np.array(encoder, dtype=object)
    latent_dict["affine"] = first_affine

    base = f_template
    stem = (
        base[:-7]
        if base.endswith(".nii.gz")
        else (base[:-4] if base.endswith(".nii") else base)
    )
    npz_path = os.path.join(f_output_dir, f"{stem}_nerve.npz")
    np.savez_compressed(npz_path, **latent_dict)
    print(f"Latents saved to: {npz_path}")

    # --- Optionally merge and save patches
    if save_patch:
        merged = np.stack(patch_arrays, axis=-1).astype(np.float32)  # (D,H,W,#patch)
        patch_img = nib.Nifti1Image(merged, first_affine)
        patch_path = os.path.join(f_output_dir, f"{stem}_nervepatch.nii.gz")
        nib.save(patch_img, patch_path)
        print(f"Merged patch saved to: {patch_path}")

def nerve(input, output=None, model=None, GPU=False, method='NERVE', save_patch=False):
    from argparse import Namespace
    args = Namespace()
    if not isinstance(input, list):
        input = [input]
    args.input = input
    args.output = output
    args.model = model
    args.gpu = GPU
    args.method = method
    args.save_patch = save_patch
    return run_args(args)




def run_args(args):

    run_d = vars(args) #store all arg in dict

 
    input_file_list = args.input
    if os.path.isdir(args.input[0]):
        input_file_list = glob.glob(join(args.input[0], '*.nii'))
        input_file_list += glob.glob(join(args.input[0], '*.nii.gz'))

    elif '*' in args.input[0]:
        input_file_list = glob.glob(args.input[0])

    output_dir = args.output
    omodel = dict()


    if args.method == 'NERVE':
        omodel['encode'] = 'nerve_lp4_encoder.onnx'
    else:
        omodel['encode'] = 'nerve_lp4_encoder.onnx'
        #omodel['decoder'] = 'nerve_lp4_decoder.onnx'
    omodel['encode'] = lib_tool.get_model(omodel['encode'])


    # if you want to use other models
    if isinstance(args.model, dict):
        for mm in args.model.keys():
            omodel[mm] = args.model[mm]
    elif isinstance(args.model, str):
        import ast
        model_dict = ast.literal_eval(args.model)
        for mm in model_dict.keys():
            omodel[mm] = model_dict[mm]


    print('Total nii files:', len(input_file_list))

    #check duplicate basename
    #for detail, check get_template
    base_ffs = [basename(f) for f in input_file_list]
    common_folder = None
    if len(base_ffs) != len(set(base_ffs)):
        common_folder = commonpath(input_file_list)

    ftemplate, f_output_dir = get_ftemplate(input_file_list[0], output_dir, common_folder)

    os.makedirs(f_output_dir, exist_ok=True)
        
    count = 0
    fcount = len(input_file_list)
    for f in input_file_list:
        count += 1

        print(f'Preprocessing {count}/{fcount}:\n', os.path.basename(f))
        t = time.time()
        
        encode_nii(f,
            encoder=omodel['encode'],
            output_dir=f_output_dir,
            GPU=args.gpu,
            save_patch=args.save_patch,
            f_template=ftemplate)

    
        print('Processing time: %d seconds' %  (time.time() - t))
    return 1


if __name__ == "__main__":
    main()
    if platform.system() == 'Windows':
        os.system('pause')