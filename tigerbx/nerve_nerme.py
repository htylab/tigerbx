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
from tigerbx.lib_nerve import onnx_encode, onnx_decode

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
    parser.add_argument('-r', '--save_recon', action='store_true', help='Saving recon patches')

# ------------------------------------------------------------------
# 2. Encode-only helper (save all latents into one NPZ; optional patch merge)
# ------------------------------------------------------------------
def encode_nii(
    raw_path,
    encoder=None,
    decoder=None,
    output_dir="NERVE_latent",
    GPU=False,
    save_patch= False,
    f_template=None
):
    if GPU:
        provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        bx_string = 'bag'
    else:
        provider = ["CPUExecutionProvider"]
        bx_string = 'ba'
    
    f_output_dir = output_dir


    # --- Output folder
    if f_template==None: f_template = basename(raw_path)
    

    # --- Build encoder session
    onnx.checker.check_model(onnx.load(encoder))
    enc_sess = ort.InferenceSession(encoder, providers=provider)

    if decoder:
        onnx.checker.check_model(onnx.load(decoder))
        dec_sess = ort.InferenceSession(decoder, providers=provider)
    #input_name = enc_sess.get_inputs()[0].name
    #output_name = enc_sess.get_outputs()[0].name

    # --- Temporary workspace (auto-deleted)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = tigerbx.run(bx_string, raw_path, tmpdir, silent=True)
    #print('Cleaning up temporary files...')

    patches = nerve_preprocess_nib(result["aseg"], result["tbet"])

    # --- Encode each patch
    latent_dict = {}
    patch_arrays = []
    recon_arrays = []
    first_affine = None


    for tag, img in patches.items():
        #vol = img.get_fdata().astype(np.float32)[None, None]     # → [1,1,D,H,W]
        #latent = enc_sess.run([output_name], {input_name: vol})[0]
        latent = onnx_encode(enc_sess, img)[0]
        latent_dict[tag] = latent
        print(f"{tag} ▸ latent shape {latent.shape}")

        patch_arrays.append(img.get_fdata())
        if first_affine is None:
            first_affine = img.affine.copy()

        if decoder:            
            recon = onnx_decode(dec_sess, latent[None, ...])
            recon_vol = recon.squeeze()
            recon_arrays.append(recon_vol)




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

    if decoder and save_patch:
        merged = np.stack(recon_arrays, axis=-1).astype(np.float32)  # (D,H,W,#patch)
        patch_img = nib.Nifti1Image(merged, first_affine)
        patch_path = os.path.join(f_output_dir, f"{stem}_reconpatch.nii.gz")
        nib.save(patch_img, patch_path)
        print(f"Merged recon patch saved to: {patch_path}")

def nerve(input, output=None, model=None, GPU=False, method='NERVE', save_patch=False, save_recon=False):
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
    args.save_recon = save_recon
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
        omodel['decode'] = 'nerve_lp4_decoder.onnx'
    else:
        omodel['encode'] = 'nerve_lp4_encoder.onnx'
        omodel['decode'] = 'nerve_lp4_decoder.onnx'
        #omodel['decoder'] = 'nerve_lp4_decoder.onnx'
    omodel['encode'] = lib_tool.get_model(omodel['encode'])
    omodel['decode'] = lib_tool.get_model(omodel['decode'])

    if not args.save_recon: omodel['decode'] = None

    print(omodel['decode'])


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
        ftemplate, f_output_dir = get_ftemplate(f, output_dir, common_folder)

        print(f'Preprocessing {count}/{fcount}:\n', os.path.basename(f))
        t = time.time()
        
        encode_nii(f,
            encoder=omodel['encode'],
            decoder=omodel['decode'],
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