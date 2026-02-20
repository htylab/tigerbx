---
name: tigerbx
description: >
  Use tigerbx when the user needs to process structural brain MRI (T1-weighted NIfTI) images.
  Tigerbx provides deep-learning models for brain extraction, tissue segmentation, cortical
  thickness, registration to MNI space, EPI distortion correction, and hippocampus/amygdala
  embedding. Trigger this skill whenever the user mentions tasks such as brain extraction (BET),
  skull-stripping, segmentation, parcellation, cortical thickness, VBM, MNI registration,
  DTI/EPI distortion correction, or hippocampus encoding on NIfTI files.
argument-hint: "[module] [input] [output_dir]"
allowed-tools: Bash, Read, Glob
---

# TigerBx skill

TigerBx (`tigerbx`) is a Python package and CLI tool for deep-learning-based brain MRI analysis.

Install:
```bash
pip install --no-cache-dir "tigerbx[cpu] @ https://github.com/htylab/tigerbx/archive/release.zip"
# GPU (CUDA 12):
pip install --no-cache-dir "tigerbx[cu12] @ https://github.com/htylab/tigerbx/archive/release.zip"
```

CLI entry point: `tiger <subcommand> ...`

---

## When to use each module

| Situation | Module |
|-----------|--------|
| User wants to extract the brain / skull-strip a T1 image | `bx` |
| User wants brain mask, ASEG, cortical thickness, deep GM, WMH, CGW, tumor | `bx` |
| User wants hierarchical parcellation (56-region) or tissue probability maps (CSF/GM/WM) | `hlc` |
| User wants to register a T1 image to MNI space, run VBM, or apply a warp | `reg` |
| User wants to correct geometric distortions in a DTI / EPI image | `gdm` |
| User wants to encode hippocampus/amygdala patches to latent vectors | `nerve` |

---

## Module: `bx` — brain extraction and segmentation

### Python API
```python
import tigerbx
tigerbx.run(argstring, input, output=None, model=None, silent=False)
```

**`argstring` flags** (combine freely, e.g. `'bmad'`):

| Flag | Output (suffix) | Description |
|------|-----------------|-------------|
| `b`  | `_tbet`         | Brain-extracted image |
| `m`  | `_tbetmask`     | Binary brain mask |
| `a`  | `_aseg`         | ASEG 43-region tissue segmentation |
| `c`  | `_ct`           | Cortical thickness map |
| `C`  | `_cgw_pve0/1/2` | CSF / GM / WM probability maps (3 files) |
| `d`  | `_dgm`          | Deep gray matter mask (12 structures) |
| `S`  | `_syn`          | SynthSeg-style ASEG |
| `W`  | `_wmh`          | White matter hypointensity mask |
| `t`  | `_tumor`        | Tumor mask |
| `q`  | `_qc-N.log`     | QC score log |
| `g`  | —               | Use GPU |
| `p`  | —               | Patch-based inference |
| `z`  | —               | Force `.nii.gz` output |

Default (no flag): brain extraction (`b`).

```python
# examples
tigerbx.run('bm', 'T1w.nii.gz', 'output/')        # brain + mask
tigerbx.run('bmad', 'T1w.nii.gz', 'output/')      # recommended pipeline
tigerbx.run('bmacdCSWtq', '/data/T1w_dir')        # all outputs, save in-place
tigerbx.run('bmag', '/data/**/*.nii.gz', 'out/')  # GPU, glob input
tigerbx.run('clean_onnx')                          # remove cached models
```

### CLI
```bash
tiger bx T1w.nii.gz -bmad -o output/
tiger bx T1w.nii.gz -bm -o output/
tiger bx T1w.nii.gz -bmacdCSWtq -o output/
tiger bx /data/T1w_dir -bmag -o output/
tiger bx --clean_onnx
```

---

## Module: `hlc` — hierarchical label consolidation

Maps FreeSurfer-style labels to 56 hierarchical regions. Also computes cortical thickness and tissue probability maps. Developed by Pin-Chuan Chen.

### Python API
```python
tigerbx.hlc(input=None, output=None, model=None, save='h', GPU=False, gz=True, patch=False)
```

**`save` letters**:

| Letter | Output (suffix) | Description |
|--------|-----------------|-------------|
| `m`    | `_tbetmask`     | Brain mask |
| `b`    | `_tbet`         | Brain-extracted image |
| `h`    | `_hlc`          | HLC 56-region parcellation |
| `t`    | `_ct`           | Cortical thickness |
| `c`    | `_csf`          | CSF probability map |
| `g`    | `_gm`           | GM probability map |
| `w`    | `_wm`           | WM probability map |
| `all`  | all above       | Shorthand for `mbhtcgw` |

```python
tigerbx.hlc('T1w.nii.gz', 'output/')                      # HLC labels only
tigerbx.hlc('T1w.nii.gz', 'output/', save='all')          # all outputs
tigerbx.hlc('T1w.nii.gz', 'output/', save='tcgw', GPU=True)
```

### CLI
```bash
tiger hlc T1w.nii.gz -o output/
tiger hlc T1w.nii.gz --save all -o output/
tiger hlc T1w.nii.gz --save htcgw -g -o output/
tiger hlc /data/T1w_dir --save all -p -o output/
```

---

## Module: `reg` — registration and VBM

Affine and nonlinear registration to MNI space, VBM pipeline. Developed by Pei-Mao Sun.

### Python API
```python
tigerbx.reg(argstring, input=None, output=None, model=None,
            template=None, save_displacement=False, affine_type='C2FViT')
```

**`argstring` flags**:

| Flag | Description |
|------|-------------|
| `A`  | Affine registration (C2FViT or ANTs) |
| `r`  | VMnet nonlinear registration |
| `s`  | SyN nonlinear registration (ANTs) |
| `S`  | SyNCC nonlinear registration (ANTs) |
| `F`  | FuseMorph nonlinear registration |
| `R`  | Rigid registration |
| `v`  | Full VBM pipeline |
| `b`  | Also save brain-extracted image |
| `g`  | GPU |

`affine_type`: `'C2FViT'` (default) or `'ANTs'`. Affects `r`, `F`, `v`.

```python
tigerbx.reg('A', 'T1w.nii.gz', 'output/')
tigerbx.reg('Ar', 'T1w.nii.gz', 'output/', affine_type='C2FViT')
tigerbx.reg('F', '/data/T1w_dir', 'output/', affine_type='ANTs')
tigerbx.reg('v', '/data/**/*.nii.gz', 'output/')
# apply saved warp field
tigerbx.transform('moving.nii.gz', 'warp.npz', 'output/', interpolation='nearest')
```

### CLI
```bash
tiger reg T1w.nii.gz -A -o output/
tiger reg T1w.nii.gz -A -r -o output/ --affine_type C2FViT
tiger reg T1w.nii.gz -F -o output/ --affine_type ANTs
tiger reg /data/T1w_dir -v -o output/
tiger reg T1w.nii.gz -r -o output/ --save_displacement
```

---

## Module: `gdm` — EPI distortion correction

Corrects geometric distortions in DTI/EPI scans using a GAN-based model. No field maps needed. [Kuo et al., Magn Reson Med 2025]

### Python API
```python
tigerbx.gdm(input, output=None, b0_index=0, dmap=False, no_resample=False, GPU=False)
```

| Parameter     | Description |
|---------------|-------------|
| `b0_index`    | Index of b0 volume, or path to `.bval` file |
| `dmap`        | Also save predicted displacement map |
| `no_resample` | Skip resampling to 1.7×1.7×1.7 mm³ |

```python
tigerbx.gdm('dti.nii.gz', 'output/')
tigerbx.gdm('dti.nii.gz', 'output/', b0_index=1)
tigerbx.gdm('dti.nii.gz', 'output/', b0_index='dti.bval')
tigerbx.gdm('dti.nii.gz', 'output/', dmap=True, GPU=True)
```

### CLI
```bash
tiger gdm dti.nii.gz -o output/
tiger gdm dti.nii.gz -b0 1 -o output/
tiger gdm dti.nii.gz -b0 dti.bval -o output/
tiger gdm dti.nii.gz -m -g -o output/
```

---

## Module: `nerve` — hippocampus/amygdala VAE embedding

Encodes hippocampus and amygdala ROI patches to latent vectors using a VAE. For downstream tasks (e.g. Alzheimer's detection). Developed by Pei-Shin Chen.

### Python API
```python
tigerbx.nerve(argstring, input, output=None, model=None, method='NERVE')
```

**`argstring` flags**:

| Flag | Description |
|------|-------------|
| `e`  | Encode to latent `.npz` files |
| `d`  | Decode `.npz` back to patch images |
| `p`  | Save ROI patch images |
| `v`  | Evaluate reconstruction quality (enables `e`, `d`, `p`) |
| `g`  | GPU |
| `s`  | Variable (sigma) reconstruction |

```python
tigerbx.nerve('e', 'T1w.nii.gz', 'output/')       # encode only
tigerbx.nerve('ep', 'T1w.nii.gz', 'output/')      # encode + save patches
tigerbx.nerve('edp', 'T1w.nii.gz', 'output/')     # encode + decode + patches
tigerbx.nerve('v', 'T1w.nii.gz', 'output/')       # full evaluation
tigerbx.nerve('d', '/data/nerve_out', 'recon/')   # decode saved .npz files
```

### CLI
```bash
tiger nerve T1w.nii.gz -e -o output/
tiger nerve T1w.nii.gz -e -p -o output/
tiger nerve T1w.nii.gz -v -o output/
tiger nerve /data/T1w_dir -e -g -o output/
tiger nerve /data/nerve_out -d -o recon/
```

---

## Input conventions

- Input can be a single `.nii` / `.nii.gz` file, a directory, or a glob pattern (`/data/**/*.nii.gz`).
- When `output` is `None`, results are saved next to each input file.
- All modules accept `GPU=True` / `-g` for GPU inference (requires compatible onnxruntime).

## Installation

```bash
# CPU
pip install --no-cache-dir "tigerbx[cpu] @ https://github.com/htylab/tigerbx/archive/release.zip"
# GPU (CUDA 12)
pip install --no-cache-dir "tigerbx[cu12] @ https://github.com/htylab/tigerbx/archive/release.zip"

# Specific version (v0.2.x and later)
pip install --no-cache-dir "tigerbx[cpu] @ https://github.com/htylab/tigerbx/archive/refs/tags/v0.2.0.tar.gz"

# Archived 0.1.x versions (no extras required)
pip install https://github.com/htylab/tigerbx/archive/refs/tags/v0.1.20.tar.gz
```
