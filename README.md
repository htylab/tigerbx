# TigerBx: Tissue Mask Generation for Brain Extraction

<img src="./doc/team.png" alt="tigerbx" width="400">

## Overview

**TigerBx** is a deep learning toolkit for brain extraction and tissue segmentation. It includes:

* Pretrained models for structural brain segmentation.
* A stand-alone application for Windows, macOS, and Linux.
* Python APIs for advanced users and scripting.
* Designed for **research purposes only**. Not for commercial or clinical use.

<img src="./doc/tigerbx.png" alt="tigerbx" width="400">

---

## Installation

### Stand-alone Version

Download the latest release:
[https://github.com/htylab/tigerbx/releases](https://github.com/htylab/tigerbx/releases)

### Python Package

```bash
pip install onnxruntime              # For CPU
# or
pip install onnxruntime-gpu          # For GPU

pip install --no-cache https://github.com/htylab/tigerbx/archive/release.zip
```

To install a specific archived version:

```bash
pip install https://github.com/htylab/tigerbx/archive/refs/tags/v0.1.18.tar.gz
```

---

## Command-Line Usage

```bash
tiger bx -bmad c:\data\*.nii.gz -o c:\output
tiger bx -c c:\data\*.nii.gz -o c:\output
tiger bx -r c:\data\*.nii.gz -o c:\output -T template.nii.gz
tiger gdm DTI.nii.gz -o c:\outputdir
```

---

## Python API Usage

### Brain Segmentation

```python
import tigerbx

# Run full segmentation
tigerbx.run('bmadk', r'C:\T1w_dir', r'C:\output_dir')

# Wildcard input
tigerbx.run('bmadk', r'C:\T1w_dir\**\*.nii.gz', r'C:\output_dir')

# Output to same directory
tigerbx.run('bmadk', r'C:\T1w_dir\**\*.nii.gz')

# Deep gray matter segmentation with GPU
tigerbx.run('dg', r'C:\T1w_dir')
```

---

### Hierarchical Label Consolidation (HLC)

This model performs segmentation across **171 anatomical labels**, based on FreeSurfer's ASEG, DKT, and WMPARC definitions. It also provides:

* **Cortical thickness (CT)** maps
* **CGW** probability maps (CSF, gray matter, white matter)

With Hierarchical Label Consolidation, these 171 labels are reduced to **56 channels**, improving efficiency and reducing memory usage.

```python
import tigerbx
tigerbx.hlc('T1w_dir', 'outputdir')
```

---

### Registration and VBM

```python
# Standard registration
tigerbx.run('r', r'C:\T1w_dir', r'C:\output_dir', template='template.nii.gz', save_displacement=False)

# FuseMorph registration
tigerbx.run('F', r'C:\T1w_dir', r'C:\output_dir', save_displacement=False)

# Voxel-Based Morphometry
tigerbx.run('v', r'C:\T1w_dir\**\*.nii.gz', r'C:\output_dir')

# Apply warp field to image
tigerbx.transform(r'C:\T1w_dir\moving.nii.gz', r'C:\T1w_dir\warp.npz', r'C:\output_dir', interpolation='nearest')
```

---

### Generative Displacement Mapping (GDM)

```python
tigerbx.gdm('dti.nii.gz')
tigerbx.gdm(r'C:\EPI_dir', r'C:\output_dir', b0_index=0)  # Specify b0 slice index
```

---

### Utilities

```python
# Clean downloaded ONNX files
tigerbx.run('clean_onnx')

# Encode to latent space
tigerbx.run('encode', r'C:\T1w_dir', r'C:\output_dir')

# Decode from latent representation
tigerbx.run('decode', r'C:\npz_dir', r'C:\output_dir')
```

---

## Supported Systems

* ✅ Windows and macOS
* ✅ Ubuntu 20.04 or newer
* ⚠️ Without GPU, deep gray matter segmentation takes about 1 minute per scan

---

## Command-Line Flags Summary

```text
-b: Brain mask
-a: ASEG segmentation
-d: Deep gray matter mask
-k: DKT segmentation (WIP)
-c: Cortical thickness
-C: CSF/GM/WM PVEs (FSL style, WIP)
-S: SynthSeg-style ASEG (WIP)
-t: Tumor segmentation (WIP)
-w: White matter parcellation (WIP)
-W: White matter hypointensity mask (WIP)
-q: Save QC score (watch for scores < 30)
-z: Output as .nii.gz
-A: Affine registration to template (default MNI152)
-r: Nonlinear registration to template (default MNI152)
-F: FuseMorph registration to template
-T: Template filename
-R: Rigid registration
-p: Enable patch inference (160×160×160)
-v: Run VBM analysis
```

---

## Citation

If you use this toolkit, please cite:

1. **Weng JS, Huang TY.**
   *Deriving a robust deep-learning model for subcortical brain segmentation by using a large-scale database: Preprocessing, reproducibility, and accuracy of volume estimation.*
   **NMR Biomed. 2022; e4880.**
   [https://doi.org/10.1002/nbm.4880](https://doi.org/10.1002/nbm.4880)

2. **Wang HC et al. (2024).**
   *Comparative Assessment of Established and Deep Learning Segmentation Methods for Hippocampal Volume Estimation in Brain MRI Analysis.*
   **NMR in Biomedicine; e5169.**
   [https://doi.org/10.1002/nbm.5169](https://doi.org/10.1002/nbm.5169)

---

## Label Definitions

See [Label definitions](doc/seglabel.md) for details on segmentation labels.

---

## Validation

See [Validation](doc/validation.md) for model accuracy and performance benchmarks.

---

## Disclaimer

This software is intended for **research use only** and has **not** been reviewed or approved by the FDA or any similar regulatory body. It must **not** be used for diagnostic or clinical purposes.

The software is provided **"as is"**, without any warranty. Use is at your own risk. The authors and contributors are not liable for any damages arising from its use.

---
