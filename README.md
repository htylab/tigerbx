# TigerBx: Tissue Mask Generation for Brain Extraction
<img src="./doc/team.png" alt="tigerbx" width="400">

## Overview

**TigerBx** is a deep learning toolkit for brain extraction and tissue segmentation. It provides:

* Pretrained models for structural brain segmentation.
* A stand-alone application for Windows, macOS, and Linux.
* Python APIs for advanced users and scripting.
* Designed strictly for **research purposes only**—not for clinical or commercial use.

<img src="./doc/tigerbx.png" alt="tigerbx" width="400">

---

## License

TigerBx is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license. Commercial use is not permitted. See `LICENSE` for details.

## Quick Start

### Install as a Python package

```bash
# CPU runtime
pip install --no-cache-dir "tigerbx[cpu] @ https://github.com/htylab/tigerbx/archive/release.zip"

# GPU runtime (CUDA 12)
pip install --no-cache-dir "tigerbx[cu12] @ https://github.com/htylab/tigerbx/archive/release.zip"
```

```python
import tigerbx

# Brain mask + brain image + ASEG + deep gray matter (recommended)
tigerbx.run('bmad', 'T1w.nii.gz', 'output_dir')
```

### Install a specific version

```bash
pip install --no-cache-dir "tigerbx[cpu] @ https://github.com/htylab/tigerbx/archive/refs/tags/v0.2.0.tar.gz"
pip install --no-cache-dir "tigerbx[cu12] @ https://github.com/htylab/tigerbx/archive/refs/tags/v0.2.0.tar.gz"
```

> **Note:** To install an archived version in the **0.1.x** series, use the simpler URL form (no extras required):
> ```bash
> pip install https://github.com/htylab/tigerbx/archive/refs/tags/v0.1.20.tar.gz
> ```

---

## Modules

### `bx` — Brain Extraction and Segmentation

```python
import tigerbx

# Brain mask + brain image + ASEG + deep gray matter (recommended)
tigerbx.run('bmad', 'T1w.nii.gz', 'output_dir')

# Full pipeline — all output types
tigerbx.run('bmacdCSWtq', 'T1w.nii.gz', 'output_dir')

# Process a directory; outputs saved next to each input file
tigerbx.run('bm', '/data/T1w_dir')

# Glob pattern, GPU
tigerbx.run('bmag', '/data/**/T1w.nii.gz', '/data/output')
```

```bash
tiger bx T1w.nii.gz -bmad -o output_dir
tiger bx T1w.nii.gz -bmacdCSWtq -o output_dir
tiger bx /data/T1w_dir -bmag -o /data/output
```

See [bx usage](doc/run.md) for a complete flag reference and output file naming.

---

### `hlc` — Hierarchical Label Consolidation

Maps FreeSurfer-style labels to 56 hierarchical regions. Also produces cortical thickness and CSF/GM/WM probability maps.

The HLC module was developed by **Pin-Chuan Chen**.

```python
import tigerbx

# Default: HLC parcellation only
tigerbx.hlc('T1w.nii.gz', 'output_dir')

# All outputs (brain mask, bet, HLC, cortical thickness, CSF/GM/WM)
tigerbx.hlc('T1w.nii.gz', 'output_dir', save='all')

# Cortical thickness + tissue probability maps with GPU
tigerbx.hlc('T1w.nii.gz', 'output_dir', save='tcgw', GPU=True)
```

```bash
tiger hlc T1w.nii.gz -o output_dir
tiger hlc T1w.nii.gz --save all -o output_dir
tiger hlc T1w.nii.gz --save tcgw -g -o output_dir
```

See [HLC usage](doc/hlc.md) for a complete description.

---

### `reg` — Registration and VBM

Supports affine (C2FViT / ANTs), VMnet, FuseMorph, SyN, and SyNCC registration, plus a full VBM pipeline.

The VBM and registration pipeline was developed by **Pei-Mao Sun**.

```python
import tigerbx

# Affine registration (C2FViT)
tigerbx.reg('A', r'C:\T1w_dir', r'C:\output_dir')

# Affine + VMnet nonlinear registration
tigerbx.reg('Ar', r'C:\T1w_dir', r'C:\output_dir', affine_type='C2FViT')

# FuseMorph with ANTs affine
tigerbx.reg('F', r'C:\T1w_dir', r'C:\output_dir', affine_type='ANTs')

# VBM pipeline
tigerbx.reg('v', r'C:\T1w_dir\**\*.nii.gz', r'C:\output_dir')

# Apply a saved warp field to a label map
tigerbx.transform(r'C:\moving.nii.gz', r'C:\warp.npz', r'C:\output_dir',
                  interpolation='nearest')
```

```bash
tiger reg T1w.nii.gz -A -o output_dir
tiger reg T1w.nii.gz -A -r -o output_dir --affine_type C2FViT
tiger reg T1w.nii.gz -F -o output_dir --affine_type ANTs
tiger reg /data/T1w_dir -v -o /data/output
```

See [registration instructions](doc/reginstruction.md) for detailed usage.

---

### `gdm` — Generative Displacement Mapping

Corrects geometric distortions in EPI scans using a GAN-based displacement field predictor, without requiring field maps or reversed-phase-encode acquisitions [Kuo et al., 2025].

```python
import tigerbx

# Correct a single DTI file
tigerbx.gdm('dti.nii.gz', 'output_dir')

# Specify b0 index or .bval file
tigerbx.gdm('dti.nii.gz', 'output_dir', b0_index=1)
tigerbx.gdm('dti.nii.gz', 'output_dir', b0_index='dti.bval')

# Save displacement map with GPU
tigerbx.gdm('dti.nii.gz', 'output_dir', dmap=True, GPU=True)
```

```bash
tiger gdm dti.nii.gz -o output_dir
tiger gdm dti.nii.gz -b0 1 -o output_dir
tiger gdm dti.nii.gz -b0 dti.bval -o output_dir
tiger gdm dti.nii.gz -m -g -o output_dir
```

See [GDM usage](doc/gdm.md) for a complete description.

---

### `nerve` — NERVE Embedding Pipeline

Extracts hippocampus and amygdala patches and encodes them into latent vectors using a variational autoencoder. Embeddings can be used for downstream tasks such as Alzheimer's disease detection.

The NERVE module was developed by **Pei-Shin Chen**.

```python
import tigerbx

# Encode to latent vectors
tigerbx.nerve('e', 'T1w.nii.gz', 'output_dir')

# Encode and save ROI patch images
tigerbx.nerve('ep', 'T1w.nii.gz', 'output_dir')

# Evaluate reconstruction quality
tigerbx.nerve('v', 'T1w.nii.gz', 'output_dir')

# Decode previously saved .npz files
tigerbx.nerve('d', '/data/nerve_out', '/data/recon_out')
```

```bash
tiger nerve T1w.nii.gz -e -o output_dir
tiger nerve T1w.nii.gz -e -p -o output_dir
tiger nerve T1w.nii.gz -v -o output_dir
tiger nerve /data/nerve_out -d -o /data/recon_out
```

See [NERVE usage](doc/nerve.md) for a complete description.

---

### `eval` — Image Quality and Segmentation Metrics

Computes quantitative metrics between a ground-truth and a predicted image.
Accepts NIfTI file paths, nibabel images, or numpy arrays.

```python
import tigerbx

# Segmentation — evaluate ASEG prediction against ground truth
result = tigerbx.run('a', 'T1w.nii.gz', 'output/')
scores = tigerbx.eval('gt_aseg.nii.gz', result['aseg'], 'dice',
                      labels=[10, 11, 17, 18])
# → {'dice': {'10': 0.91, '11': 0.89, '17': 0.94, '18': 0.92, 'mean': 0.915}}

# Multiple metrics at once
scores = tigerbx.eval('gt.nii.gz', 'pred.nii.gz', ['dice', 'hd95'],
                      labels=[1, 2, 3])

# Reconstruction quality (e.g. after GDM)
scores = tigerbx.eval('ref.nii.gz', 'pred_gdm.nii.gz', ['psnr', 'ssim', 'ncc'])
```

Supported metrics: `dice`, `iou`, `hd95`, `asd`, `mae`, `mse`, `psnr`, `ssim`, `ncc`, `mi`, `ksg_mi`, `accuracy`, `precision`, `recall`, `f1`.

See [eval usage](doc/eval.md) for the full API reference and examples.

---

## Agent Skills (Claude Code / Codex CLI)

TigerBx ships with a ready-to-use **skill pack** for AI coding assistants that support the skills standard, including [Claude Code](https://claude.ai/code) and Codex CLI.

Once installed, your assistant will automatically know when and how to call `tigerbx.run`, `tigerbx.hlc`, `tigerbx.reg`, `tigerbx.gdm`, `tigerbx.nerve`, and `tigerbx.eval` — without you having to explain the API.

### Install — Claude Code

```bash
# Project-level (this project only)
cp -r skills/tigerbx .claude/skills/

# User-level (all your projects)
cp -r skills/tigerbx ~/.claude/skills/
```

Reload Claude Code. The `/tigerbx` skill becomes available, and Claude will proactively use it for any brain MRI analysis task.

### Install — Codex CLI

In Codex CLI interactive mode, run:

```
$skill-installer https://github.com/htylab/tigerbx/tree/main/skills/tigerbx
```

### What the skill provides

| Skill file | Covers |
|---|---|
| `SKILL.md` | Environment check, module dispatch table, conventions |
| `bx.md` | `run()` flag reference, output naming |
| `hlc.md` | `hlc()` save options, tissue maps |
| `reg.md` | `reg()` / `transform()` registration modes |
| `gdm.md` | `gdm()` EPI distortion correction |
| `nerve.md` | `nerve()` hippocampus/amygdala VAE embedding |
| `eval.md` | `eval()` metrics, use cases, kwargs |
| `labels.md` | ASEG, DeepGM, HLC, SynthSeg label tables |

---

## Installation (stand-alone CLI)

Download the latest stand-alone release (no Python required):
[https://github.com/htylab/tigerbx/releases](https://github.com/htylab/tigerbx/releases)

After installation, all subcommands are available via `tiger`:

```bash
tiger bx --help
tiger hlc --help
tiger reg --help
tiger gdm --help
tiger nerve --help
```

---

## Supported Platforms

* Windows and macOS
* Ubuntu 20.04 or newer

---

## Citation

If you use TigerBx in your research, please cite the following:

1. **Weng JS, et al.** (2022) *Deriving a robust deep-learning model for subcortical brain segmentation by using a large-scale database: Preprocessing, reproducibility, and accuracy of volume estimation.* **NMR Biomed. 2022; e4880.** https://doi.org/10.1002/nbm.4880

2. **Wang HC et al.** (2024) *Comparative Assessment of Established and Deep Learning Segmentation Methods for Hippocampal Volume Estimation in Brain MRI Analysis.* **NMR in Biomedicine; e5169.** https://doi.org/10.1002/nbm.5169

3. **Kuo CC, et al.** (2025) *Referenceless reduction of spin-echo echo-planar imaging distortion with generative displacement mapping.* **Magn Reson Med.** 2025; 1–16. https://doi.org/10.1002/mrm.30577

4. **Sun PM, et al.** (2026) *DeepVBM: A fully automatic and efficient voxel-based morphometry via deep learning-based segmentation and registration methods.* **Magn Reson Imaging. 2026; 128: 110637.** https://doi.org/10.1016/j.mri.2026.110637

---

## Label Definitions

See [Label definitions](doc/seglabel.md) for a full list of anatomical regions used in segmentation outputs.

---

## Validation and Benchmarks

See [Validation](doc/validation.md) for accuracy, reproducibility, and comparison against other tools.

---

## Contributing

Contributions are welcome! See the [Developer Guide](doc/developer_guide.md) for instructions on setting up a local environment, branch and commit conventions, running tests, and submitting pull requests.

---

## Disclaimer

This software is intended solely for **research use** and has **not** been reviewed or approved by the FDA or any regulatory body. It must **not** be used for diagnostic, treatment, or other clinical purposes.

The software is provided **"as is"**, without warranty of any kind. The developers assume no responsibility for any consequences arising from the use of this software.
