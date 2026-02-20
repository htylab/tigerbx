---
name: tigerbx
description: "Use tigerbx for structural brain MRI (T1w NIfTI): brain extraction (BET), skull-stripping, tissue segmentation, ASEG, parcellation, cortical thickness, VBM, MNI registration, DTI/EPI distortion correction, hippocampus/amygdala embedding."
argument-hint: "<bx|hlc|reg|gdm|nerve> <input.nii.gz> [output_dir]"
---

# TigerBx skill

TigerBx is a CLI tool for deep-learning-based brain MRI analysis.
Entry point: `tiger <subcommand> ...`

> **Models are downloaded automatically on first use** and cached locally. No manual setup needed.

---

## Installation

```bash
# CPU (recommended — cleaner isolated environment)
uv add "tigerbx[cpu] @ https://github.com/htylab/tigerbx/archive/release.zip"

# GPU (CUDA 12)
uv add "tigerbx[cu12] @ https://github.com/htylab/tigerbx/archive/release.zip"

# Specific version
uv add "tigerbx[cpu] @ https://github.com/htylab/tigerbx/archive/refs/tags/v0.2.0.tar.gz"

# Archived 0.1.x (no extras required)
uv add https://github.com/htylab/tigerbx/archive/refs/tags/v0.1.20.tar.gz
```

---

## Module dispatch

| Task | Subcommand | Detail |
|------|-----------|--------|
| Brain extraction / skull-strip | `tiger bx` | [bx.md](bx.md) |
| Brain mask, ASEG, deep GM, WMH, cortical thickness, CGW, tumor | `tiger bx` | [bx.md](bx.md) |
| Hierarchical parcellation (56 regions) or tissue probability maps | `tiger hlc` | [hlc.md](hlc.md) |
| Register T1 to MNI space, VBM, apply warp | `tiger reg` | [reg.md](reg.md) |
| EPI / DTI geometric distortion correction | `tiger gdm` | [gdm.md](gdm.md) |
| Hippocampus/amygdala VAE embedding | `tiger nerve` | [nerve.md](nerve.md) |

---

## Common conventions

- Input: single `.nii`/`.nii.gz`, a directory, or a glob pattern (`/data/**/*.nii.gz`).
- `-o`: output directory. When omitted, results are saved next to each input file.
- `-g`: use GPU (requires `onnxruntime-gpu`).
- Output filenames follow `<input-stem>_<suffix>.nii.gz` — see each module for suffixes.
- QC score is always computed; a `.log` is written automatically if QC < 50.

---

## Label definitions

See [labels.md](labels.md) for full label tables: ASEG (43), DeepGM (12), HLC (56/171), SynthSeg.
