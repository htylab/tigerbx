---
name: tigerbx
description: "Use tigerbx for structural brain MRI (T1w NIfTI): brain extraction (BET), skull-stripping, tissue segmentation, ASEG, parcellation, cortical thickness, VBM, MNI registration, DTI/EPI distortion correction, hippocampus/amygdala embedding."
argument-hint: "<bx|hlc|reg|gdm|nerve> <input.nii.gz> [output_dir]"
---

# TigerBx skill

TigerBx is a Python package for deep-learning-based brain MRI analysis.

> **Models are downloaded automatically on first use** and cached locally. No manual setup needed.

## Environment check

**Always verify the environment before using tigerbx.** Run this first:

```python
import importlib.metadata, packaging.version
try:
    v = importlib.metadata.version('tigerbx')
    assert packaging.version.Version(v) >= packaging.version.Version('0.2.0')
    print(f'tigerbx {v} ready')
except Exception:
    print('tigerbx >= 0.2.0 not found — install required')
```

If not installed or version is too old, install with:

```bash
# CPU
uv add "tigerbx[cpu] @ https://github.com/htylab/tigerbx/archive/release.zip"
# GPU (CUDA 12)
uv add "tigerbx[cu12] @ https://github.com/htylab/tigerbx/archive/release.zip"
```

---

## Module dispatch

| Task | Function | Detail |
|------|----------|--------|
| Brain extraction, skull-stripping | `tigerbx.run()` | [bx.md](bx.md) |
| Brain mask, ASEG, deep GM, WMH, cortical thickness, CGW, tumor | `tigerbx.run()` | [bx.md](bx.md) |
| Hierarchical parcellation (56 regions) or tissue probability maps | `tigerbx.hlc()` | [hlc.md](hlc.md) |
| Register T1 to MNI space, VBM, apply warp | `tigerbx.reg()` / `tigerbx.transform()` | [reg.md](reg.md) |
| EPI / DTI geometric distortion correction | `tigerbx.gdm()` | [gdm.md](gdm.md) |
| Hippocampus/amygdala VAE embedding | `tigerbx.nerve()` | [nerve.md](nerve.md) |

---

## Common conventions

- `input`: single `.nii`/`.nii.gz` path, a directory path, a glob pattern, or a list of paths.
- `output`: output directory string. When `None`, results are saved next to each input file.
- `GPU=True`: enable GPU inference (requires `onnxruntime-gpu`).
- Return value: dict of nibabel images (single input) or list of filename dicts (batch). Use `.get_fdata()` to access arrays.
- QC score is always computed. A `.log` is auto-written if QC < 50.

---

## Label definitions

See [labels.md](labels.md) for full label tables: ASEG (43), DeepGM (12), HLC (56/171), SynthSeg.
