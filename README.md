# TigerBx: Tissue Mask Generation for Brain Extraction
<img src="./doc/team.png" alt="tigerbx" width="400">

## Overview

**TigerBx** is a deep learning toolkit for brain extraction, tissue segmentation and template registration of
structural brain MRI. It provides:

* **Pipelines**, the recommended way to use TigerBx: `bx` (brain extraction), `seg` (segmentation), `affine` and
  `norm` (template normalisation), `vbm` (voxel-based morphometry) and `template` (population templates), from Python
  and from the stand-alone `tigerbx` executable (Windows, macOS, Linux).
* **Modules** (`tigerbx.run`, `tigerbx.reg`, `tigerbx.hlc`, ...): the lower-level building blocks of the pipelines.
* Designed strictly for **research purposes only**—not for clinical or commercial use.

---

## License

TigerBx, including the model weights it downloads, is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license. Commercial use is not permitted, and adaptations must be shared under the same license. See `LICENSE` for details.

## Quick Start

### Install as a Python package

```bash
# CPU runtime
pip install --no-cache-dir "tigerbx[cpu] @ https://github.com/htylab/tigerbx/archive/release.zip"

# GPU runtime (CUDA 12)
pip install --no-cache-dir "tigerbx[cu12] @ https://github.com/htylab/tigerbx/archive/release.zip"

# GPU runtime (CUDA 13, NVIDIA driver >= 580)
pip install --no-cache-dir "tigerbx[cu13] @ https://github.com/htylab/tigerbx/archive/release.zip"
```

```python
import tigerbx

tigerbx.bx('T1w.nii.gz', 'output_dir')          # _tbx
tigerbx.seg('T1w.nii.gz', 'output_dir')         # _tbx, _tseg
tigerbx.affine('T1w.nii.gz', 'output_dir')      # _tbx, _affine, _warp.npz
tigerbx.norm('T1w.nii.gz', 'output_dir')        # _tbx, _norm, _warp.npz
tigerbx.vbm('/data/T1w_dir', 'output_dir')      # _SmoothedGM, _norm, vbm_summary.csv
tigerbx.template('/data/T1w_dir', 'tpl_dir')    # template, template_mask, template_tseg (a population template)
```

### Install a specific version

```bash
pip install --no-cache-dir "tigerbx[cpu] @ https://github.com/htylab/tigerbx/archive/refs/tags/v0.3.0.tar.gz"
pip install --no-cache-dir "tigerbx[cu12] @ https://github.com/htylab/tigerbx/archive/refs/tags/v0.3.0.tar.gz"
pip install --no-cache-dir "tigerbx[cu13] @ https://github.com/htylab/tigerbx/archive/refs/tags/v0.3.0.tar.gz"
```

> **Note:** To install an archived version in the **0.1.x** series, use the simpler URL form (no extras required):
> ```bash
> pip install https://github.com/htylab/tigerbx/archive/refs/tags/v0.1.20.tar.gz
> ```

---

## Pipelines

The pipelines are TigerBx's main functions, the same in Python and in the stand-alone executable.

| Pipeline | Python | Executable | Outputs |
|----------|--------|------------|---------|
| `bx`       | `tigerbx.bx(...)`       | `tigerbx in.nii.gz`          | `_tbx` |
| `seg`      | `tigerbx.seg(...)`      | `tigerbx in.nii.gz --seg`    | `_tbx`, `_tseg` |
| `affine`   | `tigerbx.affine(...)`   | `tigerbx in.nii.gz --affine` | `_tbx`, `_affine`, `_warp.npz` |
| `norm`     | `tigerbx.norm(...)`     | `tigerbx in.nii.gz --norm`   | `_tbx`, `_norm`, `_warp.npz` |
| `vbm`      | `tigerbx.vbm(...)`      | —                            | `_SmoothedGM`, `_norm`, `vbm_summary.csv` |
| `template` | `tigerbx.template(...)` | —                            | `template`, `template_mask`, `template_tseg` |

```python
import tigerbx
tigerbx.seg(r'C:\T1w_dir', r'C:\output_dir', gpu=True)    # a directory of T1w images on the GPU
tigerbx.norm(r'C:\T1w_dir', r'C:\output_dir')             # template normalisation
tigerbx.vbm(r'C:\T1w_dir', r'C:\output_dir')              # voxel-based morphometry
```

See [pipeline usage](doc/pipelines.md) for the steps, options and outputs of each pipeline.

### Stand-alone executable

The `tigerbx` executable is native (no Python) and runs the `bx`, `seg`, `affine` and `norm` pipelines with the same
models.
Download the zip for Linux (x64, glibc 2.28+), macOS (arm64) or Windows (x64) from
[https://github.com/htylab/tigerbx/releases](https://github.com/htylab/tigerbx/releases) and unpack it; keep the
files together (Linux / macOS: the ONNX Runtime library next to `tigerbx`).

```bash
tigerbx T1w.nii.gz                 # bx:   _tbx
tigerbx data/ -o out/ --seg        # seg:  _tbx, _tseg
tigerbx data/ -o out/ --affine     # affine: _tbx, _affine, _warp.npz
tigerbx data/ -o out/ --norm       # norm: _tbx, _norm, _warp.npz
tigerbx data/ -o out/ --norm --intensity gnorm   # T1 tissue intensities (CSF 1/3, GM 2/3, WM 1)
tigerbx data/ -o out/ --all        # seg + norm
tigerbx data/ -o out/ --seg -g     # on the GPU
tigerbx --help
```

See [rust/README.md](rust/README.md) for the single-output options and details.

---

## Modules

The modules (`tigerbx.run`, `tigerbx.hlc`, `tigerbx.reg`, `tigerbx.embed`, `tigerbx.gdm`, `tigerbx.eval`,
`tigerbx.val`) are the lower-level building blocks of the pipelines; each gives a model's own output. See the
[documentation](doc/) for their options.

---

## Agent Skills (Claude Code / Codex CLI)

TigerBx ships a skill for AI coding assistants. Just ask Claude Code or Codex:

```
Install the tigerbx skill from https://github.com/htylab/tigerbx/tree/main/skills/tigerbx
```

After that the assistant knows when and how to use the TigerBx API.

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

Contributions are welcome! See the [Developer Guide](doc/developer_guide.md) for setting up a local environment,
branch and commit conventions, running tests and submitting pull requests. The people who built TigerBx are listed
in [CONTRIBUTORS.md](CONTRIBUTORS.md).

---

## Disclaimer

This software is intended solely for **research use** and has **not** been reviewed or approved by the FDA or any regulatory body. It must **not** be used for diagnostic, treatment, or other clinical purposes.

The software is provided **"as is"**, without warranty of any kind. The developers assume no responsibility for any consequences arising from the use of this software.
