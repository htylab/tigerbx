# TigerBx: Brain Extraction, Segmentation and Registration of Brain MRI
<img src="./doc/team.png" alt="tigerbx" width="400">

## Overview

**TigerBx** is a deep learning toolkit for brain extraction, tissue segmentation and template registration of
structural brain MRI. It provides:

* **Pipelines**, the recommended way to use TigerBx: `inspect` (what each image is), `bx` (brain extraction), `seg`
  (segmentation), `affine` and `norm` (template normalisation), `vbm` (voxel-based morphometry) and `template`
  (population templates), from Python and from the stand-alone `tigerbx` executable (Windows, macOS, Linux).
* **Modules** (`tigerbx.infer`, `tigerbx.run`, `tigerbx.reg`, `tigerbx.hlc`, ...): the lower-level building blocks of the pipelines.
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

Check the installation with `tigerbx.check()` (`gpu=True`: whether the models run on the GPU): versions, ONNX
Runtime, threads, and the cached models (each checked against its release sha256; a damaged or outdated copy is
downloaded again).

### Install a specific version

```bash
pip install --no-cache-dir "tigerbx[cpu] @ https://github.com/htylab/tigerbx/archive/refs/tags/v0.5.1.tar.gz"
pip install --no-cache-dir "tigerbx[cu12] @ https://github.com/htylab/tigerbx/archive/refs/tags/v0.5.1.tar.gz"
pip install --no-cache-dir "tigerbx[cu13] @ https://github.com/htylab/tigerbx/archive/refs/tags/v0.5.1.tar.gz"
```

### Upgrading from 0.3.x or 0.2.x

From 0.5.0 TigerBx has two sets of models, and each set has its own output names, so results of the two are never
mixed up:

| | Current models | Earlier models (as in 0.3.x) |
|---|---|---|
| Python | `tigerbx.infer()` and the pipelines (`tigerbx.bx()`, `seg()`, `norm()`, ...) | `tigerbx.run()` |
| Executable | all options | — |
| Brain extraction | tBX: `_tbx`, `_tbxmask` | tBET: `_tbet`, `_tbetmask` |
| Whole-brain segmentation | TSEG: `_tseg` | ASEG: `_aseg` |
| DGM12, CT | `_dgm`, `_ct` inside the tBX mask | `_dgm`, `_ct` inside the tBET mask |

* **The tBX brain mask follows the brain surface**, as the tBET mask does. The mask that also includes the CSF
  around the brain, up to the inner surface of the skull, is the input of TSEG; `tigerbx.infer('x', ...)` or the
  executable's `-x` saves it (`_tbxoutermask`). TIV is the volume of all TSEG labels, as in SynthSeg.
* **The bundled templates were rebuilt** (T1, its labels, and the SE / BOLD EPI templates): the T1 template follows
  the brain surface as the tBX mask, and its labels are a vote of TSEG. Registered and normalised images
  (`_affine`, `_treg`, `_norm`) differ from 0.3.x.
* **Scripts written for the earlier models** keep running them with `tigerbx.run()`: `run('bmad')` gives the same
  images as 0.3.3. The brain files are named as in 0.2.x, `_tbet` and `_tbetmask` (0.3.x named them `_tbx` and
  `_tbxmask`).
* **`_tseg` and `_aseg`** come from two segmentation models, each with its own label table; see
  [Label definitions](doc/seglabel.md).
* **Registration plan `V` (VMnet) and `affine_type='C2FViT'` were removed**; use `AD` (numreg affine + tReg),
  which is more accurate.
* Within one study, use one set of models.

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
| `template` | `tigerbx.template(...)` | `tigerbx data/ -o tpl/ --template` | `template`, `template_mask`, `template_tseg` |
| `inspect`  | `tigerbx.inspect(...)`  | `tigerbx in.nii.gz --inspect` | `inspect_summary.csv` |
| `duope`    | `tigerbx.duope(fwd, rev, ...)` | `tigerbx --duope fwd rev` | `<input>_duope` (forward, reverse, series) |
| `dwimc`    | `tigerbx.dwimc(dwi, reverse=pa)` | `tigerbx --dwimc dwi --reverse pa` | `<dwi>_dwimc`, `.bval`, `.bvec`, `.csv` |
| `epinorm`  | `tigerbx.epinorm(series, t1, ap, pa)` | —          | `<series>_epi2t1`, `<series>_epi2mni` |
| `to_epi`   | `tigerbx.to_epi(t1, epi, labels=...)` | —          | `<t1>_t12epi`, `<label>_t12epi` |

```python
import tigerbx
tigerbx.seg(r'C:\T1w_dir', r'C:\output_dir', gpu=True)    # a directory of T1w images on the GPU
tigerbx.norm(r'C:\T1w_dir', r'C:\output_dir')             # template normalisation
tigerbx.vbm(r'C:\T1w_dir', r'C:\output_dir')              # voxel-based morphometry
```

See [pipeline usage](doc/pipelines.md) for the steps, options and outputs of each pipeline.

`tigerbx.duope` corrects EPI susceptibility distortion from a pair of scans with opposite phase encoding (AP / PA,
LR / RL), as FSL topup does, in seconds on the CPU, and can unwarp a DWI or BOLD series with the field. See
[duope](doc/duope.md). `tigerbx.dwimc` corrects head motion and eddy currents of a DWI series (and its
susceptibility distortion, given reversed-PE b0s) in one resampling: [dwimc](doc/dwimc.md). `tigerbx.epinorm`
brings a DWI / BOLD series onto the subject's T1 and the template in one resampling: [epinorm](doc/epinorm.md).

### Inspect your data first

`tigerbx.inspect` tells what each image is (contrast: T1w, T2w, FLAIR, b0, DWI or other), whether it is already
skull-stripped, whether it is EPI, whether its header orientation is right, whether the head is cut by the field of
view, and scores motion artifacts, plus header / data checks (4D, thick slices, oblique, NaN, label map), in about
1 s per image on the CPU. See [inspect](doc/inspect.md).

```python
tigerbx.inspect(r'C:\data_dir', r'C:\output_dir')    # -> inspect_summary.csv
```

### Stand-alone executable

The `tigerbx` executable is native (no Python) and runs the `bx`, `seg`, `affine`, `norm`, `inspect` and `template`
pipelines with the same models.
Download the zip for Linux (x64, glibc 2.28+), macOS (arm64) or Windows (x64) from
[https://github.com/htylab/tigerbx/releases](https://github.com/htylab/tigerbx/releases) and unpack it; keep the
files together (Linux / macOS: the ONNX Runtime library next to `tigerbx`).

```bash
tigerbx T1w.nii.gz                 # bx:   _tbx
tigerbx data/ -o out/ --seg        # seg:  _tbx, _tseg
tigerbx data/ -o out/ --affine     # affine: _tbx, _affine, _warp.npz
tigerbx data/ -o out/ --norm       # norm: _tbx, _norm, _warp.npz
tigerbx data/ -o out/ --norm --intensity gnorm   # T1 tissue intensities (CSF 1/3, GM 2/3, WM 1)
tigerbx data/ -o out/ --seg --norm # seg + norm
tigerbx data/ -o out/ --seg -g     # on the GPU
tigerbx --check -g                 # version, ONNX Runtime, threads, GPU, embedded models
tigerbx --help
```

See [rust/README.md](rust/README.md) for the single-output options and details.

---

## Modules

The modules (`tigerbx.infer`, `tigerbx.hlc`, `tigerbx.reg`, `tigerbx.embed`, `tigerbx.gdm`, `tigerbx.eval`,
`tigerbx.val`) are the lower-level building blocks of the pipelines; each gives a model's own output.
`tigerbx.run` runs the earlier models (see [Upgrading](#upgrading-from-03x-or-02x)). See the
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


## Contributing

Contributions are welcome! See the [Developer Guide](doc/developer_guide.md) for setting up a local environment,
branch and commit conventions, running tests and submitting pull requests. The people who built TigerBx are listed
in [CONTRIBUTORS.md](CONTRIBUTORS.md).

---

## Disclaimer

This software is intended solely for **research use** and has **not** been reviewed or approved by the FDA or any regulatory body. It must **not** be used for diagnostic, treatment, or other clinical purposes.

The software is provided **"as is"**, without warranty of any kind. The developers assume no responsibility for any consequences arising from the use of this software.
