# TigerBx Release Notes — v0.3.0

> **Breaking release.** Output names, registration defaults, the template and the license change, and several
> options are removed. Do not mix 0.2.x and 0.3.0 results in one study. See [Migration](#migration-from-02x).

## Highlights

- **Pipelines first**: `tigerbx.bx`, `seg`, `affine`, `norm`, `vbm` and `template` are plain functions with their
  own options; the lower-level modules are in `tigerbx.modules` (`tigerbx.run`, `reg`, `hlc` keep their names).
- **Stand-alone `tigerbx` executable** (Rust + ONNX Runtime, no Python) for Linux x64, macOS arm64 and Windows x64,
  with the same models and results as the Python package; `-g` runs on the GPU (CUDA on Linux, DirectML on
  Windows). It replaces the PyInstaller `tiger` binary and the `tiger` command. See [rust/README.md](../../rust/README.md).
- **New segmentation model, ASEG and TSEG**: the whole-brain segmentation uses the new model `mprage_aseg43_v0011`
  (same 43 labels; volumes change, so do not mix with 0.2.x results). `tigerbx.run('a')` still writes `_aseg` (the model
  output), and `run('t')` writes TSEG (`_tseg`: ASEG + label 99 `Unlabeled-Intracranial`, so TIV = all non-zero
  voxels). `tigerbx.seg()` also takes the deep-GM structures from DGM12. Brain extraction tBET is renamed tBX (`_tbx`,
  `_tbxmask`). See the [label table](../seglabel.md) for compatibility with earlier versions.
- **New default registration `AD`**: numreg affine (numpy / scipy, replaces ANTs; antspyx and optuna are no longer
  needed) + VFA deformable registration, on a new bundled template in the same MNI frame.
- **`affine` / `norm` for any contrast**: `intensity='raw'` (default), `'minmax'` or `'gnorm'` (T1 only), `mask=`,
  `template=`; every image is sampled once from its input grid; `_warp.npz` holds the registration as plain arrays.
- **CSF / GM / WM maps from TSEG** (`seg(save_tissue=True)`, used by VBM) instead of a separate model.
- **VBM for real batches**: one bad input no longer stops a run; `vbm_summary.csv` with TIV, tissue volumes,
  image-quality measures and `qc_flags`.
- **New `tigerbx.embed`** (regional VFA brain embedding) and **`tigerbx.template()`** (population templates).
- **Faster and consistent inference**: optimised default models return labels only; CPU and GPU give the same
  results; much lower GPU memory (optional cap `TIGERBX_GPU_MEM_LIMIT_GB`); `tigerbx[cu13]` install extra.
- **Batch robustness**: `tigerbx.val()` reports a failing file instead of skipping the set; previous tigerbx outputs in
  an input folder are no longer picked up as new inputs.
- **License**: CC BY-NC-SA 4.0 (was CC BY-NC 4.0).

## Migration from 0.2.x

| 0.2.x | 0.3.0 |
|---|---|
| `tigerbx.run('a')`, `_aseg` | unchanged (new model); for TIV and new analyses `tigerbx.run('t')` or `tigerbx.seg()`, `_tseg` |
| `tigerbx.run('b' / 'm')`, `_tbet`, `_tbetmask` | `tigerbx.bx()`, `_tbx`, `_tbxmask` |
| `tigerbx.pipeline('vbm', ...)` | `tigerbx.vbm(...)` |
| `tigerbx.reg('AV', ...)` (C2FViT + VMnet), `'AF'`, `'AN'`, `'AC'` | `tigerbx.reg('AD', ...)` or `tigerbx.norm()` |
| `affine_type='ANTs'` | `affine_type='numreg'` (alias, warns) |
| `_Af`, `_reg` | `_affine`, `_VXM` |
| summing all `_aseg` labels as brain volume | leave out label 99; with it the sum is TIV |
| `run` flags `q`, `S`, `W`, `C`; HLC `save` `c`, `g`, `w`, `t` | removed (`ValueError` names the replacement) |
| `result['QC']`, `tigerbx.qc_stat`; `tigerbx.nerve` | VBM `qc_flags`, `tigerbx.core.qa`; `tigerbx.embed` |
| `tiger` command / binary | Python API or the `tigerbx` executable |
| results tied to the old template | pass the old template file with `template=` |

---

- package version is bumped from **`0.2.3`** to **`0.3.0`**
