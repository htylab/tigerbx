---
name: tigerbx
description: "Use TigerBx 0.3+ for research-use structural brain MRI processing: brain extraction, whole-brain and deep-GM segmentation, template normalisation (affine or nonlinear, any contrast), VBM, population templates, HLC parcellation, registration, EPI distortion correction, regional brain embeddings and image metrics. Trigger when writing or reviewing code that calls tigerbx, choosing a TigerBx pipeline, running the tigerbx executable, or interpreting its output files and labels."
---

# TigerBx

Research use only: never make clinical or diagnostic claims from TigerBx outputs.

## Pick the API

Prefer the pipelines; use the modules only for a model's raw output or custom workflows.

| Task | Python | Executable | Outputs |
|---|---|---|---|
| Brain extraction | `tigerbx.bx(input, output)` | `tigerbx in.nii.gz` | `_tbx` (`save_mask=True`: `_tbxmask`) |
| Whole-brain segmentation | `tigerbx.seg(input, output)` | `--seg` | `_tbx`, `_tseg` (`save_tissue=True`: `_pve_csf/gm/wm`) |
| Affine to the 1 mm template (keeps size and shape) | `tigerbx.affine(input, output)` | `--affine` | `_tbx`, `_affine`, `_warp.npz` |
| Nonlinear normalisation (numreg affine + VFA) | `tigerbx.norm(input, output)` | `--norm` | `_tbx`, `_norm`, `_warp.npz` |
| VBM | `tigerbx.vbm(input, output)` | — | `_SmoothedGM`, `_norm`, `vbm_summary.csv` |
| Population template | `tigerbx.template(input, output)` | `--template` | `template`, `template_mask`, `template_tseg` |
| Raw model outputs | `tigerbx.run('bmatcd', input, output)` | `-b -m -t -d` | `_tbx _tbxmask _aseg _tseg _ct _dgm` |
| Carry an image to the template | `tigerbx.transform(image, warp_npz, output)` | — | template-space image |
| HLC parcellation | `tigerbx.hlc(input, output)` | — | `_hlc` |
| Registration plans (`R` rigid, `A` affine, `V` VMnet, `D` VFA) | `tigerbx.reg('AD', input, output)` | — | `_affine`, `_VFA`, `_warp.npz` |
| Regional embedding, volumes, TIV | `tigerbx.embed(input, output)` | — | `<stem>_embed.npz` |
| What an image is (contrast, skull-stripped, EPI, orientation, truncation, motion score) | `tigerbx.inspect(input, output)` | `--inspect` | `inspect_summary.csv` |
| EPI / DTI distortion correction | `tigerbx.gdm(input, output)` | — | `_gdmi` (`dmap=True`: `_gdm`) |
| Metrics (Dice, HD95, PSNR, SSIM, ...) | `tigerbx.eval(y_true, y_pred, metrics)` | — | dict |

`input` is a NIfTI file, a directory, a glob or a list. Install: `pip install "tigerbx[cpu]"` (or `[cu12]` / `[cu13]`)
from the release tarball linked in the README; the stand-alone executable is on the GitHub Releases page.

## Rules that are easy to get wrong

- **Outputs are written to disk.** With `output=None` files go next to each input; pass an output folder.
  `tigerbx.run(..., save_outputs=False)` only returns images.
- **Return values:** one input returns a dict of nibabel images (`.get_fdata()` for arrays); several inputs return a
  list of dicts of written file paths. A failing file in a batch is logged and skipped, not raised; a `RuntimeWarning` at the end lists the skipped
  files. `tigerbx.check()` (executable: `tigerbx --check`) shows versions, ONNX Runtime, GPU and the model cache.
- **GPU:** pipelines, `reg`, `vbm`, `embed` take `gpu=True`; `hlc`, `gdm`, `val` take `GPU=True`; `tigerbx.run` uses
  the `g` flag; the executable uses `-g`. If onnxruntime-gpu does not match the CUDA install it falls back to CPU
  with a `RuntimeWarning`.
- **Labels:** `_tseg` = ASEG + label 99 `Unlabeled-Intracranial`, so TIV = all non-zero voxels; leave 99 out of
  structure volumes. `run('a')` writes `_aseg` without 99, as in 0.2.x. Label numbers follow FreeSurfer, but TSEG is
  not a FreeSurfer aseg. `seg()` takes the 12 deep-GM structures from DGM12 (`merge_dgm=False` keeps the raw model).
  Labels 30, 31, 62, 63, 77, 85 are deprecated (unreliable).
- **Intensities:** `affine` / `norm` keep the input's units by default (any contrast); `intensity='gnorm'` (T1 only)
  maps CSF / GM / WM to 1/3, 2/3, 1; `'minmax'` scales to [0, 1]. `mask='none'` for already skull-stripped inputs.
- **`_warp.npz`** holds the registration as plain arrays; apply it to other images (labels with
  `interpolation='nearest'`) with `tigerbx.transform`.
- **Check `_tbx` first** when any result looks wrong: every pipeline starts from it.
- Models download automatically on first use; do not install or download anything unless the user asks to run.
- Use the public `tigerbx.*` API; do not import `tigerbx.core` / `tigerbx.modules` internals.

## Details

Fetch only the page you need:

- Pipelines (steps, all options, outputs): https://github.com/htylab/tigerbx/blob/main/doc/pipelines.md
- `run` flags and outputs: https://github.com/htylab/tigerbx/blob/main/doc/bx.md
- Labels (ASEG / TSEG, DGM, HLC, compatibility table): https://github.com/htylab/tigerbx/blob/main/doc/seglabel.md
- Registration and `transform`: https://github.com/htylab/tigerbx/blob/main/doc/reg.md
- Executable: https://github.com/htylab/tigerbx/blob/main/rust/README.md
- Other modules: https://github.com/htylab/tigerbx/tree/main/doc (inspect.md, hlc.md, embed.md, gdm.md, eval.md, validation.md,
  models.md)
