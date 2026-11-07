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
| Nonlinear normalisation (numreg affine + tReg) | `tigerbx.norm(input, output)` | `--norm` | `_tbx`, `_norm`, `_warp.npz` |
| VBM | `tigerbx.vbm(input, output)` | — | `_SmoothedGM`, `_norm`, `vbm_summary.csv` |
| Population template | `tigerbx.template(input, output)` | `--template` | `template`, `template_mask`, `template_tseg` |
| Raw model outputs (current models) | `tigerbx.infer('bmxtcd', input, output)` | `-b -m -x -t -d` | `_tbx _tbxmask _tbxoutermask _tseg _ct _dgm` |
| Raw model outputs (earlier models, 0.2.x files) | `tigerbx.run('bmacd', input, output)` | — | `_tbet _tbetmask _aseg _ct _dgm` |
| Carry an image to the template | `tigerbx.transform(image, warp_npz, output)` | — | template-space image |
| HLC parcellation | `tigerbx.hlc(input, output)` | — | `_hlc` |
| Registration plans (`R` rigid, `A` affine, `D` tReg) | `tigerbx.reg('AD', input, output)` | — | `_affine`, `_treg`, `_warp.npz` |
| Regional embedding, volumes, TIV | `tigerbx.embed(input, output)` | — | `<stem>_embed.npz` |
| What an image is (contrast, skull-stripped, EPI, orientation, truncation, motion score) | `tigerbx.inspect(input, output)` | `--inspect` | `inspect_summary.csv` |
| EPI distortion from a reverse phase-encoding pair (AP/PA, LR/RL; topup-like), optionally a DWI/BOLD series | `tigerbx.duope(forward, reverse, output, series=...)` | `--duope fwd rev --series ...` | `<input>_duope` (forward, reverse, series); `save_field=True`: `_duope_field` (Hz) |
| A DWI / BOLD series onto the subject's T1 and the template (distortion from a reverse pair, rigid to the T1, T1 affine + tReg, one resampling) | `tigerbx.epinorm(series, t1, ap, pa, output)` | — | `_epi2t1`, `_epi2mni`, `_epinorm.npz` |
| EPI / DTI distortion correction | `tigerbx.gdm(input, output)` | — | `_gdmi` (`dmap=True`: `_gdm`) |
| Metrics (Dice, HD95, PSNR, SSIM, ...) | `tigerbx.eval(y_true, y_pred, metrics)` | — | dict |

`input` is a NIfTI file, a directory, a glob or a list. Install: `pip install "tigerbx[cpu]"` (or `[cu12]` / `[cu13]`)
from the release tarball linked in the README; the stand-alone executable is on the GitHub Releases page.

## Rules that are easy to get wrong

- **Outputs are written to disk.** With `output=None` files go next to each input; pass an output folder.
  `tigerbx.infer(..., save_outputs=False)` only returns images.
- **Return values:** one input returns a dict of nibabel images (`.get_fdata()` for arrays); several inputs return a
  list of dicts of written file paths. A failing file in a batch is logged and skipped, not raised; a `RuntimeWarning` at the end lists the skipped
  files. `tigerbx.check()` (executable: `tigerbx --check`) shows versions, ONNX Runtime, GPU and the model cache.
- **GPU:** pipelines, `reg`, `vbm`, `embed`, `gdm` take `gpu=True`; `hlc`, `val` take `GPU=True`; `tigerbx.infer` / `run`
  use the `g` flag; the executable uses `-g`. If onnxruntime-gpu does not match the CUDA install it falls back to CPU
  with a `RuntimeWarning`.
- **`infer` vs `run`:** `infer` = the current models (tBX `_tbx`/`_tbxmask` following the brain surface, `x`
  `_tbxoutermask` with the surrounding CSF, TSEG inside it); `run` =
  the earlier models with the 0.2.x files (tBET `_tbet`/`_tbetmask`, ASEG `_aseg`). Use `infer` or the pipelines
  unless a study must match earlier results; never mix the two masks in one study.
- **Labels:** `_tseg` (`seg()`, `infer('t')`) and `_aseg` (`run('a')`, the earlier model) are two models with their
  own label tables (doc/seglabel.md): TSEG 32 structures; ASEG 43 labels (+ corpus callosum 251-255 and the
  deprecated, unreliable 30, 31, 62, 63, 77, 85). Label numbers follow FreeSurfer, but neither is a FreeSurfer
  aseg. TIV = the volume of all TSEG labels, as in SynthSeg.
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
- Other modules: https://github.com/htylab/tigerbx/tree/main/doc (inspect.md, duope.md, hlc.md, embed.md, gdm.md, eval.md,
  models.md)
