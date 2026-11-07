# TigerBx Release Notes — v0.5.0

See [Upgrading from 0.3.x or 0.2.x](../../README.md#upgrading-from-03x-or-02x) before updating a running study.

## Changes

- **Two sets of models.** `tigerbx.infer()` and the pipelines use the current models (tBX `_tbx` / `_tbxmask`,
  TSEG `_tseg`); `tigerbx.run()` keeps the earlier ones with their 0.2.x names (tBET `_tbet` / `_tbetmask`, ASEG
  `_aseg`): `run('bmad')` gives the same images as 0.3.3.
- **tBX** (brain extraction) gives two masks: the brain (`_tbxmask`, default, following the brain surface) and the
  brain with the surrounding CSF up to the inner skull (`infer('x')`, executable `-x`: `_tbxoutermask`), the input of
  TSEG. TIV is the volume of all TSEG labels.
- **TSEG** (whole-brain segmentation, FreeSurfer label numbers) replaces ASEG in `seg()`, `infer('t')` and the
  executable's `--seg`. The executable embeds tBX, TSEG, tReg and inspect, and downloads DGM12 on first use.
- **tReg**, the deformable registration (plan step `D`, `norm`, `template`), is our own network; lower memory and
  time on the CPU.
- **Rebuilt templates**: the bundled T1 template follows the brain surface and its labels are a vote of TSEG; the SE
  and BOLD EPI templates were rebuilt on it. Registered and normalised images differ from 0.3.x.
- **Removed**: registration plan `V` (VMnet) and `affine_type='C2FViT'` (use `AD`; `'C2FViT'` runs numreg with a
  warning). A `_warp.npz` of 0.3.x holding either is refused with a message to re-run.
- **Executable**: options for pipelines and outputs only; algorithm settings run at their defaults (removed: `--all`,
  `-A`, `-n`, the model overrides, `--stat`, `--labels`, `--series-pe`, `--no-series-align`, `--joint`,
  `--save-disp`, `--alpha`, `--max-mm`, `--ec-order`, `--dwimc-iters`).
- **GPU**: a brain mask input too large for the GPU (an oblique image resampled onto a large grid) runs on the CPU
  with a warning instead of failing.
- **Fixed**: an image with only a qform (sform code 0) is read with its stored orientation. Since 0.3.0, importing
  tigerbx changed a nibabel setting that, with nibabel 5, rotated such images (and any image nibabel loaded
  afterwards in the same process).
- **embed** uses the tReg v3 feature network (`mprage_embed_v002.onnx`, 2 mm and 4 mm levels, 72 columns); its
  tables are not comparable with those of 0.3.x.
  `embed(gpu=True)` works (it failed on the GPU since 0.3.0).
- **CPU memory**: ONNX Runtime runs without its CPU memory arena (Python and executable): about 30% lower peak
  memory for tBX + TSEG on the CPU, about 20% longer, same results.
