# TigerBx Release Notes — v0.3.0

## Highlights

- **New default registration `AD`** — numreg affine + VFA deformable
  registration (reg60 mean Dice 0.852, vs 0.797 for the 0.2.4 default
  C2FViT + VMnet)
- **No more antspyx / optuna** — affine and rigid registration run on a
  built-in numpy/scipy registration (numreg); the install pulls 10 packages
  instead of 39 (13 with `[cpu]`)
- **New bundled template** — `mni_template.nii.gz` (same MNI frame as before)
  with a group-voted aseg
- **Removed steps** — FuseMorph (`F`), SyN (`N`), SyNCC (`C`)
- **Renamed outputs** — `_Af` → `_affine`, `_reg` → `_VXM`; VBM writes `_norm`
- **New `tigerbx.embed`** — VFA brain embedding: mean encoder feature per ASEG structure ([37, 96]) + volumes, TIV
  and image quality per T1w (`tiger embed`); AD vs CN AUC 0.857 fused with volumes
- **VBM for real batches** — one bad input no longer stops a run; `vbm_summary.csv` with QC, TIV, GM / WM / CSF
  volumes and image-quality measures per input
- **TIV from aseg** — new label 99 `Unlabeled-Intracranial`; TIV = all non-zero `_aseg` voxels
- **License** — CC BY-NC-SA 4.0 (was CC BY-NC 4.0)

> **Breaking release.** Registration defaults, output file names, the
> template and the license change. Results from 0.2.x and 0.3.0
> registration / VBM should not be mixed in one study. See
> [Migration](#7-migration-from-02x).

---

## 1. VFA deformable registration (plan step `D`)

- New step `D` (`mprage_reg_v004_vfa.onnx`, 8.4 MB), output `_VFA`; the
  field is stored in `warp.npz` as `VFA_dense_warp` and applied by
  `tigerbx.transform`.
- Needs a prior `A`, like `V`. `V` and `D` can run in the same plan.
- `tigerbx.reg`, `tigerbx.vbm` and the CLIs default to `AD`.

| plan (reg60, 26 structures, mean Dice to the template aseg) | Dice |
|---|---|
| **`AD`: numreg affine + VFA (0.3.0 default)** | **0.852** |
| `AD` with the ANTs affine (removed) | 0.852 |
| `AV`: numreg affine + VMnet | 0.834 |
| `AV`: C2FViT + VMnet (0.2.4 default) | 0.797 |

C2FViT and VMnet were trained with the previous MNI152 template as fixed
image; for them `-T` set to that file works best.

## 2. numreg: affine and rigid without ANTs

- `tigerbx/core/numreg.py`: deterministic numpy/scipy registration (Mattes
  mutual information, symmetric cost, 6/3/1.5 mm pyramid, L-BFGS,
  multi-start). Default `affine_type='numreg'`.
- Same accuracy as the ANTs affine it replaces (reg60 `AD` 0.8519 vs 0.8518,
  29/60 subjects better, p = 0.79) and several times faster on CPU (single
  BLAS thread: ~3–5 s per subject on an idle machine).
- Step `R` (rigid) uses numreg with 6 degrees of freedom instead of the
  `mprage_rigid_v002` ONNX model; it agrees with ANTs Rigid at Dice 0.985 on
  same-subject T1→T2 alignment, where a 12-dof affine shrinks the image by
  1–3%.
- `affine_type='ANTs'` is kept as an alias of `numreg` with a
  `FutureWarning`.
- `warp.npz` stores the affine as a 4×4 world matrix (`M`) plus the reference
  grid. `tigerbx.transform` still reads ANTs affines from older `warp.npz`
  files.

## 3. New template

`tigerbx/template/mni_template.nii.gz` replaces `MNI152_T1_1mm_brain`: same
frame (1 mm, 160×224×192, identical affine), intensity scaled to the old
file's brain median. It was built in-house by unbiased group-wise VFA
registration of public-domain (CC0) MP2RAGE T1w data. The template aseg is a
group vote of per-subject aseg maps.

## 4. Removed

| removed | replacement |
|---|---|
| FuseMorph, plan step `F` (and the `optuna` dependency) | `AD` |
| ANTs SyN / SyNCC, plan steps `N` / `C` | `AD` |
| `antspyx` dependency | numreg (adds the small pure-Python `threadpoolctl`, used to keep its BLAS single-threaded) |
| `mprage_rigid_v002` rigid model | numreg `dof=6` |
| transform ONNX models (`mprage_transform_v002_*`, `mprage_affinetransform_v002_*`) | numpy/scipy resampling in `core/warp.py`, identical output |

Removed steps raise a `ValueError` that names the replacement.
`tigerbx.transform` still applies `Fuse_dense_warp` from an older
`warp.npz` and skips old SyN entries with a warning.

## 5. Output names

| step | 0.2.x | 0.3.0 |
|---|---|---|
| `R` | `_rigid` | `_rigid` |
| `A` | `_Af` | `_affine` |
| `V` (VMnet) | `_reg` | `_VXM` |
| `D` (VFA) | — | `_VFA` |
| VBM, T1 after the whole plan | — (affine image `_Af`) | `_norm` |

`result_dict` keys follow the same names. `save_reg_output` is now honoured
for the rigid and affine images too.

### VBM pipeline

- **Batch-safe**: a failing input is logged (`FAILED <file>: <error>`),
  gets a `<stem>_vbmfail.log` next to its outputs and is skipped instead of
  stopping the run; this includes file errors such as an unwritable folder
  or a full disk (a single input still raises, after writing its row to the
  summary).
- **`vbm_summary.csv`** in the output folder: status, QC, TIV and native
  GM / WM / CSF volumes per input (and the error of failed ones), ready to
  use as group-analysis covariates.
- **One tBET per input**: the BET / CGW result is reused by the
  registration stage (about 40% less CPU time per subject in our test).
- **Flat output folder**: `_SmoothedGM` sits next to `_norm`
  (before: `<stem>/<stem>_SmoothedGM`).
- New keyword-only `fwhm` (mm, default 7.065 = sigma 3 mm, unchanged) and
  `save_modulated` (`_ModulatedGM`) options (after `verbose`, so positional
  0.2.x calls are unaffected); CLI `--fwhm`,
  `--save_modulated`. Smoothing now converts the FWHM with the voxel size
  (same result on the 1 mm template).
- **Image quality and homogeneity** in `vbm_summary.csv`: `noise`, `bias`,
  `contrast`, `edge`, `res` (no-reference, from the image and its ASEG; no
  model) and `homog_z` (each SmoothedGM map's agreement with the batch;
  z < -3 is logged, mostly misregistration / orientation errors). Helpers in
  `tigerbx.core.qa`.
- Docs: modulation uses the nonlinear Jacobian only (affine scaling not
  included), return-value convention, summary columns.
- Tested on 224 sessions of a public multi-site dementia cohort (AD 114, CN
  80, FTD 30): AD < CN peaks in
  the amygdala / hippocampus (t = -11.6, FDR q < 0.05).

### `tigerbx.embed` (new)

- `tigerbx.embed(input, output)` / `tiger embed`: tBET + ASEG, numreg affine, then the encoder of the VFA
  registration model (`mprage_embed_v001_vfa.onnx`, 6.8 MB, same weights as the `D` step) gives 2 mm / 4 mm feature
  maps; their mean inside each of the 37 ASEG structures is saved with the structure volumes, TIV, QC and image
  quality in `<stem>_embed.npz` (~15 KB). Batch-safe like VBM (`_embedfail.log`, `embed_summary.csv`).
- Dementia cohort, 527 sessions: AD vs CN AUC 0.836 (embedding), 0.849 (volumes), 0.857 (late fusion);
  random encoder 0.765. See `doc/embed.md`.
- NERVE stays for now; the embedding is meant to replace it in a later release.

## 6. Fixes

- Registration of int16 images with a large `scl_slope` no longer
  overflows.
- `tigerbx.transform()` no longer writes every output to the same file.
- `tigerbx.vbm()`: fixed the reg module import and the moved Jacobian
  helper.
- `tigerbx.val(task='reg')` works with the new defaults.
- `warp.npz` is smaller (165 MB instead of 366 MB per subject with C2FViT):
  the C2FViT init flow is stored as its 3-value shift.

### ASEG labels

- **New label 99 `Unlabeled-Intracranial`** in the `_aseg` output: voxels
  inside the tBET mask that no ASEG structure claims (sulcal CSF, pial
  partial volume, meninges, vessels; ~200-350 ml). TIV is now simply all
  non-zero `_aseg` voxels. Against manually corrected intracranial masks
  (n = 200, T1/T2/PD) this TIV has r = 0.979 and a -0.7% bias (SD 2.4%),
  vs r = 0.872 and -18.7% for the sum of the ASEG structures. Structure
  volumes do not change, but **a sum over all labels now includes label 99**.

- `doc/seglabel.md` lists 6 of the 43 ASEG labels as **deprecated in 0.3.0**:
  vessel (30, 62), choroid plexus (31, 63), WM hypointensities (77) and
  optic chiasm (85). The model still outputs them (they are kept for
  backward compatibility), but they are not reliable for volumes or analyses
  and may be removed from a later ASEG model.

### New internal helper

- `tigerbx.core.tissuenorm.tissue_norm`: tissue-anchored T1 intensity
  normalisation (CSF 1/3, GM 2/3, WM 1) for images on the template grid.
  Not used by any pipeline yet; see `doc/core/tissuenorm.md`.

## 7. Migration from 0.2.x

| 0.2.x | 0.3.0 |
|---|---|
| `tigerbx.reg('AV', ...)` (C2FViT + VMnet) | `tigerbx.reg('AD', ...)`; old behaviour: `tigerbx.reg('AV', ..., affine_type='C2FViT')` |
| `affine_type='ANTs'` | `affine_type='numreg'` (alias, warns) |
| `tigerbx.reg('AF', ...)`, `'AN'`, `'AC'` | `tigerbx.reg('AD', ...)` |
| `*_Af.nii.gz`, `*_reg.nii.gz` | `*_affine.nii.gz`, `*_VXM.nii.gz` |
| `tigerbx.vbm(..., reg_plan='AF')` | `tigerbx.vbm(...)` (default `AD`) |
| `pip install antspyx optuna` for reg | not needed |
| summing all `_aseg` labels as brain volume | exclude label 99 (it is the rest of the intracranial volume); with it the sum is TIV |
| results tied to the old bundled template | template changed; to keep the old one, pass `-T MNI152_T1_1mm_brain.nii.gz` (shipped with 0.2.4, not with 0.3.0) |

`tiger hlc`, `tiger gdm` and the ASEG model are unchanged from 0.2.4;
`tiger bx -a` only adds label 99 to the `_aseg` file.

## 8. License

TigerBx is now licensed under **CC BY-NC-SA 4.0** (Attribution –
NonCommercial – ShareAlike). Derivatives must be shared under the same
license.

---

- package version is bumped from **`0.2.4`** to **`0.3.0`**
