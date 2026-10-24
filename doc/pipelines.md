# TigerBx: `pipelines`

High-level multi-stage workflows are exposed through `tigerbx.pipeline()`.
Pipelines compose lower-level modules into a more opinionated workflow.

Currently available pipelines:

| Pipeline | Dispatcher | Convenience alias |
|----------|------------|-------------------|
| `vbm`    | `tigerbx.pipeline('vbm', ...)` | `tigerbx.vbm(...)` |

The VBM pipeline was developed by **Pei-Mao Sun**.

---

## Dispatcher API

```python
tigerbx.pipeline(name, input=None, output=None, **kwargs)
```

| Parameter | Type  | Default | Description |
|-----------|-------|---------|-------------|
| `name`    | `str` | —       | Pipeline name. Currently only `vbm` is available |
| `input`   | `str` or `list` | `None` | Input NIfTI file, directory, or glob pattern |
| `output`  | `str` | `None`  | Output directory; if `None`, saves next to each input file |
| `**kwargs`| —     | —       | Forwarded to the selected pipeline implementation |

If you prefer a direct function call, `tigerbx.vbm(...)` is kept as a
convenience alias for the `vbm` pipeline.

---

## VBM Pipeline API

```python
tigerbx.pipeline('vbm', input=None, output=None,
                 model=None, template=None, reg_plan='AD',
                 gpu=False, gz=False, save_displacement=False,
                 affine_type='numreg', verbose=0,
                 *, fwhm=7.065, save_modulated=False)

# convenience alias
tigerbx.vbm(input=None, output=None, model=None, template=None,
            reg_plan='AD', gpu=False, gz=False,
            save_displacement=False, affine_type='numreg', verbose=0,
            *, fwhm=7.065, save_modulated=False)
```

`fwhm` and `save_modulated` are keyword-only (positional calls written for 0.2.x keep their meaning). `fwhm`
must be a finite number > 0; `None` means the default.

| Parameter           | Type            | Default     | Description |
|---------------------|-----------------|-------------|-------------|
| `input`             | `str` or `list` | `None`      | Input NIfTI file, directory, or glob pattern |
| `output`            | `str`           | `None`      | Output directory; if `None`, saves next to each input file |
| `model`             | `str` or `dict` | `None`      | Registration model override; `None` uses bundled defaults |
| `template`          | `str`           | `None`      | Custom template NIfTI path; `None` uses the bundled 1 mm MNI-space brain template (`mni_template.nii.gz`) |
| `reg_plan`          | `str`           | `'AD'`      | Registration plan used inside the VBM pipeline |
| `gpu`               | `bool`          | `False`     | Use GPU for inference |
| `gz`                | `bool`          | `False`     | Force `.nii.gz` output |
| `save_displacement` | `bool`          | `False`     | Save the displacement field produced by the registration stage |
| `affine_type`       | `str`           | `'numreg'`  | Affine preprocessing method used by `D` / `V` steps (`'numreg'` or `'C2FViT'`) |
| `fwhm`              | `float`         | `7.065`     | Gaussian smoothing FWHM in mm for the GM map (7.065 mm = sigma 3 mm) |
| `save_modulated`    | `bool`          | `False`     | Also save the unsmoothed modulated GM map (`_ModulatedGM`) |
| `verbose`           | `int`           | `0`         | Verbosity: `0` = tqdm only, `1` = progress, `2` = debug |

**Return value** (same convention as `tigerbx.run` / `tigerbx.reg`): for a single input, a `dict` of nibabel images
(`'Smoothed_GM'`, `'Modulated_GM'`, `'Reg_GM'`, `'norm'`, registration outputs) plus `'TIV_ml'`, `'GM_ml'`,
`'WM_ml'`, `'CSF_ml'`; for several inputs, a `list` with one dict of output file paths per **successful** input.
Use `vbm_summary.csv` (below) to match results to inputs and to find failed ones.

`reg_plan` uses the same uppercase plan syntax as `tigerbx.reg()`. See
[Registration usage](reg.md) for the step definitions and plan rules.

---

## Examples

```python
import tigerbx

# Recommended dispatcher entry point
tigerbx.pipeline('vbm', r'C:\T1w_dir', r'C:\output_dir')

# Convenience alias
tigerbx.vbm(r'C:\T1w_dir', r'C:\output_dir')

# Customize the registration stage used inside the pipeline
tigerbx.pipeline('vbm', r'C:\T1w_dir', r'C:\output_dir',
                 reg_plan='AV', affine_type='C2FViT')

# Save the displacement field produced by the registration stage
tigerbx.pipeline('vbm', r'C:\T1w_dir', r'C:\output_dir',
                 save_displacement=True)
```

---

## Outputs

The VBM pipeline runs tBET + CSF/GM/WM segmentation once, registers the T1 to the template with `reg_plan`, warps the
native GM map to the template, modulates it and smooths it. All files go to the output folder (next to each input when
`output` is `None`).

For example, inputs `sub-001_T1w.nii.gz` and `sub-002_T1w.nii.gz` produce:

```text
output_dir/
  sub-001_T1w_SmoothedGM.nii.gz     # modulated, smoothed GM in template space (the VBM input)
  sub-001_T1w_norm.nii.gz           # T1 after the whole registration plan (for checking the alignment)
  sub-001_T1w_ModulatedGM.nii.gz    # only with save_modulated=True
  sub-002_T1w_...
  sub-003_T1w_vbmfail.log           # only if sub-003 failed
  vbm_summary.csv
```

`warp.npz` is also written with `save_displacement=True`.

### `vbm_summary.csv`

One row per input:

| Column   | Description |
|----------|-------------|
| `file`   | Input path |
| `status` | `ok` or `failed` |
| `qc_flags` | What to check, `;`-separated (empty = nothing flagged): `fail`, `noise`, `bias`, `contrast`, `edge`, `homog` (see below) |
| `TIV_ml` | Total intracranial volume: tBET brain-mask volume in native space (ml) |
| `GM_ml`, `WM_ml`, `CSF_ml` | Native tissue volumes: sum of the CSF / GM / WM partial-volume maps (ml) |
| `noise`, `bias`, `contrast`, `edge`, `res` | No-reference image quality of the input T1w (see below) |
| `homog_z` | Sample homogeneity z of the SmoothedGM map (batches of >= 10 successful inputs) |
| `error`  | Exception message of a failed input |

### Image quality and homogeneity

`noise`, `bias`, `contrast`, `edge` and `res` come from `tigerbx.core.qa.image_quality` on the input image and its
ASEG (no model; T1 ordering assumed):

| Measure    | Meaning | Worse when |
|------------|---------|------------|
| `noise`    | robust local std in WM / (WM - CSF) | higher (noise, motion ghosting) |
| `bias`     | P90 - P10 of the smooth WM field (12 mm) / (WM - CSF) | higher (intensity inhomogeneity) |
| `contrast` | (WM - GM) / (WM - CSF) | lower (poor GM/WM contrast, e.g. low field) |
| `edge`     | GM/WM border gradient x 1 mm / (WM - GM) | lower (blur, thick slices, upsampled low-resolution data) |
| `res`      | RMS voxel size (mm) | higher |

They have no absolute thresholds (those would not hold across scanners and contrasts). In batches of >= 10 successful
inputs, `qc_flags` names a measure when it is an outlier of the batch: robust z (median / MAD,
`tigerbx.core.qa.quality_outliers`) > 3 in the worse direction (`noise`, `bias` higher; `contrast`, `edge` lower).
`res` is reported but not flagged, as a different voxel size is not poor quality. `homog` is added when `homog_z`
< -3 or the SmoothedGM map is unusable; `fail` marks a failed input. The end-of-run warning counts the flags.

`edge` subtracts the expected noise contribution from the border gradient (estimated from `noise`), so noisy
low-resolution scans no longer look sharp; with very strong noise it errs slightly low.

Reference values (medians of 25 public T1w scans per group, tigerbx 0.3.0 ASEG):

| Group | noise | bias | contrast | edge | res (mm) |
|---|---|---|---|---|---|
| 3T research, adult | 0.035 | 0.182 | 0.442 | 0.323 | 0.98 |
| 1.5/3T multi-site dementia cohort | 0.062 | 0.065 | 0.392 | 0.339 | 0.98 |
| 3T research, children | 0.052 | 0.156 | 0.525 | 0.350 | 0.80 |
| Glioma, pre-processed | 0.028 | 0.115 | 0.329 | 0.283 | 1.00 |
| Clinical, thick slices | 0.047 | 0.125 | 0.298 | 0.133 | 3.84 |
| Clinical, brain metastases (contrast-enhanced) | 0.046 | 0.180 | 0.152 | 0.223 | 2.95 |
| 0.3T low field | 0.162 | 0.207 | 0.202 | 0.216 | 3.55 |

Low field stands out on `noise` (3-5x), contrast-enhanced and low-field scans on `contrast`, and all
low-resolution scans (thick slices, low field) on `edge` (0.13-0.22 vs 0.28-0.35 at ~1 mm). In synthetic
degradations of 20 healthy 1 mm scans, `noise` followed added noise (Spearman +0.98) and motion ghosting (+0.95);
`edge` followed blur (-0.94) and slice thickness (-0.74), and moved little with added noise (-0.32; +0.42 without the
noise correction).

`homog_z` is the robust z (median / MAD over the batch) of each SmoothedGM map's mean correlation with all other maps
(on a 4 mm copy). `z < -3` adds `homog` to `qc_flags`: it mostly catches misregistration or orientation errors (10 degree
rotation, 4 mm shift, left-right flip), not segmentation bias or noise. Two or three normal maps in a few hundred can
also fall below -3, so treat it as a flag to inspect `_norm`, not an exclusion rule. A map that is not finite or
is flat gets an empty `homog_z` and the `homog` flag, and is left out of the others' reference. Quality flags have the
same caveat: in a large batch a few normal scans exceed z = 3.

A failing input does not stop a batch: it is logged as `FAILED <file>: <error>`, recorded in the summary, and a
`<stem>_vbmfail.log` (time, input path, error) is written where its outputs would go; then it is skipped. With a single
input the fail log is written and the exception is raised. A later successful run of the same input deletes its
`_vbmfail.log`. Find failures with e.g. `find output_dir -name '*_vbmfail.log'`.

### Modulation and covariates

`_SmoothedGM` is modulated by the Jacobian determinant of the **nonlinear** warp only (`D` / `V` steps); the affine
scaling to the template is not included (CAT12's `m0wp1`-style, "non-linear only" modulation). Global head-size
differences are therefore already removed, and the maps are not absolute volumes. Adding `TIV_ml` (and age, sex,
scanner) as covariates in the group model is still common practice and harmless.

The default smoothing FWHM of 7.065 mm equals a Gaussian sigma of 3 mm (3 voxels on the 1 mm template); 8 mm is another
common choice (`fwhm=8`).

Before 0.3.0 the smoothed map was written to a per-subject subfolder (`<stem>/<stem>_SmoothedGM.nii.gz`).
