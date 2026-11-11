# TigerBx: `pipelines`

The pipelines are the recommended way to use TigerBx: `tigerbx.bx`, `tigerbx.seg`, `tigerbx.affine`,
`tigerbx.norm` and `tigerbx.vbm`, each a function with its own options. The modules (`tigerbx.run`, `tigerbx.reg`,
...) give each model's own output, which shows its raw performance; post-processing that makes the results more
accurate is done in the pipelines.

| Pipeline | Python | Executable | Outputs |
|----------|--------|------------|---------|
| `bx`     | `tigerbx.bx(...)`     | `tigerbx in.nii.gz` | `_tbx` |
| `seg`    | `tigerbx.seg(...)`    | `--seg` | `_tbx`, `_tseg` |
| `affine` | `tigerbx.affine(...)` | `--affine` | `_tbx`, `_affine`, `_warp.npz` |
| `norm`   | `tigerbx.norm(...)`   | `--norm` | `_tbx`, `_norm`, `_warp.npz` |
| `long`   | `tigerbx.long(baseline, followup, ...)` | — | `<followup>_long.csv`, `<followup>_logjac` (volume change between two T1 scans of one head; [below](#long)) |
| `vbm`    | `tigerbx.vbm(...)`    | — | `_SmoothedGM`, `_norm`, `vbm_summary.csv` |
| `template` | `tigerbx.template(...)` | `--template` | `template`, `template_mask`, `template_tseg`, `template_log.json` |
| `inspect` | `tigerbx.inspect(...)` | `--inspect` | `inspect_summary.csv` (what each image is; [inspect](inspect.md)) |
| `duope` | `tigerbx.duope(forward, reverse, ...)` | `--duope` | `<input>_duope` for forward, reverse and series, `duope_summary.csv` (EPI distortion from a reverse phase-encoding pair; [duope](duope.md)) |
| `dwimc` | `tigerbx.dwimc(dwi, reverse=...)` | `--dwimc` | `<dwi>_dwimc`, `<dwi>_dwimc.bval` / `.bvec` (rotated) / `.csv` (head motion + eddy-current correction of a DWI series, with the susceptibility field from reversed-PE b0s; [dwimc](dwimc.md)) |
| `epinorm` | `tigerbx.epinorm(series, t1, ap, pa)` | — | `<series>_epi2t1`, `<series>_epi2mni`, `_epinorm.npz`, `epinorm_summary.csv` (a DWI / BOLD series onto the T1 and the template: duope from the AP / PA pair, rigid to the T1, the T1's affine + tReg, one resampling; [epinorm](epinorm.md)) |
| `to_epi` | `tigerbx.to_epi(t1, epi, labels=...)` | — | `<t1>_t12epi`, `<label>_t12epi` (the T1 and label maps onto an EPI's grid, rigid: tBX on both, Mattes MI, labels nearest; [epinorm](epinorm.md)) |

Common arguments: `input` (a NIfTI file, a directory, a glob pattern or a list), `output` (output directory;
`None` writes next to each input), `gpu`, `gz` (`.nii.gz` outputs), `model` (model overrides), `verbose`. A single
input returns a dict of images; several return a list of dicts of the written files (a failing input is logged
and skipped, and a `RuntimeWarning` at the end lists the skipped inputs with their errors). `_tbx` (all but `vbm`) is the brain on the input grid, the first thing to check when a result
looks wrong.

The VBM pipeline was developed by **Pei-Mao Sun**.

---

## bx

Brain extraction (the executable with no option): tBX -> `_tbx` (the brain image on the input grid). Same as
`tigerbx.infer('b')`.

```python
tigerbx.bx(input=None, output=None, *, save_mask=False, gpu=False, gz=True, model=None, resume=False, verbose=0)
```

| Parameter   | Type   | Default | Description |
|-------------|--------|---------|-------------|
| `save_mask` | `bool` | `False` | Also write `_tbxmask` |
| `model`     | `dict` | `None`  | `{'tbx': ...}` override |
| `resume` | `bool` | `False` | Skip inputs whose `_tbx` already exists (continuing a batch); they are left out of the returned list |

---

## seg

Segmentation (`tigerbx --seg`): tBX -> TSEG (the tBX brain as input; labels in [seglabel.md](seglabel.md)).

```python
tigerbx.seg(input=None, output=None, *, save_tbx=True, save_tissue=False, gpu=False, gz=True, model=None,
            resume=False, verbose=0)
```

| Parameter   | Type   | Default | Description |
|-------------|--------|---------|-------------|
| `resume` | `bool` | `False` | Skip inputs whose `_tseg` already exists (continuing a batch); they are left out of the returned list |
| `save_tbx`  | `bool` | `True`  | Also write `_tbx` |
| `save_tissue` | `bool` | `False` | Also write the CSF / GM / WM partial-volume maps `_pve_csf`, `_pve_gm`, `_pve_wm` (T1 only; see below) |
| `model`     | `dict` | `None`  | Overrides: `tbx`, `tseg` |

Outputs per input (input grid): `_tseg` (label numbers follow FreeSurfer) and `_tbx`.

**CSF / GM / WM maps** (`save_tissue`, and the GM map of `vbm`) are computed, not predicted by a model
(`tigerbx.core.cgw.tissue_pve`): TSEG gives each voxel its tissue (cortex and deep GM -> GM, white matter / brain stem
/ ventral DC -> WM, ventricles / CSF -> CSF; inside the TSEG labels, so the CSF around the brain is included); a voxel next to another tissue whose T1 intensity lies between the
two tissues' local mean intensities (Gaussian 12 mm, so the bias field drops out) is split linearly between them. It is more
stable under repositioning than the former CGW model.

---

## affine and norm

Template normalisation onto the 1 mm template grid:

* `affine` (executable `--affine`): brain mask -> numreg affine. Keeps each brain's size and shape (e.g. ventricle
  volume, atrophy), which a nonlinear warp removes.
* `norm` (executable `--norm`): brain mask -> numreg affine -> tReg. Aligns the anatomy voxel by voxel.

Any contrast: numreg uses mutual information, and the intensities stay in the input's units by default. tReg is a T1
model; other contrasts register somewhat less well. Each image is sampled once from the input grid, however oblique or anisotropic: `_affine` at the affine
and `_norm` at the affine + tReg warp composed, both cubic B-spline (clipped to the input range). tReg itself gets
the trilinear affine image, as it expects.

```python
tigerbx.affine(input=None, output=None, *, mask=None, template=None, intensity='raw', save_warp_file=True,
               gpu=False, gz=True, model=None, resume=False, verbose=0)
tigerbx.norm(input=None, output=None, *, mask=None, template=None, intensity='raw', save_affine=False,
             save_warp_file=True, gpu=False, gz=True, model=None, resume=False, verbose=0)
```

| Parameter        | Type            | Default | Description |
|------------------|-----------------|---------|-------------|
| `mask`           | `None`, `str`, image or `'none'` | `None` | `None`: tBX; a NIfTI path / image: your brain mask (non-zero; resampled nearest if on another grid); `'none'`: the input is already skull-stripped |
| `template`       | `str`           | `None`  | Registration template (default: the bundled 1 mm template); a custom one keeps its own grid (see [reg](reg.md)) |
| `intensity`      | `str`           | `'raw'` | `'raw'`: the input's units (any contrast); `'minmax'`: [0, 1]; `'gnorm'`: T1 tissues CSF 1/3, GM 2/3, WM 1 (`tigerbx.core.tissuenorm.tissue_norm`; bundled template only) |
| `save_affine`    | `bool`          | `False` | `norm` only: also write `_affine` |
| `save_warp_file` | `bool`          | `True`  | Write `_warp.npz` |
| `refine`         | `bool`          | `False` | `norm` only: refine the tReg warp against the template (`mprage_refine_v001`: local CC, 20 compositive iterations, no folding; T1 only, about 10 s on the CPU / 1 s on the GPU) |
| `model`          | `dict`          | `None`  | Overrides: `tbx`, `treg`, `refine` (`norm`) |
| `resume`  | `bool`          | `False` | Skip inputs whose `_affine` / `_norm` already exists (continuing a batch); they are left out of the returned list |

Outputs per input: `_tbx` (input grid; when tBX ran), `_affine` or `_norm` (template grid) and `_warp.npz` (plain
arrays, see [registration usage](reg.md): `affine_M`, plus `treg_warp` for `norm`). `tigerbx.transform(image,
'<stem>_warp.npz', out)` carries any other image on the input grid (labels, PVE maps) to the template.

```python
tigerbx.norm(r'C:\T1w_dir', r'C:\output_dir', intensity='gnorm')            # T1
tigerbx.norm(r'C:\T2w_dir', r'C:\output_dir')                               # any contrast, input units
tigerbx.affine(r'C:\brains', r'C:\output_dir', mask='none')                 # already skull-stripped
tigerbx.transform(r'C:\T1w_dir\sub-01_tseg.nii.gz', r'C:\output_dir\sub-01_T1w_warp.npz', r'C:\output_dir')
```

---

## template

A population template from a group of images, on the tigerbx template grid (1 mm, 160 x 224 x 192, the bundled
template's frame), the way the bundled template was built:

1. each image: brain (tBX, or `mask='none'` for skull-stripped inputs), numreg affine to the bundled template
   (mutual information: any contrast), intensity divided by the brain median;
2. `iters` rounds: tReg-register every subject to the current template (round 0: the affine images), take the
   voxel-wise median (`stat`) and move it by the inverse of the mean displacement, so the template stays in the
   group's mean frame; the last round samples every subject once from its input grid (affine + warp, cubic);
3. brain mask = majority of the subjects' tBX masks; for T1 (`labels='auto'`) the labels are the majority vote of
   the subjects' TSEG (as `tigerbx.seg`) inside the mask, among foreground labels.

The bundled template was built this way (8 rounds, median). Its mask follows the brain surface, as the tBX mask, so
its CSF label (24) is the CSF inside that mask (sulci), not the CSF up to the skull.

```python
tigerbx.template(input=None, output=None, *, contrast='t1', iters=6, stat='median', mask=None, labels='auto',
                 gpu=False, gz=True, model=None, verbose=0)
```

| Parameter  | Type   | Default    | Description |
|------------|--------|------------|-------------|
| `output`   | `str`  | —          | Output directory: `template.nii.gz`, `template_mask.nii.gz`, `template_tseg.nii.gz` (labels), `template_log.json` |
| `contrast` | `str`  | `'t1'`     | `'t1'` builds labels by default; other contrasts (e.g. `'t2'`) give the image and mask only |
| `iters`    | `int`  | `6`        | Registration rounds |
| `stat`     | `str`  | `'median'` | `'median'` or `'mean'` of the warped images |
| `mask`     | `None` / `'none'` | `None` | `None`: tBX; `'none'`: the inputs are skull-stripped |
| `labels`   | `'auto'` / `bool` | `'auto'` | TSEG labels (`'auto'`: for T1) |

The template intensities are scaled to the bundled template's brain median. Use it as the registration target of
the other pipelines: `tigerbx.norm(..., template='tpl_dir/template.nii.gz')` (its `_tseg` file is found by name).
tReg is a T1 model; a T2 group builds, but registering T2 to its own template did not align better than to the
bundled T1 template.

---

## long

```python
tigerbx.long(baseline, followup, output=None, *, refine=True, save_logjac=True, gpu=False, gz=True,
             model=None, verbose=0)
```

Volume change between two T1 scans of one head (research use), from the deformation between them rather than from
two separate segmentations. Symmetric in the two scans:

1. tBX + TSEG of each scan.
2. Rigid follow-up -> baseline (numreg, the heads inside the tBX outer masks) and its halfway point: both scans are
   resampled once (cubic) onto the template grid, each the same distance from its own grid.
3. tReg both ways, each refined by `mprage_refine_v001` (local-CC force, 20 iterations of at most 0.2 voxel, each
   composed with the deformation, so it never folds; `refine=True`). tReg alone underestimates small changes; the
   refinement recovers most of them.
4. Per TSEG region: change % = 100 (sqrt(r1 r2) - 1), r1 / r2 the follow-up / baseline volume ratio of the region
   through each of the two deformations.

| Parameter     | Type            | Default | Description |
|---------------|-----------------|---------|-------------|
| `baseline`, `followup` | `str` or `list` | — | Two T1 files, or two lists of the same length (pairs in order) |
| `output`      | `str`           | `None`  | Output directory; if `None`, next to the follow-up file |
| `refine`      | `bool`          | `True`  | Refine the tReg deformations (`False`: tReg alone) |
| `save_logjac` | `bool`          | `True`  | Write `_logjac` (template grid: log of the local volume ratio follow-up / baseline) |
| `model`       | `dict`          | `None`  | Overrides: `tbx`, `tseg`, `treg`, `refine` |

Outputs, named after the follow-up: `_long.csv` with one row per region (`brain`, `gm_cortex`, `wm`, `ventricles`,
`hippocampus`, `amygdala`, `thalamus`, `caudate`, `putamen`, `cerebellum`): `change_pct` (the deformation),
`tseg_change_pct` (the difference of the two TSEG volumes, for comparison), `tseg_baseline_ml`,
`tseg_followup_ml`; and `_logjac`. A single pair returns `{'regions': {...}, 'logjac': image, 'rigid': 4 x 4}`.

Head motion blurs a scan and shrinks its measured brain slightly; compare scans of similar quality.

---

## vbm

```python
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
| `affine_type`       | `str`           | `'numreg'`  | Affine method; only `'numreg'` (`'C2FViT'` was removed in 0.5.0 and runs numreg with a warning) |
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

tigerbx.vbm(r'C:\T1w_dir', r'C:\output_dir')

# Customize the registration stage used inside the pipeline
tigerbx.vbm(r'C:\T1w_dir', r'C:\output_dir', reg_plan='RAD')

# Save the displacement field produced by the registration stage
tigerbx.vbm(r'C:\T1w_dir', r'C:\output_dir', save_displacement=True)
```

---

## Outputs

The VBM pipeline runs tBX + TSEG once, computes the CSF / GM / WM partial-volume maps from them (below), registers the
T1 to the template with `reg_plan`, warps the native GM map to the template, modulates it and smooths it. All files go to the output folder (next to each input when
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

`_warp.npz` is also written with `save_displacement=True`.

### `vbm_summary.csv`

One row per input:

| Column   | Description |
|----------|-------------|
| `file`   | Input path |
| `status` | `ok` or `failed` |
| `qc_flags` | What to check, `;`-separated (empty = nothing flagged): `fail`, `noise`, `bias`, `contrast`, `edge`, `homog` (see below) |
| `TIV_ml` | Total intracranial volume: the volume of all TSEG labels in native space (ml), as in SynthSeg |
| `GM_ml`, `WM_ml`, `CSF_ml` | Native tissue volumes: sum of the CSF / GM / WM partial-volume maps (ml) |
| `noise`, `bias`, `contrast`, `edge`, `res` | No-reference image quality of the input T1w (see below) |
| `homog_z` | Sample homogeneity z of the SmoothedGM map (batches of >= 10 successful inputs) |
| `error`  | Exception message of a failed input |

### Image quality and homogeneity

`noise`, `bias`, `contrast`, `edge` and `res` come from `tigerbx.core.qa.image_quality` on the input image and its
TSEG (no model; T1 ordering assumed):

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

`_SmoothedGM` is modulated by the Jacobian determinant of the **nonlinear** warp only (the `D` step); the affine
scaling to the template is not included (CAT12's `m0wp1`-style, "non-linear only" modulation). Global head-size
differences are therefore already removed, and the maps are not absolute volumes. Adding `TIV_ml` (and age, sex,
scanner) as covariates in the group model is still common practice and harmless.

The default smoothing FWHM of 7.065 mm equals a Gaussian sigma of 3 mm (3 voxels on the 1 mm template); 8 mm is another
common choice (`fwhm=8`).

Before 0.3.0 the smoothed map was written to a per-subject subfolder (`<stem>/<stem>_SmoothedGM.nii.gz`).
