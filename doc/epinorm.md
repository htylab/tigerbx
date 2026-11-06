# TigerBx: `epinorm` — a DWI / BOLD series onto its subject's T1 and the template

`tigerbx.epinorm` brings a DWI or BOLD series onto the subject's T1 grid and the template grid, the susceptibility
distortion corrected from a reverse phase-encoding pair, every output voxel sampled once from the raw series.

```python
import tigerbx
r = tigerbx.epinorm('sub-01_task-rest_bold.nii.gz', 'sub-01_T1w.nii.gz',
                    'sub-01_dir-AP_epi.nii.gz', 'sub-01_dir-PA_epi.nii.gz', 'out/')
r = tigerbx.epinorm(dwi, t1, ap, pa, 'out/', space='t1')     # the T1 grid only
r = tigerbx.epinorm(bold, t1)                                 # no pair: rigid + template only (a warning)
```

## Steps

1. **Distortion.** The field from the pair (`forward`, `reverse`; e.g. AP and PA), as [duope](duope.md): phase
   encoding and readout time from the BIDS sidecars, else `pe` / `readout`. The field is carried onto the series:
   its grid, its phase-encoding polarity, its readout time (scaled) and, with `series_align`, its head position
   (the series' b0 / mean registered to the corrected pair). Without a pair nothing is corrected (a warning).
2. **EPI → T1.** The corrected forward scan's brain (tBX; without a pair the series' b0 / mean) registered rigidly
   to the T1's brain (numreg, Mattes MI, 6 dof).
3. **T1 → template.** tBX, numreg affine and tReg, as [norm](pipelines.md).
4. **Output.** Field (with its Jacobian), head position, rigid, affine and tReg warp composed: one cubic B-spline
   sample of the raw series per output voxel and volume, read and written volume by volume.

The EPI → T1 step is rigid on purpose. After the distortion correction, a deformable EPI → T1 registration (tReg,
or a deformation restricted to the phase axis) raised the agreement with the T1's appearance but moved the field
away from an independently measured field map: it matched the image, not the residual distortion. Boundary-based
registration and tReg-encoder features did not beat Mattes MI for the rigid step either. The deformable part belongs
to T1 → template, where tReg is a T1 → T1 model.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `series` | — | The DWI / BOLD series (3D or 4D, any grid), or a list (one per subject) |
| `t1` | — | The subject's T1w (raw; tBX runs on it), or a list |
| `forward`, `reverse` | `None`, `None` | The reverse phase-encoding pair (e.g. AP, PA), or lists; `None`: no distortion correction |
| `output` | `None` | Output directory (`None`: next to each input) |
| `pe`, `readout` | `None` | The forward scan's direction / readout time without JSON, as [duope](duope.md) (without either, the phase axis comes from the data) |
| `series_pe` | `None` | The series' direction, always followed (without it: the JSON, checked against the data, which win when clearly the other way; without either, from the data), as [duope](duope.md) |
| `series_align`, `joint` | `True`, `'auto'` | As [duope](duope.md) |
| `space` | `('t1', 'mni')` | The outputs: `'t1'` (the T1 grid) and / or `'mni'` (the template grid) |
| `res` | `2` | Output voxel size in mm over the same field of view (the T1's / the template's); `'epi'`: the series' smallest voxel size; `None`: the T1's / template's own grid (a long series at 1 mm takes tens of GB) |
| `template` | `None` | Registration template (default: the bundled 1 mm template) |
| `resume` | `False` | Skip series whose first output (`_epi2<space>`) already exists (continuing a batch); they are left out of the returned list |
| `gpu`, `gz`, `verbose` | `False`, `True`, `0` | As the other pipelines |

## Outputs

| Output | Content |
|--------|---------|
| `<series>_epi2t1`, `<series>_epi2mni` | The series, every volume, on the T1 / template field of view at `res` mm |
| `<t1>_tbx` | The T1's brain |
| `<series>_epinorm.npz` | The transform chain (plain arrays): `field_disp` (the series' displacement in its voxels along `pe_axis`, `polarity`), `series_affine`, `head_M` (pair world → series world), `rigid_M` (T1 world → pair world), `t1_affine` / `t1_shape`, and for the template `affine_M`, `treg_warp`, `template_affine` / `template_shape` (as `_warp.npz`) |
| `epinorm_summary.csv` | One row per series: phase axis, polarities, the series direction's source and NCCs (as duope), readout time and scale, motion within the pair, the series' head position, the rigid EPI → T1 transform's size, spaces, time |

The returned dict (one series) has the images under `epi2t1`, `epi2mni` and `t1_tbx`. Research use only.

## EPI templates

Two skull-stripped, distortion-corrected EPI templates on the grid of the bundled template, built with this pipeline
(each EPI through its subject's T1 to the bundled template, averaged; rebuilt with the template in 0.5.0):

| Template | File | `tigerbx.core.templatereg` |
|---|---|---|
| Spin-echo EPI (T2-weighted: field maps, DWI b0) | `tigerbx/template/mni_template_se_epi.nii.gz` | `epi_template('se')` |
| Gradient-echo BOLD (temporal mean) | `tigerbx/template/mni_template_bold_epi.nii.gz` | `epi_template('bold')` |

They are not used by a pipeline yet; they are meant for registering EPI without a T1, to the template of the same
contrast (affine and deformable step). A deformable model has to accept an EPI as the fixed image for the second.

## to_epi: the T1 and its label maps onto an EPI

```python
r = tigerbx.to_epi('sub-01_T1w.nii.gz', 'sub-01_dwi.nii.gz', 'out/', labels='sub-01_T1w_tseg.nii.gz')
```

The EPI's b0 / mean (the b < 50 volumes of a DWI, else all) and the T1 are skull-stripped (tBX), the EPI brain is
registered rigidly to the T1 brain (numreg, Mattes MI, 6 dof: the EPI → T1 step of `epinorm`), and the T1 is
resampled onto the EPI's grid (cubic), each label map nearest. Outputs `<t1>_t12epi` and `<label>_t12epi` (next to
the T1 / each label, or in `output`); the returned dict has the images under `t1_t12epi`, `<label>_t12epi` and the
rigid matrix `rigid_M` (T1 world → EPI world). The EPI is taken as it is: its distortion is not corrected (run
[duope](duope.md) first, or use `epinorm` for the series itself). Research use only.
