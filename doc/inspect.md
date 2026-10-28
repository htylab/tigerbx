# TigerBx: `inspect` — what kind of image is this?

`tigerbx.inspect` tells, for each input, its contrast and a few properties that decide how to process it. One small
3D CNN (`inspect_v003.onnx`, 3.6 M parameters) does it in about 1 s per image on one CPU thread, no GPU needed. Any
contrast and any orientation can be given; of a 4D image the first volume is used.

```python
import tigerbx
r = tigerbx.inspect('sub-01_T2w.nii.gz')          # one input -> dict
rows = tigerbx.inspect('data/', 'out/')           # several -> list of dicts, and out/inspect_summary.csv
rows = tigerbx.inspect('data/', 'out/', detail=True)   # every probability and check value
```

The stand-alone executable gives the same `inspect_summary.csv`: `tigerbx data/ -o out/ --inspect` (`--detail` for
every value).

By default each row has the decisions: `file`, `status`, `contrast`, `p_contrast` (probability of that class),
`skull_stripped`, `epi`, `orientation`, `truncated`, `motion`, `ndim`, `matrix`, `voxel_mm`, `warnings`, `error`. `warnings` spells out
what needs a look: an orientation that does not match the header, a head cut by the field of view, motion artifacts
and the rule-based findings below. `detail=True` adds every class probability, the probability behind each decision,
`motion_flag` and all rule-check values.

| Output | Values |
|--------|--------|
| `contrast` | `T1w` (contrast-enhanced T1 included), `T2w`, `FLAIR`, `b0`, `DWI` (b ≥ ~800), `other` (PD, GRE, ASL, angiography, quantitative maps, ...); `p_<class>` per class |
| `skull_stripped` | the image is already skull-stripped (`p_skull_stripped`) |
| `epi` | an EPI acquisition: diffusion, ASL (`p_epi`) |
| `orientation` | anatomical direction of the image's +x, +y, +z after its header: `RAS` = the header is right; `RPS` = anterior-posterior flipped, `RAI` = superior-inferior flipped, `RSA` = two axes swapped, ... (`p_orientation`) |
| `truncated` | part of the head lies outside the field of view (`p_truncated`) |
| `motion` | motion artifact score 0-1 (ringing, blur from head motion) and `motion_flag` (score > 0.5); T1w, T2w and FLAIR only, empty for other contrasts |

Rule-based checks (header and data, no model):

| Output | Values |
|--------|--------|
| `n_volumes` | volumes of a 4D image (only the first is inspected) |
| `ndim` | number of array dimensions (3, 4, ...): sort by it to find the inputs that differ |
| `matrix`, `voxel_mm` | the full array shape (a 4D image shows its 4th dimension) and the voxel size, e.g. `256x256x176`, `1.00x1.00x1.00` |
| `thick_slices` | thick slices / low resolution: an axis ≥ 3 mm or largest / smallest ≥ 3 |
| `oblique_deg` | angle of the voxel axes to the world axes (warning above 20°) |
| `fov_mm` | field of view (warning when an axis covers < 80 mm: a slab) |
| `nonfinite_frac` | fraction of NaN / Inf voxels |
| `label_map` | integer values only, few and sparse: a segmentation or mask, not an image |
| `warnings` | the checks that apply, plus an empty / constant image and a missing spatial header (sform and qform codes 0) |

`inspect_summary.csv` has one row per input, in the order of the input (a list of files keeps its order; a folder or
a glob is in file name order); a failing input is logged and skipped (a single input raises).

## Using the results

The pipelines do not read `inspect_summary.csv`; run `tigerbx.inspect` first and choose their options from it.

| Finding | What to do |
|---------|------------|
| `skull_stripped` True | the pipelines' `mask='none'` (`--skull-stripped` for `--template`): the brain is the non-zero voxels, no tBX |
| `contrast` not T1w | `tigerbx.vbm` and `intensity='gnorm'` assume T1; `norm` / `affine` work with `intensity='raw'` (VFA is a T1 model, other contrasts register less well); `template(contrast=...)` builds `template_tseg` only for T1 |
| `orientation` not `RAS` | fix the header before processing; every output follows the header |
| `truncated` | volumes (TIV, VBM) and the template registration miss the cut part |
| `thick_slices`, `n_volumes` > 1 | 2D / thick-slice data limit segmentation and VBM; of a 4D image only the first volume is processed |
| `label_map` | not an image: leave it out |
| `motion` high | look at the image; compare scores within the study |

Process a group (VBM, template) the same way throughout: decide `mask` for the whole study, not per image.

## Notes

- Left and right cannot be told apart (a brain is near symmetric): an R-L axis is always reported as `R`, so a
  left-right flip in the header is not detected.
- `orientation` and `truncated` judge the image against its own header; a slab that covers only part of the brain
  on purpose (e.g. a thin sagittal acquisition) can be reported as truncated or, rarely, as reoriented.
- The contrast comes from the image, not from the header or the file name. `b0` and `DWI` are the volumes of a
  diffusion series; give a 4D series as one file per volume to classify each.
- `motion` is a screening score, not a rating: compare scores within a study and look at the flagged images.
- The network input is a 256 mm cube at 2.67 mm (96³) on RAS axes (from the header) centred on the intensity centre
  of mass: intensity / p99, the fraction of empty (0 or non-finite) voxels, the inside-field-of-view mask and
  |image - 3x3x3 box mean| / p99 from the native grid (motion ringing and blur), plus header features (voxel size,
  matrix, obliqueness, sform / qform codes, float data) that only the skull-stripped and EPI outputs use.
- Research use only.
