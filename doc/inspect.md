# TigerBx: `inspect` — what kind of image is this?

`tigerbx.inspect` tells, for each input, its contrast and a few properties that decide how to process it. One small
3D CNN (`inspect_v003.onnx`, 3.6 M parameters) does it in about 1 s per image on one CPU thread, no GPU needed. Any
contrast and any orientation can be given; of a 4D image the first volume is used.

```python
import tigerbx
r = tigerbx.inspect('sub-01_T2w.nii.gz')          # one input -> dict
rows = tigerbx.inspect('data/', 'out/')           # several -> list of dicts, and out/inspect_summary.csv
```

| Output | Values |
|--------|--------|
| `contrast` | `T1w` (contrast-enhanced T1 included), `T2w`, `FLAIR`, `b0`, `DWI` (b ≥ ~800), `other` (PD, GRE, ASL, angiography, quantitative maps, ...); `p_<class>` per class |
| `skull_stripped` | the image is already skull-stripped (`p_skull_stripped`) |
| `epi` | an EPI acquisition: diffusion, ASL (`p_epi`) |
| `orientation` | anatomical direction of the image's +x, +y, +z after its header: `RAS` = the header is right; `RPS` = anterior-posterior flipped, `RAI` = superior-inferior flipped, `RSA` = two axes swapped, ... (`p_orientation`) |
| `truncated` | part of the head lies outside the field of view (`p_truncated`) |
| `motion` | motion artifact score 0-1 (ringing, blur from head motion) and `motion_flag` (score > 0.5); T1w, T2w and FLAIR only, empty for other contrasts |

`inspect_summary.csv` has one row per input (`file`, `status`, the outputs above, `error`); a failing input is logged
and skipped (a single input raises).

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
