# TigerBx: `duope` — distortion correction from a reverse phase-encoding pair

`tigerbx.duope` (DuoPE, *dual phase-encoding*) estimates the susceptibility field of a pair of spin-echo EPI images
acquired with opposite phase-encoding polarity (AP / PA, LR / RL, ...) and unwarps them, and optionally a DWI or BOLD
series, with it. It does what FSL `topup` does, in numpy / scipy only, in seconds on one CPU thread. For a single
b0 without a reverse scan see [gdm](gdm.md).

```python
import tigerbx
r = tigerbx.duope('sub-01_dir-AP_epi.nii.gz', 'sub-01_dir-PA_epi.nii.gz', 'out/')
r = tigerbx.duope(ap, pa, 'out/', series='sub-01_task-rest_bold.nii.gz') # also unwarp a BOLD / DWI series
r = tigerbx.duope([ap1, ap2], [pa1, pa2], 'out/')                          # several pairs -> list of file dicts
r = tigerbx.duope('ap.nii.gz', 'pa.nii.gz', 'out/', pe='AP', readout=0.05)   # no JSON sidecars
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `forward`, `reverse` | — | The two scans (NIfTI), or equal-length lists of pairs. A 4D input contributes the mean of its b < 50 volumes (from its `.bval`) or of all its volumes; the reverse scan is resampled onto the forward grid if needed |
| `output` | `None` | Output directory (`None`: next to the forward scan) |
| `series` | `None` | A DWI / BOLD series (or a list) to unwarp with the field, each on its own grid and by its own polarity (the field resampled onto it, scaled by the readout times); for a list of pairs, one entry per pair |
| `series_align` | `True` | Move the field to each series' head position: its b0 (DWI, from the `.bval`) or mean image, unwarped, is registered rigidly to the corrected forward scan (twice, the second time with the moved field). The series itself is not resampled into the forward space. `False`: the same head position is assumed |
| `pe` | `None` | The forward scan's phase-encoding direction when there is no JSON sidecar: a voxel axis (`'i'`, `'i-'`, `'j'`, `'j-'`, `'k'`, `'k-'`, as BIDS) or an anatomical direction, from → to (`'AP'`, `'PA'`, `'LR'`, `'RL'`, `'IS'`, `'SI'`), converted with the affine. A reverse scan without a sidecar is taken as the opposite |
| `readout` | `None` | Total readout time (s) when there is no JSON sidecar (needed for `save_field` and to scale the field to a series with another readout time) |
| `series_pe` | `None` | The direction of a series without a sidecar (same values as `pe`; default: the forward scan's); a list for a list of series |
| `joint` | `'auto'` | Estimate field and motion jointly: `'auto'` when the motion between the scans exceeds 1 mm or 1°, `True`, `False` |
| `alpha` | `0.15` | Bending weight of the field (intensities scaled to p99 = 1); smaller follows the data more closely |
| `max_mm` | `3.5` | Finest estimation grid of the field in mm: the field is solved on block means of f voxels (f the largest power of 2 with f × voxel ≤ `max_mm`) and interpolated; `0` = the native grid (slower). The defaults are tuned for correcting images; a field map for its own sake (QSM, ...) may want `max_mm=0` and a smaller `alpha` |
| `save_field` | `False` | Also write the field in Hz (needs the readout time) |
| `save_disp` | `False` | Also write the forward scan's displacement in voxels |
| `resume` | `False` | Skip pairs whose `<forward>_duope` already exists (continuing a batch); they are left out of the returned list |
| `gz` | `True` | `.nii.gz` outputs |

The phase-encoding direction of each scan and the readout time are read from the BIDS JSON sidecars
(`PhaseEncodingDirection`, `TotalReadoutTime`, else `EffectiveEchoSpacing` × (phase-encoding matrix − 1)) — never from a `dir-` label, which datasets use inconsistently. Without
a sidecar, `pe` / `readout` are used, else the phase encoding comes from the data (with a warning; the header's
`dim_info` is only compared): the phase axis is the one along which the corrected pair agrees best, the forward scan's
polarity is then taken as positive, and a series without a sidecar runs along the forward or the reverse scan's
direction by which of the two unwarps it onto the corrected forward scan. A series' sidecar direction is checked the
same way: when the data clearly say the other direction (NCC higher by 0.03 or more; a wrong direction doubles the
distortion, so the gap is large) the data are used, with a warning; `series_pe` is always followed. The summary's
`series_direction` gives, per series, the source (`json`, `series_pe`, `data`, `data_over_json`) and the two NCCs
(along the forward / the reverse scan's direction). For a list of pairs, `pe`, `readout` and `series_pe` take either one value
for all pairs or a list with one per pair. The two directions of a pair are compared in world space, so the reverse
scan may be stored in another voxel orientation.

Some vendors do not record the polarity in the DICOM header, and its JSON value is then entered by hand; it can
contradict the `dir-` label. For a reverse pair the corrected images are the same either way; only the sign of the
field output changes. The summary CSV lists the anatomical direction used for the forward scan (`forward_direction`)
so that it can be checked.

The stand-alone executable runs the same pipeline (`tigerbx --duope FORWARD REVERSE [FORWARD REVERSE ...]`, options
`--series`, `--pe`, `--readout`, `--save-field`, the other arguments at their defaults; see
[rust/README.md](../rust/README.md)), with the same outputs and summary.

## Outputs

By default only the corrected images, each named after its input:

| Output | Content |
|--------|---------|
| `<forward>_duope`, `<reverse>_duope` | The two scans corrected, every volume, each in one cubic resampling with its motion (on the forward grid) |
| `<series>_duope` | Each series corrected with the field (scaled by its readout time when both are known) |
| `<forward>_duope_field` | With `save_field`: the field in Hz; a positive-polarity scan is displaced by field × readout time voxels along +axis |
| `<forward>_duope_disp` | With `save_disp`: the forward scan's displacement in voxels along its phase axis: corrected(x) = forward(x + disp) |
| `duope_summary.csv` | One row per pair in the output folder (a pair run again replaces its row): phase axis, polarity, anatomical direction, readout time, motion between the scans, joint step, time, each series' motion (mm / degrees) |

The returned dict (one pair) has the images under `forward`, `reverse`, each series' file name, `field`, `disp`.

### The field: units, sign, resolution

- `core.duope.estimate` returns the forward scan's displacement `d` in **voxels** along its phase axis (the
  `_duope_disp` output); `_duope_field` is `d / readout time` in **Hz**.
- Sign: a positive field moves a positive-polarity scan's signal towards +axis (the corrected image is
  `forward(x + d)`), the negative-polarity scan's towards −axis. This agrees in sign with a gradient-echo field map,
  and the synthetic tests (`tests/test_duope.py`) pin it.
- All displacements are in NIfTI voxel axes. FSL works in radiological voxel order and flips voxel x of images with
  a positive-determinant affine (e.g. LPS): with the phase along x, topup's `field_hz` then has the opposite sign in
  NIfTI voxels — compare fields with that in mind.
- The field is written on the forward scan's grid, but is solved on the `max_mm` grid (block means, default
  3.5 mm) under the bending prior `alpha` and interpolated, so it is smooth: for a field map as the result itself
  (QSM, ...) use `max_mm=0` (native grid) and a smaller `alpha`.

## Method

Model as in topup: one displacement `d` along the phase axis moves the forward image by `+d` and the reverse one by
`-d`, the signal modulated by the Jacobian `1 ± ∂d/∂pe`; the cost is the squared difference of the two unwarped
images plus a bending-energy prior. Differences from topup:

- **Physics prior.** The field is harmonic where the susceptibility is constant and bends only at tissue / air
  interfaces, so the bending weight is lowered on a shell at every signal boundary and is stronger in tissue.
- **Dipole start.** On a coarse grid (≥ 4 mm) the field is modelled as `D χ + H c`: the dipole kernel
  `1/3 − (k·b₀)²/|k|²` (B₀ from the header) applied to a susceptibility source on the boundary shell and outside the
  head, plus the solid harmonics of order ≤ 2 (shim). It is fitted to a closed-form per-line start (cumulative-signal
  matching) and then to the data.
- **Solver.** Gauss–Newton on the dense field at the finest level (≤ 3.5 mm), matrix-free conjugate gradients with a
  DCT preconditioner.
- **Motion.** Rigid motion of the reverse scan from the two unwarped images (numreg); the translation along the phase
  axis is a constant field offset and is not estimated. A motion above 1 mm / 1° is refined jointly with the field
  (the motion block eliminated by its Schur complement).
- **Intensities.** Each scan is normalised separately, and a smooth intensity ratio between the scans (coil / gain
  differences) is removed.
- **Output.** Motion and field are applied in one cubic resampling. A 4D input is read in blocks and written volume by volume, so a long BOLD series needs little memory (a 657-volume 88 × 88 × 64 series: about 1.2 GB).

The field was validated against independently measured gradient-echo field maps and against the session T1 on
public reverse phase-encoding data (AP / PA and LR / RL; DWI, BOLD and field-map scans). Research use
only.
