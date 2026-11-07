# TigerBx: `dwimc` — head motion and eddy-current correction of DWI

`tigerbx.dwimc` corrects head motion and eddy-current distortion of a diffusion series acquired in one
phase-encoding direction, and its susceptibility distortion when a few b0 volumes with the reversed direction are
given. Every volume is resampled once (cubic, with the Jacobian of its distortion). numpy / scipy only, CPU.

```python
import tigerbx
r = tigerbx.dwimc('sub-01_dwi.nii.gz', 'out/', reverse='sub-01_dir-PA_epi.nii.gz')   # + susceptibility field
r = tigerbx.dwimc('sub-01_dwi.nii.gz', 'out/')                                       # motion + eddy currents only
r = tigerbx.dwimc('dwi.nii.gz', 'out/', reverse='pa.nii.gz', pe='AP')                # no JSON sidecar
r = tigerbx.dwimc([dwi1, dwi2], 'out/', reverse=[pa1, pa2])                          # several series
```

The executable runs the same pipeline (`rust/src/dwimc.rs`, about 3x faster):

```bash
tigerbx --dwimc sub-01_dwi.nii.gz -o out/ --reverse sub-01_dir-PA_epi.nii.gz
tigerbx --dwimc dwi1.nii.gz dwi2.nii.gz -o out/ --reverse pa1.nii.gz --reverse pa2.nii.gz --pe AP
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input` | — | A 4D DWI series (NIfTI), or a list. Needs at least one b0 (b < 50) and six diffusion-weighted volumes |
| `output` | `None` | Output directory (`None`: next to the input) |
| `reverse` | `None` | b0 volume(s) acquired with the reversed phase encoding (a list: one per input); the susceptibility field is estimated from them and the series' b0s ([duope](duope.md)) and moves with the head |
| `field` | `None` | Instead of `reverse`: a field in Hz (as `tigerbx.duope(..., save_field=True)`), with the readout time from the JSON sidecar or `readout` |
| `bval`, `bvec` | `None` | FSL-format files (default: the input's `.bval` / `.bvec`) |
| `pe` | `None` | The series' phase-encoding direction when there is no JSON sidecar (`'j'`, `'j-'`, `'AP'`, ... as in duope) |
| `readout` | `None` | Total readout time (s) for `field` without a sidecar |
| `order` | `2` | Eddy-current polynomial along the phase axis: `1` linear, `2` quadratic |
| `iters` | `3` | Outer iterations (the first on images smoothed to 8 mm FWHM) |
| `workers` | 70% of the CPUs | Processes for the per-volume steps (one process where `fork` is unavailable) |
| `mask` | `'threshold'` | Voxels the estimation uses: `'threshold'` (b0 intensity head mask, holes filled), `'tbx'` (tBX on the mean b0, unwarped by the field when there is one; falls back to the threshold mask with a warning) or a mask file on the input grid. Python only |
| `gpu` | `False` | Run tBX on the GPU (`mask='tbx'`) |
| `resume` | `False` | Skip inputs whose `<input>_dwimc` already exists |
| `gz`, `verbose` | `True`, `0` | |

## Outputs

| File | Content |
|------|---------|
| `<input>_dwimc.nii.gz` | The corrected series (float32), in the frame of its first b0 |
| `<input>_dwimc.bval` | The b-values |
| `<input>_dwimc.bvec` | The b-vectors rotated by each volume's head rotation (FSL convention) |
| `<input>_dwimc.csv` | One row per volume: b, translation (mm) and rotation (degrees) in voxel axes, eddy-current coefficients (voxels along the phase axis; terms in voxel axes i / j / k scaled to [-1, 1] over the grid), intensity scale, RMS displacement by motion and eddy currents in the brain (mm) |

## Method

For each volume the observed position of a head point is its rigidly moved position plus a displacement along the
phase axis: the susceptibility field (attached to the head) plus a polynomial in scanner coordinates (the eddy
currents). The signal is multiplied by the Jacobian of that displacement.

1. **Prediction.** Every volume is registered to a prediction of itself from the other volumes of its shell: an even
   spherical-harmonic model whose per-order variances and noise are fitted by marginal likelihood, applied
   leave-one-out. The b0s register to the first b0, which defines the frame.
2. **Gauss-Newton per volume** in the observed space with analytic derivatives (motion, eddy-current terms,
   intensity scale), the displacement inverted line by line along the phase axis, volumes in parallel.
3. **Second level.** The eddy-current terms follow the diffusion gradient: c = B g (g scaled by sqrt(b / b_max)), B
   shared by all volumes; the constant term is merged with the translation along the phase axis.
4. **Anchors to the b0s.** Volume-to-volume consistency cannot see an error that varies smoothly with the gradient
   direction, nor the position of all diffusion-weighted volumes relative to the b0s. The mean diffusion-weighted
   image is registered rigidly to the mean b0, and every volume is registered to the mean b0 (rigid + linear eddy
   currents); the part of those per-volume offsets that is a smooth function of the gradient direction is applied.
   Both use the correlation ratio (a binned intensity mapping between the contrasts), over the head mask dilated by
   7 voxels (the brain edge and some background).
5. The first outer iteration runs on images smoothed to 8 mm FWHM, the others at full resolution.

The corrected series is one cubic resampling of each raw volume. Negative interpolation overshoot is set to 0.
