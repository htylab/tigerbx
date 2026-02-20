# `tiger gdm` — EPI Distortion Correction

Corrects geometric distortions in DTI/EPI scans using a GAN-based displacement field predictor.
No field maps or reversed-phase-encode acquisitions required.

Reference: Kuo et al., *Magn Reson Med* 2025. https://doi.org/10.1002/mrm.30577

```
tiger gdm <input> [input ...] [-o OUTPUT] [-b0 INDEX_OR_BVAL] [-m] [-n] [-g]
```

---

## Flags

| Flag | Description |
|------|-------------|
| `-b0 N` | Index of b0 volume (integer). Default: `0` (first volume) |
| `-b0 FILE` | Path to a `.bval` file — b0 is identified automatically |
| `-m` | Also save the predicted displacement map (`_dmap.nii.gz`) |
| `-n` | Skip resampling to 1.7×1.7×1.7 mm³ |
| `-g` | Use GPU |

---

## Examples

```bash
# Correct a DTI file (b0 = first volume)
tiger gdm dti.nii.gz -o output/

# Specify b0 index
tiger gdm dti.nii.gz -b0 1 -o output/

# Use .bval file to find b0 automatically
tiger gdm dti.nii.gz -b0 dti.bval -o output/

# Save displacement map, GPU
tiger gdm dti.nii.gz -m -g -o output/
```

---

## Output naming

For input `dti.nii.gz`:

| Flag | Output file |
|------|-------------|
| (default) | `dti_gdm.nii.gz` — distortion-corrected volume |
| `-m`      | `dti_dmap.nii.gz` — displacement map |
