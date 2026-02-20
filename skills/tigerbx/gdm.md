# `tigerbx.gdm()` — EPI Distortion Correction

Corrects geometric distortions in DTI/EPI scans using a GAN-based displacement field predictor.
No field maps or reversed-phase-encode acquisitions required.

Reference: Kuo et al., *Magn Reson Med* 2025. https://doi.org/10.1002/mrm.30577

```python
import tigerbx

tigerbx.gdm(input, output=None, b0_index=0, dmap=False, no_resample=False, GPU=False)
```

| Parameter      | Type         | Default | Description |
|----------------|--------------|---------|-------------|
| `input`        | `str`        | —       | Path to DTI/EPI NIfTI file (4D) |
| `output`       | `str`        | `None`  | Output directory; `None` saves next to input |
| `b0_index`     | `int` or `str` | `0`   | Index of b0 volume, or path to a `.bval` file |
| `dmap`         | `bool`       | `False` | Also save predicted displacement map |
| `no_resample`  | `bool`       | `False` | Skip resampling to 1.7×1.7×1.7 mm³ |
| `GPU`          | `bool`       | `False` | Use GPU |

---

## Examples

```python
import tigerbx

# Correct a DTI file (b0 = first volume)
tigerbx.gdm('dti.nii.gz', 'output/')

# Specify b0 by index
tigerbx.gdm('dti.nii.gz', 'output/', b0_index=1)

# Identify b0 automatically from .bval file
tigerbx.gdm('dti.nii.gz', 'output/', b0_index='dti.bval')

# Save displacement map, use GPU
tigerbx.gdm('dti.nii.gz', 'output/', dmap=True, GPU=True)

# Skip resampling (keep original resolution)
tigerbx.gdm('dti.nii.gz', 'output/', no_resample=True)
```

---

## Output file naming

For input `dti.nii.gz`:

| Parameter | Output file |
|-----------|-------------|
| (default) | `dti_gdm.nii.gz` — distortion-corrected volume |
| `dmap=True` | `dti_dmap.nii.gz` — predicted displacement map |

---

## CLI (for simple one-off tasks)

```bash
tiger gdm dti.nii.gz -o output/
tiger gdm dti.nii.gz -b0 1 -o output/
tiger gdm dti.nii.gz -b0 dti.bval -o output/
tiger gdm dti.nii.gz -m -g -o output/
```
