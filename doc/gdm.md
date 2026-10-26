# TigerBx: `gdm` Module

Generative Displacement Mapping (GDM) corrects geometric distortions in EPI (echo-planar imaging) scans using a GAN-based displacement field predictor. It improves spatial alignment between diffusion-weighted and T1-weighted images without requiring field maps or reversed-phase-encode acquisitions [Kuo et al., 2025].

---

## Python API

```python
tigerbx.gdm(input, output=None, b0_index=0, dmap=False,
            no_resample=False, GPU=False, verbose=0)
```

| Parameter     | Type            | Default | Description |
|---------------|-----------------|---------|-------------|
| `input`       | `str` or `list` | —       | Input EPI/DTI NIfTI file or directory |
| `output`      | `str`           | `None`  | Output directory; if `None`, saves next to each input file |
| `b0_index`    | `int` or `str`  | `0`     | Index of the b0 volume within the 4-D image, or path to a `.bval` file |
| `dmap`        | `bool`          | `False` | Also save the predicted displacement map |
| `no_resample` | `bool`          | `False` | Skip resampling to 1.7 x 1.7 x 1.7 mm³ |
| `GPU`         | `bool`          | `False` | Use GPU for inference |
| `verbose`     | `int`           | `0`     | Verbosity: `0` = tqdm only, `1` = progress, `2` = debug |

**Returns:** for one input, a dictionary of nibabel images (`gdmi`, and `gdm`
when requested). For multiple inputs, returns a list of dictionaries containing
the corresponding saved file paths.

---

## Examples

```python
import tigerbx

# Correct a single DTI file (b0 is the first volume)
tigerbx.gdm('dti.nii.gz', 'output_dir')

# Specify the b0 volume index (e.g., second volume)
tigerbx.gdm('dti.nii.gz', 'output_dir', b0_index=1)

# Use a .bval file to locate the b0 volume automatically
tigerbx.gdm('dti.nii.gz', 'output_dir', b0_index='dti.bval')

# Save the displacement map alongside the corrected image
tigerbx.gdm('dti.nii.gz', 'output_dir', dmap=True)

# Process an entire directory with GPU
tigerbx.gdm('/data/EPI_dir', '/data/output', GPU=True)
```

---

## Reference

Kuo CC, et al. (2025). Referenceless reduction of spin-echo echo-planar imaging distortion with generative displacement mapping. *Magn Reson Med.* 2025; 1–16. https://doi.org/10.1002/mrm.30577
