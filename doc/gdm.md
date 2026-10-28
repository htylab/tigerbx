# TigerBx: `gdm` Module

Generative Displacement Mapping (GDM) corrects geometric distortions in EPI (echo-planar imaging) scans using a GAN-based displacement field predictor. It improves spatial alignment between diffusion-weighted and T1-weighted images without requiring field maps or reversed-phase-encode acquisitions [Kuo et al., 2025].


With a reverse phase-encoding pair (AP / PA, LR / RL) use [duope](duope.md) instead: it measures the field from
the two scans rather than predicting it from one.

---

## Python API

```python
tigerbx.gdm(input, output=None, b0_index=0, dmap=False,
            no_resample=False, gpu=False, verbose=0, gz=True, model=None)
```

| Parameter     | Type            | Default | Description |
|---------------|-----------------|---------|-------------|
| `input`       | `str` or `list` | —       | Input EPI/DTI NIfTI file, directory, glob pattern or list |
| `output`      | `str`           | `None`  | Output directory; if `None`, saves next to each input file |
| `b0_index`    | `int` or `str`  | `0`     | Index of the b0 volume within the 4-D image, or the series' `.bval` file (its first b = 0 volume; any whitespace between the values) |
| `dmap`        | `bool`          | `False` | Also save the predicted displacement map |
| `no_resample` | `bool`          | `False` | Skip resampling to 1.7 x 1.7 x 1.7 mm³ (the image must already be about 1.7 mm and 150 x 150 x 120) |
| `gpu`         | `bool`          | `False` | Use GPU for inference (`GPU=` still works, with a deprecation warning) |
| `verbose`     | `int`           | `0`     | Verbosity: `0` = tqdm only, `1` = progress, `2` = debug |
| `gz`          | `bool`          | `True`  | `.nii.gz` outputs |
| `model`       | `str`           | `None`  | Another GDM ONNX model (default `vdm_unet3d_v002_opt`) |

**Returns:** for one input, a dictionary of nibabel images (`gdmi`, and `gdm`
when requested). For multiple inputs, returns a list of dictionaries containing
the corresponding saved file paths. A failing input of a batch is logged and
skipped, and a `RuntimeWarning` at the end lists the skipped inputs; a single
input raises.

**Outputs:** `_gdmi`, the corrected image (every volume of a 4-D series, the
input's data type, integers rounded); `_gdm` (`dmap=True`), the displacement map
in input voxels along the phase-encoding axis (float32). NaN / Inf voxels are set
to 0 with a warning.

**Phase encoding:** the network corrects along the second array axis, the A-P
axis of an axial acquisition with A-P (or P-A) phase encoding. A warning is given
when the header says otherwise (NIfTI `dim_info` phase dimension, or a second axis
that does not run A-P); such images are not corrected correctly.

Run [`tigerbx.inspect`](inspect.md) first to find the EPI inputs: `contrast` `b0`
/ `DWI` and `epi` True.

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
tigerbx.gdm('/data/EPI_dir', '/data/output', gpu=True)
```

---

## Reference

Kuo CC, et al. (2025). Referenceless reduction of spin-echo echo-planar imaging distortion with generative displacement mapping. *Magn Reson Med.* 2025; 1–16. https://doi.org/10.1002/mrm.30577
