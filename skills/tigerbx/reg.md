# `tigerbx.reg()` / `tigerbx.transform()` — Registration and VBM

Affine and nonlinear registration to MNI space; full VBM pipeline.
Developed by Pei-Mao Sun.

---

## `tigerbx.reg()`

```python
import tigerbx

tigerbx.reg(argstring, input, output=None, model=None,
            template=None, save_displacement=False, affine_type='C2FViT')
```

| Parameter           | Type   | Default    | Description |
|---------------------|--------|------------|-------------|
| `argstring`         | `str`  | —          | Registration method flags (see table) |
| `input`             | `str`  | —          | NIfTI file, directory, or glob pattern |
| `output`            | `str`  | `None`     | Output directory |
| `affine_type`       | `str`  | `'C2FViT'` | `'C2FViT'` or `'ANTs'` — affects `r`, `F`, `v` |
| `save_displacement` | `bool` | `False`    | Save warp field as `.npz` for reuse |
| `template`          | `str`  | `None`     | Custom template NIfTI (for VBM) |

### `argstring` flags

| Flag | Description |
|------|-------------|
| `A`  | Affine registration (C2FViT or ANTs) |
| `r`  | VMnet nonlinear registration |
| `s`  | SyN nonlinear registration (ANTs) |
| `S`  | SyNCC nonlinear registration (ANTs) |
| `F`  | FuseMorph nonlinear registration |
| `R`  | Rigid registration |
| `v`  | Full VBM pipeline |
| `b`  | Also save brain-extracted image |
| `g`  | GPU |

### Examples

```python
import tigerbx

# Affine registration
tigerbx.reg('A', 'T1w.nii.gz', 'output/')

# Affine + VMnet nonlinear (C2FViT affine)
tigerbx.reg('Ar', 'T1w.nii.gz', 'output/', affine_type='C2FViT')

# FuseMorph with ANTs affine
tigerbx.reg('F', 'T1w.nii.gz', 'output/', affine_type='ANTs')

# Save warp field for later use
tigerbx.reg('Ar', 'T1w.nii.gz', 'output/', save_displacement=True)

# Full VBM pipeline on a directory
tigerbx.reg('v', '/data/T1w_dir/', '/data/output/')
```

---

## `tigerbx.transform()`

Apply a previously saved `.npz` warp field to any image (e.g. a label map).

```python
tigerbx.transform(moving, warp_npz, output=None, interpolation='linear')
```

| Parameter       | Description |
|-----------------|-------------|
| `moving`        | Path to the image to warp |
| `warp_npz`      | Path to the `.npz` warp field saved by `reg(..., save_displacement=True)` |
| `output`        | Output directory |
| `interpolation` | `'linear'` for continuous images; `'nearest'` for label/segmentation maps |

```python
import tigerbx

# Apply warp to a label map
tigerbx.transform('moving_labels.nii.gz', 'warp.npz', 'output/', interpolation='nearest')

# Apply warp to a continuous image
tigerbx.transform('moving_T1w.nii.gz', 'warp.npz', 'output/', interpolation='linear')
```

---

## CLI (for simple one-off tasks)

```bash
tiger reg T1w.nii.gz -A -o output/
tiger reg T1w.nii.gz -A -r -o output/ --affine_type C2FViT
tiger reg T1w.nii.gz -F -o output/ --affine_type ANTs
tiger reg /data/T1w_dir/ -v -o /data/output/
tiger reg T1w.nii.gz -A -r -o output/ --save_displacement
```
