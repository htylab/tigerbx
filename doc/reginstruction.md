# TigerBx: `reg` Module

Brain image registration and voxel-based morphometry (VBM) via `tigerbx.reg()` or the `tiger reg` CLI. The module supports deep-learning and classical affine/nonlinear registration methods.

The VBM and registration pipeline was developed by **Pei-Mao Sun**.

---

## Python API

```python
tigerbx.reg(argstring, input=None, output=None, model=None,
            template=None, save_displacement=False, affine_type='C2FViT')
```

| Parameter           | Type            | Default     | Description |
|---------------------|-----------------|-------------|-------------|
| `argstring`         | `str`           | —           | One or more flag characters (see table below) |
| `input`             | `str` or `list` | `None`      | Input NIfTI file, directory, or glob pattern |
| `output`            | `str`           | `None`      | Output directory; if `None`, saves next to each input file |
| `model`             | `str` or `dict` | `None`      | Custom model override; `None` uses bundled defaults |
| `template`          | `str`           | `None`      | Custom template NIfTI path; `None` uses the bundled MNI152 1 mm brain |
| `save_displacement` | `bool`          | `False`     | Save the displacement field alongside the registered image |
| `affine_type`       | `str`           | `'C2FViT'`  | Affine preprocessing method: `'C2FViT'` (deep learning) or `'ANTs'` (classical) |

### Apply a saved warp field

```python
tigerbx.transform(image_path, warp_path, output_dir=None, GPU=False, interpolation='nearest')
```

| Parameter      | Type   | Default     | Description |
|----------------|--------|-------------|-------------|
| `image_path`   | `str`  | —           | Path to the moving image to warp |
| `warp_path`    | `str`  | —           | Path to the `.npz` displacement field |
| `output_dir`   | `str`  | `None`      | Output directory |
| `GPU`          | `bool` | `False`     | Use GPU |
| `interpolation`| `str`  | `'nearest'` | `'nearest'` or `'linear'` |

---

## CLI Usage

```
tiger reg <input> [input ...] [-o OUTPUT] [method flags] [options]
```

---

## Flags

| API flag | CLI flag | Description |
|----------|----------|-------------|
| `A`      | `-A`     | Affine registration to template (C2FViT or ANTs) |
| `r`      | `-r`     | VMnet nonlinear registration to template |
| `s`      | `-s`     | SyN nonlinear registration (ANTs-based) |
| `S`      | `-S`     | SyNCC nonlinear registration (ANTs-based) |
| `F`      | `-F`     | FuseMorph nonlinear registration |
| `R`      | `-R`     | Rigid registration |
| `v`      | `-v`     | Full VBM analysis pipeline |
| `b`      | `-b`     | Also save a brain-extracted image |
| `g`      | `-g`     | Use GPU |
| `z`      | `-z`     | Force `.nii.gz` output |

Additional options:

| CLI flag                      | Python parameter    | Description |
|-------------------------------|---------------------|-------------|
| `-T TEMPLATE`                 | `template`          | Custom template NIfTI file (default: bundled MNI152) |
| `--save_displacement`         | `save_displacement` | Save the displacement field |
| `--affine_type {C2FViT,ANTs}` | `affine_type`       | Affine method used by `-r`, `-F`, and `-v` (default: `C2FViT`) |

> `-s` and `-S` include their own affine preprocessing and are not affected by `--affine_type`.
> `-R` (rigid) operates independently. Combining rigid with other methods is only supported when *not* saving displacement fields.

---

## Examples

### Python API

```python
import tigerbx

# Affine registration only (C2FViT)
tigerbx.reg('A', r'C:\T1w_dir', r'C:\output_dir')

# VMnet nonlinear registration with C2FViT affine preprocessing
tigerbx.reg('r', r'C:\T1w_dir', r'C:\output_dir',
            template='template.nii.gz', affine_type='C2FViT')

# Affine + VMnet in one pass
tigerbx.reg('Ar', r'C:\T1w_dir', r'C:\output_dir')

# FuseMorph registration with ANTs affine preprocessing
tigerbx.reg('F', r'C:\T1w_dir', r'C:\output_dir', affine_type='ANTs')

# SyN registration (ANTs), save displacement field
tigerbx.reg('s', r'C:\T1w_dir', r'C:\output_dir', save_displacement=True)

# Full VBM pipeline
tigerbx.reg('v', r'C:\T1w_dir\**\*.nii.gz', r'C:\output_dir', affine_type='C2FViT')

# Apply a saved warp field to a new image
tigerbx.transform(r'C:\moving.nii.gz', r'C:\warp.npz', r'C:\output_dir',
                  interpolation='nearest')
```

### CLI

```bash
# Affine registration only
tiger reg T1w.nii.gz -A -o output_dir

# VMnet nonlinear registration
tiger reg T1w.nii.gz -r -o output_dir -T template.nii.gz --affine_type C2FViT

# Affine + VMnet in one pass
tiger reg T1w.nii.gz -A -r -o output_dir

# FuseMorph registration with ANTs affine preprocessing
tiger reg T1w.nii.gz -F -o output_dir --affine_type ANTs

# SyN registration
tiger reg T1w.nii.gz -s -o output_dir

# Full VBM pipeline on a directory
tiger reg /data/T1w_dir -v -o /data/output --affine_type C2FViT

# Save displacement field alongside registered image
tiger reg T1w.nii.gz -r -o output_dir --save_displacement
```

---

## Notes

- **Default template**: the bundled MNI152 1 mm brain is used when `-T` / `template` is not specified.
- **Affine preprocessing** (`--affine_type`) affects `-r`, `-F`, and `-v`. Use `C2FViT` (default) for a fully deep-learning pipeline; use `ANTs` for a classical affine step.
- **Displacement fields** saved by `-s` and `-S` are stored as file paths rather than arrays — take extra care when re-applying them.
- **Interpolation** in `tigerbx.transform`: use `'nearest'` for label maps and `'linear'` for intensity images.
