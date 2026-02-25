# TigerBx: `reg` Module

Brain image registration via `tigerbx.reg()` or the `tiger reg` CLI. The module supports deep-learning and classical affine/nonlinear registration methods.

VBM is available as a separate subcommand: `tigerbx.vbm()` / `tiger vbm`.

The VBM and registration pipeline was developed by **Pei-Mao Sun**.

---

## Python API

```python
tigerbx.reg(plan, input=None, output=None, model=None,
            template=None, gpu=False, gz=False,
            save_displacement=False, affine_type='C2FViT', verbose=0)
```

| Parameter           | Type            | Default     | Description |
|---------------------|-----------------|-------------|-------------|
| `plan`              | `str`           | —           | Ordered registration steps (see table below) |
| `input`             | `str` or `list` | `None`      | Input NIfTI file, directory, or glob pattern |
| `output`            | `str`           | `None`      | Output directory; if `None`, saves next to each input file |
| `model`             | `str` or `dict` | `None`      | Custom model override; `None` uses bundled defaults |
| `template`          | `str`           | `None`      | Custom template NIfTI path; `None` uses the bundled MNI152 1 mm brain |
| `gpu`               | `bool`          | `False`     | Use GPU for inference |
| `gz`                | `bool`          | `False`     | Force `.nii.gz` output |
| `save_displacement` | `bool`          | `False`     | Save the displacement field alongside the registered image |
| `affine_type`       | `str`           | `'C2FViT'`  | Affine preprocessing method: `'C2FViT'` (deep learning) or `'ANTs'` (classical) |
| `verbose`           | `int`           | `0`         | Verbosity: 0=quiet, 1=progress, 2=debug |

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
tiger reg <PLAN> <input> [input ...] [-o OUTPUT] [options]
```

---

## Plan Steps

The `plan` string is an ordered sequence of uppercase letters specifying which registration steps to run and in what order.

| Step | Name       | Description |
|------|------------|-------------|
| `R`  | Rigid      | Rigid registration to template |
| `A`  | Affine     | Affine registration to template (C2FViT or ANTs) |
| `V`  | VMnet      | VMnet nonlinear registration (requires prior `A`) |
| `N`  | SyN        | SyN nonlinear registration (ANTs-based) |
| `C`  | SyNCC      | SyNCC nonlinear registration (ANTs-based) |
| `F`  | FuseMorph  | FuseMorph nonlinear registration (requires prior `A`) |

**Rules:**
- `V` and `F` require a prior `A` step (e.g. `AV`, `AF`).
- `R` must be the first step when combined with other steps (e.g. `RAF`).
- `N` and `C` include their own affine preprocessing and are not affected by `--affine_type`.
- Duplicate steps are not allowed.

Additional CLI options:

| CLI flag                      | Python parameter    | Description |
|-------------------------------|---------------------|-------------|
| `-g` / `--gpu`                | `gpu`               | Use GPU |
| `-z` / `--gz`                 | `gz`                | Force `.nii.gz` output |
| `-T TEMPLATE`                 | `template`          | Custom template NIfTI file (default: bundled MNI152) |
| `--save_displacement`         | `save_displacement` | Save the displacement field |
| `--affine_type {C2FViT,ANTs}` | `affine_type`       | Affine method (default: `C2FViT`) |

---

## Examples

### Python API

```python
import tigerbx

# Affine registration only (C2FViT)
tigerbx.reg('A', r'C:\T1w_dir', r'C:\output_dir')

# Affine + VMnet nonlinear registration
tigerbx.reg('AV', r'C:\T1w_dir', r'C:\output_dir')

# Affine + FuseMorph with ANTs affine preprocessing
tigerbx.reg('AF', r'C:\T1w_dir', r'C:\output_dir', affine_type='ANTs')

# SyN registration (ANTs), save displacement field
tigerbx.reg('AN', r'C:\T1w_dir', r'C:\output_dir', save_displacement=True)

# SyNCC registration (ANTs)
tigerbx.reg('AC', r'C:\T1w_dir', r'C:\output_dir')

# Apply a saved warp field to a new image
tigerbx.transform(r'C:\moving.nii.gz', r'C:\warp.npz', r'C:\output_dir',
                  interpolation='nearest')

# VBM pipeline (separate API)
tigerbx.vbm(r'C:\T1w_dir', r'C:\output_dir')
```

### CLI

```bash
# Affine registration only
tiger reg A T1w.nii.gz -o output_dir

# Affine + VMnet nonlinear registration
tiger reg AV T1w.nii.gz -o output_dir -T template.nii.gz

# Affine + FuseMorph with ANTs affine preprocessing
tiger reg AF T1w.nii.gz -o output_dir --affine_type ANTs

# SyN registration
tiger reg AN T1w.nii.gz -o output_dir

# Save displacement field alongside registered image
tiger reg AV T1w.nii.gz -o output_dir --save_displacement

# VBM pipeline (separate subcommand)
tiger vbm /data/T1w_dir -o /data/output
```

---

## Migration from v0.1.x

The `reg` module now uses a **positional plan string** instead of individual flags.

| Old (v0.1.x)              | New (v0.2.x)                    |
|---------------------------|---------------------------------|
| `tigerbx.reg('r', ...)`  | `tigerbx.reg('AV', ...)`       |
| `tigerbx.reg('Ar', ...)` | `tigerbx.reg('AV', ...)`       |
| `tigerbx.reg('F', ...)`  | `tigerbx.reg('AF', ...)`       |
| `tigerbx.reg('s', ...)`  | `tigerbx.reg('AN', ...)`       |
| `tigerbx.reg('S', ...)`  | `tigerbx.reg('AC', ...)`       |
| `tigerbx.reg('v', ...)`  | `tigerbx.vbm(...)`             |
| `tiger reg -A -r ...`    | `tiger reg AV ...`             |
| `tiger reg -F ...`       | `tiger reg AF ...`             |
| `tiger reg -v ...`       | `tiger vbm ...`                |
| `-g` in argstring        | `gpu=True` / `--gpu`           |
| `-z` in argstring        | `gz=True` / `--gz`             |

---

## Notes

- **Default template**: the bundled MNI152 1 mm brain is used when `-T` / `template` is not specified.
- **Affine preprocessing** (`--affine_type`) affects `V` and `F` steps. Use `C2FViT` (default) for a fully deep-learning pipeline; use `ANTs` for a classical affine step.
- **Interpolation** in `tigerbx.transform`: use `'nearest'` for label maps and `'linear'` for intensity images.
