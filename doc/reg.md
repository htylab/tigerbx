# TigerBx: `reg` Module

Brain image registration via `tigerbx.reg()` or the `tiger reg` CLI. The
module supports deep-learning and classical affine/nonlinear registration
methods.

DeepVBM is implemented as a separate high-level pipeline in
`tigerbx.pipelines.vbm`. See [Pipeline usage](pipelines.md) for
`tigerbx.pipeline('vbm', ...)`, `tigerbx.vbm(...)`, and `tiger vbm`.

The registration pipeline was developed by **Pei-Mao Sun**.

---

## Registration API

```python
tigerbx.reg(plan, input=None, output=None, model=None,
            template=None, gpu=False, gz=False,
            save_displacement=False, affine_type='numreg', verbose=0)
```

| Parameter           | Type            | Default     | Description |
|---------------------|-----------------|-------------|-------------|
| `plan`              | `str`           | —           | Ordered registration steps (see table below) |
| `input`             | `str` or `list` | `None`      | Input NIfTI file, directory, or glob pattern |
| `output`            | `str`           | `None`      | Output directory; if `None`, saves next to each input file |
| `model`             | `str` or `dict` | `None`      | Custom model override; `None` uses bundled defaults |
| `template`          | `str`           | `None`      | Custom template NIfTI path; `None` uses the bundled 1 mm MNI-space brain template (`mni_template.nii.gz`) |
| `gpu`               | `bool`          | `False`     | Use GPU for inference |
| `gz`                | `bool`          | `False`     | Force `.nii.gz` output |
| `save_displacement` | `bool`          | `False`     | Save the displacement field (`warp.npz`) alongside the registered image; the C2FViT `init_flow` is stored as its 3-value translation |
| `affine_type`       | `str`           | `'numreg'`  | Affine preprocessing method: `'numreg'` (classical, numpy/scipy, default) or `'C2FViT'` (deep learning); `'ANTs'` is a deprecated alias of `'numreg'` |
| `verbose`           | `int`           | `0`         | Verbosity: `0` = tqdm only, `1` = progress, `2` = debug |

### Apply a saved warp field

```python
tigerbx.transform(image_path, warp_path, output_dir=None,
                  GPU=False, interpolation='nearest')
```

| Parameter       | Type   | Default     | Description |
|-----------------|--------|-------------|-------------|
| `image_path`    | `str`  | —           | Path to the moving image to warp |
| `warp_path`     | `str`  | —           | Path to the `.npz` displacement field |
| `output_dir`    | `str`  | `None`      | Output directory |
| `GPU`           | `bool` | `False`     | Ignored (resampling runs in numpy on the CPU); kept for compatibility |
| `interpolation` | `str`  | `'nearest'` | `'nearest'` or `'linear'` |

---

## CLI Usage

```
tiger reg <PLAN> <input> [input ...] [-o OUTPUT] [options]
```

---

## Plan Steps

The `plan` string is an ordered sequence of uppercase letters specifying which
registration steps to run and in what order.

| Step | Name       | Description |
|------|------------|-------------|
| `R`  | Rigid      | Rigid (rotation + translation) registration to template with numreg; no model needed |
| `A`  | Affine     | Affine registration to template (numreg or C2FViT) |
| `D`  | VFA        | VFA nonlinear registration (requires prior `A`); recommended |
| `V`  | VMnet      | VMnet nonlinear registration (requires prior `A`) |

**Rules**

- `D` and `V` require a prior `A` step (e.g. `AD`, `AV`)
- `R` must be the first step when combined with other steps (e.g. `RAD`)
- Duplicate steps are not allowed

## Registration CLI options

| CLI flag                      | Python parameter    | Description |
|-------------------------------|---------------------|-------------|
| `--model MODEL`               | `model`             | Registration model override |
| `-g` / `--gpu`                | `gpu`               | Use GPU |
| `-z` / `--gz`                 | `gz`                | Force `.nii.gz` output |
| `-T TEMPLATE`                 | `template`          | Custom template NIfTI file (default: bundled MNI-space template) |
| `--save_displacement`         | `save_displacement` | Save the displacement field |
| `--affine_type {numreg,C2FViT}` | `affine_type`     | Affine method (default: `numreg`) |
| `--verbose N`                 | `verbose`           | Verbosity: `0` = tqdm only, `1` = progress (default), `2` = debug |

---

## Examples

### Python API

```python
import tigerbx

# Affine registration only (numreg)
tigerbx.reg('A', r'C:\T1w_dir', r'C:\output_dir')

# Affine + VFA nonlinear registration (recommended)
tigerbx.reg('AD', r'C:\T1w_dir', r'C:\output_dir')

# Affine (C2FViT) + VMnet nonlinear registration
tigerbx.reg('AV', r'C:\T1w_dir', r'C:\output_dir', affine_type='C2FViT')

# Affine + VFA, save displacement field
tigerbx.reg('AD', r'C:\T1w_dir', r'C:\output_dir', save_displacement=True)

# Apply a saved warp field to a new image
tigerbx.transform(r'C:\moving.nii.gz', r'C:\warp.npz', r'C:\output_dir',
                  interpolation='nearest')
```

### CLI

```bash
# Affine registration only
tiger reg A T1w.nii.gz -o output_dir

# Affine + VFA nonlinear registration (recommended)
tiger reg AD T1w.nii.gz -o output_dir

# Affine + VMnet nonlinear registration
tiger reg AV T1w.nii.gz -o output_dir -T template.nii.gz

# Save displacement field alongside registered image
tiger reg AV T1w.nii.gz -o output_dir --save_displacement
```

---

## Migration from v0.1.x

The `reg` module now uses a **positional plan string** instead of individual
flags.

| Old (v0.1.x)              | New (v0.2.x)                    |
|---------------------------|---------------------------------|
| `tigerbx.reg('r', ...)`   | `tigerbx.reg('AV', ...)`        |
| `tigerbx.reg('Ar', ...)`  | `tigerbx.reg('AV', ...)`        |
| `tigerbx.reg('F', ...)`   | removed (FuseMorph); use `'AD'` |
| `tigerbx.reg('s', ...)`   | removed (SyN); use `'AD'`       |
| `tigerbx.reg('S', ...)`   | removed (SyNCC); use `'AD'`     |
| `tigerbx.reg('v', ...)`   | `tigerbx.pipeline('vbm', ...)`  |
| `tiger reg -A -r ...`     | `tiger reg AV ...`              |
| `tiger reg -F ...`        | removed (FuseMorph); use `AD`   |
| `tiger reg -v ...`        | `tiger vbm ...`                 |
| `-g` in argstring         | `gpu=True` / `--gpu`            |
| `-z` in argstring         | `gz=True` / `--gz`              |

---

## Notes

- **Default template**: the bundled 1 mm MNI-space brain template (`mni_template.nii.gz`) is used when `-T` /
  `template` is not specified
- **Affine preprocessing** (`--affine_type`) affects `D` and `V` steps. `numreg` (default) is a
  deterministic numpy/scipy affine (Mattes MI, symmetric cost, 6/3/1.5 mm pyramid); it replaced the ANTs affine with
  the same accuracy (reg60 0.852 with `AD`, 0.834 with `AV`) and removed the antspyx dependency. `C2FViT` is a fully
  deep-learning affine step. `tigerbx.transform` still applies the affine stored in an older ANTs `warp.npz`
- **VFA vs VMnet**: VFA (`D`) is more accurate than VMnet (`V`); C2FViT and VMnet were
  trained with the previous MNI152 template as fixed image and work best with `-T` set to that file
- **Output names**: `_rigid` (R), `_affine` (A), `_VXM` (V, VMnet), `_VFA` (D); before 0.3 the affine and VMnet
  outputs were `_Af` and `_reg`. `tigerbx.vbm` writes the T1 after the whole plan as `_norm`
- **`warp.npz`**: the VFA field is stored as `VFA_dense_warp` and the VMnet field as `dense_warp`; `tigerbx.transform`
  writes them as `_VFA` / `_VXM`
- **SyN (`N`) and SyNCC (`C`) were removed** (they were kept for ablations; `AD` is faster and more
  accurate)
- **FuseMorph (`F`) was removed** (it tuned VMnet warps on the template aseg with optuna and was less
  accurate than `AD`). `tigerbx.transform` still applies `Fuse_dense_warp` from an older `warp.npz`
- **Pipelines**: DeepVBM is documented separately in [Pipeline usage](pipelines.md)
- **Interpolation** in `tigerbx.transform`: use `'nearest'` for label maps and
  `'linear'` for intensity images
