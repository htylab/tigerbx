# TigerBx: `reg` Module

Brain image registration via `tigerbx.reg()`. The
module supports deep-learning and classical affine/nonlinear registration
methods.

DeepVBM is implemented as a separate high-level pipeline in
`tigerbx.pipelines.vbm`. See [Pipeline usage](pipelines.md) for
`tigerbx.vbm(...)`.
Stand-alone executable for numreg affine + VFA template normalisation (tissue-normalised `_norm`):
`tigerbx data/ -o out/ --norm`, see [rust/README.md](../rust/README.md).

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
| `save_displacement` | `bool`          | `False`     | Save the registration as `<stem>_warp.npz` (plain arrays, see Notes) for `tigerbx.transform` |
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
| `warp_path`     | `str`  | —           | `_warp.npz` of `tigerbx.reg(save_displacement=True)`, `tigerbx.affine` / `tigerbx.norm` or the `tigerbx` executable (`--affine`, `--norm`) |
| `output_dir`    | `str`  | `None`      | Output directory |
| `GPU`           | `bool` | `False`     | Ignored (resampling runs in numpy on the CPU); kept for compatibility |
| `interpolation` | `str`  | `'nearest'` | `'nearest'` or `'linear'` |

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

---

## Examples

```python
import tigerbx

# Affine registration only (numreg)
tigerbx.reg('A', r'C:\T1w_dir', r'C:\output_dir')

# Affine + VFA nonlinear registration (recommended)
tigerbx.reg('AD', r'C:\T1w_dir', r'C:\output_dir')

# Affine (C2FViT) + VMnet nonlinear registration
tigerbx.reg('AV', r'C:\T1w_dir', r'C:\output_dir', affine_type='C2FViT')

# Affine + VMnet with a custom template
tigerbx.reg('AV', r'C:\T1w_dir', r'C:\output_dir', template='template.nii.gz')

# Affine + VFA, save displacement field
tigerbx.reg('AD', r'C:\T1w_dir', r'C:\output_dir', save_displacement=True)

# Apply a saved warp field to a new image
tigerbx.transform(r'C:\moving.nii.gz', r'C:\warp.npz', r'C:\output_dir',
                  interpolation='nearest')
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
| `tigerbx.reg('v', ...)`   | `tigerbx.vbm(...)`              |
| `-g` in argstring         | `gpu=True`                      |
| `-z` in argstring         | `gz=True`                       |

---

## Notes

- **Default template**: the bundled 1 mm MNI-space brain template (`mni_template.nii.gz`) is used when
  `template` is not specified
- **Affine preprocessing** (`affine_type`) affects `D` and `V` steps. `numreg` (default) is a
  deterministic numpy/scipy affine (Mattes MI, symmetric cost, 6/3/1.5 mm pyramid); it replaced the ANTs affine with
  the same accuracy (reg60 0.852 with `AD`, 0.834 with `AV`) and removed the antspyx dependency. `C2FViT` is a fully
  deep-learning affine step
- **VFA vs VMnet**: VFA (`D`) is more accurate than VMnet (`V`); C2FViT and VMnet were
  built for the previous MNI152 template and work best with `template` set to that file
- **Output names**: `_rigid` (R), `_affine` (A), `_VXM` (V, VMnet), `_VFA` (D); before 0.3 the affine and VMnet
  outputs were `_Af` and `_reg`. `tigerbx.vbm` writes the T1 after the whole plan as `_norm`
- **`_warp.npz`**: plain arrays, `np.load(path)` without pickle (`tigerbx.core.warpfile`): `template_affine`,
  `template_shape`, `input_affine`, `input_shape`, and per step `rigid_M` / `affine_M` (4x4, template world -> input
  world), `vfa_warp` / `vxm_warp` ((3, X, Y, Z) displacement in template voxels) or `c2fvit_matrix` +
  `c2fvit_init_shift`. `tigerbx.transform` writes `_rigid`, `_affine`, `_VXM`, `_VFA`; the VBM modulation of a map
  is the Jacobian determinant of `vfa_warp` (`tigerbx.core.deform.jacobian_determinant`). The pickled `warp.npz` of
  earlier versions is no longer read: re-run the registration
- **SyN (`N`) and SyNCC (`C`) were removed** (they were kept for ablations; `AD` is faster and more
  accurate)
- **FuseMorph (`F`) was removed** (it tuned VMnet warps on the template tseg with optuna and was less
  accurate than `AD`)
- **Pipelines**: DeepVBM is documented separately in [Pipeline usage](pipelines.md)
- **Interpolation** in `tigerbx.transform`: use `'nearest'` for label maps and
  `'linear'` for intensity images. Each output (and `tigerbx.modules.reg.apply_warp`) samples the image once from its own
  grid at the affine (+ dense warp) composed, not reorder -> affine -> warp: on reg60 the carried labels score
  0.866 instead of 0.852
