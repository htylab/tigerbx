# TigerBx: `pipelines`

High-level multi-stage workflows are exposed through `tigerbx.pipeline()` and
dedicated CLI subcommands. Pipelines compose lower-level modules into a more
opinionated workflow.

Currently available pipelines:

| Pipeline | Dispatcher | Convenience alias | CLI |
|----------|------------|-------------------|-----|
| `vbm`    | `tigerbx.pipeline('vbm', ...)` | `tigerbx.vbm(...)` | `tiger vbm ...` |

The VBM pipeline was developed by **Pei-Mao Sun**.

---

## Dispatcher API

```python
tigerbx.pipeline(name, input=None, output=None, **kwargs)
```

| Parameter | Type  | Default | Description |
|-----------|-------|---------|-------------|
| `name`    | `str` | —       | Pipeline name. Currently only `vbm` is available |
| `input`   | `str` or `list` | `None` | Input NIfTI file, directory, or glob pattern |
| `output`  | `str` | `None`  | Output directory; if `None`, saves next to each input file |
| `**kwargs`| —     | —       | Forwarded to the selected pipeline implementation |

If you prefer a direct function call, `tigerbx.vbm(...)` is kept as a
convenience alias for the `vbm` pipeline.

---

## VBM Pipeline API

```python
tigerbx.pipeline('vbm', input=None, output=None,
                 model=None, template=None, reg_plan='AF',
                 gpu=False, gz=False, save_displacement=False,
                 affine_type='C2FViT', verbose=0)

# convenience alias
tigerbx.vbm(input=None, output=None, model=None, template=None,
            reg_plan='AF', gpu=False, gz=False,
            save_displacement=False, affine_type='C2FViT', verbose=0)
```

| Parameter           | Type            | Default     | Description |
|---------------------|-----------------|-------------|-------------|
| `input`             | `str` or `list` | `None`      | Input NIfTI file, directory, or glob pattern |
| `output`            | `str`           | `None`      | Output directory; if `None`, saves next to each input file |
| `model`             | `str` or `dict` | `None`      | Registration model override; `None` uses bundled defaults |
| `template`          | `str`           | `None`      | Custom template NIfTI path; `None` uses the bundled MNI152 1 mm brain |
| `reg_plan`          | `str`           | `'AF'`      | Registration plan used inside the VBM pipeline |
| `gpu`               | `bool`          | `False`     | Use GPU for inference |
| `gz`                | `bool`          | `False`     | Force `.nii.gz` output |
| `save_displacement` | `bool`          | `False`     | Save the displacement field produced by the registration stage |
| `affine_type`       | `str`           | `'C2FViT'`  | Affine preprocessing method used by `V` / `F` steps |
| `verbose`           | `int`           | `0`         | Verbosity: `0` = tqdm only, `1` = progress, `2` = debug |

`reg_plan` uses the same uppercase plan syntax as `tigerbx.reg()`. See
[Registration usage](reg.md) for the step definitions and plan rules.

---

## CLI Usage

```
tiger vbm <input> [input ...] [-o OUTPUT] [options]
```

## CLI options

| CLI flag                      | Python parameter    | Description |
|-------------------------------|---------------------|-------------|
| `--model MODEL`               | `model`             | Registration model override |
| `-g` / `--gpu`                | `gpu`               | Use GPU |
| `-z` / `--gz`                 | `gz`                | Force `.nii.gz` output |
| `--reg-plan PLAN`             | `reg_plan`          | Registration plan used inside DeepVBM (default: `AF`) |
| `-T TEMPLATE`                 | `template`          | Custom template NIfTI file (default: bundled MNI152) |
| `--save_displacement`         | `save_displacement` | Save the displacement field |
| `--affine_type {C2FViT,ANTs}` | `affine_type`       | Affine method (default: `C2FViT`) |
| `--verbose N`                 | `verbose`           | Verbosity: `0` = tqdm only, `1` = progress (default), `2` = debug |

---

## Examples

### Python API

```python
import tigerbx

# Recommended dispatcher entry point
tigerbx.pipeline('vbm', r'C:\T1w_dir', r'C:\output_dir')

# Convenience alias
tigerbx.vbm(r'C:\T1w_dir', r'C:\output_dir')

# Customize the registration stage used inside the pipeline
tigerbx.pipeline('vbm', r'C:\T1w_dir', r'C:\output_dir',
                 reg_plan='AF', affine_type='ANTs')

# Save the displacement field produced by the registration stage
tigerbx.pipeline('vbm', r'C:\T1w_dir', r'C:\output_dir',
                 save_displacement=True)
```

### CLI

```bash
# Run DeepVBM on a directory
tiger vbm /data/T1w_dir -o /data/output

# Customize the registration plan used internally
tiger vbm /data/T1w_dir -o /data/output --reg-plan AF

# Use ANTs for the affine stage inside the pipeline
tiger vbm /data/T1w_dir -o /data/output --affine_type ANTs

# Save the displacement field from the registration stage
tiger vbm /data/T1w_dir -o /data/output --save_displacement
```

---

## Outputs

The VBM pipeline runs registration first, then generates a modulated and
smoothed gray-matter map in template space.

- Registration-stage outputs follow the selected `reg_plan`
- A subject-specific subdirectory is created for the VBM result
- The final smoothed GM image is saved as `*_SmoothedGM.nii.gz`

For example, an input named `sub-001_T1w.nii.gz` produces:

```text
output_dir/
  sub-001_T1w_Af.nii.gz
  sub-001_T1w_Fuse.nii.gz
  sub-001_T1w/
    sub-001_T1w_SmoothedGM.nii.gz
```

The exact registration outputs depend on `reg_plan`. If `save_displacement=True`
is set, the registration stage also writes a `warp.npz` file.
