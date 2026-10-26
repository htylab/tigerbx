# TigerBx: `hlc` Module

Hierarchical Label Consolidation maps FreeSurfer-style ASEG/DKT labels into 56 hierarchically organised regions, reducing memory use while preserving anatomical detail.

The HLC module was developed by **Pin-Chuan Chen**.

---

## Python API

```python
tigerbx.hlc(input=None, output=None, model=None, save='h', GPU=False, gz=True, patch=False, verbose=0)
```

| Parameter | Type            | Default | Description |
|-----------|-----------------|---------|-------------|
| `input`   | `str` or `list` | `None`  | Input NIfTI file, directory, or glob pattern |
| `output`  | `str`           | `None`  | Output directory; if `None`, saves next to each input file |
| `model`   | `dict`          | `None`  | Custom model override dict; `None` uses bundled defaults |
| `save`    | `str`           | `'h'`   | Letters specifying which outputs to generate (see table below) |
| `GPU`     | `bool`          | `False` | Use GPU for inference (requires at least 32 GB VRAM) |
| `gz`      | `bool`          | `True`  | Save in `.nii.gz` format |
| `patch`   | `bool`          | `False` | Enable patch-based inference |
| `verbose` | `int`           | `0`     | Verbosity: `0` = tqdm only, `1` = progress, `2` = debug |

---

## `save` Options

| Letter | Output suffix  | Description |
|--------|----------------|-------------|
| `m`    | `_tbxmask`    | Binary brain mask |
| `b`    | `_tbx`        | Brain-extracted image |
| `h`    | `_hlc`         | HLC parcellation (56 hierarchical regions) |
| `all`  | all of the above | Shorthand for `mbh` |

The cortical thickness (`t`) and CSF / GM / WM maps (`c`, `g`, `w`) were removed: use `tigerbx.run('c')` (the CT
model) and `tigerbx.seg(save_tissue=True)` (computed from TSEG).

---

## Examples

```python
import tigerbx

# Default: HLC parcellation only
tigerbx.hlc('T1w.nii.gz', 'output_dir')

# Brain mask + HLC labels
tigerbx.hlc('T1w.nii.gz', 'output_dir', save='mh')

# All outputs
tigerbx.hlc('T1w.nii.gz', 'output_dir', save='all')

# HLC labels with GPU
tigerbx.hlc('T1w.nii.gz', 'output_dir', GPU=True)

# Process a whole directory, save all outputs
tigerbx.hlc('/data/T1w_dir', '/data/output', save='all')

# Patch-based inference
tigerbx.hlc('/data/T1w_dir', '/data/output', save='all', patch=True)
```

---

## Output Files

For an input named `sub-001_T1w.nii.gz`:

| Letter | Output file |
|--------|-------------|
| `m`    | `sub-001_T1w_tbxmask.nii.gz` |
| `b`    | `sub-001_T1w_tbx.nii.gz` |
| `h`    | `sub-001_T1w_hlc.nii.gz` |

---

For label definitions used in the HLC parcellation, see [Label definitions](seglabel.md).
For registration tools, see [Registration usage](reg.md).
For VBM and other high-level workflows, see [Pipeline usage](pipelines.md).
