# TigerBx: `hlc` Module

Hierarchical Label Consolidation maps FreeSurfer-style ASEG/DKT labels into 56 hierarchically organised regions, reducing memory use while preserving anatomical detail. The module can also output cortical thickness and tissue probability maps.

The HLC module was developed by **Pin-Chuan Chen**.

---

## Python API

```python
tigerbx.hlc(input=None, output=None, model=None, save='h', GPU=False, gz=True, patch=False)
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

---

## CLI Usage

```
tiger hlc <input> [input ...] [-o OUTPUT] [--save LETTERS] [-g] [-z] [-p]
```

---

## `save` Options

| Letter | Output suffix  | Description |
|--------|----------------|-------------|
| `m`    | `_tbetmask`    | Binary brain mask |
| `b`    | `_tbet`        | Brain-extracted image |
| `h`    | `_hlc`         | HLC parcellation (56 hierarchical regions) |
| `t`    | `_ct`          | Cortical thickness map |
| `c`    | `_csf`         | CSF probability map |
| `g`    | `_gm`          | GM probability map |
| `w`    | `_wm`          | WM probability map |
| `all`  | all of the above | Shorthand for `mbhtcgw` |

---

## Examples

### Python API

```python
import tigerbx

# Default: HLC parcellation only
tigerbx.hlc('T1w.nii.gz', 'output_dir')

# Brain mask + HLC labels
tigerbx.hlc('T1w.nii.gz', 'output_dir', save='mh')

# All outputs
tigerbx.hlc('T1w.nii.gz', 'output_dir', save='all')

# Cortical thickness + tissue probability maps with GPU
tigerbx.hlc('T1w.nii.gz', 'output_dir', save='tcgw', GPU=True)

# Process a whole directory, save all outputs
tigerbx.hlc('/data/T1w_dir', '/data/output', save='all')

# Glob pattern with patch-based inference
tigerbx.hlc('/data/**/T1w.nii.gz', '/data/output', save='all', patch=True)
```

### CLI

```bash
# Default: HLC parcellation only
tiger hlc T1w.nii.gz -o output_dir

# All outputs (brain mask, bet, hlc, cortical thickness, CSF/GM/WM)
tiger hlc T1w.nii.gz --save all -o output_dir

# HLC + cortical thickness + tissue probability maps with GPU
tiger hlc T1w.nii.gz --save htcgw -g -o output_dir

# Process a whole directory with patch-based inference
tiger hlc /data/T1w_dir --save all -p -o /data/output

# Force .nii.gz output (already the default; use -z to ensure it when gz=False elsewhere)
tiger hlc T1w.nii.gz --save mh -z -o output_dir
```

---

## Output Files

For an input named `sub-001_T1w.nii.gz`:

| Letter | Output file |
|--------|-------------|
| `m`    | `sub-001_T1w_tbetmask.nii.gz` |
| `b`    | `sub-001_T1w_tbet.nii.gz` |
| `h`    | `sub-001_T1w_hlc.nii.gz` |
| `t`    | `sub-001_T1w_ct.nii.gz` |
| `c`    | `sub-001_T1w_csf.nii.gz` |
| `g`    | `sub-001_T1w_gm.nii.gz` |
| `w`    | `sub-001_T1w_wm.nii.gz` |

---

For label definitions used in the HLC parcellation, see [Label definitions](seglabel.md).
For registration tools and VBM analyses, see [Registration instructions](reginstruction.md).
