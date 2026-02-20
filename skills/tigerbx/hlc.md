# `tigerbx.hlc()` — Hierarchical Label Consolidation

Maps FreeSurfer-style ASEG/DKT labels into 56 hierarchically organised regions.
Also produces cortical thickness and CSF/GM/WM tissue probability maps.

Developed by Pin-Chuan Chen.

```python
import tigerbx

result = tigerbx.hlc(input, output=None, model=None, save='h', GPU=False, gz=True, patch=False)
```

| Parameter | Type            | Default | Description |
|-----------|-----------------|---------|-------------|
| `input`   | `str` or `list` | —       | NIfTI file, directory, glob pattern, or list of paths |
| `output`  | `str`           | `None`  | Output directory; `None` saves next to each input |
| `model`   | `dict`          | `None`  | Custom model override dict |
| `save`    | `str`           | `'h'`   | Letters specifying which outputs to generate (see table) |
| `GPU`     | `bool`          | `False` | Use GPU (requires ≥ 32 GB VRAM for hlc) |
| `gz`      | `bool`          | `True`  | Save as `.nii.gz` |
| `patch`   | `bool`          | `False` | Patch-based inference |

---

## `save` options

| Letter | Dict key / output suffix | Description |
|--------|--------------------------|-------------|
| `m`    | `tbetmask` / `_tbetmask` | Binary brain mask |
| `b`    | `tbet` / `_tbet`         | Brain-extracted image |
| `h`    | `hlc` / `_hlc`           | HLC 56-region parcellation |
| `t`    | `ct` / `_ct`             | Cortical thickness map |
| `c`    | `csf` / `_csf`           | CSF probability map |
| `g`    | `gm` / `_gm`             | GM probability map |
| `w`    | `wm` / `_wm`             | WM probability map |
| `all`  | all above                | Shorthand for `'mbhtcgw'` |

---

## Examples

```python
import tigerbx

# HLC parcellation only (default)
result = tigerbx.hlc('T1w.nii.gz', 'output/')
hlc_arr = result['hlc'].get_fdata()

# All outputs
result = tigerbx.hlc('T1w.nii.gz', 'output/', save='all')
ct_arr  = result['ct'].get_fdata()    # cortical thickness
csf_arr = result['csf'].get_fdata()   # CSF probability

# Cortical thickness + tissue maps, GPU
result = tigerbx.hlc('T1w.nii.gz', 'output/', save='tcgw', GPU=True)

# Whole directory
tigerbx.hlc('/data/T1w_dir/', '/data/output/', save='all')
```

---

## Output file naming

For input `sub-001_T1w.nii.gz`:

| Letter | Output file |
|--------|-------------|
| `m`    | `sub-001_T1w_tbetmask.nii.gz` |
| `b`    | `sub-001_T1w_tbet.nii.gz` |
| `h`    | `sub-001_T1w_hlc.nii.gz` |
| `t`    | `sub-001_T1w_ct.nii.gz` |
| `c`    | `sub-001_T1w_csf.nii.gz` |
| `g`    | `sub-001_T1w_gm.nii.gz` |
| `w`    | `sub-001_T1w_wm.nii.gz` |

See [labels.md](labels.md) for HLC label definitions (56-region and 171-label tables).
