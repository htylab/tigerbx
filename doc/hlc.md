# TigerBx: `hlc` Module

This guide describes how to consolidate labels using `tigerbx.hlc`.
The HLC module was developed by **Pin-Chuan Chen**.

## Function

`hlc(input, output=None, model=None, save='h', GPU=False, gz=True, patch=False)`

The function merges segmentation labels into a hierarchy and can optionally output cortical thickness or CSF/GM/WM probability maps.

### Parameters

- **input**: Path to a NIfTI file, directory, or wildcard pattern.
- **output**: Destination directory for results. Defaults to the input location if not specified.
- **model**: Custom model overrides in dictionary form. Leave `None` for default weights.
- **save**: Letters indicating which outputs to save. Options:
  - `b` – brain-extracted image
  - `m` – brain mask
  - `h` – HLC segmentation labels
  - `t` – cortical thickness map
  - `c`, `g`, `w` – CSF, GM and WM probability maps
  Use `'all'` to generate the whole set (`bmhtcgw`).
- **GPU**: Use GPU if `True`. Note: at least 32G VRAM required.
- **gz**: Save files in `.nii.gz` format.
- **patch**: Enable patch‑based inference.

### Example

```python
import tigerbx

# Run HLC with default settings and save brain mask + HLC labels
result = tigerbx.hlc('T1w_dir', 'out_dir', save='bh')
```

### CLI Example

```bash
tiger hlc T1w_dir -o out_dir --save bh -g
```

The function returns either a dictionary of NIfTI objects (single file) or a list of output filenames when processing multiple inputs.

---

For a list of label IDs used in segmentation, see [Label definitions](seglabel.md). For registration tools and VBM analyses, refer to [registration instructions](reginstruction.md).


Additional Examples
-------------------

```python
# Process a folder and only save HLC labels
# All outputs will go to the specified directory
files = tigerbx.hlc('/data/subj*/T1w.nii.gz', 'hlc_dir', save='h')

# Generate cortical thickness and CSF/GM/WM maps using GPU
# Patch mode reduces memory usage for large volumes
results = tigerbx.hlc('T1w.nii.gz', 'out_dir', save='tcgw', GPU=True, patch=True)
```

These snippets show how to select outputs, use wildcards and enable
patch inference when running the HLC module.
