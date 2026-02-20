# `tigerbx.run()` — Brain Extraction and Segmentation

```python
import tigerbx

result = tigerbx.run(argstring, input, output=None, model=None, silent=False)
```

| Parameter   | Type                    | Default | Description |
|-------------|-------------------------|---------|-------------|
| `argstring` | `str`                   | —       | Flag characters specifying outputs (see table below) |
| `input`     | `str`, `list`, or glob  | —       | NIfTI file, directory, glob pattern, or list of paths |
| `output`    | `str`                   | `None`  | Output directory; `None` saves next to each input |
| `model`     | `str` or `dict`         | `None`  | Custom model path/dict; `None` uses bundled defaults |
| `silent`    | `bool`                  | `False` | Suppress console output |

**Return value:**
- Single input → `dict` of nibabel images keyed by output type (e.g. `result['tbetmask']`, `result['aseg']`).
- Multiple inputs → `list` of filename dicts.
- Use `.get_fdata()` on any nibabel image to get a NumPy array.

---

## `argstring` flags

Combine freely as a string, e.g. `'bmad'`.

| Flag | Dict key / output suffix | Description |
|------|--------------------------|-------------|
| `b`  | `tbet` / `_tbet`           | Brain-extracted image |
| `m`  | `tbetmask` / `_tbetmask`   | Binary brain mask |
| `a`  | `aseg` / `_aseg`           | ASEG 43-region tissue segmentation |
| `c`  | `ct` / `_ct`               | Cortical thickness map |
| `C`  | `cgw` / `_cgw_pve0/1/2`    | CSF / GM / WM probability maps (list of 3) |
| `d`  | `dgm` / `_dgm`             | Deep gray matter mask (12 structures) |
| `S`  | `syn` / `_syn`             | SynthSeg-style ASEG |
| `W`  | `wmh` / `_wmh`             | White matter hypointensity mask |
| `t`  | `tumor` / `_tumor`         | Tumor mask |
| `q`  | `QC` / `_qc-N.log`         | QC score (also auto-written to disk when QC < 50) |
| `g`  | —                          | Use GPU |
| `p`  | —                          | Patch-based inference (for high-res inputs) |
| `z`  | —                          | Force `.nii.gz` output |

Default when no flag given: `'b'` (brain extraction only).

---

## Examples

```python
import tigerbx
import numpy as np

# Brain extraction only
result = tigerbx.run('b', 'T1w.nii.gz', 'output/')
brain = result['tbet']                    # nibabel image
brain_arr = brain.get_fdata()             # numpy array

# Recommended: brain + mask + ASEG + deep GM
result = tigerbx.run('bmad', 'T1w.nii.gz', 'output/')
mask_arr = result['tbetmask'].get_fdata()
aseg_arr = result['aseg'].get_fdata()

# All outputs, GPU, silent
result = tigerbx.run('bmacdCSWtq', 'T1w.nii.gz', 'output/', GPU=True, silent=True)
csf, gm, wm = result['cgw']              # cgw returns a list of 3 nibabel images
qc_score = result['QC']

# Process a whole directory (returns list of filename dicts)
results = tigerbx.run('bmad', '/data/T1w_dir/', '/data/output/')
for r in results:
    print(r['aseg'])                      # path to saved aseg file

# Glob pattern
tigerbx.run('bmad', '/data/**/T1w.nii.gz', '/data/output/')

# Remove cached ONNX models
tigerbx.run('clean_onnx')
```

---

## Output file naming

For input `sub-001_T1w.nii.gz` with an output directory:

| Flag | Output file |
|------|-------------|
| `b`  | `sub-001_T1w_tbet.nii.gz` |
| `m`  | `sub-001_T1w_tbetmask.nii.gz` |
| `a`  | `sub-001_T1w_aseg.nii.gz` |
| `c`  | `sub-001_T1w_ct.nii.gz` |
| `C`  | `sub-001_T1w_cgw_pve0.nii.gz`, `_pve1.nii.gz`, `_pve2.nii.gz` |
| `d`  | `sub-001_T1w_dgm.nii.gz` |
| `S`  | `sub-001_T1w_syn.nii.gz` |
| `W`  | `sub-001_T1w_wmh.nii.gz` |
| `t`  | `sub-001_T1w_tumor.nii.gz` |
| `q`  | `sub-001_T1w_qc-<score>.log` |

See [labels.md](labels.md) for ASEG and DeepGM label definitions.

---

## CLI (for simple one-off tasks)

Flags can be combined into a single argument (e.g. `-bmad`).

```bash
tiger bx T1w.nii.gz -bmad -o output/
tiger bx T1w.nii.gz -bmacdCSWtq -o output/
tiger bx /data/T1w_dir/ -bmag -o /data/output/
tiger bx --clean_onnx
```
