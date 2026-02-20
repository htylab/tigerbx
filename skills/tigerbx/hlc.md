# `tiger hlc` — Hierarchical Label Consolidation

Maps FreeSurfer-style ASEG/DKT labels into 56 hierarchically organised regions.
Also produces cortical thickness and CSF/GM/WM tissue probability maps.

Developed by Pin-Chuan Chen.

```
tiger hlc <input> [input ...] [-o OUTPUT] [--save LETTERS] [-g] [-p] [-z]
```

---

## `--save` letters

| Letter | Output suffix | Description |
|--------|---------------|-------------|
| `m`    | `_tbetmask`   | Binary brain mask |
| `b`    | `_tbet`       | Brain-extracted image |
| `h`    | `_hlc`        | HLC 56-region parcellation |
| `t`    | `_ct`         | Cortical thickness map |
| `c`    | `_csf`        | CSF probability map |
| `g`    | `_gm`         | GM probability map |
| `w`    | `_wm`         | WM probability map |
| `all`  | all above     | Shorthand for `mbhtcgw` |

Default: `--save h` (HLC parcellation only).

---

## Examples

```bash
# HLC parcellation only (default)
tiger hlc T1w.nii.gz -o output/

# All outputs
tiger hlc T1w.nii.gz --save all -o output/

# HLC + cortical thickness + tissue maps, GPU
tiger hlc T1w.nii.gz --save htcgw -g -o output/

# Whole directory, patch-based inference
tiger hlc /data/T1w_dir --save all -p -o /data/output/
```

---

## Output naming

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
