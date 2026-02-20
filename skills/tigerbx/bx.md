# `tiger bx` — Brain Extraction and Segmentation

```
tiger bx <input> [input ...] [-o OUTPUT] [flags]
```

Flags can be combined into a single argument (e.g. `-bmad`).

---

## Flags

| Flag | Output suffix | Description |
|------|---------------|-------------|
| `-b` | `_tbet`           | Brain-extracted image |
| `-m` | `_tbetmask`       | Binary brain mask |
| `-a` | `_aseg`           | ASEG 43-region tissue segmentation |
| `-c` | `_ct`             | Cortical thickness map |
| `-C` | `_cgw_pve0/1/2`   | CSF / GM / WM probability maps (3 files) |
| `-d` | `_dgm`            | Deep gray matter mask (12 structures) |
| `-S` | `_syn`            | SynthSeg-style ASEG |
| `-W` | `_wmh`            | White matter hypointensity mask |
| `-t` | `_tumor`          | Tumor mask |
| `-q` | `_qc-<score>.log` | QC score log |
| `-g` | —                 | Use GPU |
| `-p` | —                 | Patch-based inference (for high-res inputs) |
| `-z` | —                 | Force `.nii.gz` output |

Default when no flag given: `-b` (brain extraction only).

QC score is always computed internally. A `.log` is written automatically if QC < 50, even without `-q`.

---

## Examples

```bash
# Recommended: brain + mask + ASEG + deep GM
tiger bx T1w.nii.gz -bmad -o output/

# Brain + mask only
tiger bx T1w.nii.gz -bm -o output/

# All outputs
tiger bx T1w.nii.gz -bmacdCSWtq -o output/

# Whole directory with GPU
tiger bx /data/T1w_dir -bmag -o /data/output/

# Glob pattern
tiger bx '/data/**/T1w.nii.gz' -bmad -o /data/output/

# Patch-based inference (high-resolution inputs)
tiger bx T1w.nii.gz -bmp -o output/

# Remove cached ONNX models
tiger bx --clean_onnx
```

---

## Output naming

For input `sub-001_T1w.nii.gz`:

| Flag | Output file |
|------|-------------|
| `-b` | `sub-001_T1w_tbet.nii.gz` |
| `-m` | `sub-001_T1w_tbetmask.nii.gz` |
| `-a` | `sub-001_T1w_aseg.nii.gz` |
| `-c` | `sub-001_T1w_ct.nii.gz` |
| `-C` | `sub-001_T1w_cgw_pve0.nii.gz`, `_pve1.nii.gz`, `_pve2.nii.gz` |
| `-d` | `sub-001_T1w_dgm.nii.gz` |
| `-S` | `sub-001_T1w_syn.nii.gz` |
| `-W` | `sub-001_T1w_wmh.nii.gz` |
| `-t` | `sub-001_T1w_tumor.nii.gz` |
| `-q` | `sub-001_T1w_qc-<score>.log` |

See [labels.md](labels.md) for ASEG and DeepGM label definitions.
