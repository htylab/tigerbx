# `tiger reg` — Registration and VBM

Affine and nonlinear registration to MNI space; full VBM pipeline.
Developed by Pei-Mao Sun.

```
tiger reg <input> [input ...] [-o OUTPUT] [flags] [--affine_type C2FViT|ANTs]
```

---

## Flags

| Flag | Description |
|------|-------------|
| `-A` | Affine registration (C2FViT or ANTs) |
| `-r` | VMnet nonlinear registration |
| `-s` | SyN nonlinear registration (ANTs) |
| `-S` | SyNCC nonlinear registration (ANTs) |
| `-F` | FuseMorph nonlinear registration |
| `-R` | Rigid registration |
| `-v` | Full VBM pipeline |
| `-b` | Also save brain-extracted image |
| `-g` | GPU |
| `--affine_type` | `C2FViT` (default) or `ANTs` — affects `-r`, `-F`, `-v` |
| `--save_displacement` | Save warp field as `.npz` for later use with `transform` |

---

## Examples

```bash
# Affine registration (C2FViT)
tiger reg T1w.nii.gz -A -o output/

# Affine + VMnet nonlinear
tiger reg T1w.nii.gz -A -r -o output/ --affine_type C2FViT

# FuseMorph with ANTs affine
tiger reg T1w.nii.gz -F -o output/ --affine_type ANTs

# Full VBM pipeline on a directory
tiger reg /data/T1w_dir -v -o /data/output/

# Save displacement field for later reuse
tiger reg T1w.nii.gz -A -r -o output/ --save_displacement
```

---

## Applying a saved warp field

Use the Python API `tigerbx.transform()` to apply a previously saved `.npz` warp to another image (e.g. a label map):

```bash
# Via Python one-liner
python -c "import tigerbx; tigerbx.transform('moving.nii.gz', 'warp.npz', 'output/', interpolation='nearest')"
```

Use `interpolation='nearest'` for segmentation/label maps, `'linear'` for continuous images.
