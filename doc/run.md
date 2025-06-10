# TigerBx: `run` Module

This guide explains how to use the Python API `tigerbx.run` for brain extraction and related segmentation tasks.

## Function

`run(argstring, input, output=None, model=None)`

The `argstring` argument is a string of flags specifying which models to apply. Multiple flags can be combined, e.g. `'bd'` for brain extraction plus deep gray matter segmentation.

### Common Flags

| Flag | Description |
| ---- | ----------- |
| `b`  | Brain extraction (default if no other flags are given) |
| `m`  | Brain mask only |
| `a`  | ASEG tissue segmentation |
| `d`  | Deep gray matter mask |
| `k`  | DKT cortical parcellation |
| `c`  | Cortical thickness map |
| `C`  | CSF/GM/WM probability maps |
| `w`  | White matter parcellation |
| `W`  | White matter hypointensity mask |
| `t`  | Tumor segmentation |
| `q`  | Save a QC score to file |
| `z`  | Force `.nii.gz` output |
| `p`  | Enable patch inference |
| `g`  | Use GPU for inference |
| `clean_onnx` | Delete downloaded model files |
| `encode` / `decode` | Convert volumes to latent space or back |

### Example

```python
import tigerbx

# Perform brain extraction and deep GM segmentation
# Equivalent to running flags "bd" on the input file
result = tigerbx.run('bd', 'T1w.nii.gz', 'out_dir')
```

If a directory or wildcard pattern is provided as the `input`, all matching files are processed. When `output` is `None`, results are saved next to each input file.

