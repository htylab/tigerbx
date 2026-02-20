# TigerBx: `bx` Module

Brain extraction and tissue segmentation via `tigerbx.run()` or the `tiger bx` CLI.

---

## Python API

```python
tigerbx.run(argstring, input=None, output=None, model=None, silent=False)
```

| Parameter   | Type            | Default | Description |
|-------------|-----------------|---------|-------------|
| `argstring` | `str`           | —       | One or more flag characters specifying which outputs to produce (see table below) |
| `input`     | `str` or `list` | `None`  | Input NIfTI file, directory, or glob pattern |
| `output`    | `str`           | `None`  | Output directory; if `None`, results are saved next to each input file |
| `model`     | `str` or `dict` | `None`  | Custom model path or override dict; `None` uses bundled defaults |
| `silent`    | `bool`          | `False` | Suppress all console output |

If no segmentation flag is provided, brain extraction (`b`) runs by default.

---

## CLI Usage

```
tiger bx <input> [input ...] [-o OUTPUT] [flags ...]
```

---

## Flags

| API flag | CLI flag | Output suffix        | Description |
|----------|----------|----------------------|-------------|
| `b`      | `-b`     | `_tbet`              | Brain-extracted image |
| `m`      | `-m`     | `_tbetmask`          | Binary brain mask |
| `a`      | `-a`     | `_aseg`              | ASEG tissue segmentation (43 regions) |
| `c`      | `-c`     | `_ct`                | Cortical thickness map |
| `C`      | `-C`     | `_cgw_pve0/1/2`      | CSF / GM / WM probability maps (3 files, FSL-style PVE) |
| `d`      | `-d`     | `_dgm`               | Deep gray matter mask (12 structures) |
| `S`      | `-S`     | `_syn`               | SynthSeg-style ASEG segmentation |
| `W`      | `-W`     | `_wmh`               | White matter hypointensity mask |
| `t`      | `-t`     | `_tumor`             | Tumor mask |
| `q`      | `-q`     | `_qc-<score>.log`    | QC score log (also written automatically when QC < 50) |
| `g`      | `-g`     | —                    | Use GPU for inference |
| `z`      | `-z`     | —                    | Force `.nii.gz` output format |
| `p`      | `-p`     | —                    | Enable patch-based inference |

Additional CLI-only options:

| CLI flag        | Description |
|-----------------|-------------|
| `--model MODEL` | Specify a custom ONNX model path or dict string |
| `--clean_onnx`  | Delete all downloaded ONNX model files |
| `--silent`      | Suppress console output |

---

## Examples

### Python API

```python
import tigerbx

# Brain extraction only (default when no flag is given)
tigerbx.run('b', 'T1w.nii.gz', 'output_dir')

# Brain mask + brain image + ASEG + deep gray matter (recommended)
tigerbx.run('bmad', 'T1w.nii.gz', 'output_dir')

# Full pipeline — all output types
tigerbx.run('bmacdCSWtq', 'T1w.nii.gz', 'output_dir')

# Process an entire directory; outputs saved next to each input file
tigerbx.run('bm', '/data/T1w_dir')

# Glob pattern, GPU inference, silent mode
tigerbx.run('bmag', '/data/**/T1w.nii.gz', '/data/output', silent=True)

# Delete downloaded ONNX model files
tigerbx.run('clean_onnx')
```

### CLI

```bash
# Brain mask + brain image + ASEG + deep gray matter (recommended)
tiger bx T1w.nii.gz -bmad -o output_dir

# Brain extraction and brain mask only
tiger bx T1w.nii.gz -b -m -o output_dir

# Full pipeline — all output types
tiger bx T1w.nii.gz -b -m -a -c -C -d -S -W -t -q -o output_dir

# Process a whole directory with GPU
tiger bx /data/T1w_dir -b -m -a -g -o /data/output

# Patch-based inference for high-resolution inputs
tiger bx T1w.nii.gz -b -m -p -o output_dir

# Delete downloaded ONNX model files
tiger bx --clean_onnx
```

---

## Output Files

For an input named `sub-001_T1w.nii.gz`, outputs are:

| Flag | Output file |
|------|-------------|
| `-b` | `sub-001_T1w_tbet.nii.gz` |
| `-m` | `sub-001_T1w_tbetmask.nii.gz` |
| `-a` | `sub-001_T1w_aseg.nii.gz` |
| `-c` | `sub-001_T1w_ct.nii.gz` |
| `-C` | `sub-001_T1w_cgw_pve0.nii.gz`, `_pve1.nii.gz`, `_pve2.nii.gz` (CSF / GM / WM) |
| `-d` | `sub-001_T1w_dgm.nii.gz` |
| `-S` | `sub-001_T1w_syn.nii.gz` |
| `-W` | `sub-001_T1w_wmh.nii.gz` |
| `-t` | `sub-001_T1w_tumor.nii.gz` |
| `-q` | `sub-001_T1w_qc-<score>.log` |

---

For label definitions used in ASEG, DeepGM, and SynthSeg outputs, see [Label definitions](seglabel.md).
