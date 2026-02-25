# TigerBx: `bx` Module

Brain extraction and tissue segmentation via `tigerbx.run()` or the `tiger bx` CLI.

---

## Python API

```python
tigerbx.run(argstring, input=None, output=None, model=None,
            verbose=0, chunk_size=50, continue_=False, silent=False,
            save_outputs=True)
```

| Parameter    | Type            | Default | Description |
|--------------|-----------------|---------|-------------|
| `argstring`  | `str`           | —       | One or more flag characters specifying which outputs to produce (see table below) |
| `input`      | `str` or `list` | `None`  | Input NIfTI file, directory, glob pattern, or list of paths |
| `output`     | `str`           | `None`  | Output directory; if `None`, results are saved next to each input file |
| `model`      | `str` or `dict` | `None`  | Custom model path or override dict; `None` uses bundled defaults |
| `verbose`    | `int`           | `0`     | Verbosity level: `0` = tqdm progress bars only, `1` = progress messages, `2` = debug |
| `chunk_size` | `int`           | `50`    | Number of files per session-cache batch. Larger values reuse the ONNX session more aggressively but use more RAM |
| `continue_`  | `bool`          | `False` | Skip files whose output files already exist on disk |
| `silent`     | `bool`          | `False` | *(Deprecated)* Suppress all output. Use `verbose=0` instead |
| `save_outputs` | `bool`        | `True`  | Write output files to disk. Set `False` to get in-memory nibabel objects only |

If no segmentation flag is provided, brain extraction (`b`) runs by default.

### Return value

- **Single file:** `dict` containing output nibabel images (keys match the flag names, e.g. `'tbetmask'`, `'tbet'`, `'aseg'`) plus `'QC'` and `'QC_raw'`.
- **Multiple files:** `list` of `dict`, one per input file, containing output file paths and `'QC'` and `'QC_raw'` values.

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
| `q`      | `-q`     | `_qc-<score>.log`    | QC score log (also written automatically when QC < 50) |
| `g`      | `-g`     | —                    | Use GPU for inference |
| `z`      | `-z`     | —                    | Force `.nii.gz` output format |
| `p`      | `-p`     | —                    | Enable patch-based inference |

Additional CLI-only options:

| CLI option              | Default | Description |
|-------------------------|---------|-------------|
| `--verbose N`           | `1`     | Verbosity: `0` = tqdm only, `1` = progress messages (default), `2` = debug |
| `--chunk-size N`        | `50`    | Files per session-cache batch |
| `--continue`            | off     | Skip files whose outputs already exist |
| `--model MODEL`         | —       | Specify a custom ONNX model path or dict string |
| `--clean_onnx`          | —       | Delete all downloaded ONNX model files |

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
tigerbx.run('bmacdCSW', 'T1w.nii.gz', 'output_dir')

# Process an entire directory; outputs saved next to each input file
tigerbx.run('bm', '/data/T1w_dir')

# Batch processing with session cache — 100 files per chunk
tigerbx.run('bmad', '/data/T1w_dir', '/data/output', chunk_size=100)

# Resume an interrupted run — skip already-processed files
tigerbx.run('bmad', '/data/T1w_dir', '/data/output', continue_=True)

# Verbose progress messages
tigerbx.run('bm', 'T1w.nii.gz', 'output_dir', verbose=1)

# Delete downloaded ONNX model files
tigerbx.run('clean_onnx')
```

### CLI

```bash
# Brain mask + brain image + ASEG + deep gray matter (recommended)
tiger bx T1w.nii.gz -bmad -o output_dir

# Brain extraction and brain mask only
tiger bx T1w.nii.gz -bm -o output_dir

# Full pipeline — all output types
tiger bx T1w.nii.gz -bmacdCSW -o output_dir

# Process a whole directory with GPU
tiger bx /data/T1w_dir -bmag -o /data/output

# Large batch: 100 files per session-cache chunk
tiger bx /data/T1w_dir -bmad -o /data/output --chunk-size 100

# Resume a previously interrupted run
tiger bx /data/T1w_dir -bmad -o /data/output --continue

# Patch-based inference for high-resolution inputs
tiger bx T1w.nii.gz -bmp -o output_dir

# Quiet output (tqdm bars only)
tiger bx /data/T1w_dir -bm -o /data/output --verbose 0

# Debug output (disable tqdm, print per-file details)
tiger bx T1w.nii.gz -bm -o output_dir --verbose 2

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
| `-q` | `sub-001_T1w_qc-<score>.log` |

---

## QC Score

Every run computes a QC confidence score (0–100) for the brain extraction step:

- **100**: high-confidence extraction (predicted Dice ≥ 0.95)
- **< 50**: low confidence — a `[WARNING]` is printed immediately and again in a summary at the end of the run; a `.log` file is also written automatically
- The `QC_raw` float value (0–1) is always available in the return dict

---

For label definitions used in ASEG, DeepGM, and SynthSeg outputs, see [Label definitions](seglabel.md).
