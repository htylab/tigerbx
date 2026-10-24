# TigerBx: `bx` Module

Brain extraction and tissue segmentation via `tigerbx.run()`.
Stand-alone executable for tBET / ASEG / template normalisation: `tigerbx data/ -o out/ --seg`, see
[rust/README.md](../rust/README.md).

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

- **Single file:** `dict` containing output nibabel images, keyed by the output suffix (not the flag letter): `m` →
  `'tbetmask'`, `b` → `'tbet'`, `a` → `'aseg'`, `C` → `'cgw'` (list of CSF / GM / WM maps).
  For example the brain mask of `run('m', ...)` is `result['tbetmask']`, not `result['tbet']`.
- **Multiple files:** `list` of `dict`, one per input file, containing output file paths.

---

## Flags

| Flag | Output suffix        | Description |
|------|----------------------|-------------|
| `b`  | `_tbet`              | Brain-extracted image |
| `m`  | `_tbetmask`          | Binary brain mask |
| `a`  | `_aseg`              | ASEG segmentation (43 regions + 99 Unlabeled-Intracranial; TIV = all non-zero voxels) |
| `c`  | `_ct`                | Cortical thickness map |
| `C`  | `_cgw_pve0/1/2`      | CSF / GM / WM probability maps (3 files, FSL-style PVE) |
| `d`  | `_dgm`               | Deep gray matter mask (12 structures) |
| `g`  | —                    | Use GPU for inference |
| `z`  | —                    | Force `.nii.gz` output format |
| `p`  | —                    | Enable patch-based inference |

The flags `q` (tBET QC score), `S` (SynthSeg-style ASEG) and `W` (white-matter hypointensity) have been removed; an
argstring containing one of them raises a `ValueError`. For image quality use `tigerbx.core.qa.image_quality`.

---

## Examples

```python
import tigerbx

# Brain extraction only (default when no flag is given)
tigerbx.run('b', 'T1w.nii.gz', 'output_dir')

# Brain mask + brain image + ASEG + deep gray matter (recommended)
tigerbx.run('bmad', 'T1w.nii.gz', 'output_dir')

# Full pipeline — all output types
tigerbx.run('bmacdC', 'T1w.nii.gz', 'output_dir')

# Process an entire directory; outputs saved next to each input file
tigerbx.run('bm', '/data/T1w_dir')

# Batch processing with session cache — 100 files per chunk
tigerbx.run('bmad', '/data/T1w_dir', '/data/output', chunk_size=100)

# Resume an interrupted run — skip already-processed files
tigerbx.run('bmad', '/data/T1w_dir', '/data/output', continue_=True)

# In-memory outputs only (no files written)
tigerbx.run('mC', 'T1w.nii.gz', save_outputs=False)

# Patch-based inference for high-resolution inputs
tigerbx.run('bmp', 'T1w.nii.gz', 'output_dir')

# Verbose progress messages
tigerbx.run('bm', 'T1w.nii.gz', 'output_dir', verbose=1)

# Delete downloaded ONNX model files
tigerbx.run('clean_onnx')
```

---

## Output Files

For an input named `sub-001_T1w.nii.gz`, outputs are:

| Flag | Output file |
|------|-------------|
| `b`  | `sub-001_T1w_tbet.nii.gz` |
| `m`  | `sub-001_T1w_tbetmask.nii.gz` |
| `a`  | `sub-001_T1w_aseg.nii.gz` |
| `c`  | `sub-001_T1w_ct.nii.gz` |
| `C`  | `sub-001_T1w_cgw_pve0.nii.gz`, `_pve1.nii.gz`, `_pve2.nii.gz` (CSF / GM / WM) |
| `d`  | `sub-001_T1w_dgm.nii.gz` |

---

For label definitions used in ASEG and DeepGM outputs, see [Label definitions](seglabel.md).
