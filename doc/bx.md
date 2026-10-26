# TigerBx: `run` Module

The models' own outputs via `tigerbx.run()` (`tigerbx.modules.bx`): tBX mask and brain, TSEG, DGM12, CT. CSF / GM / WM maps:
`tigerbx.seg(save_tissue=True)` (computed from TSEG, `tigerbx.core.cgw`; the CGW model and flag `C` were removed).
For brain extraction and segmentation use the pipelines `tigerbx.bx()` / `tigerbx.seg()` ([pipelines](pipelines.md)).
Stand-alone executable for tBX / TSEG / template normalisation: `tigerbx data/ -o out/ --seg`, see
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
  `'tbxmask'`, `b` → `'tbx'`, `t` → `'tseg'`.
  For example the brain mask of `run('m', ...)` is `result['tbxmask']`, not `result['tbx']`.
- **Multiple files:** `list` of `dict`, one per input file, containing output file paths.

---

## Flags

| Flag | Output suffix        | Description |
|------|----------------------|-------------|
| `b`  | `_tbx`              | Brain-extracted image |
| `m`  | `_tbxmask`          | Binary brain mask |
| `a`  | `_aseg`              | ASEG: the TSEG model output (43 labels, FreeSurfer numbers), as in 0.2.x; kept for compatibility |
| `t`  | `_tseg`              | TSEG, the TigerBx whole-brain segmentation: ASEG + 99 Unlabeled-Intracranial (TIV = all non-zero voxels) |
| `c`  | `_ct`                | Cortical thickness map |
| `d`  | `_dgm`               | Deep gray matter mask (12 structures) |
| `g`  | —                    | Use GPU for inference |
| `z`  | —                    | Force `.nii.gz` output format |
| `p`  | —                    | Enable patch-based inference |

The flags `q` (tBX QC score), `S` (SynthSeg-style segmentation) and `W` (white-matter hypointensity) have been removed; an
argstring containing one of them raises a `ValueError`. For image quality use `tigerbx.core.qa.image_quality`.
`a` and `t` share one TSEG inference (`'at'` writes both). New analyses should use `t` or `tigerbx.seg()`.

---

## Examples

```python
import tigerbx

# Brain extraction only (default when no flag is given)
tigerbx.run('b', 'T1w.nii.gz', 'output_dir')

# Brain mask + brain image + TSEG + deep gray matter (recommended)
tigerbx.run('bmtd', 'T1w.nii.gz', 'output_dir')

# Full pipeline — all output types
tigerbx.run('bmtcdC', 'T1w.nii.gz', 'output_dir')

# Process an entire directory; outputs saved next to each input file
tigerbx.run('bm', '/data/T1w_dir')

# Batch processing with session cache — 100 files per chunk
tigerbx.run('bmtd', '/data/T1w_dir', '/data/output', chunk_size=100)

# Resume an interrupted run — skip already-processed files
tigerbx.run('bmtd', '/data/T1w_dir', '/data/output', continue_=True)

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
| `b`  | `sub-001_T1w_tbx.nii.gz` |
| `m`  | `sub-001_T1w_tbxmask.nii.gz` |
| `a`  | `sub-001_T1w_aseg.nii.gz` |
| `t`  | `sub-001_T1w_tseg.nii.gz` |
| `c`  | `sub-001_T1w_ct.nii.gz` |
| `d`  | `sub-001_T1w_dgm.nii.gz` |

---

For label definitions used in ASEG / TSEG and DeepGM outputs, see [Label definitions](seglabel.md).
