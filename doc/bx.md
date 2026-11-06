# TigerBx: `infer` and `run` Modules

The models' own outputs (`tigerbx.modules.bx`), from two entry points:

* `tigerbx.infer()`: the current models: tBX brain extraction (`_tbx`, `_tbxmask`, following the brain surface;
  `_tbxoutermask`, with the surrounding CSF), TSEG (`_tseg`, inside the outer mask), DGM12 and CT inside the tBX mask. The pipelines (`tigerbx.bx()`, `tigerbx.seg()`, ...,
  [pipelines](pipelines.md)) and the executable are built on it.
* `tigerbx.run()`: the earlier models, with the outputs of 0.2.x: tBET brain extraction (`_tbet`, `_tbetmask`, a
  mask that follows the brain surface), ASEG (`_aseg`), DGM12 and CT inside the tBET mask. Scripts written for those
  models keep giving the same files.

The two never share an output name, so their results cannot be mixed up. CSF / GM / WM maps:
`tigerbx.seg(save_tissue=True)` (computed from TSEG, `tigerbx.core.cgw`). Stand-alone executable (current models
only): `tigerbx data/ -o out/ --seg`, see [rust/README.md](../rust/README.md).

---

## Python API

```python
tigerbx.infer(argstring, input=None, output=None, model=None,
              verbose=0, chunk_size=1, continue_=False, silent=False,
              save_outputs=True)
tigerbx.run(argstring, ...)          # same parameters
```

| Parameter    | Type            | Default | Description |
|--------------|-----------------|---------|-------------|
| `argstring`  | `str`           | —       | One or more flag characters specifying which outputs to produce (see table below) |
| `input`      | `str` or `list` | `None`  | Input NIfTI file, directory, glob pattern, or list of paths |
| `output`     | `str`           | `None`  | Output directory; if `None`, results are saved next to each input file |
| `model`      | `str` or `dict` | `None`  | Model override dict, e.g. `{'tbx': 'my.onnx'}`; `None` uses the defaults |
| `verbose`    | `int`           | `0`     | Verbosity level: `0` = tqdm progress bars only, `1` = progress messages, `2` = debug |
| `chunk_size` | `int`           | `1`     | Files processed together per ONNX session. `1` (default) runs one file at a time with the lowest memory use; larger values reuse each session across the batch (faster) but keep every file's intermediates of the batch in memory |
| `continue_`  | `bool`          | `False` | Skip files whose output files already exist on disk |
| `silent`     | `bool`          | `False` | *(Deprecated)* Suppress all output. Use `verbose=0` instead |
| `save_outputs` | `bool`        | `True`  | Write output files to disk. Set `False` to get in-memory nibabel objects only |

If no model flag is given, brain extraction (`b`) runs.

### Return value

- **Single file:** `dict` containing output nibabel images, keyed by the output suffix (not the flag letter): with
  `infer`, `m` → `'tbxmask'`, `b` → `'tbx'`, `t` → `'tseg'`; with `run`, `m` → `'tbetmask'`, `b` → `'tbet'`,
  `a` → `'aseg'`.
- **Multiple files:** `list` of `dict`, one per input file, containing output file paths.

---

## Flags

| Flag | `tigerbx.infer` (current models) | `tigerbx.run` (earlier models) |
|------|----------------------------------|--------------------------------|
| `b`  | `_tbx`: brain image (tBX)        | `_tbet`: brain image (tBET) |
| `m`  | `_tbxmask`: brain mask that follows the brain surface | `_tbetmask`: brain mask that follows the brain surface |
| `x`  | `_tbxoutermask`: the brain with the surrounding CSF, up to the inner surface of the skull (the TSEG input); with `b` also `_tbxouter` | — |
| `t`  | `_tseg`: TSEG, 32 structures, inside the outer mask (FreeSurfer numbers; see [seglabel.md](seglabel.md)) | — |
| `a`  | — | `_aseg`: ASEG, 43 labels (FreeSurfer numbers; see [seglabel.md](seglabel.md)) |
| `d`  | `_dgm`: deep gray matter, 12 structures, inside the tBX mask | `_dgm`, inside the tBET mask |
| `c`  | `_ct`: cortical thickness map, inside the tBX mask | `_ct`, inside the tBET mask |
| `g`  | Use the GPU | Use the GPU |
| `z`  | Force `.nii.gz` output | Force `.nii.gz` output |
| `p`  | Patch-based inference | Patch-based inference |

A flag of the other entry point (`t` or `x` in `run`, `a` in `infer`) raises a `ValueError`. The flags `q` (tBX QC
score), `S` (SynthSeg-style segmentation), `W` (white-matter hypointensity) and `C` (CGW model) have been removed; an
argstring containing one of them raises a `ValueError`. For image quality use `tigerbx.core.qa.image_quality`.

---

## Examples

```python
import tigerbx

# Brain extraction only (default when no flag is given)
tigerbx.infer('b', 'T1w.nii.gz', 'output_dir')

# Brain mask + brain image + TSEG + deep gray matter
tigerbx.infer('bmtd', 'T1w.nii.gz', 'output_dir')

# The mask with the surrounding CSF (up to the inner surface of the skull) as well
tigerbx.infer('bmx', 'T1w.nii.gz', 'output_dir')

# The earlier models, files as in 0.2.x: _tbet, _tbetmask, _aseg, _dgm
tigerbx.run('bmad', 'T1w.nii.gz', 'output_dir')

# Process an entire directory; outputs saved next to each input file
tigerbx.infer('bm', '/data/T1w_dir')

# Faster batches: reuse each session across 100 files (needs more memory)
tigerbx.infer('bmtd', '/data/T1w_dir', '/data/output', chunk_size=100)

# Resume an interrupted run — skip already-processed files
tigerbx.infer('bmtd', '/data/T1w_dir', '/data/output', continue_=True)

# In-memory outputs only (no files written)
tigerbx.infer('mt', 'T1w.nii.gz', save_outputs=False)

# Patch-based inference for high-resolution inputs
tigerbx.infer('bmp', 'T1w.nii.gz', 'output_dir')

# Verbose progress messages
tigerbx.infer('bm', 'T1w.nii.gz', 'output_dir', verbose=1)

# Delete downloaded ONNX model files
tigerbx.run('clean_onnx')
```

---

## Output Files

For an input named `sub-001_T1w.nii.gz`, outputs are:

| Flag | `tigerbx.infer` | `tigerbx.run` |
|------|-----------------|---------------|
| `b`  | `sub-001_T1w_tbx.nii.gz` | `sub-001_T1w_tbet.nii.gz` |
| `m`  | `sub-001_T1w_tbxmask.nii.gz` | `sub-001_T1w_tbetmask.nii.gz` |
| `x`  | `sub-001_T1w_tbxoutermask.nii.gz` (with `b` also `_tbxouter`) | — |
| `t`  | `sub-001_T1w_tseg.nii.gz` | — |
| `a`  | — | `sub-001_T1w_aseg.nii.gz` |
| `c`  | `sub-001_T1w_ct.nii.gz` | `sub-001_T1w_ct.nii.gz` |
| `d`  | `sub-001_T1w_dgm.nii.gz` | `sub-001_T1w_dgm.nii.gz` |

---

For label definitions used in ASEG / TSEG and DeepGM outputs, see [Label definitions](seglabel.md).
