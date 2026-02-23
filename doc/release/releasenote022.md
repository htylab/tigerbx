# TigerBx Release Notes — v0.2.2

## Highlights

- **ONNX Session Cache** — dramatic speedup for large batches
- **tqdm progress bars** with configurable verbosity
- **`--continue`** flag for fault-tolerant batch processing
- **`val()` session cache** — validation now reuses sessions across files
- **Low-QC end-of-run summary** — warnings are never missed

---

## 1. ONNX Session Cache (bx module)

### Problem

Previous versions created a new `ort.InferenceSession` for every input file.
For a batch of N files with M models, this caused **N × M** session loads —
the most expensive step in ONNX inference.

### Solution: chunked model-first processing

`run_args()` is restructured from a *file-first* loop into a *model-first
chunked* loop:

```
for each chunk of chunk_size files:
    Phase 1 — tBET:   one session, all files in chunk
    Phase 2 — aseg:   one session, all files in chunk
    Phase 3 — dgm:    one session, all files in chunk
    ...
```

Session loads drop from **N × M** to **ceil(N / chunk_size) × M**.

| Scenario | Before | After (chunk = 50) |
|---|---|---|
| 2000 files, 5 models | 10 000 loads | 200 loads |
| Peak RAM | O(1) | O(chunk_size) |
| Fault tolerance | none | chunk granularity |

### New `lib_tool` API

```python
from tigerbx import lib_tool

# create a reusable session
session = lib_tool.create_session(model_ff, GPU)

# pass it to predict() — skips session creation
lib_tool.predict(model_ff, data, GPU, session=session)
```

`produce_betmask()` and `produce_mask()` in `bx.py` also accept `session=`.

---

## 2. New CLI and API Options

### `--chunk-size N` / `chunk_size=N`

Number of files per session-cache batch. Default **50**.
Increase for faster throughput; decrease to reduce peak RAM.

```bash
tiger bx /data/T1w_dir -bmad -o /data/output --chunk-size 100
```

```python
tigerbx.run('bmad', '/data/T1w_dir', '/data/output', chunk_size=100)
```

### `--continue` / `continue_=True`

Skip files whose output files already exist on disk.
Enables safe resumption of interrupted runs.

```bash
tiger bx /data/T1w_dir -bmad -o /data/output --continue
```

```python
tigerbx.run('bmad', '/data/T1w_dir', '/data/output', continue_=True)
```

---

## 3. Progress Display (tqdm + verbose)

All phases now display **tqdm progress bars** by default.

| `--verbose` | CLI default | Behaviour |
|---|---|---|
| `0` | — | tqdm bars only (clean for piped output) |
| `1` | ✓ | tqdm bars + summary messages |
| `2` | — | debug mode: per-file text log, no tqdm |

The tBET bar shows the current filename (up to 18 characters) and QC score in
the postfix:

```
tBET: 45% 9/20 [01:18<01:35, 9.5s/file, sub-001_T1w.ni | QC=98]
```

### API `verbose` parameter

```python
# Default (verbose=0): tqdm bars only
tigerbx.run('bmad', input_dir, output_dir)

# verbose=1: add summary messages
tigerbx.run('bmad', input_dir, output_dir, verbose=1)
```

### `silent=` deprecated

`silent=True` is deprecated. Use `verbose=0` (already the default).

---

## 4. Low-QC Warning Improvements

When QC score < 50 (low-confidence brain extraction):

1. **Inline warning** — printed immediately via `tqdm.write()` so it does not
   disrupt the progress bar.
2. **End-of-run summary** — after all files are processed, a consolidated list
   of all low-QC files is printed at verbosity ≥ 1:

```
[WARNING] 3 file(s) with low QC score — check results carefully:
  QC=38: sub-042_T1w.nii.gz
  QC=21: sub-107_T1w.nii.gz
  QC=45: sub-198_T1w.nii.gz
```

### `QC_raw` now in multi-file results

Multi-file calls (`run()` with N > 1 inputs) now include `'QC_raw'` in each
result dict alongside `'QC'`:

```python
results = tigerbx.run('m', file_list, output_dir)
for r in results:
    print(r['QC'], r['QC_raw'])   # both available
```

---

## 5. `val()` Session Cache

Validation functions (`val_bet_synstrip`, `val_bet_NFBS`, `_val_seg_123`) now
run all files in a single `tigerbx.run()` call instead of one call per file.
The session cache in `run_args()` handles the reuse automatically.

New internal helper:

```python
# validate.py
_run_bx_batch(ffs, argstr, GPU, model, compute_metrics_fn, ...)
```

Session loads for a 500-file NFBS validation drop from 500 to 10
(with default chunk_size=50).

---

## 6. Internal Changes

- **Ensemble support removed** — `lib_bx.run()` now takes a single `model_ff`
  instead of a list. All default models are single `.onnx` files.
- `lib_tool.create_session()` extracted from `predict()` as a public helper.
- `produce_betmask()` and `produce_mask()` simplified (no list handling).
- `_run_seg`, `_run_cgw`, `_run_ct` module-level helpers with a
  `_MODEL_RUNNERS` dispatch dict for clean per-model iteration.
