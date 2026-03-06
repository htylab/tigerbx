# TigerBx Release Notes — v0.2.3

## Highlights

- **VBM moved into the pipeline layer** — implemented in
  `tigerbx.pipelines.vbm`
- **Registration / VBM split** — `tiger reg PLAN ...` and `tiger vbm ...`
- **New `tigerbx.pipeline('vbm', ...)` dispatcher** with `tigerbx.vbm(...)`
  alias
- **New `tigerbx.core.*` foundation modules** for IO, ONNX, resampling,
  deformation, metrics, and spatial ops
- **Dependency reduction** — removed `nilearn`, `SimpleITK`, and `pandas`
  runtime usage
- **`bx.run(..., save_outputs=False)`** for in-memory workflows
- **manylinux build workflow** replaces the old lite compile workflow

---

## 1. Registration API Split and VBM Pipeline Move

### Problem

`reg_vbm.py` previously mixed image registration and VBM into one module, and
the CLI/API exposed several mode-specific flags and legacy lowercase codes.

### Solution: explicit registration plans + dedicated pipeline module

Registration now lives in `tigerbx.reg` and uses an **ordered positional plan
string** with uppercase steps:

| Step | Meaning |
|---|---|
| `R` | rigid |
| `A` | affine |
| `V` | VMnet nonlinear registration |
| `N` | SyN |
| `C` | SyNCC |
| `F` | FuseMorph |

`V` and `F` now explicitly require a prior `A` step.

```python
tigerbx.reg('AV', input_dir, output_dir)
tigerbx.pipeline('vbm', input_dir, output_dir)
tigerbx.vbm(input_dir, output_dir)   # alias kept for convenience
```

```bash
tiger reg AV T1w.nii.gz -o output_dir
tiger vbm /data/T1w_dir -o /data/output
```

### Migration examples

| Old usage | New usage |
|---|---|
| `tigerbx.reg('Ar', ...)` | `tigerbx.reg('AV', ...)` |
| `tigerbx.reg('F', ...)` | `tigerbx.reg('AF', ...)` |
| `tigerbx.reg('v', ...)` | `tigerbx.pipeline('vbm', ...)` |
| `tiger reg -A -r ...` | `tiger reg AV ...` |
| `tiger reg -v ...` | `tiger vbm ...` |

The VBM code is moved out of `reg_vbm.py` into `tigerbx.pipelines.vbm`, and a
new `tigerbx.pipeline()` dispatcher is added for high-level pipelines.

`tigerbx.vbm(...)` is kept as a convenience alias, while `tiger vbm` becomes
the dedicated CLI entry point for the pipeline.

---

## 2. New `tigerbx.core` Foundation Layer

A new shared `core` package now consolidates helpers that used to be scattered
across `lib_*` modules and ad-hoc utilities:

- `tigerbx.core.io` — input discovery, duplicate-basename handling, output
  templates, NIfTI writing
- `tigerbx.core.onnx` — session creation and prediction helpers
- `tigerbx.core.resample` — image reorder / resample helpers
- `tigerbx.core.deform` — deformation utilities
- `tigerbx.core.spatial` — crop / restore / pad helpers
- `tigerbx.core.metrics` — evaluation metrics used by `eval()` and registration

This refactor also simplifies public module naming:

- `tigerbx.hlc171` → `tigerbx.hlc`
- `tigerbx.nerve_nerme` → `tigerbx.nerve`
- `tigerbx.eval` now acts as a stable facade over `tigerbx.core.metrics`

A full developer-facing docs set is added under `doc/core/`.

---

## 3. Dependency Reduction and Runtime Portability

### `nilearn` removed from resampling

The old Nilearn-based image operations are replaced with internal
`numpy`/`scipy`/`nibabel` implementations in `tigerbx.core.resample`.

This removes the `nilearn` runtime dependency while keeping the same
reordering/resampling responsibilities used by `bx`, `reg`, `nerve`, `hlc`,
and validation code.

### `SimpleITK` removed from GDM

`gdmi.py` is rewritten to use shared core helpers instead of `lib_gdm` /
`SimpleITK`-based code:

- displacement application now goes through `tigerbx.core.deform`
- output writing now goes through `tigerbx.core.io`
- distortion-map zoom handling is corrected during output generation

### `pandas` removed from validation summaries

`validate.val_bet_synstrip()` no longer needs `pandas` just to print category
means. The summary is now computed with stdlib + `numpy`, and the function
returns plain structured data plus the aggregate metric.

### Packaging changes

Runtime requirements are lighter, and package metadata is updated to:

- **Python `>=3.10`**
- `nilearn` removed from `install_requires`
- `SimpleITK` removed from `install_requires`

---

## 4. `bx` Pipeline Improvements

### New `save_outputs` API

`bx.run()` now supports `save_outputs=False`, allowing callers to keep
results in memory without writing files to disk.

```python
result = tigerbx.run('mC', 'T1w.nii.gz', save_outputs=False)
```

This is used internally by the new VBM pipeline to fetch CGW outputs directly
before registration/modulation/smoothing.

### Shared BET-derived crop preparation

Non-BET models now reuse a single BET-derived `tbet111_crop` preparation path.
The logic distinguishes between:

- strict native 1 mm handling for `cgw` / `ct`
- slightly looser native-space reuse for other downstream masks

This removes duplicated crop/resample branches and makes follow-up model inputs
more consistent.

### Better `--continue` handling

Batch resume logic now correctly handles:

- duplicate basenames saved into a shared output directory
- multi-file folder inputs resolved via common-folder headers
- CGW requests only counting as complete when **all three**
  `cgw_pve0`, `cgw_pve1`, and `cgw_pve2` outputs exist

There are also smaller usability fixes, including cleaner progress-bar behavior
for single-file runs.

---

## 5. Metrics, Validation, and Internal Cleanup

### Shared metric implementations

Segmentation / reconstruction / classification metrics are moved into
`tigerbx.core.metrics` and reused across the codebase.

One user-visible fix: `dice()` now infers labels from the union of `y_true`
and `y_pred` when `labels=None`, which avoids missing labels that only appear
in predictions.

### Registration cleanup

`lib_reg.py` is simplified to reuse shared helpers instead of keeping local
duplicate implementations for Dice and FWHM conversion. This reduces drift
between evaluation and registration scoring.

### API polish

Additional cleanup in this release includes:

- updated README examples for current `bx` flags and GPU usage
- CLI loaders updated to point at renamed modules (`hlc.py`, `nerve.py`)
- tests updated to import spatial helpers from `tigerbx.core.spatial`

---

## 6. Docs and Build Workflow

- New documentation pages are added for `io`, `onnx`, `resample`, `spatial`,
  `deform`, and `metrics` under `doc/core/`
- `README.md`, `doc/reg.md`, and `doc/pipelines.md` are updated for the new
  registration plan syntax and the separate VBM entry points
- the old `compile_lite.yml` GitHub Actions workflow is replaced with
  `compile_manylinux.yml`
- package version is bumped from **`0.2.2`** to **`0.2.3`**

This release is mainly a **structural and API cleanup release**: registration
and VBM are easier to understand, shared internals are centralized, the runtime
dependency set is smaller, and the `bx` / validation workflows are more useful
for batch and in-memory processing.
