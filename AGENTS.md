# Repository Guidelines

## Overview

TigerBx is a neuroimaging toolkit (brain extraction / segmentation / registration) with:
- Python API: `tigerbx.*`
- Stand-alone executable: `tigerbx` (`rust/`)

Current baseline:
- Version: `0.5.1`
- Python: `>=3.10`
- Stack: ONNX Runtime, nibabel, numpy, scipy

## Project Structure (key folders)

- `tigerbx/`: core library (`bx`, `hlc`, `reg`, `vbm`, `embed`, `gdm`, `eval`, `validate`)
- `tigerbx/pipelines/vbm.py`: VBM pipeline (separate from `reg`)
- `tigerbx/pipelines/duope.py` + `tigerbx/core/duope.py`: EPI distortion from a reverse phase-encoding pair
- `tigerbx/pipelines/epinorm.py`: a DWI / BOLD series (with its AP / PA pair) onto the subject's T1 and the template in one resampling
  (executable: `rust/src/duope.rs`, `duope_pipe.rs`; `tests/duope_parity.py`)
- `tigerbx/core/`: shared I/O / ONNX / resample / spatial helpers; `core/models.py` the default models and sha256
- `rust/`: the stand-alone `tigerbx` executable (tBX, TSEG, template normalisation; see `rust/README.md`)
- `tests/run_and_compare.py`: primary integration/regression runner
- `doc/`: user-facing docs
- `temp/`: temporary local work files (gitignored)

## Core Coding Conventions (must follow)

### Paths

- Use `os.path` (`join`, `basename`, `dirname`, `relpath`) and `os.makedirs(...)`.
- Do **not** use `pathlib`.

### Imports

- Order: stdlib -> third-party -> internal.
- Heavy imports (`onnxruntime`) should be lazy (inside functions).

### Typing / Docstrings

- **No type annotations by default** (`typing` / parameter annotations / return annotations).
- Do not add type hints to new or existing functions unless explicitly requested.
- Do not mass-add docstrings to existing private helpers.

### API Pattern

- Public API functions build args/namespace and call `run_args(args)`.
- There is no Python CLI (`tiger` was removed); the stand-alone executable is `tigerbx` (`rust/`).

### Modules vs Pipelines

- Modules (`tigerbx.infer`, `tigerbx.reg`, ...) and the executable's single output options (`-t`, `-d`, `-A`, ...) give
  each model's own output, so its raw performance can be measured. No post-processing there.
- `tigerbx.infer` = the current models (tBX `_tbx`/`_tbxmask`, TSEG `_tseg`); `tigerbx.run` = the earlier models,
  frozen, with their own file names (tBET `_tbet`/`_tbetmask`, ASEG `_aseg`). Never give an earlier and a current
  model the same output name; new models go into `infer` and the pipelines, never into `run`.
- Post-processing that improves results belongs to the pipelines (`tigerbx.bx` / `seg` / `affine` / `norm` /
  `vbm`, each a function with its own options; the executable's no option / `--seg` / `--affine` / `--norm` / `--all`), and the docs recommend the
  pipelines to users (README: pipelines first, modules below as building blocks).
- TSEG (`_tseg`) is the TigerBx segmentation, not a FreeSurfer aseg; its label numbers follow FreeSurfer.

### `reg` / `vbm` Contracts

- `reg` uses uppercase plan strings (positional): `R/A/D` (default `AD`)
  - `R` rigid, `A` affine (numreg), `D` tReg; `V` (VMnet) and `affine_type='C2FViT'` were removed in 0.5.0
- `VBM` is a separate pipeline (`tigerbx.vbm`), not a `reg` mode.

### Output Path Contract (important)

- `get_template()` returns `(ftemplate, output_dir)` where `ftemplate` is a **full path template** containing `@@@@`.
- `save_nib(data_nib, ftemplate, postfix)` only replaces `@@@@` with `postfix` and writes the file.
- Callers must not join a directory onto `ftemplate` again.

### Result Return Convention

- Single input file -> return `result_dict` (nibabel objects)
- Multiple input files -> return list of `result_filedict` (saved file paths)
- Maintain `result_dict` and `result_filedict` in parallel during processing

## Build / Test / Dev Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,cpu]"   # or [dev,cu12]

# Syntax/compile check
python -m compileall -q tigerbx tests

# Integration/regression runner
python tests/run_and_compare.py all --run-only
```

## Validation Checklist

Run before a release and after changes to `bx`, `core/resample`, `core/numreg` or `rust/`:

- **Rust vs Python parity**: `python tests/rust_python_parity.py <T1w> <tigerbx exe> <out_dir>` on at least a 1 mm
  RAS T1w, an anisotropic one (e.g. a seg18 scan) and an oblique one. `_tbxmask`, `_tseg`,
  `_dgm` (modules) and the seg pipeline `_tseg` must be voxel-identical (2026-10-25: 0 differing voxels on all).
  Rust and Python share the resampling (scipy-compatible cubic spline, nearest back onto the input grid), crops and
  model inputs; a mismatch means the two diverged.
- **Accuracy**: `tigerbx.val()` seg18 (`tseg_18`, 30 structures), seg123 (12 deep-GM structures), two skull-stripping sets (tBX), reg60 (`reg`, plan `AD`).

## Docs / Change Policy

- If you change API arguments, plans, or output naming, update `README.md` and relevant files in `doc/`.
- Keep diffs focused; avoid repo-wide formatting/style churn.
- Use research-use wording only (no clinical/diagnostic claims).
- Commit messages carry no `Claude-Session:` line (no links to AI sessions).

## Safety / Data

- Do not commit large binaries, model files, or PHI.
- **Never describe any dataset, anywhere**: no dataset names, subject ids, cohort descriptions (population, disease,
  scanner, site, number of subjects or scans) or training data in commit messages, docs (validation docs included),
  release notes, GitHub release text, code comments / docstrings, or ONNX `metadata_props`. Only the validation set
  labels `seg18`, `seg123` and `reg60` may appear. Check every message, diff and exported model before pushing.
- Model files are downloaded via `lib_tool.get_model()` and cached locally; a release model name must match its
  sha256 in `tigerbx/core/models.py`, the one place for the default model names (see doc/developer_guide.md §7).
  Model build tools (network definitions, ONNX export and optimisation) live outside this repository.
- Put temporary artifacts under `temp/`.

## Don’ts (quick list)

- No `pathlib`
- No new dependencies unless necessary
- No heavy imports at module top level
- Do not create ONNX sessions directly; use `tigerbx.core.onnx.create_session()`
- Do not change the single-vs-multi result return convention
