# Repository Guidelines

## Overview

TigerBx is a neuroimaging toolkit (brain extraction / segmentation / registration) with:
- Python API: `tigerbx.*`
- Stand-alone executable: `rustbx` (`rust/`)

Current baseline:
- Version: `0.3.0`
- Python: `>=3.10`
- Stack: ONNX Runtime, nibabel, numpy, scipy

## Project Structure (key folders)

- `tigerbx/`: core library (`bx`, `hlc`, `reg`, `vbm`, `embed`, `gdm`, `eval`, `validate`)
- `tigerbx/pipelines/vbm.py`: VBM pipeline (separate from `reg`)
- `tigerbx/core/`: shared I/O / ONNX / resample / spatial helpers
- `rust/`: `rustbx`, the stand-alone executable (tBET, ASEG, template normalisation; see `rust/README.md`)
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
- There is no Python CLI (`tiger` was removed); the stand-alone executable is `rustbx` (`rust/`).

### `reg` / `vbm` Contracts

- `reg` uses uppercase plan strings (positional): `R/A/V/D` (default `AD`)
  - `R` rigid, `A` affine, `D` VFA, `V` VMnet
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

## Docs / Change Policy

- If you change API arguments, plans, or output naming, update `README.md` and relevant files in `doc/`.
- Keep diffs focused; avoid repo-wide formatting/style churn.
- Use research-use wording only (no clinical/diagnostic claims).

## Safety / Data

- Do not commit large binaries, model files, or PHI.
- Model files are downloaded via `lib_tool.get_model()` and cached locally.
- Put temporary artifacts under `temp/`.

## Don’ts (quick list)

- No `pathlib`
- No new dependencies unless necessary
- No heavy imports at module top level
- Do not create ONNX sessions directly; use `tigerbx.core.onnx.create_session()`
- Do not change the single-vs-multi result return convention
