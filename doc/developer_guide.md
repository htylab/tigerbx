# TigerBx Developer Guide

This guide covers setting up a local development environment, making changes, and submitting contributions.

---

## 1. Prerequisites

- [uv](https://docs.astral.sh/uv/) (replaces pip + venv; install once per machine)
- Git
- (Optional) CUDA 12 toolkit for GPU testing

Install uv if you haven't already:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## 2. Clone and Set Up

```bash
# Clone the repository
git clone https://github.com/htylab/tigerbx.git
cd tigerbx

# Create a virtual environment and install in editable mode (CPU)
uv venv
uv pip install -e ".[dev,cpu]"
```

After this, any change to a `.py` file under `tigerbx/` or `tigerbx_cli/` takes effect immediately — no reinstall needed.

To use GPU (CUDA 12) instead:

```bash
uv pip install -e ".[dev,cu12]"
```

> **Why uv?** uv resolves and installs dependencies ~10–100× faster than pip and manages the `.venv` automatically. You do not need to activate the venv manually — prefix commands with `uv run` or activate once with `source .venv/bin/activate` (macOS/Linux) / `.venv\Scripts\activate` (Windows).

---

## 3. Verify the Installation

```bash
# CLI smoke test
uv run tiger bx --help

# Python API smoke test
uv run python -c "import tigerbx; print(tigerbx.__version__)"

# Run the test suite
uv run pytest tests/
```

---

## 4. Project Structure

```
tigerbx/
├── tigerbx/             # Core library
│   ├── bx.py            # Brain extraction pipeline
│   ├── hlc.py           # HLC parcellation
│   ├── reg.py           # Registration (plan-driven)
│   ├── pipeline/        # High-level pipelines (e.g., VBM)
│   ├── gdmi.py          # EPI distortion correction
│   ├── nerve.py         # NERVE hippocampus/amygdala pipeline
│   ├── eval.py          # Image quality and segmentation metrics
│   ├── lib_tool.py      # ONNX inference, model download
│   ├── lib_bx.py        # BET preprocessing helpers
│   └── lib_reg.py       # Registration helpers
├── tigerbx_cli/         # CLI entry points
│   ├── tiger.py         # Main dispatcher
│   ├── bx_cli.py
│   ├── reg_cli.py
│   └── ...
├── tests/               # pytest test suite
├── doc/                 # Documentation
├── pyinstaller_hooks/   # PyInstaller hooks
└── setup.py             # Package metadata and dependencies
```

---

## 5. Daily Development Workflow

### 5.1 Create a feature branch

Always work on a branch, not directly on `main`:

```bash
git checkout main
git pull origin main                   # sync with upstream
git checkout -b feature/my-new-feature
```

Branch naming conventions:
- `feature/` — new features
- `fix/` — bug fixes
- `docs/` — documentation only
- `refactor/` — code cleanup without behaviour change

### 5.2 Make changes and test

Edit files, then verify:

```bash
# Run affected tests
uv run pytest tests/test_bx.py -v

# Run the full suite
uv run pytest tests/ -v

# Quick CLI check
uv run tiger bx T1w.nii.gz -bm -o /tmp/test_out
```

### 5.3 Commit

```bash
# Stage specific files (avoid git add -A to prevent accidental inclusions)
git add tigerbx/bx.py tigerbx/lib_tool.py

# Commit with a clear message
git commit -m "fix: handle edge case when input voxel size is anisotropic"
```

Commit message prefixes:
| Prefix | Use |
|--------|-----|
| `feat:` | new feature |
| `fix:` | bug fix |
| `docs:` | documentation change |
| `refactor:` | code cleanup, no behaviour change |
| `test:` | add or update tests |
| `chore:` | build system, CI, dependency updates |

### 5.4 Push and open a Pull Request

```bash
git push -u origin feature/my-new-feature
```

Then open a PR on GitHub:

```bash
# Using GitHub CLI
gh pr create \
  --title "fix: handle anisotropic voxel edge case in bx pipeline" \
  --body "## Summary
- Fixed crash when input voxel spacing is non-isotropic (e.g. 0.5×0.5×1.0 mm)
- Added regression test

## Test plan
- [ ] pytest tests/test_bx.py passes
- [ ] Manual test with anisotropic sample image"
```

Or open it from the GitHub web UI.

---

## 6. Bumping the Version

Version is defined in two places:

- [`tigerbx/__init__.py`](../tigerbx/__init__.py) — `__version__ = 'X.Y.Z'`
- [`setup.py`](../setup.py) — `version='X.Y.Z'`

Update both before tagging a release. TigerBx follows [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Patch: bug fixes, no API change
- Minor: new features, backwards-compatible
- Major: breaking API changes

---

## 7. Adding a New Model

1. Train and export the model to ONNX format.
2. Upload the `.onnx` file to the model hub release on GitHub (`modelhub` tag).
3. Register the model name in `bx.py` (or the relevant module) under `omodel`.
4. The model is downloaded on first use via `lib_tool.get_model()`.

---

## 8. Running Tests Against Two Versions

To regression-test a change against a known-good version:

```bash
# Create a separate venv for the baseline release
uv venv .venv_old
uv pip install --python .venv_old \
    "tigerbx[cpu] @ https://github.com/htylab/tigerbx/archive/refs/tags/v0.2.1.tar.gz"

# Run comparison script using each venv's Python
uv run --python .venv_old python tests/run_and_compare.py --input T1w.nii.gz --tag old
uv run pytest tests/ -v   # current version
```

---

## 9. Building the Standalone Executable (CI)

Executables are built automatically by GitHub Actions on `workflow_dispatch`:

- [`compile10.yml`](../.github/workflows/compile10.yml) — full build (all features)
- [`compile_lite.yml`](../.github/workflows/compile_lite.yml) — lite build (excludes `antspyx`, `optuna`)

To trigger a build, go to **Actions → Compile10 → Run workflow** on GitHub.

---

## 10. Coding Conventions

- **No top-level heavy imports** — `ants`, `optuna`, `onnxruntime` must be imported inside the functions that use them (lazy import), to keep startup time fast.
- **NIfTI I/O** — use `nibabel` for reading/writing; avoid hard-coding voxel assumptions.
- **Model inference** — always go through `lib_tool.predict()`, never create `ort.InferenceSession` directly in feature modules.
- **No pandas in core** — pandas is a dev-only dependency; use `csv` module for any file output in production code.
