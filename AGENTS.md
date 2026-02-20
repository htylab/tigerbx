# Repository Guidelines

## Project Structure & Module Organization

- `tigerbx/`: core Python package (APIs like `tigerbx.run`, `tigerbx.hlc`, `tigerbx.reg`, `tigerbx.gdm`, `tigerbx.nerve`)
- `tigerbx_cli/`: CLI entrypoint (`tiger`) and subcommand implementations
- `doc/`: user-facing docs and images referenced by `README.md`
- `tests/`: integration/regression runner (`tests/run_and_compare.py`)
- `pyinstaller_hooks/` and `.github/workflows/`: standalone build plumbing (PyInstaller + CI)
- `skills/`: optional AI-assistant skill definitions (not required to use the library/CLI)

## Build, Test, and Development Commands

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install -e .

# Inference runtime (pick one)
pip install onnxruntime        # CPU
pip install onnxruntime-gpu    # GPU (optional)

# CLI smoke check
tiger --help
tiger bx T1w.nii.gz -bmad -o out/
```

Build a wheel/sdist:

```bash
python -m pip install build
python -m build
```

## Coding Style & Naming Conventions

- Python: 4-space indentation; follow existing file style and keep diffs focused (avoid repo-wide reformatting).
- Naming: `snake_case` for modules/functions, `CapWords` for classes, `UPPER_SNAKE_CASE` for constants.
- When changing CLI flags or output naming, update the relevant docs in `doc/` and examples in `README.md`.

## Testing Guidelines

- Primary test entrypoint is an integration/regression script (not `pytest`):
  - Run inference only: `python tests/run_and_compare.py all --run-only`
  - Compare outputs: place reference data in `test_output/old` (gitignored); new outputs default to `test_output/new`
- Prefer small, deterministic inputs for checks (see the template used by the test script in `tigerbx/template/`).

## Commit & Pull Request Guidelines

- Commit history mostly uses short, imperative summaries (e.g., `Update hlc.md`, `update doc`); prefer `Update <area>: <change>` for clarity.
- PRs should include: what/why, how you tested (commands), and sample outputs/screenshots when changing CLI UX.

## Data, Models, and Safety Notes

- Do not commit large binaries or sensitive data (PHI). Common outputs and model caches are ignored (e.g., `test_output/`, `models/`).
- TigerBx is for research use; avoid adding language that implies clinical use or diagnosis.

## Agent Notes (Temp Artifacts)

- Store temporary working files and analysis outputs under `temp/`.
- The latest deep code review report lives at `temp/code_review.md`.
