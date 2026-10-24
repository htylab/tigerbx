# TigerBx Release Notes — v0.3.1

## Highlights

- **Stand-alone `tigerbx` executable** (Rust, no Python) replaces the PyInstaller `tiger` binary: Linux x64
  (glibc 2.28+), macOS arm64 and Windows x64 (a single `tigerbx.exe`). Same models as the Python package.
  - `tigerbx T1w.nii.gz` → `_tbet`; `--seg` → `_tbet`, `_aseg`; `--norm` → `_tbet`, `_norm`; `--all` → all three
  - single outputs: `-b` `_tbet`, `-m` `_tbetmask`, `-a` `_aseg`, `-d` `_dgm`, `-A` `_affine`, `-n` `_norm`
  - its `_aseg` takes the 12 deep-GM structures from DGM12, so it differs from `tigerbx.run('a')` there
  - its `_norm` is tissue-normalised (CSF 1/3, GM 2/3, WM 1; T1 only); `--norm-minmax` gives the
    `tigerbx.reg('AD')` scaling
  - see [rust/README.md](../../rust/README.md)
- **`tiger` command removed** from the pip package: use the Python API (`tigerbx.run`, `reg`, `hlc`, `vbm`,
  `embed`, `gdm`) or the `tigerbx` executable.
- **DGM12**: default model `mprage_dgm12_v002_opt` (optimised graph, same results).
- **`tigerbx.val(task='aseg_ibsr18')`** reproduces the IBSR18 ASEG validation (30 structures).

Python API results are unchanged from 0.3.0.
