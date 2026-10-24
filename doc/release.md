# TigerBx Release Notes

## Latest releases

- [`v0.3.2`](release/releasenote032.md)
  - GPU (`-g`) for the `tigerbx` executable: CUDA on Linux, DirectML on Windows
  - Much lower GPU memory; CPU and GPU now give the same results
  - `tigerbx[cu13]` install extra
- [`v0.3.1`](release/releasenote031.md)
  - Stand-alone `tigerbx` executable (Rust; Linux, macOS, Windows) replaces the PyInstaller `tiger` binary
  - `tiger` command removed: use the Python API or the `tigerbx` executable
  - DGM12 default `mprage_dgm12_v002_opt` (same results); `tigerbx.val(task='aseg_ibsr18')`
- [`v0.3.0`](release/releasenote030.md)
  - New default registration `AD`: numreg affine + VFA (reg60 Dice 0.852 vs 0.797)
  - numreg replaces ANTs for affine and rigid; antspyx and optuna are no longer needed
  - New bundled MNI-space template; FuseMorph, SyN and SyNCC removed
  - Outputs renamed `_Af` → `_affine`, `_reg` → `_VXM`; VBM writes `_norm`; license CC BY-NC-SA 4.0
  - ASEG label 99 `Unlabeled-Intracranial`: TIV = all non-zero `_aseg` voxels
  - New `tigerbx.embed`; VBM batch-safe with `vbm_summary.csv` and `qc_flags`
  - Faster tBET / ASEG / HLC (optimised models, no softmax); HLC uint8 label fix
  - Removed: tBET QC score, `bx` flags `q` / `S` / `W`, NERVE
- [`v0.2.4`](release/releasenote024.md)
  - New default ASEG model `mprage_aseg43_v0011_allseg.onnx` (IBSR18 0.8129 vs 0.8076, same labels)
  - No GPU out-of-memory across many input shapes; `TIGERBX_GPU_MEM_LIMIT_GB`
  - `tigerbx.val()` reports failing files instead of skipping the dataset
  - Previous outputs in an input folder are no longer treated as new subjects
- [`v0.2.3`](release/releasenote023.md)
  - Registration and VBM are split more cleanly
  - DeepVBM is moved into `tigerbx.pipelines.vbm`
  - New dispatcher entry point: `tigerbx.pipeline('vbm', ...)`
  - Core IO / ONNX / resample / metrics helpers are consolidated under `tigerbx.core`
- [`v0.2.2`](release/releasenote022.md)
  - ONNX session cache for faster batch inference
  - `--chunk-size` and `--continue` for large batch workflows
  - Improved progress display and low-QC reporting
