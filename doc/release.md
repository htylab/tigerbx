# TigerBx Release Notes

## Latest releases

- [`v0.3.2`](release/releasenote032.md)
  - New `tigerbx.inspect`: contrast, skull-stripped, EPI, orientation, truncation, motion and header / data checks
  - Executable `--inspect`, `--template` and `--check`; models verified by sha256
  - Resampling keeps an input's edge plane that floating-point error used to drop

- [`v0.3.1`](release/releasenote031.md)
  - Registration (`norm`, `reg('AD')`, executable `--norm`) uses the low-memory VFA model v005: same results, much
    lower peak memory

- [`v0.3.0`](release/releasenote030.md)
  - Pipelines as functions: `tigerbx.bx`, `seg`, `affine`, `norm`, `vbm`, `template`
  - Stand-alone `tigerbx` executable (Rust; Linux, macOS, Windows; GPU with `-g`) replaces the `tiger` binary
  - New segmentation model v0011; TSEG (`_tseg`, label 99 and TIV), `run('a')` still writes `_aseg`; tBET is now
    **tBX** (`_tbx`); see the label table for compatibility
  - New default registration `AD` (numreg affine + VFA); antspyx and optuna are no longer needed
  - New `tigerbx.embed`; VBM batch-safe; CSF / GM / WM maps from TSEG; faster optimised models
  - License CC BY-NC-SA 4.0
- [`v0.2.3`](release/releasenote023.md)
  - Registration and VBM are split more cleanly
  - DeepVBM is moved into `tigerbx.pipelines.vbm`
  - New dispatcher entry point: `tigerbx.pipeline('vbm', ...)`
  - Core IO / ONNX / resample / metrics helpers are consolidated under `tigerbx.core`
- [`v0.2.2`](release/releasenote022.md)
  - ONNX session cache for faster batch inference
  - `--chunk-size` and `--continue` for large batch workflows
  - Improved progress display and low-QC reporting
