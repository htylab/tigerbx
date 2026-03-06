# TigerBx Release Notes

## Latest releases

- [`v0.2.3`](release/releasenote023.md)
  - Registration and VBM are split more cleanly
  - DeepVBM is moved into `tigerbx.pipelines.vbm`
  - New dispatcher entry point: `tigerbx.pipeline('vbm', ...)`
  - Core IO / ONNX / resample / metrics helpers are consolidated under `tigerbx.core`
- [`v0.2.2`](release/releasenote022.md)
  - ONNX session cache for faster batch inference
  - `--chunk-size` and `--continue` for large batch workflows
  - Improved progress display and low-QC reporting
