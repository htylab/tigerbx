# TigerBx Release Notes

## Latest releases

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
