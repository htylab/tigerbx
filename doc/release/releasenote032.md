# TigerBx Release Notes — v0.3.2

## Changes

- **New `tigerbx.inspect`** (pipeline): for each input, its contrast (T1w, T2w, FLAIR, b0, DWI, other),
  skull-stripped, EPI, header orientation, truncated field of view and a motion artifact score, plus rule-based
  header / data checks (number of volumes, matrix, voxel size, thick slices, obliqueness, slab field of view,
  non-finite voxels, label map instead of an image, empty image, missing spatial header). Writes
  `inspect_summary.csv`; `detail=True` adds every probability and check value. Model `inspect_v003` is downloaded
  automatically on first use. See [`doc/inspect.md`](../inspect.md).
- **Executable**: `--inspect` (`--detail`), the same summary as `tigerbx.inspect`, and `--template`, a port of
  `tigerbx.template` (population template from a group of images).
- **Model checks**: every downloaded model is verified by its sha256; `tigerbx.check()` / `tigerbx --check` report
  the installation (version, ONNX Runtime, models). The executable gives an error message instead of a crash when
  ONNX Runtime cannot be loaded; a batch with failed inputs ends with a warning.
- **Resampling**: matrix entries within 1e-9 of an integer are set to it, so an image whose grid starts exactly on an
  input plane keeps that plane (Python and the executable). Oblique inputs can change very slightly at the brain
  edge.
