# TigerBx Release Notes — v0.3.3

## Changes

- **New `tigerbx.duope`** (pipeline): EPI susceptibility distortion correction from a pair of scans with opposite
  phase encoding (AP / PA, LR / RL, ...); also unwarps a DWI or BOLD series with the estimated field. Executable:
  `--duope`. See [`doc/duope.md`](../duope.md).
- **`tigerbx.gdm`** follows the pipeline conventions (`gpu=`, `gz`, `model`; a failing input in a batch is skipped
  with a warning). `_gdm`, the displacement map, is saved as float32 (it was cast to the input's integer type and lost
  its sub-voxel values); `.bval` files with decimal values are read; a warning when the header's phase-encoding axis
  is unexpected. Default model `vdm_unet3d_v002_opt` (faster; the CPU gives the GPU's result), downloaded on first use.
- **Executable zips** carry `THIRD_PARTY_LICENSES.txt` (license and notice files of the Rust crates and of ONNX
  Runtime).
