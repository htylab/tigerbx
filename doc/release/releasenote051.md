# TigerBx Release Notes — v0.5.1

## Changes

- **tReg v4** (`mprage_treg_v004.onnx`), the default of registration plan step `D`, `norm` and `template`: the v3
  network fine-tuned to be diffeomorphic (each level's displacement is the exponential of a velocity field, by
  scaling and squaring), so the warp no longer folds (det J > 0 everywhere). Registration accuracy is that of v3;
  `_norm` and `_warp.npz` differ slightly from 0.5.0.
- **GPU**: tReg runs as `mprage_treg_v004_gpu.onnx`, the same network with its sampling written as Gather ops
  (the ONNX Runtime CUDA provider has no 5-D GridSample, so the earlier model ran that step on the CPU): faster on
  the GPU, and its warp agrees with the CPU model's.
- **Fold repair**: a warp from an earlier tReg (`model=`) has its few folded voxels repaired (where det J <= 0.05,
  the displacement is smoothed locally until none is left; `tigerbx.core.templatereg.unfold`). With v4 there is
  nothing to repair.
- **Executable**: embeds tReg v4 and does the same fold repair.
