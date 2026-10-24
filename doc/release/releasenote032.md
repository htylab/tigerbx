# TigerBx Release Notes — v0.3.2

## Highlights

- **GPU for the `tigerbx` executable** (`-g`): CUDA on Linux (the zip carries the GPU build of ONNX Runtime; CUDA 12
  and cuDNN 9 libraries come from the system), DirectML on Windows (any DirectX 12 GPU). Falls back to the CPU with a
  warning. See [rust/README.md](../../rust/README.md).
- **Much lower GPU memory** in the Python package and the executable: the `labels` branch of the optimised models is
  not run on the GPU (its CUDA ArgMax needed ~10x the logits as workspace), cuDNN uses heuristic algorithm search
  without its maximum workspace, and the GPU memory arena is released after every run.
- **CPU and GPU give the same results**: the optimised tBET / ASEG / DGM12 / HLC models compute GroupNorm statistics in
  stages (the CPU's flat reduction lost precision). Same file names; results change slightly on the CPU.
- **`tigerbx[cu13]`**: install extra for CUDA 13 (onnxruntime-gpu >= 1.27).
