# rustbx

tBET brain extraction as a native executable (Rust + ONNX Runtime), with the same model and preprocessing as
`tigerbx.run('m')`: resample to 1 mm when a voxel size is outside 0.9–1.1 mm, reorder to a diagonal affine
(TigerBx `read_bet_input`, scipy-compatible cubic spline), run the model, keep the largest component and bring the
mask back onto the input grid. Masks match TigerBx except for voxels whose sampling point lies on the edge of the
field of view (floating-point rounding of the 4x4 affine inverse).

Work in progress: the executable is not part of a release yet.

## Build

```bash
mkdir -p models
curl -L -o models/mprage_bet_v005_opt.onnx \
  https://github.com/htylab/tigerbx/releases/download/modelhub/mprage_bet_v005_opt.onnx
cargo build --release
```

The model is embedded in the binary. ONNX Runtime is loaded at run time (`ort` with `load-dynamic`): from
`ORT_DYLIB_PATH`, else an `onnxruntime` library next to the executable, else the system library path. Use the
official ONNX Runtime release (e.g. `libonnxruntime.so.1.24.x`), which runs on older glibc versions.

## Usage

```bash
rustbx T1w.nii.gz                      # writes T1w_tbet.nii.gz and T1w_tbetmask.nii.gz next to the input
rustbx data/*.nii.gz -o out/           # glob, several files or a directory (non-recursive)
rustbx data/ -o out/ -t 8              # intra-op threads (default: 70% of the available CPUs)
rustbx T1w.nii.gz -m other_bet.onnx    # another tBET ONNX model (labels output optional)
```

Originally developed as a separate repository (htylab/rustbx); see the TigerBx license.
