# rustbx

TigerBx tBET and ASEG (`tigerbx.run('bma')`) as a native executable (Rust + ONNX Runtime), with the same models
and preprocessing: resample to 1 mm when a voxel size is outside 0.9–1.1 mm, reorder to a diagonal affine
(TigerBx `read_bet_input`, scipy-compatible cubic spline), tBET + largest component, ASEG on the tBET brain cropped
to the brain + 16 voxels, nearest-neighbour back onto the input grid, label 99 for unlabeled tBET-mask voxels.
Outputs match TigerBx 0.3.0 (identical `_aseg` / `_tbetmask` on IBSR18; `_tbet` in the input data type); on oblique
images a few voxels on the edge of the field of view can differ (floating-point rounding of the 4x4 affine inverse).

Work in progress: the executable is not part of a release yet.

## Build

```bash
mkdir -p models
for m in mprage_bet_v005_opt mprage_aseg43_v0011_opt; do
  curl -L -o models/$m.onnx https://github.com/htylab/tigerbx/releases/download/modelhub/$m.onnx
done
cargo build --release
```

Both models are embedded in the binary (~65 MB). ONNX Runtime is loaded at run time (`ort` with `load-dynamic`): from
`ORT_DYLIB_PATH`, else an `onnxruntime` library next to the executable, else the system library path. Use the
official ONNX Runtime release (e.g. `libonnxruntime.so.1.24.x`), which runs on older glibc versions.

## Usage

```bash
rustbx T1w.nii.gz                      # no option = -b -m -a: T1w_tbet, T1w_tbetmask, T1w_aseg next to the input
rustbx data/*.nii.gz -o out/ -m        # brain mask only (glob, several files or a directory, non-recursive)
rustbx data/ -o out/ -b -a -t 8        # brain + ASEG, 8 intra-op threads (default: 70% of the available CPUs)
rustbx T1w.nii.gz --aseg-model other_aseg43.onnx   # other models (--bet-model / --aseg-model; labels output optional)
```

Earlier outputs (`_tbet`, `_tbetmask`, `_aseg`) in an input folder are skipped; a failing input is reported and the
batch continues.

Originally developed as a separate repository (htylab/rustbx); see the TigerBx license.
