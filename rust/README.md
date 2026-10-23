# rustbx

TigerBx tBET, ASEG (`tigerbx.run('bma')`) and template registration (`tigerbx.reg('AD')`, the VBM `_norm`) as a
native executable (Rust + ONNX Runtime), with the same models and preprocessing. Segmentation: resample to 1 mm when a voxel size is outside 0.9–1.1 mm, reorder to a diagonal affine
(TigerBx `read_bet_input`, scipy-compatible cubic spline), tBET + largest component, ASEG on the tBET brain cropped
to the brain + 16 voxels, nearest-neighbour back onto the input grid, label 99 for unlabeled tBET-mask voxels.
Outputs match TigerBx 0.3.0 (identical `_aseg` / `_tbetmask` on IBSR18; `_tbet` in the input data type); on oblique
images a few voxels on the edge of the field of view can differ (floating-point rounding of the 4x4 affine inverse).

Registration: a Rust port of numreg (Mattes MI, symmetric cost, 6 / 3 / 1.5 mm pyramid, multi-start L-BFGS) gives
`_affine` (tBET brain on the template grid, min-max normalised), then VFA (`mprage_reg_v004_vfa`) gives `_norm`.
It cannot be bit-identical to numreg (another random generator for the 1.5 mm sample subset, another line search);
on 20 reg60 subjects the template-space Dice of carried labels is 0.7217 vs 0.7218 (affine) and 0.8451 vs 0.8452
(affine + VFA), image correlation >= 0.9993.

Work in progress: the executable is not part of a release yet.

## Build

```bash
mkdir -p models
for m in mprage_bet_v005_opt mprage_aseg43_v0011_opt mprage_reg_v004_vfa; do
  curl -L -o models/$m.onnx https://github.com/htylab/tigerbx/releases/download/modelhub/$m.onnx
done
cargo build --release
```

The models and the TigerBx template (`../tigerbx/template/mni_template.nii.gz`) are embedded in the binary (~80 MB). ONNX Runtime is loaded at run time (`ort` with `load-dynamic`): from
`ORT_DYLIB_PATH`, else an `onnxruntime` library next to the executable, else the system library path. Use the
official ONNX Runtime release (e.g. `libonnxruntime.so.1.24.x`), which runs on older glibc versions.

## Usage

```bash
rustbx T1w.nii.gz                      # no option = all: _tbet, _tbetmask, _aseg, _affine, _norm next to the input
rustbx data/*.nii.gz -o out/ -m        # brain mask only (glob, several files or a directory, non-recursive)
rustbx data/ -o out/ -b -a -t 8        # brain + ASEG, 8 intra-op threads (default: 70% of the available CPUs)
rustbx data/ -o out/ -A -n             # template registration only: _affine (numreg) and _norm (+ VFA)
rustbx T1w.nii.gz --aseg-model other_aseg43.onnx   # other models (--bet-model / --aseg-model; labels output optional)
```

Earlier outputs (`_tbet`, `_tbetmask`, `_aseg`, `_affine`, `_norm`) in an input folder are skipped; a failing input is reported and the
batch continues.

`RUSTBX_DUMP_REG=<dir>` also writes `<stem>_M.txt` (4x4 world matrix template -> input) and `<stem>_grid.f32`
(VFA voxel sampling grid, float32 [3, 160, 224, 192]) for validation.

Originally developed as a separate repository (htylab/rustbx); see the TigerBx license.
