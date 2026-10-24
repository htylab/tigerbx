# tigerbx executable

TigerBx tBET, ASEG (`tigerbx.run('bma')`) and template registration (`tigerbx.reg('AD')`, the VBM `_norm`) as a
native executable (Rust + ONNX Runtime), with the same models and preprocessing. Segmentation: resample to 1 mm when a voxel size is outside 0.9–1.1 mm, reorder to a diagonal affine
(TigerBx `read_bet_input`, scipy-compatible cubic spline), tBET + largest component, ASEG on the tBET brain cropped
to the brain + 16 voxels, nearest-neighbour back onto the input grid, label 99 for unlabeled tBET-mask voxels.
DGM12 (`tigerbx.run('d')`) runs on the same crop, and its 12 deep-GM structures replace those of `_aseg` (DGM12
is more accurate there: IBSR18 deep GM 0.846 vs 0.837, seg123 0.844 vs 0.835); ASEG voxels of a deep-GM label
that DGM12 does not keep take the nearest non-deep-GM label. IBSR18 30-structure Dice: 0.8180 (TigerBx `_aseg`
0.8131). `_tbetmask` and `_dgm` match TigerBx 0.3.0 (identical on IBSR18; `_tbet` in the input data type); on oblique
images a few voxels on the edge of the field of view can differ (floating-point rounding of the 4x4 affine inverse).

Registration: a Rust port of numreg (Mattes MI, symmetric cost, 6 / 3 / 1.5 mm pyramid, multi-start L-BFGS) gives
`_affine` (tBET brain on the template grid, min-max normalised), then VFA (`mprage_reg_v004_vfa`) gives `_norm`.
It cannot be bit-identical to numreg (another random generator for the 1.5 mm sample subset, another line search);
on 20 reg60 subjects the template-space Dice of carried labels is 0.7217 vs 0.7218 (affine) and 0.8451 vs 0.8452
(affine + VFA), image correlation >= 0.9993.

`_norm` intensities: gnorm by default, a port of `tigerbx.core.tissuenorm.tissue_norm` (prior mode: GMM with the
template tissue priors + bias field; background 0, CSF 1/3, GM 2/3, WM 1). The brain is the tBET mask carried
through the same affine + VFA warp (oblique inputs leave cubic-resampling ringing of ~1e-40 to 1e-3 outside the
brain, so `img > 0` is not the brain). On 20 reg60 subjects the peaks match the Python `tissue_norm` (same mask)
within 6.4e-4 relative and the voxel values within 5e-4. T1 only: when the peaks are not ordered CSF < GM < WM
(T2, PD, ...) `_norm` falls back to img / p99 of the brain with a warning, and the batch goes on.
`--norm-minmax` keeps the min-max `_norm` of `tigerbx.reg('AD')`.

Prebuilt executables: the `exe` GitHub workflow builds Linux (glibc 2.28+, x64) and macOS (arm64) zips with the
ONNX Runtime library next to the executable, and Windows (x64) as a single `tigerbx.exe` (ONNX Runtime 1.23.2 from
pyke linked statically; needs the VC++ 2015-2022 runtime and Windows 10 1903+ for DirectML.dll, both usually
present), as workflow artifacts on every change under `rust/` and attached to the draft release of each `v*` tag
(Compile10). ONNX Runtime 1.23.2 and 1.24.4 give identical outputs (6 IBSR18 subjects, all outputs, 0 voxels differ).

## Build

```bash
mkdir -p models
for m in mprage_bet_v005_opt mprage_aseg43_v0011_opt mprage_dgm12_v002_opt mprage_reg_v004_vfa; do
  curl -L -o models/$m.onnx https://github.com/htylab/tigerbx/releases/download/modelhub/$m.onnx
done
cargo build --release
# Windows, single executable (pyke's static ONNX Runtime build; it uses the DLL C runtime, so no +crt-static):
#   cargo build --release --no-default-features --features static-ort
```

The models and the TigerBx template (`../tigerbx/template/mni_template.nii.gz`) are embedded in the binary (~100 MB). ONNX Runtime is loaded at run time (`ort` with `load-dynamic`): from
`ORT_DYLIB_PATH`, else an `onnxruntime` library next to the executable, else the system library path. Use the
official ONNX Runtime release (e.g. `libonnxruntime.so.1.24.x`), which runs on older glibc versions.

## Usage

```bash
tigerbx T1w.nii.gz                      # no option: _tbet next to the input
tigerbx T1w.nii.gz --seg                # pipeline tBET + ASEG: _tbet, _aseg
tigerbx T1w.nii.gz --norm               # pipeline tBET + template normalisation: _tbet, _norm (gnorm)
tigerbx T1w.nii.gz --all                # --seg + --norm: _tbet, _aseg, _norm
tigerbx data/ -o out/ --seg -m          # output options add to a pipeline: + _tbetmask
tigerbx data/*.nii.gz -o out/ -m        # single outputs: -b _tbet, -m _tbetmask, -a _aseg, -d _dgm, -A _affine, -n _norm
tigerbx data/ -o out/ -a -A -t 8        # ASEG + _affine only, 8 intra-op threads (default: 70% of the available CPUs)
tigerbx data/ -o out/ --norm --norm-minmax  # _norm min-max scaled as tigerbx.reg('AD') instead of gnorm
tigerbx T1w.nii.gz --aseg-model other_aseg43.onnx   # other models (--bet-/--aseg-/--dgm-/--vfa-model)
tigerbx data/ -o out/ --seg -g         # GPU (see below)
```

GPU (`-g`): Linux uses CUDA. The Linux zip carries the GPU build of ONNX Runtime 1.24.4 (same CPU results as the
CPU build); the CUDA 12 and cuDNN 9 libraries must be on the library path (a CUDA 12 toolkit, or e.g.
`pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-runtime-cu12` plus
`LD_LIBRARY_PATH` to their `lib` folders). Windows uses DirectML (any DirectX 12 GPU, nothing to install). macOS and
any setup where the GPU provider cannot load print a warning and run on the CPU. On the GPU the models' `labels`
output is dropped (CUDA ArgMax needs ~10x the logits as workspace) and the argmax runs on the CPU; cuDNN uses
heuristic algorithm selection without its maximum workspace, and the GPU arena is released after every run.
`--seg -g` on IBSR18: ~6 s per subject, GPU peak ~8 GB; CPU and GPU agree (IBSR_01: tBET mask identical, 4 `_aseg`
voxels differ) since the optimised models compute GroupNorm statistics in stages (see `tigerbx.dev.onnx_optimize`).

Earlier outputs (`_tbet`, `_tbetmask`, `_aseg`, `_dgm`, `_affine`, `_norm`) in an input folder are skipped; a failing input is reported and the
batch continues.

`TIGERBX_DUMP_REG=<dir>` also writes `<stem>_M.txt` (4x4 world matrix template -> input) and `<stem>_grid.f32`
(VFA voxel sampling grid, float32 [3, 160, 224, 192]) for validation.

Originally developed as a separate repository (htylab/rustbx, the former name of this executable); see the TigerBx license.
