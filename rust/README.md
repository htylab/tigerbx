# tigerbx executable

TigerBx tBX, TSEG (`tigerbx.run('bmt')`) and template registration (`tigerbx.reg('AD')`, the VBM `_norm`) as a
native executable (Rust + ONNX Runtime), with the same models and preprocessing. Segmentation: resample to 1 mm when a voxel size is outside 0.9–1.1 mm, reorder to a diagonal affine
(TigerBx `read_tbx_input`, scipy-compatible cubic spline), tBX + largest component, TSEG on the tBX brain cropped
to the brain + 16 voxels (to the tBX mask + 32 when that box exceeds 256^3 voxels), nearest-neighbour back onto the
input grid, label 99 for unlabeled tBX-mask voxels. DGM12 (`-d`, `tigerbx.run('d')`) runs on the same crop as a
separate `_dgm`. The output options give the models' own results, as the Python modules: `_tbxmask`, `_tseg` and
`_dgm` match `tigerbx.run('bmtd')` (identical on seg18; `_tbx` in the input data type). The `--seg` / `--all`
pipelines post-process as `tigerbx.seg()`: the 12 deep-GM structures of `_tseg` come from DGM12 (seg18
30 structures 0.8178 vs 0.8129), voxel-identical to Python. Check: `tests/rust_python_parity.py`. On oblique images a
few voxels on the edge of the field of view can differ (floating-point rounding of the 4x4 affine inverse).

Registration: a Rust port of numreg (Mattes MI, symmetric cost, 6 / 3 / 1.5 mm pyramid, multi-start L-BFGS) gives
`_affine` (tBX brain on the template grid, min-max normalised), then VFA (`mprage_reg_v005_vfa`) gives `_norm`.
As in Python, `_affine` and `_norm` (affine + VFA warp composed) are sampled once from the input grid with a cubic
B-spline clipped to the input range, the VFA input trilinearly; with the same matrix and warp they match
`tigerbx.core.numreg.resample` within 5e-6.
It cannot be bit-identical to numreg (another random generator for the 1.5 mm sample subset, another line search);
on a reg60 subset the template-space Dice of carried labels is 0.7217 vs 0.7218 (affine) and 0.8451 vs 0.8452
(affine + VFA), image correlation >= 0.9993.

`_norm` / `--affine` `_affine` intensities (`--intensity`, as `tigerbx.norm(intensity=...)`): `raw` (default; the
input's units, any contrast), `minmax` ([0, 1]) or `gnorm`, a port of `tigerbx.core.tissuenorm.tissue_norm` (prior
mode: GMM with the template tissue priors + bias field; background 0, CSF 1/3, GM 2/3, WM 1). `-A` alone writes the
min-max `_affine` of `tigerbx.reg('A')`. For gnorm the brain is the tBX mask carried
through the same affine + VFA warp (oblique inputs leave cubic-resampling ringing of ~1e-40 to 1e-3 outside the
brain, so `img > 0` is not the brain). On a reg60 subset the peaks match the Python `tissue_norm` (same mask)
within 6.4e-4 relative and the voxel values within 5e-4. T1 only: when the peaks are not ordered CSF < GM < WM
(T2, PD, ...) `_norm` falls back to img / p99 of the brain with a warning, and the batch goes on.
`_norm` comes with `_warp.npz`, the registration as plain arrays (affine `affine_M` + VFA `vfa_warp`, TigerBx
`core.warpfile`), which `tigerbx.transform` and `tigerbx.norm` share.

Prebuilt executables: the `exe` GitHub workflow builds Linux (glibc 2.28+, x64) and macOS (arm64) zips with the
ONNX Runtime library next to the executable, and Windows (x64) as a single `tigerbx.exe` (ONNX Runtime 1.23.2 from
pyke linked statically; needs the VC++ 2015-2022 runtime and Windows 10 1903+ for DirectML.dll, both usually
present), as workflow artifacts on every change under `rust/` and attached to the draft release of each `v*` tag
(Compile10). ONNX Runtime 1.23.2 and 1.24.4 give identical outputs (seg18, all outputs, 0 voxels differ).

## Build

```bash
mkdir -p models
for m in mprage_tbx_v005_opt mprage_tseg_v0011_opt mprage_dgm12_v002_opt mprage_reg_v005_vfa; do
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
tigerbx T1w.nii.gz                      # no option: _tbx next to the input
tigerbx T1w.nii.gz --seg                # pipeline tBX + TSEG: _tbx, _tseg (deep GM from DGM12)
tigerbx T1w.nii.gz --affine             # pipeline affine-only normalisation: _tbx, _affine, _warp.npz
tigerbx T1w.nii.gz --norm               # pipeline template normalisation (affine + VFA): _tbx, _norm, _warp.npz
tigerbx T1w.nii.gz --all                # --seg + --norm: _tbx, _tseg, _norm, _warp.npz
tigerbx data/ -o out/ --seg -m          # output options add to a pipeline: + _tbxmask
tigerbx data/*.nii.gz -o out/ -m        # single outputs: -b _tbx, -m _tbxmask, -t _tseg, -d _dgm, -A _affine, -n _norm
tigerbx data/ -o out/ -t -A -j 8        # TSEG + _affine only, 8 intra-op threads (default: 70% of the available CPUs)
tigerbx data/ -o out/ --norm --intensity gnorm  # T1 tissue intensities (CSF 1/3, GM 2/3, WM 1); raw (default) or minmax
tigerbx T1w.nii.gz --tseg-model other_tseg.onnx   # other models (--tbx-/--tseg-/--dgm-/--vfa-model)
tigerbx data/ -o out/ --seg -g         # GPU (see below)
```

GPU (`-g`): Linux uses CUDA. The Linux zip carries the GPU build of ONNX Runtime 1.24.4 (same CPU results as the
CPU build); the CUDA 12 and cuDNN 9 libraries must be on the library path (a CUDA 12 toolkit, or e.g.
`pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-runtime-cu12` plus
`LD_LIBRARY_PATH` to their `lib` folders). Windows uses DirectML (any DirectX 12 GPU, nothing to install). macOS and
any setup where the GPU provider cannot load print a warning and run on the CPU. On the GPU only the models' logits
output is kept (CUDA ArgMax of `labels` needs ~10x the logits as workspace) and the argmax runs on the CPU; cuDNN uses
heuristic algorithm selection without its maximum workspace, and the GPU arena is released after every run.
TF32 is on (convolutions 1.5-3x faster on Ampere and newer NVIDIA GPUs; ~100 `_tseg` voxels per subject then differ
from the CPU); `TIGERBX_TF32=0` turns it off, and the GPU then matches the CPU. With TF32 off, newer cuDNN (9.2x) logs
"No execution plans support the graph" and falls back to another algorithm (results are fine).
`--seg -g` on seg18: ~6 s per subject; a GPU peak of ~16 GB per process was measured on an H200. With
`TIGERBX_TF32=0` CPU and GPU agree (a seg18 scan: tBX mask identical, 4 `_tseg` voxels differ), since the optimised
models compute GroupNorm statistics in stages (see `tigerbx.dev.onnx_optimize`).

Earlier outputs (`_tbx`, `_tbxmask`, `_tseg`, `_dgm`, `_affine`, `_norm`) in an input folder are skipped; a failing input is reported and the
batch continues.

`TIGERBX_DUMP_REG=<dir>` also writes `<stem>_M.txt` (4x4 world matrix template -> input) and `<stem>_grid.f32`
(VFA voxel sampling grid, float32 [3, 160, 224, 192]) for validation.

Originally developed as a separate repository (htylab/rustbx, the former name of this executable); see the TigerBx license.
