# TigerBx Release Notes

## Latest release

- [`v0.5.1`](release/releasenote051.md)
  - tReg v4: diffeomorphic, the warp no longer folds; a faster GPU model
  - Fold repair of earlier tReg warps (Python and executable)

## Earlier releases

- [`v0.5.0`](release/releasenote050.md)
  - Two sets of models: `tigerbx.infer()` and the pipelines use the current tBX / TSEG; `tigerbx.run()` keeps the
    earlier tBET / ASEG with their 0.2.x file names
  - tBX gives the brain mask and the mask with the surrounding CSF (TSEG input); TSEG replaces ASEG
  - tReg, our own deformable registration network; rebuilt T1 and EPI templates
  - Removed: registration plan `V` (VMnet) and `affine_type='C2FViT'`; fewer executable options
  - Fixed: qform-only images keep their stored orientation; `embed(gpu=True)`; lower CPU memory
