# Registration, transforms, and VBM

## Registration

```python
result = tigerbx.reg(plan, input, output=None, model=None, template=None,
                     gpu=False, gz=False, save_displacement=False,
                     affine_type='numreg', verbose=0)
```

`plan` is an ordered uppercase string:

| Step | Method |
|---|---|
| `R` | rigid (numreg, 6 dof); must be first when combined |
| `A` | affine |
| `D` | VFA nonlinear (default/recommended); requires earlier `A` |
| `V` | VMnet nonlinear; requires earlier `A` |

No duplicate steps. Typical plans: `AD` (recommended), `A`, `AV`, `RAD`.
`affine_type` defaults to `'numreg'` (numpy/scipy affine); `'C2FViT'` is the deep-learning alternative. `'ANTs'` is a deprecated alias of `'numreg'`.

```python
affine = tigerbx.reg('A', 'T1w.nii.gz', 'out/')
nonlinear = tigerbx.reg('AD', 'T1w.nii.gz', 'out/', gpu=True,
                        save_displacement=True)
```

## Apply a saved warp

```python
tigerbx.transform('labels.nii.gz', 'subject_warp.npz', 'out/',
                  interpolation='nearest')
tigerbx.transform('T1w.nii.gz', 'subject_warp.npz', 'out/',
                  interpolation='linear')
```

Use nearest for labels and linear for continuous images.

## VBM is separate

```python
result = tigerbx.vbm('T1w.nii.gz', 'out/', reg_plan='AD', gpu=False)
# equivalent dispatcher:
result = tigerbx.pipeline('vbm', input='T1w.nii.gz', output='out/', reg_plan='AD')
```

Never encode VBM as a reg plan step.

Outputs (flat, in the output folder): `_SmoothedGM` (modulated GM, FWHM 7.065 mm), `_norm` (T1 after the plan),
`_ModulatedGM` with `save_modulated=True`, and `vbm_summary.csv` (status, `qc_flags`, TIV, GM/WM/CSF ml, image quality,
`homog_z`). `fwhm=` and `save_modulated=` are keyword-only. A failing input gets `<stem>_vbmfail.log` and is
skipped in a batch.
