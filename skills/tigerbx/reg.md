# Registration, transforms, and VBM

## Registration

```python
result = tigerbx.reg(plan, input, output=None, model=None, template=None,
                     gpu=False, gz=False, save_displacement=False,
                     affine_type='C2FViT', verbose=0)
```

`plan` is an ordered uppercase string:

| Step | Method |
|---|---|
| `R` | rigid; must be first when combined |
| `A` | affine |
| `V` | VMnet nonlinear; requires earlier `A` |
| `N` | ANTs SyN |
| `C` | ANTs SyNCC |
| `F` | FuseMorph; requires earlier `A` |

No duplicate steps. Typical plans: `A`, `AV`, `AN`, `AC`, `AF`, `RAV`.

```python
affine = tigerbx.reg('A', 'T1w.nii.gz', 'out/')
nonlinear = tigerbx.reg('AV', 'T1w.nii.gz', 'out/', gpu=True,
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
result = tigerbx.vbm('T1w.nii.gz', 'out/', reg_plan='AF', gpu=False)
# equivalent dispatcher:
result = tigerbx.pipeline('vbm', input='T1w.nii.gz', output='out/', reg_plan='AF')
```

Never encode VBM as a reg plan step.

CLI:

```bash
tiger reg AV T1w.nii.gz -o out/ --save_displacement
tiger vbm T1w.nii.gz -o out/ --reg-plan AF
```
