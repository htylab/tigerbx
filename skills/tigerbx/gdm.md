# EPI/DTI distortion correction

```python
result = tigerbx.gdm(input, output=None, b0_index=0, dmap=False,
                     no_resample=False, GPU=False, verbose=0)
```

- Input is a 3D/4D EPI NIfTI; for 4D input, select b0 with an integer index or `.bval` path.
- Default resamples model input to 1.7 mm; use `no_resample=True` only when appropriate.
- `dmap=True` also returns/saves the predicted displacement map.

```python
corrected = tigerbx.gdm('dti.nii.gz', 'out/', b0_index='dti.bval')
both = tigerbx.gdm('dti.nii.gz', 'out/', dmap=True, GPU=True)
```

Return contract:

- Single input: nibabel dict with `gdmi` (corrected image) and optional `gdm` (displacement map).
- Batch: list of saved-path dicts with the same keys.
- Saved suffixes: `_gdmi.nii.gz` and optional `_gdm.nii.gz`.
