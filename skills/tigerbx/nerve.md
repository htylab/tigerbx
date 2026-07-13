# NERVE embedding and reconstruction

```python
result = tigerbx.nerve(flags, input, output=None, model=None,
                       method='NERVE', verbose=0)
```

| Flag | Operation |
|---|---|
| `e` | encode T1w to one latent NPZ |
| `d` | decode latent NPZ to one stacked reconstruction NIfTI |
| `p` | save stacked ROI patches during encode |
| `v` | evaluate; implies encode + decode + patches |
| `s` | stochastic reconstruction using sigma |
| `g` | GPU |

```python
encoded = tigerbx.nerve('ep', 'T1w.nii.gz', 'out/')
# {'latent': '..._nerve.npz', 'patch': '..._nerve_patch.nii.gz'}
decoded = tigerbx.nerve('d', 'out/T1w_nerve.npz', 'recon/')
# {'recon': '..._nerve_recon.nii.gz'}
evaluation = tigerbx.nerve('v', 'T1w.nii.gz', 'out/')
# keys: records, average, csv, outputs
```

Single input returns a named path dict; batch input returns a list of such dicts. The NPZ contains `LH`, `LA`, `RH`, `RA`, corresponding `_sigma` arrays, `encoder`, and `affine`. Do not assume separate left/right files or a generic `z` key.

CLI:

```bash
tiger nerve T1w.nii.gz -e -p -o out/
tiger nerve out/T1w_nerve.npz -d -o recon/
```
