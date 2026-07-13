# Brain extraction and segmentation

```python
result = tigerbx.run(flags, input, output=None, model=None, verbose=0,
                     chunk_size=50, continue_=False, save_outputs=True)
```

Combine case-sensitive flags; no flags defaults to `b`.

| Flag | Single-input key | Meaning |
|---|---|---|
| `b` | `tbet` | brain-extracted T1w |
| `m` | `tbetmask` | binary brain mask |
| `a` | `aseg` | 43-label ASEG |
| `d` | `dgm` | 12-label deep gray matter |
| `W` | `wmh` | white-matter hypointensity |
| `S` | `syn` | SynthSeg-style segmentation |
| `c` | `ct` | cortical thickness |
| `C` | `cgw` | `[CSF, GM, WM]` PVE images |
| `q` | `QC`, `QC_raw` | QC values; always computed even without `q` |
| `g` | — | GPU |
| `p` | — | patch inference |
| `z` | — | gzip saved NIfTI |

Common calls:

```python
import tigerbx

brain = tigerbx.run('b', 'T1w.nii.gz', 'out/')['tbet']
seg = tigerbx.run('bmad', 'T1w.nii.gz', 'out/')
gpu_all = tigerbx.run('bmacCdSWg', 'T1w.nii.gz', 'out/')
batch_paths = tigerbx.run('bma', '/data/T1w/', 'out/', continue_=True)
```

Do not pass `GPU=True`; add `g` to `flags`. `silent` is deprecated; use `verbose=0`.

Saved suffixes: `_tbet`, `_tbetmask`, `_aseg`, `_dgm`, `_wmh`, `_syn`, `_ct`, and `_cgw_pve0/1/2`. Read [labels.md](labels.md) only for label lookup.

CLI:

```bash
tiger bx T1w.nii.gz -bmad -o out/
tiger bx /data/T1w/ -bmag --continue -o out/
```
