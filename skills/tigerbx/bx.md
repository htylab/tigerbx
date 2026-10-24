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
| `c` | `ct` | cortical thickness |
| `C` | `cgw` | `[CSF, GM, WM]` PVE images |
| `g` | — | GPU |
| `p` | — | patch inference |
| `z` | — | gzip saved NIfTI |

Common calls:

```python
import tigerbx

brain = tigerbx.run('b', 'T1w.nii.gz', 'out/')['tbet']
seg = tigerbx.run('bmad', 'T1w.nii.gz', 'out/')
gpu_all = tigerbx.run('bmacCdg', 'T1w.nii.gz', 'out/')
batch_paths = tigerbx.run('bma', '/data/T1w/', 'out/', continue_=True)
```

Do not pass `GPU=True`; add `g` to `flags`. Pass an output folder (default writes next to the input) or `save_outputs=False`. `silent` is deprecated; use `verbose=0`.

Saved suffixes: `_tbet`, `_tbetmask`, `_aseg`, `_dgm`, `_ct`, and `_cgw_pve0/1/2`. Read [labels.md](labels.md) only for label lookup.

Stand-alone executable (no Python): `rustbx data/ -o out/ --seg` gives `_tbet` + `_aseg` (`-m` adds `_tbetmask`,
`-d` `_dgm`); see `rust/README.md`.
