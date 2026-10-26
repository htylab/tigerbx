# Brain extraction and segmentation

```python
result = tigerbx.run(flags, input, output=None, model=None, verbose=0,
                     chunk_size=50, continue_=False, save_outputs=True)
```

Combine case-sensitive flags; no flags defaults to `b`.

| Flag | Single-input key | Meaning |
|---|---|---|
| `b` | `tbx` | brain-extracted T1w |
| `m` | `tbxmask` | binary brain mask |
| `t` | `tseg` | 43-label TSEG |
| `d` | `dgm` | 12-label deep gray matter |
| `c` | `ct` | cortical thickness |
| `g` | — | GPU |
| `p` | — | patch inference |
| `z` | — | gzip saved NIfTI |

Common calls:

```python
import tigerbx

brain = tigerbx.run('b', 'T1w.nii.gz', 'out/')['tbx']
seg = tigerbx.run('bmtd', 'T1w.nii.gz', 'out/')
gpu_all = tigerbx.run('bmtcdg', 'T1w.nii.gz', 'out/')
batch_paths = tigerbx.run('bmt', '/data/T1w/', 'out/', continue_=True)
```

Do not pass `GPU=True`; add `g` to `flags`. Pass an output folder (default writes next to the input) or `save_outputs=False`. `silent` is deprecated; use `verbose=0`.

Saved suffixes: `_tbx`, `_tbxmask`, `_tseg`, `_dgm`, `_ct`. CSF / GM / WM maps: `tigerbx.seg(save_tissue=True)` (`_pve_csf/_pve_gm/_pve_wm`, from TSEG); flag `C` was removed. Read [labels.md](labels.md) only for label lookup.

Stand-alone executable (no Python): `tigerbx data/ -o out/ --seg` gives `_tbx` + `_tseg` (`-m` adds `_tbxmask`,
`-d` `_dgm`); see `rust/README.md`.
