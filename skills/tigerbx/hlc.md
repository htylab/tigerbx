# HLC parcellation

```python
result = tigerbx.hlc(input, output=None, model=None, save='h', GPU=False,
                     gz=True, patch=False, verbose=0)
```

| `save` letter | Key | Output |
|---|---|---|
| `m` | `tbetmask` | brain mask |
| `b` | `tbet` | brain-extracted image |
| `h` | `hlc` | HLC parcellation |
| `t` | `ct` | cortical thickness |
| `c` | `csf` | CSF PVE |
| `g` | `gm` | GM PVE |
| `w` | `wm` | WM PVE |
| `all` | all above | shorthand |

```python
hlc = tigerbx.hlc('T1w.nii.gz', 'out/')['hlc']
all_outputs = tigerbx.hlc('T1w.nii.gz', 'out/', save='all')
```

Use `GPU=True` here (unlike `tigerbx.run`, where GPU is flag `g`). HLC GPU inference may require substantial VRAM; fall back to CPU or `patch=True` when appropriate.

Read [labels.md](labels.md) only when interpreting HLC label values.
