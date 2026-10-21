# Custom / replacement models

Override any default model by filename or path:

```python
tigerbx.run('bag', 'T1w.nii.gz', 'out/', model={'aseg': '/models/mprage_aseg43_v010_test.onnx'})
```

Model keys: `bet`, `aseg`, `dgm`, `wmh`, `syn`, `cgw`, `ct` (for `run`); `HLC` (uppercase, for `hlc`).

Filename = `<prefix>_<mode>_<version>_<tag>.onnx`. The `<mode>` field (`bet`, `aseg43`, `dgm12`, `wmh`, `synthseg`,
...) selects normalisation, softmax vs sigmoid and the label table — a wrong or missing mode raises `ValueError`.
Do not rename a model to "look nicer".

Input given to non-BET models: tBET brain, RAS, ~1 mm, cropped to the brain bounding box + 16 voxels (arbitrary, odd
size), float32, divided by its max (`synthseg`: min-max). Output must be logits `[1, C, X, Y, Z]` with channel 0 =
background and channel k = `label_all[mode][k-1]`.

Full contract, export checklist and how to validate: `doc/models.md`, then [val.md](val.md).
