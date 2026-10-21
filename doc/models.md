# Model files: naming and I/O contract

This page is for anyone who trains or swaps a `bet` / `aseg` / `dgm` / `wmh` / `syn` model and wants it to be a
drop-in replacement (`model={'aseg': 'my.onnx'}` in `tigerbx.run()`, `seg_model=` in `tigerbx.val()`).

## Filename

```
<prefix>_<mode>_<version>_<tag>.onnx        e.g. mprage_aseg43_v0011_allseg.onnx
```

`lib_bx.get_mode()` splits the basename on `_`; **`<mode>` is not cosmetic** — it selects

| mode | channels (incl. background) | normalisation | activation | label table |
|---|---|---|---|---|
| `bet` | 1 | `/ max` | sigmoid, largest component | — |
| `aseg43` | 44 | `/ max` | softmax + argmax | `label_all['aseg43']` |
| `dgm12` | 13 | `/ max` | softmax + argmax | 1..12 |
| `wmh` | 2 | `/ max` | softmax + argmax | 1 |
| `synthseg` | 33 | min-max | softmax + argmax | `label_all['synthseg']` |
| `dkt`, `wmp`, `seg3` | 63 / 74 / 4 | `/ max` | softmax + argmax | `label_all[mode]` |

A name such as `mprage_aseg_v009.onnx` (mode `aseg`, only 3 fields) raises a `ValueError` naming the expected pattern.

## Input the model receives (non-BET models)

1. tBET brain (T1w × BET mask) of the input image.
2. Reoriented to RAS; kept at native resolution if every zoom is within 0.9–1.1 mm (0.99–1.01 for `ct`/`cgw`),
   otherwise resampled to 1 mm.
3. Cropped to the bounding box of brain > 0 with 16 voxels padding → **arbitrary, often odd, spatial size**.
4. `float32`, shape `[1, 1, X, Y, Z]`, normalised as in the table.

So the network must accept any X/Y/Z (no fixed size, not necessarily divisible by 16): e.g. upsample to the skip
tensor's size instead of a fixed factor, or pad/crop inside the exported graph.

## Output

`[1, C, X, Y, Z]` **logits** (not probabilities), same spatial size as the input. Channel 0 is background, channel
`k` is `label_all[mode][k-1]`. tigerbx applies softmax/argmax/LUT, resamples back to the input grid and masks with
the brain mask.

## Export checklist

- ONNX opset ≤ 17, dynamic axes on batch and the three spatial dims.
- Check ONNX vs. PyTorch on at least two odd shapes (e.g. 141×180×138 and 160×201×171).
- Store provenance in `metadata_props` (training data/config, iteration, code hash, validation scores): the
  repository's test models are otherwise indistinguishable except by weight diffs.
- Validate with `tigerbx.val(task='aseg', seg_model='mprage_aseg43_<ver>_<tag>.onnx', full=True, GPU=True)`
  (see [validation.md](validation.md)).
