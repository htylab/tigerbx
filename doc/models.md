# Model files: naming and I/O contract

This page is for anyone who trains or swaps a `tbx` / `tseg` / `dgm` model and wants it to be a
drop-in replacement (`model={'tseg': 'my.onnx'}` in `tigerbx.run()`, `seg_model=` in `tigerbx.val()`).

## Filename

```
<prefix>_<mode>_<version>_<tag>.onnx        e.g. mprage_tseg_v0011_opt.onnx
```

`lib_bx.get_mode()` splits the basename on `_`; **`<mode>` is not cosmetic** — it selects

| mode | channels (incl. background) | normalisation | activation | label table |
|---|---|---|---|---|
| `tbx` | 1 | `/ max` | logit ≥ 0 (sigmoid ≥ 0.5), largest component | — |
| `tseg` | 44 | `/ max` | argmax | `label_all['tseg']` |
| `dgm12` | 13 | `/ max` | argmax | 1..12 |
| `dkt`, `wmp`, `seg3` | 63 / 74 / 4 | `/ max` | argmax | `label_all[mode]` |

A name such as `mprage_tseg_v009.onnx` (mode `tseg`, only 3 fields) raises a `ValueError` naming the expected pattern.

## Input the model receives (non-tBX models)

1. tBX brain (T1w × tBX mask) of the input image.
2. Reoriented to RAS; kept at native resolution if every zoom is within 0.9–1.1 mm (0.99–1.01 for `ct`),
   otherwise resampled to 1 mm.
3. Cropped to the bounding box of brain > 0 with 16 voxels padding → **arbitrary, often odd, spatial size**.
4. `float32`, shape `[1, 1, X, Y, Z]`, normalised as in the table.

So the network must accept any X/Y/Z (no fixed size, not necessarily divisible by 16): e.g. upsample to the skip
tensor's size instead of a fixed factor, or pad/crop inside the exported graph.

## Output

`[1, C, X, Y, Z]` **logits** (not probabilities), same spatial size as the input. Channel 0 is background, channel
`k` is `label_all[mode][k-1]`. tigerbx takes the argmax (tBX: logit ≥ 0) and the LUT, resamples back to the input
grid and masks with the brain mask; no softmax / sigmoid is computed, as it does not change the labels.

Optional second output `labels` (uint8, `[1, X, Y, Z]`): the argmax over channels (tBX: logit ≥ 0). When a model has
it, tigerbx requests only that output, so the logits (1.15 GB for TSEG on a typical crop) are not copied out of ONNX
Runtime; patch inference (`p`) still uses the logits, which must stay the first output.

## Optimising an exported model

```
python -m tigerbx.dev.onnx_optimize model.onnx model_opt.onnx --labels --shape 1,1,160,192,160
```

rewrites each exported GroupNorm / InstanceNorm into ops that ONNX Runtime runs faster on CPU (same result: two-pass
variance, folded affine), adds the `labels` output, and compares both models on a random input (largest logit
difference, run time). Needs the `onnx` package. For the 0.3.0 tBX and TSEG models it gives 1.2-1.3x faster
inference and, together with the `labels` output, about 1.7x faster TSEG inference + post-processing, with identical
masks / labels on real T1w images. Always check the speed: a model with few, small normalisation layers (e.g. VFA
registration) can get slower.

## Export checklist

- ONNX opset ≤ 17, dynamic axes on batch and the three spatial dims.
- Check ONNX vs. PyTorch on at least two odd shapes (e.g. 141×180×138 and 160×201×171).
- Store provenance in `metadata_props` (architecture, iteration, code hash; never data, see AGENTS.md): the
  repository's test models are otherwise indistinguishable except by weight diffs.
- Run `tigerbx.dev.onnx_optimize ... --labels` (above) and check that the labels / masks of a few real images match
  the unoptimised model.
- Validate with `tigerbx.val(task='tseg', seg_model='mprage_tseg_<ver>_<tag>.onnx', full=True, GPU=True)`
  (see [validation.md](validation.md)).
