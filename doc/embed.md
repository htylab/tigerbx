# TigerBx: `embed` — VFA brain embedding

`tigerbx.embed` turns a T1w image into a small, fixed-layout table of regional features for downstream models
(classification, regression, clustering): the mean VFA encoder feature inside each of the 37 TSEG structures, plus the
structure volumes and TIV.

The encoder is the one of the VFA registration model used by `tigerbx.reg('AD')` (`mprage_reg_v005_vfa.onnx`; same weights as v004),
exported alone as `mprage_embed_v001_vfa.onnx` (6.8 MB). It is a registration encoder (no labels), so
its features describe local anatomy and shape.

---

## Python API

```python
tigerbx.embed(input=None, output=None, model=None, gpu=False, verbose=0)
```

| Parameter | Type            | Default | Description |
|-----------|-----------------|---------|-------------|
| `input`   | `str` or `list` | `None`  | T1w NIfTI file, directory, glob pattern, or list of paths (raw, not skull-stripped) |
| `output`  | `str`           | `None`  | Output folder; if `None`, results are saved next to each input |
| `model`   | `str`           | `None`  | Encoder ONNX override (default `mprage_embed_v001_vfa.onnx`) |
| `gpu`     | `bool`          | `False` | Use GPU for tBX, TSEG and the encoder |
| `verbose` | `int`           | `0`     | `0` = tqdm only, `1` = per-file timing |

**Returns:** for a single input, the embedding `dict` (below); for several inputs, a list of `_embed.npz` paths of the
successful inputs.

## Examples

```python
tigerbx.embed('/data/T1w_dir', '/data/embed_out')
tigerbx.embed('sub-001_T1w.nii.gz', 'out', gpu=True)
```

---

## Pipeline

1. tBX + TSEG (one `tigerbx.run('mt')` call; TSEG includes label 99, see [seglabel.md](seglabel.md))
2. numreg affine of the skull-stripped image to the bundled MNI-space template (as `tigerbx.reg('A')`)
3. VFA encoder on the affine image -> feature maps L1 (2 mm, 32 channels) and L2 (4 mm, 64 channels)
4. TSEG -> template space with the same affine (nearest); each structure's mask is block-averaged onto the L1 / L2
   grids (partial-volume weights) and the features are averaged inside it

## Output: `<stem>_embed.npz`

| Key         | Shape / type   | Description |
|-------------|----------------|-------------|
| `roi`       | float32 [37, 96] | Mean encoder feature per structure; columns 0-31 = L1, 32-95 = L2. A structure missing from the TSEG gives a row of NaN |
| `labels`    | int [37]       | TSEG label of each row (the 37 structures of [seglabel.md](seglabel.md); deprecated labels and 99 are left out) |
| `columns`   | str [96]       | `L1_00` ... `L2_63` |
| `volume_ml` | float32 [37]   | Native TSEG volume of each structure (ml) |
| `tiv_ml`    | float          | TIV = tBX mask volume (ml) |
| `quality`, `quality_keys` | float [5], str [5] | Image quality `noise, bias, contrast, edge, res` (see [pipelines.md](pipelines.md#image-quality-and-homogeneity)) |
| `model`, `tigerbx_version`, `template` | str | Provenance |

```python
import numpy as np
e = np.load('sub-001_T1w_embed.npz')
x = e['roi'].ravel()                                   # 3552 features
v = np.r_[e['volume_ml'], e['volume_ml'] / e['tiv_ml']]  # 74 volume features
```

`embed_summary.csv` in the output folder lists `file, status, TIV_ml`, the five quality measures and `error` per input. A failing input is logged,
gets a `<stem>_embedfail.log` and is skipped (a single input raises), as in the VBM pipeline.

## Using the features

- **Standardise per scanner / site** before pooling data from several sources (z-score each feature within site).
- The table has ~3.6k features: use a regularised linear model (e.g. L2 logistic / ridge, or PCA first).
- **Combine with volumes by late fusion**: train one model on `roi` and one on the volumes (`volume_ml`,
  `volume_ml / tiv_ml`) and average their predicted probabilities. Concatenating the 74 volume columns to the 3552
  `roi` columns lets the regularisation drown them.
- Whole-brain means of the encoder features are not stored: every encoder level ends in InstanceNorm, which makes them
  nearly identical across images.

