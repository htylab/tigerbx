# TigerBx: `embed` — tReg brain embedding

`tigerbx.embed` turns a T1w image into a small, fixed-layout table of regional features for downstream models
(classification, regression, clustering): the mean tReg encoder feature inside each of the 37 TSEG structures, plus the
structure volumes and TIV.

The features are those of the registration model (tReg v3, `mprage_treg_v003_opt.onnx`): its feature network,
exported alone up to the 2 mm and 4 mm decoder levels as `mprage_embed_v002.onnx` (14 MB). It is trained for
registration (no labels), so its features describe local anatomy and shape. 0.3.x used the encoder of the earlier
registration model (`mprage_embed_v001_vfa.onnx`, 96 columns; still accepted as `model=`); the two tables are not
comparable.

---

## Python API

```python
tigerbx.embed(input=None, output=None, model=None, gpu=False, verbose=0)
```

| Parameter | Type            | Default | Description |
|-----------|-----------------|---------|-------------|
| `input`   | `str` or `list` | `None`  | T1w NIfTI file, directory, glob pattern, or list of paths (raw, not skull-stripped) |
| `output`  | `str`           | `None`  | Output folder; if `None`, results are saved next to each input |
| `model`   | `str`           | `None`  | Encoder ONNX override (default `mprage_embed_v002.onnx`) |
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

1. tBX + TSEG (one `tigerbx.infer('mt')` call, see [seglabel.md](seglabel.md))
2. numreg affine of the skull-stripped image to the bundled MNI-space template (as `tigerbx.reg('A')`)
3. tReg feature network on the affine image -> feature maps L1 (2 mm, 24 channels) and L2 (4 mm, 48 channels)
4. TSEG -> template space with the same affine (nearest); each structure's mask is block-averaged onto the L1 / L2
   grids (partial-volume weights) and the features are averaged inside it

## Output: `<stem>_embed.npz`

| Key         | Shape / type   | Description |
|-------------|----------------|-------------|
| `roi`       | float32 [32, 72] | Mean feature per structure; columns 0-23 = L1, 24-71 = L2. A structure missing from the TSEG gives a row of NaN |
| `labels`    | int [32]       | TSEG label of each row (the TSEG structures of [seglabel.md](seglabel.md)) |
| `columns`   | str [72]       | `L1_00` ... `L2_47` |
| `volume_ml` | float32 [32]   | Native TSEG volume of each structure (ml) |
| `tiv_ml`    | float          | TIV = the volume of all TSEG labels (ml), as in SynthSeg |
| `quality`, `quality_keys` | float [5], str [5] | Image quality `noise, bias, contrast, edge, res` (see [pipelines.md](pipelines.md#image-quality-and-homogeneity)) |
| `model`, `tigerbx_version`, `template` | str | Provenance |

```python
import numpy as np
e = np.load('sub-001_T1w_embed.npz')
x = e['roi'].ravel()                                   # 2304 features
v = np.r_[e['volume_ml'], e['volume_ml'] / e['tiv_ml']]  # 64 volume features
```

`embed_summary.csv` in the output folder lists `file, status, TIV_ml`, the five quality measures and `error` per input. A failing input is logged,
gets a `<stem>_embedfail.log` and is skipped (a single input raises), as in the VBM pipeline.

## Using the features

- **Standardise per scanner / site** before pooling data from several sources (z-score each feature within site).
- The table has ~2.3k features: use a regularised linear model (e.g. L2 logistic / ridge, or PCA first).
- **Combine with volumes by late fusion**: train one model on `roi` and one on the volumes (`volume_ml`,
  `volume_ml / tiv_ml`) and average their predicted probabilities. Concatenating the 64 volume columns to the 2304
  `roi` columns lets the regularisation drown them.
- Whole-brain means of the features are not stored: every level ends in GroupNorm over the whole image, which makes
  them nearly identical across images.

