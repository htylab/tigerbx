# `tigerbx.core.metrics` — Metric Implementations

Low-level metric functions used by `tigerbx.eval()`. All functions accept file paths, nibabel images, or numpy arrays via the internal `_to_array()` converter.

For the high-level `tigerbx.eval()` API, see [eval.md](../eval.md).

---

## Segmentation Metrics

All segmentation metrics return a `dict` with integer-string keys (one per label) and a `'mean'` key.

### `dice(y_true, y_pred, labels=None, ignore_background=True)`

Dice similarity coefficient per label.

```python
from tigerbx.core.metrics import dice
scores = dice('gt.nii.gz', 'pred.nii.gz', labels=[10, 11, 17, 18])
# → {'10': 0.91, '11': 0.89, '17': 0.94, '18': 0.92, 'mean': 0.915}
```

---

### `iou(y_true, y_pred, labels=None, ignore_background=True)`

Intersection-over-union (Jaccard index) per label.

---

### `hd95(y_true, y_pred, labels=None, ignore_background=True, voxel_spacing=None)`

95th-percentile Hausdorff distance per label. Distances are in voxels by default; pass `voxel_spacing=[1.0, 1.0, 1.0]` for physical units (mm).

---

### `asd(y_true, y_pred, labels=None, ignore_background=True, voxel_spacing=None)`

Average symmetric surface distance per label.

---

## Reconstruction Metrics

All reconstruction metrics return a single `float`.

### `mae(y_true, y_pred)`

Mean absolute error.

### `mse(y_true, y_pred)`

Mean squared error.

### `psnr(y_true, y_pred, data_range=None)`

Peak signal-to-noise ratio (dB). `data_range` defaults to `max(y_true) - min(y_true)`.

### `ssim(y_true, y_pred, data_range=None, sigma=1.5)`

Structural similarity index using Gaussian-weighted statistics.

### `ncc(y_true, y_pred)`

Normalized cross-correlation (global scalar).

### `mi(y_true, y_pred, bins=64)`

Mutual information (histogram-based).

### `ksg_mi(y_true, y_pred, k=5)`

Mutual information via the KSG k-nearest-neighbour estimator.

---

## Classification Metrics

### `accuracy(y_true, y_pred)`

Classification accuracy.

### `precision(y_true, y_pred, average='macro')`

Classification precision. `average`: `'macro'`, `'micro'`, `'weighted'`, `'binary'`.

### `recall(y_true, y_pred, average='macro')`

Classification recall.

### `f1(y_true, y_pred, average='macro')`

F1 score.

---

## Common Parameters

| Parameter            | Type               | Default  | Description |
|----------------------|--------------------|----------|-------------|
| `y_true`             | str / nib / ndarray| —        | Ground truth |
| `y_pred`             | str / nib / ndarray| —        | Prediction |
| `labels`             | list or None       | `None`   | Label values to evaluate; `None` = all non-zero labels in `y_true` |
| `ignore_background`  | bool               | `True`   | Exclude label 0 from auto-discovery |
| `voxel_spacing`      | sequence or None   | `None`   | Physical voxel size for distance metrics |
| `data_range`         | float or None      | `None`   | Dynamic range for PSNR/SSIM |
| `average`            | str                | `'macro'`| Averaging mode for classification metrics |
