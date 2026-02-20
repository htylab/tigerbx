# `tigerbx.eval` — Image Quality and Segmentation Metrics

Computes common quantitative metrics between a ground-truth and a predicted image.
Accepts **NIfTI file paths, nibabel images, or numpy arrays** as input — consistent with the rest of tigerbx.

```python
tigerbx.eval(y_true, y_pred, metrics, **kwargs)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | str / nibabel image / ndarray | Ground truth |
| `y_pred` | str / nibabel image / ndarray | Prediction |
| `metrics` | str or list[str] | Metric name(s) to compute |
| `**kwargs` | — | Forwarded to each metric function (see per-metric kwargs below) |

**Returns:** flat `dict` of `{metric_name: value}`.
Segmentation metrics return a nested dict keyed by label string.

---

## Supported metrics

### Segmentation

| Metric | Key | Returns | Extra kwargs |
|--------|-----|---------|--------------|
| Dice similarity coefficient | `'dice'` | `{label: float, 'mean': float}` | `labels`, `ignore_background` |
| Intersection over union | `'iou'` | `{label: float, 'mean': float}` | `labels`, `ignore_background` |
| 95th-percentile Hausdorff distance | `'hd95'` | `{label: float, 'mean': float}` | `labels`, `ignore_background`, `voxel_spacing` |
| Average symmetric surface distance | `'asd'` | `{label: float, 'mean': float}` | `labels`, `ignore_background`, `voxel_spacing` |

**Segmentation kwargs**

| kwarg | Default | Description |
|-------|---------|-------------|
| `labels` | `None` | List of integer label values to evaluate. When `None`, all unique non-zero labels in `y_true` are used automatically. |
| `ignore_background` | `True` | Exclude label `0` from auto-discovery. Has no effect when `labels` is given explicitly. |
| `voxel_spacing` | `None` | Physical voxel size, e.g. `[1.0, 1.0, 1.0]` (mm). When `None`, distances are in voxels. Applies to `hd95` and `asd` only. |

### Reconstruction

| Metric | Key | Returns | Extra kwargs |
|--------|-----|---------|--------------|
| Mean absolute error | `'mae'` | float | — |
| Mean squared error | `'mse'` | float | — |
| Peak signal-to-noise ratio (dB) | `'psnr'` | float | `data_range` |
| Structural similarity index | `'ssim'` | float | `data_range`, `sigma` |
| Normalized cross-correlation | `'ncc'` | float | — |
| Mutual information (histogram) | `'mi'` | float | `bins` |
| Mutual information (KSG k-NN) | `'ksg_mi'` | float | `k` |

**Reconstruction kwargs**

| kwarg | Default | Description |
|-------|---------|-------------|
| `data_range` | `max(y_true) - min(y_true)` | Dynamic range for `psnr` and `ssim`. |
| `sigma` | `1.5` | Gaussian kernel σ for `ssim`. |
| `bins` | `64` | Histogram bins for `mi`. |
| `k` | `5` | Number of nearest neighbours for `ksg_mi`. |

### Classification

| Metric | Key | Returns | Extra kwargs |
|--------|-----|---------|--------------|
| Accuracy | `'accuracy'` | float | — |
| Precision | `'precision'` | float | `average` |
| Recall | `'recall'` | float | `average` |
| F1 score | `'f1'` | float | `average` |

**Classification kwargs**

| kwarg | Default | Description |
|-------|---------|-------------|
| `average` | `'macro'` | scikit-learn averaging mode: `'macro'`, `'micro'`, `'weighted'`, `'binary'`. |

---

## Examples

### Segmentation — after `tigerbx.run()`

```python
import tigerbx

# Run segmentation, then evaluate ASEG against ground truth
result = tigerbx.run('a', 'T1w.nii.gz', 'output/')

scores = tigerbx.eval('gt_aseg.nii.gz', result['aseg'], 'dice',
                      labels=[10, 11, 17, 18])
# → {'dice': {'10': 0.91, '11': 0.89, '17': 0.94, '18': 0.92, 'mean': 0.915}}
```

### Multiple segmentation metrics at once

```python
scores = tigerbx.eval('gt.nii.gz', 'pred.nii.gz', ['dice', 'hd95'],
                      labels=[1, 2, 3])
# → {'dice': {'1': ..., 'mean': ...}, 'hd95': {'1': ..., 'mean': ...}}
```

### Physical-unit Hausdorff distance

```python
scores = tigerbx.eval('gt.nii.gz', 'pred.nii.gz', 'hd95',
                      labels=[17, 18], voxel_spacing=[1.0, 1.0, 1.0])
# distances in mm
```

### Reconstruction quality (e.g. after GDM distortion correction)

```python
scores = tigerbx.eval('ref.nii.gz', 'pred_gdm.nii.gz', ['psnr', 'ssim', 'ncc'])
# → {'psnr': 38.2, 'ssim': 0.96, 'ncc': 0.97}
```

### Classification

```python
import numpy as np

y_true = np.array([0, 1, 2, 1, 0])
y_pred = np.array([0, 2, 2, 1, 0])

scores = tigerbx.eval(y_true, y_pred, ['accuracy', 'f1'], average='macro')
# → {'accuracy': 0.8, 'f1': 0.78}
```

### Numpy arrays or nibabel images

```python
import nibabel as nib
import numpy as np

gt   = nib.load('gt.nii.gz')           # nibabel image
pred = np.load('pred.npy')             # numpy array

tigerbx.eval(gt, pred, 'dice', labels=[1])
```

---

## Label auto-discovery

When `labels=None` (the default), `eval` reads all unique non-zero integer values from `y_true` and evaluates all of them. This is convenient for brain segmentation where label sets are large (e.g. all 43 ASEG labels at once):

```python
scores = tigerbx.eval('gt_aseg.nii.gz', 'pred_aseg.nii.gz', 'dice')
# evaluates all non-zero labels found in gt_aseg.nii.gz
# → {'dice': {'10': ..., '11': ..., ..., 'mean': ...}}
```

Pass `labels` explicitly to restrict evaluation to a specific subset:

```python
scores = tigerbx.eval('gt_aseg.nii.gz', 'pred_aseg.nii.gz', 'dice',
                      labels=[17, 18, 53, 54])  # hippocampus + amygdala only
```
