# `tigerbx.eval` — Image Quality and Segmentation Metrics

Computes quantitative metrics between a ground-truth and a predicted image.
Accepts **NIfTI file paths, nibabel images, or numpy arrays** — consistent with all other tigerbx functions.

```python
import tigerbx

scores = tigerbx.eval(y_true, y_pred, metrics, **kwargs)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | str / nibabel image / ndarray | Ground truth |
| `y_pred` | str / nibabel image / ndarray | Prediction |
| `metrics` | str or list[str] | Metric name(s) to compute |
| `**kwargs` | — | Forwarded to each metric (see below) |

**Returns:** `dict` keyed by metric name.
Segmentation metrics return a nested dict keyed by label string, with an extra `'mean'` key.

---

## Supported metrics

| Category | Keys | Notes |
|---|---|---|
| Segmentation | `dice`, `iou`, `hd95`, `asd` | Per-label + mean |
| Reconstruction | `mae`, `mse`, `psnr`, `ssim`, `ncc`, `mi`, `ksg_mi` | Scalar |
| Classification | `accuracy`, `precision`, `recall`, `f1` | Scalar |

**Key kwargs**

| kwarg | Used by | Default | Description |
|-------|---------|---------|-------------|
| `labels` | seg | `None` | Label values to evaluate. `None` → all unique non-zero values in `y_true` |
| `ignore_background` | seg | `True` | Exclude label 0 from auto-discovery |
| `voxel_spacing` | `hd95`, `asd` | `None` | Physical voxel size (mm), e.g. `[1.0, 1.0, 1.0]`; `None` = voxel units |
| `data_range` | `psnr`, `ssim` | auto | Signal range; defaults to `max(y_true) - min(y_true)` |
| `average` | `precision`, `recall`, `f1` | `'macro'` | scikit-learn averaging: `'macro'`, `'micro'`, `'weighted'`, `'binary'` |

---

## When to use

Use `tigerbx.eval` after any tigerbx pipeline to get quantitative scores without switching tools.

**After `tigerbx.run()`** — validate segmentation against a ground truth atlas:
```python
result = tigerbx.run('a', 'T1w.nii.gz', 'output/')
scores = tigerbx.eval('gt_aseg.nii.gz', result['aseg'], ['dice', 'hd95'],
                      labels=[10, 11, 17, 18],
                      voxel_spacing=[1.0, 1.0, 1.0])
# {'dice': {'10': 0.91, '17': 0.94, ..., 'mean': 0.915},
#  'hd95': {'10': 1.2,  '17': 0.9,  ..., 'mean': 1.25}}
```

**After `tigerbx.gdm()`** — measure reconstruction quality:
```python
tigerbx.gdm('dti.nii.gz', 'output/')
scores = tigerbx.eval('ref_b0.nii.gz', 'output/dti_gdm.nii.gz',
                      ['psnr', 'ssim', 'ncc'])
# {'psnr': 38.2, 'ssim': 0.96, 'ncc': 0.97}
```

**After `tigerbx.nerve('ep', ...)`** — evaluate VAE encoding fidelity:
```python
tigerbx.nerve('ep', 'T1w.nii.gz', 'nerve_out/')
scores = tigerbx.eval('nerve_out/T1w_nerve_patch.nii.gz',
                      'nerve_out/T1w_nerve_recon.nii.gz',
                      ['mae', 'ssim', 'psnr'])
```

---

## Label auto-discovery

When `labels=None`, all unique non-zero values in `y_true` are evaluated — useful for large label sets:

```python
# Evaluates all 43 ASEG labels at once
scores = tigerbx.eval('gt_aseg.nii.gz', 'pred_aseg.nii.gz', 'dice')
# {'dice': {'10': ..., '11': ..., ..., 'mean': ...}}
```

Restrict to a subset by passing `labels` explicitly:

```python
scores = tigerbx.eval('gt_aseg.nii.gz', 'pred_aseg.nii.gz', 'dice',
                      labels=[17, 18, 53, 54])  # hippocampus + amygdala only
```

---

## Multiple metrics at once

```python
scores = tigerbx.eval('gt.nii.gz', 'pred.nii.gz',
                      ['dice', 'hd95', 'asd'],
                      labels=[1, 2, 3],
                      voxel_spacing=[1.0, 1.0, 1.0])
```
