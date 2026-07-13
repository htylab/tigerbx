# Image and segmentation metrics

```python
scores = tigerbx.eval(y_true, y_pred, metrics, **kwargs)
```

Inputs may be NIfTI paths, nibabel images, or NumPy arrays. Output is keyed by metric name.

| Group | Metrics | Useful kwargs |
|---|---|---|
| Segmentation | `dice`, `iou`, `hd95`, `asd` | `labels`, `ignore_background`, `voxel_spacing` |
| Reconstruction | `mae`, `mse`, `psnr`, `ssim`, `ncc`, `mi`, `ksg_mi` | `data_range`, `sigma`, `bins`, `k` |
| Classification | `accuracy`, `precision`, `recall`, `f1` | `average` |

```python
seg = tigerbx.eval('gt.nii.gz', 'pred.nii.gz', ['dice', 'hd95'],
                   labels=[17, 18, 53, 54], voxel_spacing=[1, 1, 1])
iqa = tigerbx.eval('ref.nii.gz', 'pred.nii.gz', ['psnr', 'ssim', 'ncc'])
```

Rules:

- With `labels=None`, discover nonzero labels from the union of `y_true` and `y_pred`.
- Segmentation results are `{label_string: score, 'mean': score}`.
- `hd95`/`asd` use voxels unless `voxel_spacing` is supplied.
- SSIM returns `1.0` for identical constant images and `0.0` for unequal constants.
- KSG MI requires equal-length finite inputs and `1 <= k < sample_count`.
- The facade filters kwargs not accepted by a requested metric.
