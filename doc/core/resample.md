# `tigerbx.core.resample` — Image Resampling Utilities

Pure numpy/scipy reimplementation of nilearn's core resampling functions. No nilearn dependency required.

---

## Primary Functions

### `resample_to_img(source_img, target_img, interpolation="continuous", fill_value=0, clip=False, copy_header=True)`

Resample a source image onto the grid (affine + shape) of a target image.

| Parameter      | Type                     | Default        | Description |
|----------------|--------------------------|----------------|-------------|
| `source_img`   | `nib.Nifti1Image or str` | —              | Image to resample |
| `target_img`   | `nib.Nifti1Image or str` | —              | Reference image defining the output grid |
| `interpolation`| `str`                    | `"continuous"` | `"continuous"` (cubic), `"linear"`, or `"nearest"` |
| `fill_value`   | `float`                  | `0`            | Value for voxels outside the source FOV |
| `clip`         | `bool`                   | `False`        | Clip output to source data range |
| `copy_header`  | `bool`                   | `True`         | Copy header from source |

**Returns:** `nib.Nifti1Image`

```python
from tigerbx.core.resample import resample_to_img

# Resample a mask onto the input image grid
mask_resampled = resample_to_img(mask_nib, input_nib, interpolation="nearest")
```

---

### `resample_voxel(data_nib, voxelsize, target_shape=None, interpolation="continuous")`

Resample an image to a new isotropic or anisotropic voxel size.

| Parameter      | Type                | Default        | Description |
|----------------|---------------------|----------------|-------------|
| `data_nib`     | `nib.Nifti1Image`   | —              | Input image |
| `voxelsize`    | `tuple`             | —              | Target voxel size, e.g. `(1, 1, 1)` |
| `target_shape` | `tuple | None`      | `None`         | Force output shape; `None` = compute from FOV |
| `interpolation`| `str`               | `"continuous"` | Interpolation method |

**Returns:** `nib.Nifti1Image`

```python
from tigerbx.core.resample import resample_voxel

# Resample to 1mm isotropic
img_1mm = resample_voxel(img_nib, (1, 1, 1), interpolation='continuous')
```

---

### `reorder_img(img, resample=None, copy_header=True)`

Reorder image axes to RAS orientation (diagonal affine with positive entries).

| Parameter     | Type                     | Default | Description |
|---------------|--------------------------|---------|-------------|
| `img`         | `nib.Nifti1Image or str` | —       | Input image |
| `resample`    | `str | None`             | `None`  | If the affine contains rotations, resample with this interpolation; `None` raises an error |
| `copy_header` | `bool`                   | `True`  | Copy header |

**Returns:** `nib.Nifti1Image` — image with diagonal affine.

```python
from tigerbx.core.resample import reorder_img

img_ras = reorder_img(img_nib, resample='continuous')
```

---

### `reorient_and_resample_voxel(nib_obj, voxelsize=(1,1,1), target_shape=None, interpolation="linear")`

Convenience wrapper: reorder to RAS, then resample to a target voxel size.

**Returns:** `nib.Nifti1Image`

---

### `resample_to_new_resolution(data_nii, target_resolution, target_shape=None, interpolation="continuous")`

Alias for `resample_voxel`. Provided for backward compatibility.

---

## Low-Level Helpers

### `resample_img(img, target_affine=None, target_shape=None, interpolation="continuous", fill_value=0, clip=True, order="F", copy_header=True)`

Full-featured resample: transform an image to a new affine and/or shape. This is the core function used by all higher-level resampling functions.

**Interpolation modes:**
| Mode | scipy order | Use case |
|------|-------------|----------|
| `"continuous"` | 3 (cubic) | Intensity images (T1w, FLAIR) |
| `"linear"` | 1 | Probability maps, tissue PVEs |
| `"nearest"` | 0 | Label maps, masks |

---

### `to_matrix_vector(transform)` / `from_matrix_vector(matrix, vector)`

Split / combine a 4x4 homogeneous transform and a (3x3 matrix, 3-vector) pair.

### `get_bounds(shape, affine)`

Return world-space bounding box for an array given its affine.
