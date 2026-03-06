# `tigerbx.core.spatial` — Spatial Operations (Crop, Pad, Resize)

Array-level spatial operations for cropping, padding, and bounding-box manipulation.

---

## Padding

### `pad_to_shape(img, target_shape)`

Symmetrically zero-pad an array to reach the target shape.

| Parameter      | Type    | Description |
|----------------|---------|-------------|
| `img`          | ndarray | Input array |
| `target_shape` | tuple   | Desired output shape |

**Returns:** `(np.ndarray, list)` — `(padded_img, pad_width)`.

`pad_width` is a list of `(before, after)` tuples per axis, needed by `remove_padding()`.

```python
from tigerbx.core.spatial import pad_to_shape, remove_padding

padded, pw = pad_to_shape(data, (256, 256, 256))
# ... process padded ...
result = remove_padding(output, pw)
```

---

### `remove_padding(padded_img, pad_width)`

Remove padding previously applied by `pad_to_shape`.

| Parameter    | Type    | Description |
|--------------|---------|-------------|
| `padded_img` | ndarray | Padded array |
| `pad_width`  | list    | `pad_width` from `pad_to_shape` |

**Returns:** `np.ndarray`

---

## Cropping

### `crop_image(image, target_shape)`

Centrally crop an array to the target shape.

| Parameter      | Type    | Description |
|----------------|---------|-------------|
| `image`        | ndarray | Input array |
| `target_shape` | tuple   | Desired output shape |

**Returns:** `(np.ndarray, list)` — `(cropped_image, crop_slices)`.

---

### `crop_cube(ABC, tbetmask_image, padding=16, min_size=None)`

Crop the 3-D region where `tbetmask_image > 0`, with optional padding and minimum size enforcement.

| Parameter         | Type           | Default | Description |
|-------------------|----------------|---------|-------------|
| `ABC`             | ndarray        | —       | Source 3-D array to crop |
| `tbetmask_image`  | ndarray        | —       | Binary mask defining the region of interest |
| `padding`         | int            | `16`    | Voxels of padding around the bounding box |
| `min_size`        | tuple or None  | `None`  | Minimum crop size per axis, e.g. `(160, 160, 160)` |

**Returns:** `(np.ndarray, list)` — `(cube, xyz6)`.

`xyz6` is `[x_min, x_max, y_min, y_max, z_min, z_max]` (inclusive bounds), used by `restore_result()`.

```python
from tigerbx.core.spatial import crop_cube, restore_result

cube, xyz6 = crop_cube(image, mask, padding=16, min_size=(160, 160, 160))
# ... process cube ...
output = restore_result(image.shape, processed_cube, xyz6)
```

---

### `restore_result(ABC_shape, result, xyz6)`

Place a processed cube back into a zero-filled array with the original shape.

| Parameter  | Type    | Description |
|------------|---------|-------------|
| `ABC_shape`| tuple   | Original image shape `(X, Y, Z)` |
| `result`   | ndarray | Processed cube |
| `xyz6`     | list    | Bounding box from `crop_cube` |

**Returns:** `np.ndarray` — full-size array with the cube placed at the correct position.

---

## Resize

### `resize_with_pad_or_crop(image, image_size)`

Resize an array to exactly `image_size` by combining central cropping (if larger) and zero-padding (if smaller) per axis.

| Parameter    | Type    | Description |
|--------------|---------|-------------|
| `image`      | ndarray | Input array |
| `image_size` | tuple   | Target shape |

**Returns:** `np.ndarray`
