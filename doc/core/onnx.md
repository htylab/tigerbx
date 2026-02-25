# `tigerbx.core.onnx` — ONNX Session Management and Inference

Manages ONNX Runtime sessions and provides a unified `predict()` interface for all inference modes (standard, registration, patch-based, encode/decode).

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TIGERBX_PATCH_SIZE` | `128,128,128` | Override default patch size (e.g. `"96"` or `"96,96,96"`) |
| `TIGERBX_ORT_INTRA_THREADS` | `70% of CPUs` | ONNX Runtime intra-op thread count |
| `TIGERBX_ORT_INTER_THREADS` | `1` | ONNX Runtime inter-op thread count |

---

## Session Management

### `create_session(model_ff, GPU)`

Create an `ort.InferenceSession` with tuned thread settings.

| Parameter  | Type   | Description |
|------------|--------|-------------|
| `model_ff` | `str`  | Full path to the `.onnx` model file |
| `GPU`      | `bool` | Use CUDA if available |

**Returns:** `ort.InferenceSession`

```python
from tigerbx.core.onnx import create_session

session = create_session('/path/to/model.onnx', GPU=False)
# ... use session ...
del session  # explicit cleanup recommended
```

**Important:** Feature modules should use this function instead of creating `ort.InferenceSession` directly. Model paths are obtained via `lib_tool.get_model()`.

---

## Inference

### `predict(model, data, GPU, mode=None, patch_size=None, tile_step_size=0.5, gaussian=True, session=None)`

Unified inference entry point supporting multiple modes.

| Parameter        | Type                  | Default | Description |
|------------------|-----------------------|---------|-------------|
| `model`          | `str`                 | —       | Model file path (used only if `session` is None) |
| `data`           | `np.ndarray` or list  | —       | Input data (format depends on `mode`) |
| `GPU`            | `bool`                | —       | Use GPU |
| `mode`           | `str | None`          | `None`  | Inference mode (see below) |
| `patch_size`     | `tuple | int | None`  | `None`  | Patch size for `mode='patch'` |
| `tile_step_size` | `float`               | `0.5`   | Overlap ratio for patch mode |
| `gaussian`       | `bool`                | `True`  | Use Gaussian weighting for patch stitching |
| `session`        | `ort.InferenceSession`| `None`  | Reuse an existing session |

**Modes:**

| Mode | Data format | Returns | Used by |
|------|-------------|---------|---------|
| `None` | `[1, 1, D, H, W]` float | `[1, C, D, H, W]` array | bx segmentation |
| `'reg'` | `[moving, fixed]` list | list of arrays | registration |
| `'affine_transform'` | `[vol, flow, matrix]` list | list of arrays | affine transform |
| `'patch'` | `[1, 1, D, H, W]` float | `[1, C, D, H, W]` stitched logits | hlc, large inputs |
| `'encode'` | `[1, 1, D, H, W]` float | `(mu, sigma)` | NERVE encoder |
| `'decode'` | `[1, latent_dim]` float | reconstructed array | NERVE decoder |

```python
from tigerbx.core.onnx import create_session, predict

model_ff = lib_tool.get_model('mprage_bet_v005_mixsynthv4.onnx')
session = create_session(model_ff, GPU=False)
output = predict(model_ff, input_data, GPU=False, session=session)
del session
```

---

### `predict_single_output(session, data)`

Simplified single-output inference. Runs the session and returns the first output.

---

### `encode_latent(enc_sess, vol)`

Run a VAE encoder session. Returns `(mu, sigma)` arrays.

---

### `decode_latent(dec_sess, latent)`

Run a VAE decoder session. Returns the reconstructed output array.

---

## Patch Inference

### `patch_inference_3d_lite(session, vol_d, patch_size, tile_step_size, gaussian)`

Sliding-window patch inference with optional Gaussian weighting for smooth stitching.

| Parameter        | Type    | Default        | Description |
|------------------|---------|----------------|-------------|
| `session`        | session | —              | ONNX session |
| `vol_d`          | ndarray | —              | Input volume `[1, 1, D, H, W]` |
| `patch_size`     | tuple   | `(128,128,128)`| Patch dimensions |
| `tile_step_size` | float   | `0.5`          | Step size as fraction of patch size |
| `gaussian`       | bool    | `True`         | Use Gaussian importance weighting |

**Returns:** `np.ndarray` — stitched logits `[1, C, D, H, W]`.

---

### `compute_steps_for_sliding_window(image_size, tile_size, tile_step_size)`

Calculate sliding window starting positions for each axis.

**Returns:** `list[list[int]]` — start positions per axis.

---

### `compute_gaussian(tile_size, sigma_scale=1/8, value_scaling_factor=1, dtype=np.float16)`

Create a Gaussian importance map for patch weighting.

**Returns:** `np.ndarray` — Gaussian map with same shape as `tile_size`.

---

### `img_to_patches(vol_d, patch_size, tile_step_size)`

Extract patches from a volume using sliding window positions.

**Returns:** `(np.ndarray, list)` — `(patches_array, point_list)`.
