# `tigerbx.core` — Internal API Reference

The `core` package contains shared utilities used by all TigerBx modules. These are internal APIs — stable within a minor version but not part of the public `tigerbx.*` interface.

## Modules

| Module | Description | Doc |
|--------|-------------|-----|
| `core.io` | NIfTI I/O, path templates, input resolution | [io.md](io.md) |
| `core.onnx` | ONNX session management, inference modes, patch inference | [onnx.md](onnx.md) |
| `core.resample` | Image resampling, reorientation (no nilearn dependency) | [resample.md](resample.md) |
| `core.spatial` | Cropping, padding, bounding box operations | [spatial.md](spatial.md) |
| `core.metrics` | Segmentation, reconstruction, and classification metrics | [metrics.md](metrics.md) |
| `core.deform` | Displacement fields, Jacobian determinants, image warping | [deform.md](deform.md) |

## Usage from Feature Modules

```python
# I/O
from tigerbx.core.io import get_template, save_nib, resolve_inputs

# Inference
from tigerbx.core.onnx import create_session, predict

# Resampling
from tigerbx.core.resample import resample_to_img, reorder_img, resample_voxel

# Spatial ops
from tigerbx.core.spatial import crop_cube, restore_result

# Metrics (prefer tigerbx.eval() for end-user code)
from tigerbx.core.metrics import dice, ssim
```
