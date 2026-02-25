# `tigerbx.core.deform` — Displacement Fields and Warping

Low-level displacement field operations: Jacobian determinants, image warping, and VDM (voxel displacement map) application. Used by the registration and GDM modules.

---

## Jacobian Determinant

### `jacobian_determinant(disp)`

Compute the Jacobian determinant of a displacement field using `np.gradient`.

| Parameter | Type    | Description |
|-----------|---------|-------------|
| `disp`    | ndarray | Displacement field of shape `[*vol_shape, nb_dims]` (2-D or 3-D) |

**Returns:** `np.ndarray` — scalar Jacobian determinant at each voxel.

Used by `tigerbx.reg.modulate()` for VBM Jacobian modulation.

```python
from tigerbx.core.deform import jacobian_determinant

# warp shape: (D, H, W, 3)
jac = jacobian_determinant(warp_dhwc)
modulated = volume * jac
```

---

### `jacobian_det_displacement_3d(disp)`

Jacobian determinant using edge-padded central differences. Expects SITK vector order `[x, y, z]` with numpy axes `[z, y, x]`.

| Parameter | Type    | Description |
|-----------|---------|-------------|
| `disp`    | ndarray | Displacement field `[Z, Y, X, 3]` in SITK convention |

**Returns:** `np.ndarray` — Jacobian determinant.

---

## Image Warping

### `warp_displacement_linear_sitk_like(image, disp)`

Apply a displacement field to an image using linear interpolation, matching SITK `DisplacementFieldTransform + sitkLinear` behaviour in index space.

| Parameter | Type    | Description |
|-----------|---------|-------------|
| `image`   | ndarray | Input image (2-D or 3-D) |
| `disp`    | ndarray | Displacement field `[*image.shape, ndim]` |

**Returns:** `np.ndarray` — warped image. Voxels mapped outside the FOV are set to 0.

---

## VDM (Voxel Displacement Map) Operations

Used by the GDM module for EPI distortion correction.

### `build_vdm_displacement_3d(vdm, readout=1, AP_RL="AP")`

Convert a scalar VDM to a 3-D displacement field along a specific axis.

| Parameter | Type  | Default | Description |
|-----------|-------|---------|-------------|
| `vdm`     | ndarray | —     | Scalar displacement map |
| `readout` | float | `1`     | Readout scaling factor |
| `AP_RL`   | str   | `"AP"`  | Phase-encode direction: `"AP"` (axis 1) or `"RL"` (axis 2) |

**Returns:** `np.ndarray` — 3-component displacement field `[..., 3]`.

---

### `apply_vdm_3d(ima, vdm, readout=1, AP_RL="AP")`

Apply a VDM correction to a 3-D image: warp + Jacobian intensity modulation.

| Parameter | Type    | Default | Description |
|-----------|---------|---------|-------------|
| `ima`     | ndarray | —       | Input image |
| `vdm`     | ndarray | —       | Voxel displacement map |
| `readout` | float   | `1`     | Readout scaling |
| `AP_RL`   | str     | `"AP"`  | Phase-encode direction |

**Returns:** `np.ndarray` — corrected image.

---

## Low-Level Helpers

### `edge_padded_central_diff(arr, axis)`

Compute central differences along an axis with edge-padded boundaries. Used internally by `jacobian_det_displacement_3d`.
