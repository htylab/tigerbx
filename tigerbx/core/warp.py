"""Registration resampling in numpy/scipy, replacing the transform ONNX models.

The former models (mprage_transform_v002_{near,bili}) held no learned weights: they were exported torch
grid_sample graphs. dense_transform reproduces them:
'nearest' rounds half to even and clamps out-of-range coordinates to the border, 'linear' is trilinear
with zeros outside the volume (grid_sample align_corners=True semantics).
"""
import numpy as np
from scipy.ndimage import map_coordinates


def _is_nearest(interpolation):
    return str(interpolation).lower() in {'nearest', 'nearestneighbor', 'nearest_neighbor', 'nn'}


def _sample(vol, coords, nearest):
    """Sample a 3-D volume at voxel coordinates coords (3, ...)."""
    if not nearest:
        return map_coordinates(vol, coords, order=1, mode='grid-constant', cval=0.0).astype(np.float32)
    idx = 0
    for axis, n in enumerate(vol.shape):
        i = np.clip(np.rint(coords[axis]), 0, n - 1).astype(np.intp)
        idx = idx * n + i
    return np.ascontiguousarray(vol, dtype=np.float32).ravel().take(idx)


def _identity(shape):
    return np.stack(np.meshgrid(*[np.arange(n, dtype=np.float32) for n in shape], indexing='ij'))


def dense_transform(vol, warp, interpolation='linear'):
    """VoxelMorph spatial transform: out(p) = vol(p + warp(p)).

    vol : (D, H, W) array; warp : (3, D, H, W) displacement in voxels along the array axes.
    """
    vol = np.asarray(vol, dtype=np.float32)
    return _sample(vol, _identity(vol.shape) + np.asarray(warp, dtype=np.float32), _is_nearest(interpolation))
