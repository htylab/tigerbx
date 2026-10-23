"""Registration resampling in numpy/scipy, replacing the transform ONNX models.

The former models (mprage_transform_v002_{near,bili}, mprage_affinetransform_v002_{near,bili})
held no learned weights: they were exported torch grid_sample graphs. These functions reproduce them:
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


def init_shift(init_flow):
    """Translation (x, y, z order, normalised [-1, 1] units) encoded in a C2FViT init_flow.

    init_flow is the identity grid plus this shift, (1, D, H, W, 3) or (D, H, W, 3); a (3,) shift is
    returned unchanged.
    """
    flow = np.asarray(init_flow, dtype=np.float32)
    if flow.shape == (3,):
        return flow
    flow = flow.reshape(flow.shape[-4:])
    axes = [np.linspace(-1, 1, n, dtype=np.float32) for n in flow.shape[:3]]
    grid = np.meshgrid(*axes, indexing='ij')[::-1]                      # x, y, z = axes 2, 1, 0
    return np.array([(flow[..., i] - grid[i]).mean() for i in range(3)], np.float32)


def affine_transform(vol, init_flow, affine_matrix, interpolation='linear'):
    """C2FViT affine/rigid apply: translate by init_flow, then resample on affine_grid(affine_matrix).

    vol : (D, H, W); init_flow : init_flow field or its (3,) shift (see :func:`init_shift`);
    affine_matrix : (3, 4) or (1, 3, 4) theta in normalised (x, y, z) coordinates. Both steps resample,
    as the original model did.
    """
    vol = np.asarray(vol, dtype=np.float32)
    shape = np.array(vol.shape, np.float32)
    nearest = _is_nearest(interpolation)
    theta = np.asarray(affine_matrix, dtype=np.float32).reshape(3, 4)
    shift = init_shift(init_flow)[::-1]                                  # -> array axis order

    def to_voxel(g):                                                     # normalised (axis order) -> voxel
        return (g + 1) * ((shape - 1) / 2)[:, None, None, None]

    axes = [np.linspace(-1, 1, n, dtype=np.float32) for n in vol.shape]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'))                   # (3, D, H, W) in axis order
    moved = _sample(vol, to_voxel(grid + shift[:, None, None, None]), nearest)
    xyz1 = np.concatenate([grid[::-1], np.ones_like(grid[:1])])          # x, y, z, 1
    g = np.einsum('ij,j...->i...', theta, xyz1)[::-1]                    # affine_grid -> axis order
    return _sample(moved, to_voxel(g), nearest)
