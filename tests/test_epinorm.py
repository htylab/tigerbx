import importlib

import numpy as np
from scipy import ndimage

from tigerbx.core import duope as core

E = importlib.import_module('tigerbx.pipelines.epinorm')


def _volume(shape=(24, 30, 20), seed=0):
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.random(shape), 2.0) * 100


def _sample(img, src, aff, shape):
    """img through src at the voxels of the grid (aff, shape), as epinorm._write (cubic, 0 outside)."""
    p = np.indices(shape).reshape(3, -1).astype(float)
    c, jac = src.coords(aff[:3, :3] @ p + aff[:3, 3:])
    return (ndimage.map_coordinates(img, c, order=3, mode='constant') * jac).reshape(shape)


def test_source_reproduces_unwarp_with_motion():
    img = _volume()
    shape = img.shape
    aff = np.diag([2.0, 2.5, 2.5, 1.0])
    aff[:3, 3] = (-20, -35, -25)
    g = np.indices(shape).astype(float)
    d = 1.5 * np.exp(-(((g[0] - 12) / 6) ** 2 + ((g[1] - 15) / 7) ** 2 + ((g[2] - 10) / 5) ** 2))
    R = np.eye(4)
    R[:3, :3] = core.Rotation.from_rotvec(np.radians([2.0, -1.0, 1.5])).as_matrix()
    R[:3, 3] = (0.8, -0.5, 0.3)
    for sign in (1.0, -1.0):
        ref = core.unwarp(img, d, aff, 1, M=R, sign=sign)
        out = _sample(img, E.Source(aff, aff, d, 1, sign, post=R), aff, shape)
        inner = (slice(4, -4),) * 3                         # the prefilters' boundary modes differ (nearest / constant)
        assert np.allclose(out[inner], ref[inner], atol=1e-4 * img.max())


def test_series_pre_moves_the_corrected_image_with_the_head():
    # a series in another head position: pre (forward world -> series world) brings its corrected image back
    img = _volume(seed=1)
    shape = img.shape
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    d = np.zeros(shape)
    shift = np.eye(4)
    shift[:3, 3] = (4.0, 0, 0)                              # series world = forward world + 4 mm along x
    out = _sample(img, E.Source(aff, aff, d, 1, 1.0, pre=shift), aff, shape)
    assert np.allclose(out[2:-4], img[4:-2], atol=1e-6 * img.max())    # 4 mm = 2 voxels


def test_resized_grid_keeps_the_field_of_view():
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    aff[:3, 3] = (-80, -110, -90)
    q, out, shape = E._resized(aff, (160, 224, 192), 2)
    assert shape == (80, 112, 96)
    assert np.allclose(np.diag(out)[:3], 2) and np.allclose(out[:3, 3], aff[:3, 3] + 0.5)
    assert np.allclose(q[:, 0], 0.5) and np.allclose(q[:, -1], (158.5, 222.5, 190.5))   # edges coincide
    q1, out1, shape1 = E._resized(aff, (160, 224, 192), 1)                               # its own voxel size
    assert shape1 == (160, 224, 192) and np.allclose(out1, aff) and np.allclose(q1[:, 7], (0, 0, 7))
