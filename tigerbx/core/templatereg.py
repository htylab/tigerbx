"""Template registration building blocks: the template, the numreg affine and the VFA warp of a brain image.

    tpl = template()                              # bundled 1 mm template, reordered (the grid of every output)
    M = affine(brain, tpl)                        # template world -> brain world (numreg, Mattes MI)
    warp = vfa_warp(brain, tpl, M)                # (3, X, Y, Z) displacement in template voxels
    img = numreg.resample(brain, tpl, M, warp=warp, order=3)   # brain on the template, one sample

Used by the affine / norm pipelines and tigerbx.reg. Contrast-agnostic up to the VFA model (a T1 model).
"""
import numpy as np

from tigerbx.core import models, numreg
from tigerbx.core.onnx import predict
from tigerbx.core.resample import reorder_img

VFA_MODEL = models.VFA


def template(path=None):
    """The registration template on its reordered (diagonal) grid: bundled, or `path` (as tigerbx.reg)."""
    from tigerbx import lib_reg
    return reorder_img(lib_reg.get_template(path), resample='continuous')


def affine(brain, template_nib):
    """numreg affine of `brain` (skull-stripped image, any orientation) to the template: M maps template world
    to brain world. Registers the reordered image (registering on the input grid did not help)."""
    r = reorder_img(brain, resample='continuous')
    return numreg.register(r, template_nib, moving_data=r.get_fdata().astype(np.float32))


def vfa_input(brain, template_nib, M):
    """The VFA moving image: `brain` through M, trilinear from its own grid (as VFA expects), min-max."""
    from tigerbx.lib_reg import min_max_norm
    return min_max_norm(numreg.resample(brain, template_nib, M, order=1)).astype(np.float32)


def vfa_grid(moving, template_nib, model=None, gpu=False, fixed=None, session=None):
    """VFA voxel sampling grid (3, X, Y, Z) of `moving` (min-max image on the template grid) to the template, or to
    `fixed` (an image on the template grid) when given. `session`: a reused ONNX Runtime session of the model."""
    from tigerbx import lib_tool
    from tigerbx.lib_reg import min_max_norm
    f = template_nib.get_fdata() if fixed is None else fixed
    f = min_max_norm(np.asarray(f, np.float32)[None, None]).astype(np.float32)
    return np.squeeze(predict(lib_tool.get_model(model or VFA_MODEL), [moving[None, None], f], GPU=gpu, mode='reg',
                              session=session)[0])


def vfa_warp(brain, template_nib, M, model=None, gpu=False):
    """VFA displacement (3, X, Y, Z) in template voxels after the affine M: warp = sampling grid - identity."""
    grid = vfa_grid(vfa_input(brain, template_nib, M), template_nib, model, gpu)
    identity = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float32) for n in grid.shape[1:]], indexing='ij'))
    return grid - identity


def identity_grid(shape):
    return np.stack(np.meshgrid(*[np.arange(n, dtype=np.float32) for n in shape], indexing='ij'))


def warp_on_grid(data, warp, order=1):
    """An image on the template grid moved by a displacement on that grid: out(p) = data(p + warp(p))."""
    from scipy import ndimage
    c = identity_grid(data.shape) + warp
    return ndimage.map_coordinates(np.asarray(data, np.float32), c, order=order, mode='constant').astype(np.float32)


def invert_warp(warp, iters=20):
    """Displacement v with (p + v) -> (p + v) + warp(p + v) = p, by fixed-point iteration v = -warp(p + v)."""
    from scipy import ndimage
    idg = identity_grid(warp.shape[1:])
    v = -np.asarray(warp, np.float32)
    for _ in range(iters):
        c = idg + v
        v = -np.stack([ndimage.map_coordinates(warp[a], c, order=1, mode='nearest') for a in range(3)])
    return v.astype(np.float32)
