"""Template registration building blocks: the template, the numreg affine and the tReg warp of a brain image.

    tpl = template()                              # bundled 1 mm template, reordered (the grid of every output)
    etpl = epi_template('se')                     # bundled SE-EPI (or 'bold') template on the same grid
    M = affine(brain, tpl)                        # template world -> brain world (numreg, Mattes MI)
    warp = treg_warp(brain, tpl, M)               # (3, X, Y, Z) displacement in template voxels
    img = numreg.resample(brain, tpl, M, warp=warp, order=3)   # brain on the template, one sample

Used by the affine / norm pipelines and tigerbx.reg. Contrast-agnostic up to the tReg model (a T1 model).

The outputs are on the template's own grid. The tReg / VMnet models take 160 x 224 x 192 at 1 mm: a template on
another grid (e.g. a subject's T1) gets a box of that size on its brain (model_box), the model runs there and its
displacement is carried back onto the template grid (0 outside the box: the affine alone).
"""
import os

import nibabel as nib
import numpy as np

from tigerbx.core import models, numreg
from tigerbx.core.onnx import create_session, predict
from tigerbx.core.resample import reorder_img

TREG_MODEL = models.TREG
MODEL_SHAPE = (160, 224, 192)                  # the input grid of the tReg / VMnet / C2FViT models (1 mm)


def template(path=None):
    """The registration template on its reordered (diagonal) grid: bundled, or `path` (as tigerbx.reg)."""
    from tigerbx import lib_reg
    return reorder_img(lib_reg.get_template(path), resample='continuous')


EPI_TEMPLATES = {'se': 'mni_template_se_epi.nii.gz', 'bold': 'mni_template_bold_epi.nii.gz'}


def epi_template(kind='se'):
    """A bundled EPI template on the template grid, skull-stripped and distortion-corrected: 'se' (spin-echo EPI,
    T2-weighted: field maps, DWI b0) or 'bold' (gradient-echo BOLD mean)."""
    from tigerbx import lib_reg
    if kind not in EPI_TEMPLATES:
        raise ValueError(f'kind: {kind!r} (use {sorted(EPI_TEMPLATES)})')
    return reorder_img(nib.load(os.path.join(lib_reg.application_path, 'template', EPI_TEMPLATES[kind])),
                       resample='continuous')


def affine(brain, template_nib):
    """numreg affine of `brain` (skull-stripped image, any orientation) to the template: M maps template world
    to brain world. Registers the reordered image (registering on the input grid did not help)."""
    r = reorder_img(brain, resample='continuous')
    return numreg.register(r, template_nib, moving_data=r.get_fdata().astype(np.float32))


def treg_input(brain, template_nib, M):
    """The tReg moving image: `brain` through M, trilinear from its own grid (as the model expects), min-max; on the
    template grid or a model box (anything with affine and shape)."""
    from tigerbx.lib_reg import min_max_norm
    return min_max_norm(numreg.resample(brain, template_nib, M, order=1)).astype(np.float32)


def treg_grid(moving, template_nib, model=None, gpu=False, fixed=None, session=None):
    """tReg voxel sampling grid (3, X, Y, Z) of `moving` (min-max image on the template grid) to the template, or to
    `fixed` (an image on the template grid) when given. `session`: a reused ONNX Runtime session of the model."""
    from tigerbx import lib_tool
    from tigerbx.lib_reg import min_max_norm
    f = template_nib.get_fdata() if fixed is None else fixed
    f = min_max_norm(np.asarray(f, np.float32)[None, None]).astype(np.float32)
    if session is None:
        session = create_session(lib_tool.get_model(model or TREG_MODEL), gpu)
    return np.squeeze(predict(None, [moving[None, None], f], GPU=gpu, mode='reg', session=session)[0])


def model_box(template_nib):
    """None for a template on the model grid (160 x 224 x 192, 1 mm, axis-aligned: the bundled one); otherwise the
    grid (affine, shape) the models run on: that size at 1 mm along the template's (reordered) axes, centred on its
    non-background voxels."""
    from types import SimpleNamespace
    a = np.asarray(template_nib.affine, np.float64)
    if tuple(template_nib.shape[:3]) == MODEL_SHAPE and np.allclose(a[:3, :3], np.eye(3), atol=1e-6):
        return None
    data = np.asarray(template_nib.dataobj, np.float32)
    content = np.argwhere(data > data.min())
    middle = content.mean(0) if len(content) else (np.array(data.shape[:3]) - 1) / 2.0   # no background: the middle
    c = (a @ np.r_[middle, 1.0])[:3]
    box = np.eye(4)
    box[:3, :3] = a[:3, :3] / np.linalg.norm(a[:3, :3], axis=0)       # 1 mm along the template's axes
    box[:3, 3] = c - box[:3, :3] @ ((np.array(MODEL_SHAPE) - 1) / 2.0)
    return SimpleNamespace(affine=box, shape=MODEL_SHAPE)


def on_box(data, template_nib, box, order=1):
    """An array on the template grid sampled on the model box (0 outside the template)."""
    from scipy import ndimage
    V = np.linalg.inv(template_nib.affine) @ box.affine
    return ndimage.affine_transform(np.asarray(data, np.float32), V[:3, :3], V[:3, 3], output_shape=box.shape,
                                    order=order, mode='constant').astype(np.float32)


def warp_from_box(warp, box, template_nib):
    """A displacement on the model box (box voxels) as a displacement on the template grid (template voxels),
    linear, 0 outside the box: out(p) = affined(p + warp(p)) holds on either grid."""
    from scipy import ndimage
    a = np.asarray(template_nib.affine, np.float64)
    V = np.linalg.inv(box.affine) @ a                                  # template voxel -> box voxel
    shape = tuple(template_nib.shape[:3])
    c = np.einsum('ij,j...->i...', V[:3, :3], identity_grid(shape)) + V[:3, 3].reshape(3, 1, 1, 1)
    w = np.stack([ndimage.map_coordinates(np.asarray(warp[k], np.float32), c, order=1, mode='constant')
                  for k in range(3)])
    R = np.linalg.inv(a[:3, :3]) @ box.affine[:3, :3]                  # box voxels -> template voxels
    return np.einsum('ij,j...->i...', R, w).astype(np.float32)


def treg_warp(brain, template_nib, M, model=None, gpu=False):
    """tReg displacement (3, X, Y, Z) in template voxels after the affine M: warp = sampling grid - identity (on the
    model box when the template is on another grid, carried back to it)."""
    box = model_box(template_nib)
    if box is None:
        grid = treg_grid(treg_input(brain, template_nib, M), template_nib, model, gpu)
        return grid - identity_grid(grid.shape[1:])
    grid = treg_grid(treg_input(brain, box, M), template_nib, model, gpu,
                    fixed=on_box(template_nib.get_fdata(), template_nib, box))
    return warp_from_box(grid - identity_grid(grid.shape[1:]), box, template_nib)


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
