"""Refinement of a tReg sampling grid between two images of the same contrast (one ONNX model, CPU or GPU).

    g = refine(moving, fixed, g0)                 # moving(g(p)) ~ fixed(p), g0 a tReg grid (3, X, Y, Z) in voxels

Greedy local-CC force (window 9^3), the update u smoothed with a Gaussian (sigma sqrt 3 voxels) and scaled to a
largest step of 0.2 voxel, composed with the grid (g <- g(x + u): each step invertible, so no folding); the model
runs 10 iterations per call, 2 calls by default. Inputs on the model grid (160 x 224 x 192, the bundled template).
For two T1 images (two scans of one head, or a T1 and the T1 template); between contrasts it does not help.
"""
import numpy as np

from tigerbx.core import models
from tigerbx.core.onnx import create_session


def refine(moving, fixed, grid, model=None, gpu=False, session=None, calls=2):
    """Refined sampling grid (3, X, Y, Z) of `moving` to `fixed` (images on the same grid, min-max), starting from
    `grid`; `calls` runs of the model (10 iterations each). `session`: a reused ONNX Runtime session."""
    if session is None:
        from tigerbx import lib_tool
        session = create_session(lib_tool.get_model(model or models.REFINE), gpu)
    m = np.asarray(moving, np.float32)[None, None]
    f = np.asarray(fixed, np.float32)[None, None]
    g = np.asarray(grid, np.float32)[None]
    feed = {'moving': m, 'fixed': f}
    if 'identity' in [i.name for i in session.get_inputs()]:
        feed['identity'] = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float32) for n in m.shape[2:]],
                                                indexing='ij'))[None]
    for _ in range(calls):
        g = session.run(None, dict(feed, grid=g))[0]
    return g[0]
