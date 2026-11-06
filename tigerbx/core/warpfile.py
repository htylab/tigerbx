# Registration file (_warp.npz) shared by tigerbx.reg, tigerbx.affine / tigerbx.norm and the tigerbx executable.
"""Plain-array registration file, loadable with ``np.load(path)`` (no pickle).

Keys (arrays only; a key is absent when its step did not run):

    format              'tigerbx-warp/2'
    template_affine     (4, 4) float64   vox2world of the template grid (reordered, as tigerbx.reg uses it)
    template_shape      (3,) int64
    input_affine        (4, 4) float64   vox2world of the input image as stored on disk
    input_shape         (3,) int64
    affine_M            (4, 4) float64   numreg affine: template world -> input world (y_input = M @ x_template)
    rigid_M             (4, 4) float64   numreg rigid (plan step 'R'), same convention
    treg_warp           (3, X, Y, Z) float32   tReg displacement on the template grid, in template voxels along the
                                             array axes: out(p) = affined(p + treg_warp(p)) (core.warp.dense_transform)

A map on the input grid goes to the template in one sample, numreg.resample(input, template, affine_M,
warp=treg_warp) (tigerbx.transform); its VBM modulation is the Jacobian determinant of treg_warp
(core.deform.jacobian_determinant), and det(affine_M[:3, :3]) is the affine volume ratio input / template.
Files of 0.3.0 - 0.3.3 hold the tReg displacement under its old key `vfa_warp`; load_warp returns it as `treg_warp`.
Their VMnet (`vxm_warp`) and C2FViT (`c2fvit_matrix`) registrations are no longer read (both removed in 0.5.0).
"""
import numpy as np

FORMAT = 'tigerbx-warp/2'
_ARRAYS = {
    'affine_M': np.float64, 'rigid_M': np.float64, 'treg_warp': np.float32,
}
_REMOVED = {'vxm_warp': 'VMnet', 'c2fvit_matrix': 'C2FViT'}    # steps of 0.3.x files removed in 0.5.0


def save_warp(path, template_affine, template_shape, input_affine, input_shape, **arrays):
    """Write a registration file; `arrays` are keys of the format above (None values are left out)."""
    unknown = set(arrays) - set(_ARRAYS)
    if unknown:
        raise ValueError(f'unknown warp file keys: {sorted(unknown)}')
    out = dict(format=np.array(FORMAT),
               template_affine=np.asarray(template_affine, np.float64),
               template_shape=np.asarray(template_shape[:3], np.int64),
               input_affine=np.asarray(input_affine, np.float64),
               input_shape=np.asarray(input_shape[:3], np.int64))
    out.update({k: np.asarray(v, _ARRAYS[k]) for k, v in arrays.items() if v is not None})
    np.savez(path, **out)
    return path


def load_warp(path):
    """Read a registration file -> dict of arrays (keys as above). Raises ValueError for any other file,
    e.g. the pickled warp.npz of earlier tigerbx versions (re-run the registration for a new one)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
    except ValueError as exc:                    # object arrays: the old pickled warp.npz
        raise ValueError(f'{path}: not a {FORMAT} file (the pickled warp.npz of earlier tigerbx versions is no '
                         'longer read; re-run the registration)') from exc
    if 'format' not in d or str(d['format']) != FORMAT:
        raise ValueError(f'{path}: not a {FORMAT} file')
    for key, step in _REMOVED.items():
        if key in d:
            raise ValueError(f"{path}: holds a {step} registration, removed in 0.5.0; re-run the registration "
                             "(plan 'AD')")
    if 'vfa_warp' in d:                          # the key's name up to 0.3.3
        d['treg_warp'] = d.pop('vfa_warp')
    return d
