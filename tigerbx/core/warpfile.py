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
    vfa_warp            (3, X, Y, Z) float32   VFA displacement on the template grid, in template voxels along the
                                             array axes: out(p) = affined(p + vfa_warp(p)) (core.warp.dense_transform)
    vxm_warp            (3, X, Y, Z) float32   VMnet displacement (plan step 'V'), same convention
    c2fvit_matrix       C2FViT affine matrix (affine_type='C2FViT')
    c2fvit_init_shift   (3,) float32   C2FViT initial translation (core.warp.init_shift)

A map on the input grid goes to the template in one sample, numreg.resample(input, template, affine_M,
warp=vfa_warp) (tigerbx.transform); its VBM modulation is the Jacobian determinant of vfa_warp
(core.deform.jacobian_determinant), and det(affine_M[:3, :3]) is the affine volume ratio input / template.
"""
import numpy as np

FORMAT = 'tigerbx-warp/2'
_ARRAYS = {
    'affine_M': np.float64, 'rigid_M': np.float64, 'vfa_warp': np.float32, 'vxm_warp': np.float32,
    'c2fvit_matrix': np.float32, 'c2fvit_init_shift': np.float32,
}


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
    return d
