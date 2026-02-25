import numpy as np
from scipy.ndimage import map_coordinates


def edge_padded_central_diff(arr, axis):
    pads = [(0, 0)] * arr.ndim
    pads[axis] = (1, 1)
    arr_pad = np.pad(arr, pads, mode="edge")

    sl_forward = [slice(None)] * arr.ndim
    sl_backward = [slice(None)] * arr.ndim
    sl_forward[axis] = slice(2, None)
    sl_backward[axis] = slice(None, -2)
    return (arr_pad[tuple(sl_forward)] - arr_pad[tuple(sl_backward)]) * 0.5


def jacobian_det_displacement_2d(disp):
    # disp components follow SITK vector order [x, y], while numpy axes are [y, x].
    ux = disp[..., 0]
    uy = disp[..., 1]
    dux_dy = edge_padded_central_diff(ux, axis=0)
    dux_dx = edge_padded_central_diff(ux, axis=1)
    duy_dy = edge_padded_central_diff(uy, axis=0)
    duy_dx = edge_padded_central_diff(uy, axis=1)
    return (1.0 + dux_dx) * (1.0 + duy_dy) - dux_dy * duy_dx


def jacobian_det_displacement_3d(disp):
    # disp components follow SITK vector order [x, y, z], numpy axes are [z, y, x].
    ux = disp[..., 0]
    uy = disp[..., 1]
    uz = disp[..., 2]

    dux_dz = edge_padded_central_diff(ux, axis=0)
    dux_dy = edge_padded_central_diff(ux, axis=1)
    dux_dx = edge_padded_central_diff(ux, axis=2)

    duy_dz = edge_padded_central_diff(uy, axis=0)
    duy_dy = edge_padded_central_diff(uy, axis=1)
    duy_dx = edge_padded_central_diff(uy, axis=2)

    duz_dz = edge_padded_central_diff(uz, axis=0)
    duz_dy = edge_padded_central_diff(uz, axis=1)
    duz_dx = edge_padded_central_diff(uz, axis=2)

    a11 = 1.0 + dux_dx
    a12 = dux_dy
    a13 = dux_dz
    a21 = duy_dx
    a22 = 1.0 + duy_dy
    a23 = duy_dz
    a31 = duz_dx
    a32 = duz_dy
    a33 = 1.0 + duz_dz

    return (
        a11 * (a22 * a33 - a23 * a32)
        - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31)
    )


def warp_displacement_linear_sitk_like(image, disp):
    """Linear warp that matches SITK DisplacementFieldTransform + sitkLinear in index space."""
    image = np.asarray(image, dtype=np.float64)
    disp = np.asarray(disp, dtype=np.float64)

    if disp.shape[:-1] != image.shape:
        raise ValueError(
            f"displacement shape {disp.shape[:-1]} does not match image shape {image.shape}"
        )
    if disp.shape[-1] != image.ndim:
        raise ValueError(
            f"displacement last dim {disp.shape[-1]} must equal image ndim {image.ndim}"
        )

    coords = np.indices(image.shape, dtype=np.float64)
    coords = coords + np.moveaxis(disp[..., ::-1], -1, 0)

    valid = np.ones(image.shape, dtype=bool)
    coords_clipped = coords.copy()
    for axis, size in enumerate(image.shape):
        valid &= (coords[axis] >= -0.5) & (coords[axis] <= size - 0.5)
        coords_clipped[axis] = np.clip(coords_clipped[axis], 0.0, size - 1.0)

    warped = map_coordinates(
        image,
        coords_clipped,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    warped[~valid] = 0.0
    return warped


def build_vdm_displacement_2d(vdm, readout=1, AP_RL="AP"):
    if AP_RL == "AP":
        return np.stack([vdm * readout, vdm * 0], axis=-1)
    return np.stack([vdm * 0, vdm * readout], axis=-1)


def build_vdm_displacement_3d(vdm, readout=1, AP_RL="AP"):
    if AP_RL == "AP":
        return np.stack([vdm * 0, vdm * readout, vdm * 0], axis=-1)
    return np.stack([vdm * 0, vdm * 0, vdm * readout], axis=-1)


def apply_vdm_2d(ima, vdm, readout=1, AP_RL="AP"):
    disp = build_vdm_displacement_2d(
        np.asarray(vdm, dtype=np.float64), readout=readout, AP_RL=AP_RL
    )
    new_ima = warp_displacement_linear_sitk_like(ima, disp)
    jac_np = jacobian_det_displacement_2d(disp)
    return new_ima * jac_np


def apply_vdm_3d(ima, vdm, readout=1, AP_RL="AP"):
    disp = build_vdm_displacement_3d(
        np.asarray(vdm, dtype=np.float64), readout=readout, AP_RL=AP_RL
    )
    new_ima = warp_displacement_linear_sitk_like(ima, disp)
    jac_np = jacobian_det_displacement_3d(disp)
    return new_ima * jac_np


def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), "flow has to be 2D or 3D"

    grid_lst = np.meshgrid(*[np.arange(s) for s in volshape], indexing="ij")
    grid = np.stack(grid_lst, axis=-1)

    J = np.gradient(disp + grid, axis=tuple(range(nb_dims)))

    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    dfdx = J[0]
    dfdy = J[1]
    return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

