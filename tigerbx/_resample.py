"""
Resampling utilities for NIfTI images.

Reimplements the core functionality of:
  - nilearn.image.reorder_img
  - nilearn.image.resample_img  (resample to target voxel size)
  - nilearn.image.resample_to_img

Dependencies: numpy, scipy, nibabel  (NO nilearn)
"""

import warnings

import nibabel as nib
import numpy as np
from scipy import linalg
from scipy.ndimage import affine_transform


# ---------------------------------------------------------------------------
# Affine helpers
# ---------------------------------------------------------------------------

def to_matrix_vector(transform):
    """Split a 4x4 homogeneous transform into (3x3 matrix, 3-vector)."""
    ndimin = transform.shape[0] - 1
    ndimout = transform.shape[1] - 1
    matrix = transform[:ndimin, :ndimout]
    vector = transform[:ndimin, ndimout]
    return matrix, vector


def from_matrix_vector(matrix, vector):
    """Combine a 3x3 matrix and a 3-vector into a 4x4 homogeneous matrix."""
    nin, nout = matrix.shape
    t = np.zeros((nin + 1, nout + 1), matrix.dtype)
    t[:nin, :nout] = matrix
    t[nin, nout] = 1.0
    t[:nin, nout] = vector
    return t


def get_bounds(shape, affine):
    """Return world-space bounding box for an array given its affine.

    Parameters
    ----------
    shape : tuple of 3 ints
    affine : (4, 4) ndarray

    Returns
    -------
    list of (min, max) for each axis
    """
    adim, bdim, cdim = shape[:3]
    adim -= 1
    bdim -= 1
    cdim -= 1
    box = np.array([
        [0.0, 0, 0, 1],
        [adim, 0, 0, 1],
        [0, bdim, 0, 1],
        [0, 0, cdim, 1],
        [adim, bdim, 0, 1],
        [adim, 0, cdim, 1],
        [0, bdim, cdim, 1],
        [adim, bdim, cdim, 1],
    ]).T
    box = np.dot(affine, box)[:3]
    return list(zip(box.min(axis=-1), box.max(axis=-1)))


# ---------------------------------------------------------------------------
# Core resample
# ---------------------------------------------------------------------------

def _resample_one_img(data, A, b, target_shape, interpolation_order,
                      out, fill_value=0):
    """Resample a single 3-D volume using scipy.ndimage.affine_transform."""
    affine_transform(
        data,
        A,
        offset=b,
        output_shape=target_shape,
        output=out,
        cval=fill_value,
        order=interpolation_order,
    )
    return out


def resample_img(img, target_affine=None, target_shape=None,
                 interpolation='continuous', fill_value=0, clip=True,
                 order='F', copy_header=True):
    """Resample a NIfTI image to a given affine / shape.

    Parameters
    ----------
    img : nibabel.Nifti1Image or str/path
        Input image (or filename).
    target_affine : (3,3) or (4,4) ndarray, optional
        The target affine.  If (3,3), offset and shape are inferred
        from the tightest bounding box.
    target_shape : tuple of 3 ints, optional
        Target grid dimensions.  Required if target_affine is (4,4).
    interpolation : str
        'continuous' (order=3), 'linear' (order=1), or 'nearest' (order=0).
    fill_value : float
        Value for voxels outside the source FOV.
    clip : bool
        Clip resampled data to [min(source), max(source)].
    order : 'F' or 'C'
        Memory layout of the output array.
    copy_header : bool
        Copy the header from the source image.
    Returns
    -------
    nibabel.Nifti1Image
    """
    if isinstance(img, (str,)):
        img = nib.load(img)

    # --- no-op shortcut -------------------------------------------------
    if target_affine is None and target_shape is None:
        return img

    if target_shape is not None and target_affine is None:
        raise ValueError(
            "If target_shape is specified, target_affine should be specified too."
        )

    target_affine = np.asarray(target_affine, dtype=np.float64)

    if target_shape is not None and len(target_shape) != 3:
        raise ValueError(
            f"target_shape must have length 3, got {target_shape}"
        )

    # --- load data -------------------------------------------------------
    data = np.asarray(img.dataobj)
    src_affine = img.affine.copy()

    # --- embed 3x3 into 4x4 if needed -----------------------------------
    if target_affine.shape == (3, 3):
        missing_offset = True
        tmp = np.eye(4)
        tmp[:3, :3] = target_affine
        target_affine = tmp
    else:
        missing_offset = False
        target_affine = target_affine.copy()

    # --- bounding box in target voxel space ------------------------------
    transform_affine = np.linalg.inv(target_affine).dot(src_affine)
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = get_bounds(
        data.shape[:3], transform_affine
    )

    if missing_offset:
        offset = target_affine[:3, :3].dot([xmin, ymin, zmin])
        target_affine[:3, 3] = offset
        (xmin, xmax) = (0, xmax - xmin)
        (ymin, ymax) = (0, ymax - ymin)
        (zmin, zmax) = (0, zmax - zmin)

    if target_shape is None:
        target_shape = (
            int(np.ceil(xmax)) + 1,
            int(np.ceil(ymax)) + 1,
            int(np.ceil(zmax)) + 1,
        )

    if xmax < 0 or ymax < 0 or zmax < 0:
        raise ValueError(
            "Target affine FOV does not contain any of the source data."
        )

    # --- compute composite affine  src_vox -> tgt_vox --------------------
    if np.allclose(target_affine, src_affine):
        composite = np.eye(4)
    else:
        composite = linalg.inv(src_affine).dot(target_affine)

    A, b = to_matrix_vector(composite)

    # --- interpolation order ---------------------------------------------
    interp_map = {'continuous': 3, 'linear': 1, 'nearest': 0}
    if interpolation not in interp_map:
        raise ValueError(
            f"interpolation must be one of {list(interp_map)}, got '{interpolation}'"
        )
    interpolation_order = interp_map[interpolation]

    # --- output dtype ----------------------------------------------------
    out_dtype = data.dtype
    if interpolation == 'continuous' and data.dtype.kind == 'i':
        out_dtype = np.float64
        warnings.warn(
            f"Casting data from {data.dtype} to {out_dtype} for continuous interpolation."
        )

    # --- allocate output -------------------------------------------------
    target_shape = tuple(int(s) for s in target_shape)
    other_shape = list(data.shape[3:])
    resampled = np.zeros(
        list(target_shape) + other_shape, dtype=out_dtype, order=order
    )

    # If A is diagonal, pass it as 1-D for a faster path in scipy
    if np.all(np.diag(np.diag(A)) == A):
        A = np.diag(A)

    all_img = (slice(None),) * 3
    if not other_shape:
        # Pure 3-D image
        _resample_one_img(
            data, A, b, target_shape, interpolation_order,
            out=resampled, fill_value=fill_value,
        )
    else:
        # 4-D (or higher) — iterate over extra dimensions
        for ind in np.ndindex(*other_shape):
            _resample_one_img(
                data[all_img + ind], A, b, target_shape, interpolation_order,
                out=resampled[all_img + ind], fill_value=fill_value,
            )

    # --- clip ------------------------------------------------------------
    if clip:
        vmin = min(float(np.nanmin(data)), 0)
        vmax = max(float(np.nanmax(data)), 0)
        resampled = np.clip(resampled, vmin, vmax)

    # --- build output image ----------------------------------------------
    new_img = nib.Nifti1Image(resampled, target_affine,
                              header=img.header.copy() if copy_header else None)
    # update shape-related header fields
    new_img.header.set_data_shape(resampled.shape)
    zooms = np.sqrt(np.sum(target_affine[:3, :3] ** 2, axis=0))
    new_img.header.set_zooms(tuple(zooms) + img.header.get_zooms()[3:])
    return new_img


# ---------------------------------------------------------------------------
# resample_to_img
# ---------------------------------------------------------------------------

def resample_to_img(source_img, target_img, interpolation='continuous',
                    fill_value=0, clip=False, copy_header=True):
    """Resample *source_img* onto the grid of *target_img*.

    No registration is performed — the images must already be aligned.

    Parameters
    ----------
    source_img, target_img : nibabel.Nifti1Image or str/path
    interpolation : str
    fill_value : float
    clip : bool
    copy_header : bool

    Returns
    -------
    nibabel.Nifti1Image
    """
    if isinstance(target_img, str):
        target_img = nib.load(target_img)

    target_shape = target_img.shape[:3]

    return resample_img(
        source_img,
        target_affine=target_img.affine,
        target_shape=target_shape,
        interpolation=interpolation,
        fill_value=fill_value,
        clip=clip,
        copy_header=copy_header,
    )


# ---------------------------------------------------------------------------
# reorder_img
# ---------------------------------------------------------------------------

def reorder_img(img, resample=None, copy_header=True):
    """Return image with a diagonal affine (RAS orientation).

    Axes are permuted (and possibly flipped) so that the affine becomes
    diagonal with positive entries.  If the affine contains rotations that
    cannot be undone by permutation alone, a ``ValueError`` is raised unless
    *resample* is set (in which case the image is resampled).

    Parameters
    ----------
    img : nibabel.Nifti1Image or str/path
    resample : None or str ('continuous', 'linear', 'nearest')
        If None, only permute/flip axes. Otherwise resample.
    copy_header : bool

    Returns
    -------
    nibabel.Nifti1Image
    """
    if isinstance(img, str):
        img = nib.load(img)

    affine = img.affine.copy()
    A, b = to_matrix_vector(affine)

    # Check whether the affine is a permutation matrix (up to sign)
    if not np.all((np.abs(A) > 0.001).sum(axis=0) == 1):
        # Contains rotations — cannot just permute
        if resample is None:
            raise ValueError(
                "Cannot reorder the axes: the image affine contains rotations"
            )
        # Use QR decomposition to extract voxel sizes
        Q, R = np.linalg.qr(affine[:3, :3])
        target_affine = np.diag(
            np.abs(np.diag(R))[np.abs(Q).argmax(axis=1)]
        )
        return resample_img(
            img, target_affine=target_affine, interpolation=resample,
            copy_header=copy_header,
        )

    # Permute axes so that axis_numbers are sorted
    data = np.asarray(img.dataobj)
    axis_numbers = np.argmax(np.abs(A), axis=0)

    while not np.all(np.sort(axis_numbers) == axis_numbers):
        first_inv = int(np.argmax(np.diff(axis_numbers) < 0))
        ax1 = first_inv + 1
        ax2 = first_inv
        data = np.swapaxes(data, ax1, ax2)
        perm = np.array([0, 1, 2, 3])
        perm[ax1] = ax2
        perm[ax2] = ax1
        affine = affine.T[perm].T
        A, b = to_matrix_vector(affine)
        axis_numbers = np.argmax(np.abs(A), axis=0)

    # Flip axes with negative pixdim to ensure positive diagonal
    pixdim = np.diag(A).copy()
    slices = [slice(None)] * 3
    for i in range(3):
        if pixdim[i] < 0:
            b[i] = b[i] + pixdim[i] * (data.shape[i] - 1)
            pixdim[i] = -pixdim[i]
            slices[i] = slice(None, None, -1)
    data = data[tuple(slices)]
    affine = from_matrix_vector(np.diag(pixdim), b)

    new_img = nib.Nifti1Image(data, affine,
                              header=img.header.copy() if copy_header else None)
    new_img.header.set_data_shape(data.shape)
    zooms = list(pixdim) + list(img.header.get_zooms()[3:])
    new_img.header.set_zooms(tuple(zooms))
    return new_img


# ---------------------------------------------------------------------------
# Convenience: resample to specific voxel sizes
# ---------------------------------------------------------------------------

def resample_vox_sizes(img, voxel_sizes, interpolation='continuous',
                       fill_value=0, clip=True, copy_header=True):
    """Resample *img* to the given isotropic or anisotropic voxel sizes.

    This is a convenience wrapper around ``resample_img`` that constructs
    a RAS-aligned target affine with the requested voxel sizes and lets
    ``resample_img`` compute the bounding box / shape automatically.

    Parameters
    ----------
    img : nibabel.Nifti1Image or str/path
    voxel_sizes : float or (3,) sequence
        Desired voxel sizes in mm.  A scalar means isotropic.
    interpolation : str
    fill_value : float
    clip : bool
    copy_header : bool

    Returns
    -------
    nibabel.Nifti1Image
    """
    if isinstance(img, str):
        img = nib.load(img)

    voxel_sizes = np.atleast_1d(np.asarray(voxel_sizes, dtype=np.float64))
    if voxel_sizes.size == 1:
        voxel_sizes = np.repeat(voxel_sizes, 3)

    # Build a 3x3 target affine → resample_img will infer offset & shape
    target_affine = np.diag(voxel_sizes)

    return resample_img(
        img,
        target_affine=target_affine,
        interpolation=interpolation,
        fill_value=fill_value,
        clip=clip,
        copy_header=copy_header,
    )
