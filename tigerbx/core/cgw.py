"""CSF / GM / WM partial-volume maps computed from a T1 and its TSEG, without a model.

    p = tissue_pve(t1, tseg, mask, voxel_mm)        # (3, X, Y, Z) float32: CSF, GM, WM; sums to 1 in the brain

1. TSEG -> tissue class (FreeSurfer label numbers; inside the brain mask, unlabeled -> CSF, as label 99).
2. Local tissue intensities mu_k(x): Gaussian-weighted (sigma 12 mm) mean over the pure voxels of class k (the
   class eroded by one voxel), so the bias field drops out.
3. A voxel next to another class (6-neighbourhood) whose intensity lies between its own mean and that class's mean
   is a linear mix of the two: p_own = (I - mu_other) / (mu_own - mu_other). Everywhere else one class is 1.

T1 only.
"""
import numpy as np
from scipy import ndimage

GM = (3, 42, 8, 47, 10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58)
WM = (2, 41, 7, 46, 16, 28, 60, 77, 85, 251, 252, 253, 254, 255)
CSF = (4, 43, 5, 44, 14, 15, 24, 30, 31, 62, 63, 99)
CLASSES = (CSF, GM, WM)
NAMES = ('csf', 'gm', 'wm')


def tissue_classes(tseg, mask):
    """int8 class map: 0 CSF, 1 GM, 2 WM, -1 outside the brain mask."""
    c = np.full(tseg.shape, -1, np.int8)
    for k, labels in enumerate(CLASSES):
        c[np.isin(tseg, labels)] = k
    c[mask & (c < 0)] = 0
    c[~mask] = -1
    return c


def local_means(img, cls, voxel_mm, sigma_mm=12.0):
    """(3, X, Y, Z): Gaussian-weighted mean intensity of the pure (eroded) voxels of each class."""
    sigma = sigma_mm / np.asarray(voxel_mm, float)
    mu = np.zeros((3,) + img.shape, np.float32)
    for k in range(3):
        pure = ndimage.binary_erosion(cls == k)
        if pure.sum() < 50:
            pure = cls == k
        w = ndimage.gaussian_filter(pure.astype(np.float32), sigma, mode='constant')
        s = ndimage.gaussian_filter(np.where(pure, img, 0).astype(np.float32), sigma, mode='constant')
        overall = float(np.median(img[pure])) if pure.any() else 0.0
        mu[k] = np.where(w > 1e-3, s / np.maximum(w, 1e-6), overall)
    return mu


def tissue_pve(img, tseg, mask, voxel_mm, sigma_mm=12.0):
    """CSF / GM / WM fractions (3, X, Y, Z) float32 of a T1 `img` from its TSEG labels inside the brain `mask`.
    Computed on the brain's bounding box (+ 1 voxel): the same result, since everything outside the brain is 0."""
    img = np.asarray(img, np.float32)
    mask = np.asarray(mask) > 0
    p = np.zeros((3,) + img.shape, np.float32)
    if not mask.any():
        return p
    box = tuple(slice(max(int(i.min()) - 1, 0), int(i.max()) + 2) for i in np.nonzero(mask))
    p[(slice(None),) + box] = _tissue_pve(img[box], np.asarray(tseg)[box], mask[box], voxel_mm, sigma_mm)
    return p


def _tissue_pve(img, tseg, mask, voxel_mm, sigma_mm):
    cls = tissue_classes(np.asarray(tseg).astype(int), mask)
    mu = local_means(img, cls, voxel_mm, sigma_mm)
    p = np.zeros((3,) + img.shape, np.float32)
    for k in range(3):
        p[k][cls == k] = 1
    near = [ndimage.binary_dilation(cls == k, ndimage.generate_binary_structure(3, 1)) for k in range(3)]
    for own in range(3):
        for other in range(3):
            cand = (cls == own) & near[other] if other != own else None
            if cand is None or not cand.any():
                continue
            mo, mx, x = mu[own][cand], mu[other][cand], img[cand]
            lean = (x - mo) * (mx - mo) > 0                  # between the two means, on the other class's side
            f = np.clip((x - mx) / np.where(np.abs(mo - mx) > 1e-6, mo - mx, 1e-6), 0, 1)
            idx = np.flatnonzero(cand)[lean]
            cur = p[own].ravel()[idx]
            new = np.minimum(cur, f[lean])                   # with two partners, the nearer mean decides
            p[own].ravel()[idx] = new
            p[other].ravel()[idx] += cur - new
    p[:, ~mask] = 0
    return p
