"""Image-quality and sample-homogeneity checks (pure numpy / scipy, no models).

image_quality(img, aseg)   no-reference quality of a T1w image from its native ASEG (tigerbx output):
  noise     robust local std inside eroded WM / (WM - CSF) contrast        (higher = noisier)
  bias      P90 - P10 of the smooth WM intensity field (sigma 12 mm) / contrast  (higher = stronger inhomogeneity)
  contrast  (WM - GM) / (WM - CSF)                                          (lower = poor GM/WM contrast)
  edge      median gradient across the GM/WM border x 1 mm / (WM - GM),     (lower = blurred / upsampled)
            with the expected noise gradient variance subtracted
  res       RMS voxel size in mm
Tissue intensities are medians in WM (2, 41, eroded), GM (3, 42) and lateral ventricles (4, 43), so the measures
assume T1 ordering (CSF < GM < WM).

quality_outliers(values)   flags image_quality() measures that are outliers in a sample: robust z (median / MAD)
  > 3 in the worse direction (noise, bias higher; contrast, edge lower). res is not flagged (a different voxel size
  is not poor quality). No absolute thresholds: they would not hold across scanners and contrasts.

homogeneity(maps)          sample outlier check for template-space maps (e.g. VBM SmoothedGM): mean Pearson
  correlation of each map with all others and its robust z (median / MAD over the sample). z < -3 flags maps that
  disagree with the sample, which mostly catches misregistration / orientation errors (not segmentation bias).
"""
import numpy as np
from scipy import ndimage

WM, GM, CSF = (2, 41), (3, 42), (4, 43)
QUALITY_KEYS = ('noise', 'bias', 'contrast', 'edge', 'res')
# direction in which a measure gets worse (+1: higher is worse); res is reported but never flagged
QUALITY_WORSE = {'noise': 1, 'bias': 1, 'contrast': -1, 'edge': -1}
# E[g^2] per axis of np.gradient(gaussian_filter(white noise, 0.5 voxel)) per unit noise variance (measured on
# N(0, 1) volumes); used to remove the noise contribution from the GM/WM border gradient
_GRAD_NOISE_VAR = 0.1295


def _smooth_field(x, w, sigma):
    """Normalised convolution: weighted Gaussian average of x where w > 0."""
    num = ndimage.gaussian_filter(x * w, sigma, truncate=2.5)
    den = ndimage.gaussian_filter(w, sigma, truncate=2.5)
    return num / np.maximum(den, 1e-6), den


def _eroded(m, it, nmin=1000):
    """m eroded `it` times, or fewer when that would leave < nmin voxels (thick-slice / low-resolution images)."""
    for k in range(it, 0, -1):
        e = ndimage.binary_erosion(m, iterations=k)
        if e.sum() >= nmin:
            return e
    return m


def _median(x, m):
    return float(np.median(x[m])) if m.any() else float('nan')


def image_quality(img, aseg):
    """Quality measures of a T1w nibabel image given its ASEG on the same grid (see module docstring).

    Each measure is NaN when the tissue it needs is missing from the ASEG (res is always given)."""
    x = np.asarray(img.dataobj, np.float32)
    a = np.rint(np.asarray(aseg.dataobj)).astype(np.int16)
    x, a = (v.reshape(v.shape[:3]) if v.ndim > 3 and v.size == np.prod(v.shape[:3]) else v for v in (x, a))
    if x.ndim != 3 or x.shape != a.shape:
        raise ValueError(f'image {x.shape} and aseg {a.shape} must be 3D on the same grid')
    x = np.where(np.isfinite(x), x, 0).astype(np.float32)
    vox = np.array(img.header.get_zooms()[:3], float)
    nan = float('nan')
    out = dict(noise=nan, bias=nan, contrast=nan, edge=nan, res=float(np.sqrt(np.mean(vox ** 2))))
    wmb, gm = np.isin(a, WM), np.isin(a, GM)
    wm = _eroded(wmb, 2)
    csf = _eroded(np.isin(a, CSF), 1, 200)
    pw, pg, pc = _median(x, wm), _median(x, gm), _median(x, csf)
    con = pw - pc
    if not np.isfinite(con) or abs(con) < 1e-6:     # no WM or ventricles: nothing is relative to a contrast
        return out
    out['contrast'] = float((pw - pg) / con)         # NaN without GM

    # noise: residual to the 3x3x3 mean, robust std over WM (local, so independent of bias)
    k = np.ones((3, 3, 3), np.float32) / 27
    r = (x - ndimage.convolve(x, k, mode='nearest'))[wm] * np.sqrt(27 / 26)
    out['noise'] = float(1.4826 * np.median(np.abs(r - np.median(r))) / abs(con))

    # bias: smooth WM field (sigma 12 mm), robust spread relative to contrast
    f, den = _smooth_field(x, wm.astype(np.float32), 12.0 / vox)
    sup = (den > 0.3) & wmb
    if sup.sum() < 100:
        sup = (den > 0) & wmb
    q10, q90 = np.percentile(f[sup], [10, 90])
    out['bias'] = float((q90 - q10) / abs(con))

    # edge sharpness across the GM/WM border
    border = wmb & ~ndimage.binary_erosion(wmb) & ndimage.binary_dilation(gm)
    if border.any() and np.isfinite(pg):
        g2 = sum((gi / vi) ** 2 for gi, vi in zip(np.gradient(ndimage.gaussian_filter(x, 0.5)), vox))
        sigma_n = out['noise'] * abs(con)           # noise std in image units
        g2 = g2 - _GRAD_NOISE_VAR * sigma_n ** 2 * float(np.sum(1.0 / vox ** 2))   # noise-free gradient estimate
        out['edge'] = float(np.median(np.sqrt(np.maximum(g2[border], 0)))) / max(abs(pw - pg), 1e-6)
    return out


def robust_z(x):
    """(x - median) / (1.4826 MAD) over the finite values; NaN where x is not finite or the MAD is 0."""
    x = np.asarray(x, np.float64)
    z = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return z
    med = np.median(x[ok])
    mad = 1.4826 * np.median(np.abs(x[ok] - med))
    if mad > 0:
        z[ok] = (x[ok] - med) / mad
    return z


def quality_outliers(values, z_thresh=3.0):
    """values: {measure: per-input array} (image_quality() keys). Returns one list of flagged measure names per
    input: robust z beyond ``z_thresh`` in the worse direction (QUALITY_WORSE). A missing (NaN) value is not
    flagged."""
    n = len(next(iter(values.values()))) if values else 0
    flags = [[] for _ in range(n)]
    for k, sign in QUALITY_WORSE.items():
        if k not in values:
            continue
        z = robust_z(values[k]) * sign
        for i in np.flatnonzero(z > z_thresh):
            flags[i].append(k)
    return flags


def homogeneity(maps, mask=None):
    """maps: (N, V) or (N, X, Y, Z). Returns (mean correlation with the other maps, robust z) per map.

    A map that is non-finite or flat inside the mask gets NaN in both and is left out of the others' means.
    Memory is O(N x V): the N x N correlation matrix is never formed."""
    X = np.asarray(maps, np.float32).reshape(len(maps), -1)
    if mask is not None:
        X = X[:, np.asarray(mask).ravel()]
    n = len(X)
    s = np.full(n, np.nan)
    z = np.full(n, np.nan)
    if X.shape[1] < 2:
        return s, z
    ok = np.isfinite(X).all(1)
    X = X[ok] - X[ok].mean(1, keepdims=True)
    norm = np.linalg.norm(X, axis=1)
    use = norm > 1e-6
    idx = np.flatnonzero(ok)[use]
    if len(idx) < 3:
        return s, z
    X = X[use] / norm[use, None]
    tot = X.sum(0, dtype=np.float64)
    s[idx] = (X @ tot - np.einsum('ij,ij->i', X, X)) / (len(idx) - 1)      # mean correlation with the others
    med = np.median(s[idx])
    mad = 1.4826 * np.median(np.abs(s[idx] - med)) + 1e-9
    z[idx] = (s[idx] - med) / mad
    return s, z


def block_mean(vol, f=4):
    """Average f x f x f blocks (memory-light copy of a smooth map for homogeneity())."""
    v = np.asarray(vol, np.float32)
    s = [n // f * f for n in v.shape[:3]]
    v = v[:s[0], :s[1], :s[2]]
    return v.reshape(s[0] // f, f, s[1] // f, f, s[2] // f, f).mean((1, 3, 5))
