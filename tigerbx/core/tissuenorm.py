"""Tissue-anchored intensity normalisation for brain-masked T1 images affinely aligned to the MNI
template (pure numpy/scipy). The anchor idea is inspired by CAT12's global intensity normalisation,
but tissue probabilities, bias correction and peak selection are our own.

Output scale: background 0, CSF 1/3, GM 2/3, WM 1 (piecewise linear; above WM continued to 4/3, 5/3).
Tissue peaks = medians of high-probability, low-gradient voxels of each class (avoids PVE, vessels).
CSF anchor = ventricles (tseg 4, 43, 14, 15, 24); GM = cerebral + cerebellar cortex; WM = cerebral +
cerebellar white matter.

Tissue probabilities come from either
  * prior (default, seg-free): 3-class GMM whose mixing weights are the template's tissue priors
    (smoothed template tseg), alternated with a multiplicative bias-field estimate, or
  * seg: a segmentation of the image itself with FreeSurfer label numbers (TigerBx tseg, or a FreeSurfer aseg),
    same bias / peak steps.

    y, peaks = tissue_norm(t1_mni)              # skull-stripped T1 on the template grid; brain = voxels > 0
    y, peaks = tissue_norm(t1_mni, seg=tseg)

prior_tseg defaults to the bundled template tseg (tigerbx/template/mni_template_tseg.nii.gz).
T1 only: if the anchors are not ordered CSF < GM < WM (e.g. T2 / PD) or the fit fails, no exception is
raised (batch runs keep going): a warning is logged, y falls back to img / p99 of the brain, and peaks is all NaN.
Not used by any default pipeline; a model must expect tissue_norm input to benefit.
"""
import hashlib
import logging
import os

import numpy as np
from scipy import ndimage

_logger = logging.getLogger('tigerbx')

Y_NODES = np.array([0, 1 / 3, 2 / 3, 1, 4 / 3, 5 / 3])
LAB = {'csf': (4, 43, 14, 15, 24), 'gm': (3, 42, 8, 47), 'wm': (2, 41, 7, 46)}
_PRIOR = {}


def _map(x, peaks, xmax):
    csf, gm, wm = peaks
    hi = wm + (wm - csf) / 2
    xn = np.array([0, csf, gm, wm, hi, max(xmax, hi * 1.001)])
    return np.interp(x, xn, Y_NODES).astype(np.float32)


def _down(a, f=2):
    s = [n // f * f for n in a.shape]
    a = a[:s[0], :s[1], :s[2]]
    return a.reshape(s[0] // f, f, s[1] // f, f, s[2] // f, f).mean((1, 3, 5))


def _up(a, shape, f=2):
    z = ndimage.zoom(a, f, order=1)
    out = np.ones(shape, np.float32)
    s = [min(n, m) for n, m in zip(shape, z.shape)]
    out[:s[0], :s[1], :s[2]] = z[:s[0], :s[1], :s[2]]
    if any(m < n for n, m in zip(shape, z.shape)):          # odd sizes: edge-pad
        out = np.pad(z, [(0, max(0, n - m)) for n, m in zip(shape, z.shape)], mode='edge')[:shape[0], :shape[1], :shape[2]]
    return out


def template_prior(tseg, sigma=2.0):
    """(3,X,Y,Z) prior probabilities of [CSF, GM, WM] from a template tseg (cached)."""
    key = (tseg.shape, float(sigma), hashlib.blake2b(np.ascontiguousarray(tseg).tobytes(), digest_size=16).digest())
    if key not in _PRIOR:
        p = np.stack([ndimage.gaussian_filter(np.isin(tseg, LAB[k]).astype(np.float32), sigma, truncate=3.0)
                      for k in ('csf', 'gm', 'wm')])
        _PRIOR.clear()
        _PRIOR[key] = p
    return _PRIOR[key]


def _seg_prob(seg):
    return np.stack([np.isin(seg, LAB[k]).astype(np.float32) for k in ('csf', 'gm', 'wm')])


def _estep(x, lp, mu, sd):
    """Posteriors (3,N) for intensities x (N,) and log-priors lp (3,N)."""
    ll = [lp[j] - 0.5 * ((x - mu[j]) / sd[j]) ** 2 - np.log(sd[j]) for j in range(3)]
    m = np.maximum(np.maximum(ll[0], ll[1]), ll[2])
    r = np.stack([np.exp(l - m) for l in ll])
    r /= r[0] + r[1] + r[2]
    return r


def _em(x, prior, mu=None, iters=30, nfit=150000):
    """GMM on intensities x (N,) with per-voxel mixing weights prior (3,N). Returns mu, sd, resp (3,N).
    Parameters are fitted on a regular subsample of <= nfit voxels; responsibilities for all voxels."""
    if mu is not None:
        iters = min(iters, 8)
    x = x.astype(np.float32)
    lp = np.log(prior + 1e-4).astype(np.float32)
    k = max(1, len(x) // nfit)
    xs, lps = x[::k], lp[:, ::k]
    if mu is None:
        mu = np.array([np.average(xs, weights=np.exp(lps[j]) + 1e-6) for j in range(3)])
    mu = np.asarray(mu, np.float64)
    sd = np.full(3, xs.std() / 3)
    for _ in range(iters):
        r = _estep(xs, lps, mu, sd)
        n = r.sum(1) + 1e-9
        mu = r @ xs / n
        sd = np.sqrt(np.array([r[j] @ (xs - mu[j]) ** 2 for j in range(3)]) / n) + 1e-6
    return mu, sd, _estep(x, lp, mu, sd)


def _low_grad(y, core):
    g = np.sqrt(sum(gi ** 2 for gi in np.gradient(y)))
    return np.where(core, g / np.maximum(y, 1e-3), np.inf)


def _select(img, core, prob, peaks):
    """Peak voxels: probability > 0.75 and low relative gradient (CSF also dark)."""
    y0 = _map(img, peaks, float(img.max()))
    yg = _low_grad(y0, core)
    return [(prob[0] > 0.75) & core & (yg < 0.20) & (y0 < 0.45),
            (prob[1] > 0.75) & core & (yg < 0.20),
            (prob[2] > 0.75) & core & (yg < 0.10)]


def _probs(img, core, prior, mu=None):
    x = img[core]
    mu, sd, r = _em(x, np.stack([p[core] for p in prior]), mu)
    prob = np.zeros((3,) + img.shape, np.float32)
    for k in range(3):
        prob[k][core] = r[k]
    return mu, prob


def tissue_norm(img, mask=None, seg=None, prior_tseg=None, bias=True, sigma=12.0, iters=3):
    """img: skull-stripped T1 on the template grid (and seg). mask defaults to img > 0.
    prior_tseg defaults to the bundled template tseg (used unless seg is given).
    Returns (y, peaks); on failure y = img / p99 (brain) and peaks = [nan, nan, nan]."""
    try:
        x = np.asarray(img, np.float32)
        x = np.where(np.isfinite(x), x, 0).astype(np.float32)
        mask = x > 0 if mask is None else np.asarray(mask) > 0
        if mask.shape != x.shape:
            raise ValueError(f'mask {mask.shape} and image {x.shape} differ in shape')
        return _tissue_norm(x, mask, seg, prior_tseg, bias, sigma, iters)
    except Exception as e:                       # noqa: BLE001 - never stop a batch on one image
        _logger.warning('tissue_norm failed (%s); falling back to img / p99', e)
        return _fallback(img, mask), np.full(3, np.nan)


def _fallback(img, mask):
    """img / p99 of the brain (mask, or img > 0 when the mask is unusable); non-finite voxels -> 0."""
    try:
        x = np.asarray(img, np.float32)
        x = np.where(np.isfinite(x), x, 0).astype(np.float32)
        m = np.asarray(mask) > 0 if mask is not None and np.shape(mask) == x.shape else x > 0
        x = np.where(m, x, 0).astype(np.float32)
        v = x[x > 0]
        scale = float(np.percentile(v, 99)) if v.size else 1.0
        return (x / max(scale, 1e-6)).astype(np.float32)
    except Exception:                            # noqa: BLE001 - not even an array: return zeros
        return np.zeros(np.shape(img), np.float32)


def _tissue_norm(img, mask, seg, prior_tseg, bias, sigma, iters):
    img = img * mask
    core = ndimage.binary_erosion(mask, iterations=1)
    if seg is not None:
        P = _seg_prob(np.asarray(seg).astype(int))
        P = np.stack([ndimage.binary_erosion(p > 0) for p in P]).astype(np.float32)
    else:
        if prior_tseg is None:
            import nibabel as nib
            prior_tseg = np.asarray(nib.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'template',
                                                          'mni_template_tseg.nii.gz')).dataobj)
        P = template_prior(np.asarray(prior_tseg).astype(int))

    # bias field on a 2 mm grid (low-gradient GM/WM voxels / class peak == 1, smooth, divide)
    field = np.ones_like(img)
    if bias:
        d_img, d_core = _down(img), _down(core.astype(np.float32)) > 0.99
        d_P = np.stack([_down(p) for p in P])
        d_field = np.ones_like(d_img)
        mu = None
        for _ in range(iters):
            cur = d_img / d_field
            if seg is None:
                mu, prob = _probs(cur, d_core, d_P, mu)
            else:
                prob = d_P
            pk = [np.median(cur[(prob[k] > 0.75) & d_core]) for k in range(3)]
            sel = _select(cur, d_core, prob, pk)
            r = np.zeros_like(cur)
            w = np.zeros_like(cur)
            for k, wk in ((1, 0.5), (2, 1.0)):
                r[sel[k]] = cur[sel[k]] / pk[k]
                w[sel[k]] = wk
            s = sigma / 2.0
            num, den = ndimage.gaussian_filter(r * w, s), ndimage.gaussian_filter(w, s)
            upd = np.where(den > 1e-3, num / np.maximum(den, 1e-6), 1.0)
            upd /= np.median(upd[d_core])
            d_field *= upd
        field = _up(d_field.astype(np.float32), img.shape)
        img = np.where(mask, img / np.maximum(field, 1e-3), 0).astype(np.float32)

    # final peaks on the full-resolution corrected image
    if seg is None:
        _, prob = _probs(img, core, P, (np.asarray(mu) / np.median(d_field[d_core])) if bias else None)
    else:
        prob = P
    pk0 = [np.median(img[(prob[k] > 0.75) & core]) for k in range(3)]
    sel = _select(img, core, prob, pk0)
    pk = np.array([np.median(img[s]) if s.sum() > 50 else p for s, p in zip(sel, pk0)], float)
    if not (0 < pk[0] < pk[1] < pk[2]):
        raise ValueError(f'unexpected tissue order {pk}')
    return _map(img, pk, float(img.max())) * mask, pk

