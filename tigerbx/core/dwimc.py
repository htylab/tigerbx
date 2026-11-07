"""dwimc: head motion and eddy-current correction of a diffusion series (one phase-encoding direction).

Model, per volume i (voxel coordinates; the phase axis is axis 1 here, `correct` transposes other axes):

    x_s = M_i x_h                      head (reference b0) voxel -> scanner voxel, rigid in mm
    u   = x_s + F_i(x_s) e_pe          observed voxel; F_i = f(M_i^-1 x_s) + c_i . phi(x_s)

f is the susceptibility displacement (voxels, attached to the head; from a reverse phase-encoding pair, see
core.duope), phi a polynomial in scanner coordinates (order 1: 4 terms, order 2: 10), and the observed signal is the
head-space signal times the Jacobian 1 / (1 + dF/dpe).

Estimation (each step is a Gauss-Newton per volume in observed space, analytic Jacobians, volumes in parallel):

1. every volume registers to a prediction of itself from the other volumes: per shell, an even spherical-harmonic
   model with per-order variances fitted by marginal likelihood, applied leave-one-out (hat-matrix identity); b0s
   register to the reference b0 (the first one, which defines the frame);
2. the EC terms beyond the constant follow the gradient: c = B g (g the unit direction x sqrt(b / b_max)), B shared
   by all DWIs; the constant is merged with the translation along the phase axis;
3. anchors: what DWI-DWI consistency cannot see is fixed against the b0s. The mean DWI is registered rigidly to the
   mean b0 (correlation ratio: a binned intensity mapping, Gauss-Newton), and every DWI is registered to the mean b0
   (rigid + linear EC); the part of those per-volume offsets that is a smooth function of g (even SH + odd linear,
   mean removed) is applied to every DWI;
4. outer iterations: smoothed (FWHM 8 mm) first, then full resolution.

The corrected series is one cubic resampling of every volume with its Jacobian. numpy / scipy only.
"""
import multiprocessing as mp
import os
import time

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.special import sph_harm_y

PE = 1
SWAP = {0: (1, 0, 2), 1: (0, 1, 2), 2: (0, 2, 1)}         # brings the phase axis to axis 1 (its own inverse)
B0_MAX = 50.0                                             # b below this is a b0
FWHM = (8.0, 0.0)                                         # mm per outer iteration (the last value repeats)
OUTER = 3
GN_ITERS = 4
ANCHOR_L = 6
ANCHOR_DILATE = 7                                         # the anchors see the brain edge and some background

_G = {}                                                   # shared with forked workers


# ---------------------------------------------------------------- geometry
def grid(shape):
    return np.stack(np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing='ij')).reshape(3, -1)


def apply(M, x):
    return M[:3, :3] @ x + M[:3, 3:4]


def rigid(p, zooms, c_mm):
    """6 parameters (translation mm, rotation vector rad about c_mm) -> 4x4 head voxel -> scanner voxel."""
    Z, Zi = np.diag(zooms), np.diag(1.0 / np.asarray(zooms))
    R = Rotation.from_rotvec(p[3:6]).as_matrix()
    M = np.eye(4)
    M[:3, :3] = Zi @ R @ Z
    M[:3, 3] = Zi @ (c_mm - R @ c_mm + p[:3])
    return M


def params_of(M, zooms, c_mm):
    Z, Zi = np.diag(zooms), np.diag(1.0 / np.asarray(zooms))
    R = Z @ M[:3, :3] @ Zi
    return np.r_[Z @ M[:3, 3] - c_mm + R @ c_mm, Rotation.from_matrix(R).as_rotvec()]


class ECBasis:
    """Polynomial in scanner voxel coordinates scaled to [-1, 1] over the grid. order 1: 1, x, y, z; order 2 adds
    x^2, y^2, z^2, xy, xz, yz."""

    def __init__(self, shape, order=2):
        self.ctr = (np.asarray(shape) - 1) / 2.0
        self.sc = np.asarray(shape) / 2.0
        self.order = order
        self.K = 4 if order == 1 else 10

    def phi(self, xs):
        x, y, z = (xs - self.ctr[:, None]) / self.sc[:, None]
        cols = [np.ones_like(x), x, y, z]
        if self.order == 2:
            cols += [x * x, y * y, z * z, x * y, x * z, y * z]
        return np.stack(cols)

    def dphi_y(self, xs):
        x, y, z = (xs - self.ctr[:, None]) / self.sc[:, None]
        s = 1.0 / self.sc[1]
        o = np.zeros_like(x)
        cols = [o, o, np.full_like(x, s), o]
        if self.order == 2:
            cols += [o, 2 * y * s, o, x * s, o, z * s]
        return np.stack(cols)


def sh_basis(g, L):
    """Real even spherical harmonics up to order L of unit vectors g (N, 3) -> (N, n), and each column's order."""
    g = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    theta = np.arccos(np.clip(g[:, 2], -1, 1))
    phi = np.arctan2(g[:, 1], g[:, 0])
    cols, ls = [], []
    for l in range(0, L + 1, 2):
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, abs(m), theta, phi)
            cols.append(np.sqrt(2) * Y.imag if m < 0 else Y.real if m == 0 else np.sqrt(2) * Y.real)
            ls.append(l)
    return np.stack(cols, 1), np.array(ls)


def sample(vol, pts, order=1, prefiltered=False):
    return ndimage.map_coordinates(vol, pts, order=order, mode='nearest', prefilter=not prefiltered)


def line_inverse(F):
    """F (nx, ny, nz): displacement along axis 1 at the scanner grid points, u = y + F. Returns g on the grid with
    y_s = y_u + g(y_u): per line, u(y) made monotone (folds collapse) and inverted linearly."""
    ny = F.shape[1]
    y = np.arange(ny, dtype=np.float64)
    U = np.maximum.accumulate(y[None, :, None] + F, axis=1) + 1e-9 * y[None, :, None]
    out = np.empty_like(F)
    for ix in range(F.shape[0]):
        for iz in range(F.shape[2]):
            out[ix, :, iz] = np.interp(y, U[ix, :, iz], y) - y
    return out


def observed_position(xh, M, f, ec, c):
    xs = apply(M, xh)
    u = xs.copy()
    u[PE] += sample(f, xh) + c @ ec.phi(xs)
    return u


# ---------------------------------------------------------------- predictor
def sh_predictor(g, S, L):
    """Leave-one-out hat matrix (N, N) of one shell: SH basis with per-order variances and a noise variance fitted
    by marginal likelihood on the empirical covariance of S (N, V)."""
    B, ls = sh_basis(g, L)
    N = B.shape[0]
    Gm = S @ S.T / S.shape[1]
    lo = ls // 2
    scale = np.trace(Gm) / N

    def nll(theta):
        a, s2 = np.exp(theta[:-1]) * scale, np.exp(theta[-1]) * scale
        K = (B * a[lo]) @ B.T + s2 * np.eye(N)
        try:
            Lc = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e30
        return 2 * np.log(np.diag(Lc)).sum() + np.trace(np.linalg.solve(K, Gm))

    x0 = np.log(np.r_[[1.0, 0.3, 0.1, 0.03, 0.01][:L // 2 + 1], 0.05])
    theta = minimize(nll, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-3, 'fatol': 1e-6}).x
    a, s2 = np.maximum(np.exp(theta[:-1]), 1e-12) * scale, np.exp(theta[-1]) * scale
    H = B @ np.linalg.solve(B.T @ B + s2 * np.diag(1 / a[lo]), B.T)
    h = np.diag(H)
    return (H - np.diag(h)) / (1 - h)[:, None]


def sh_order(n):
    return 2 if n <= 12 else 4 if n <= 28 else 6


# ---------------------------------------------------------------- per volume (run in workers)
def _field_inverse(Minv, c):
    F = sample(_G['field'], apply(Minv, _G['grid'])) + c @ _G['ec'].phi(_G['grid'])
    return line_inverse(F.reshape(_G['shape']))[_G['obs_idx']]


def _warp(i, Minv, c, s, want_jac):
    """Prediction of volume i in its observed space (at the observed voxels) and its Jacobian columns: scale, EC
    terms (DWIs), translations (the phase-axis one only for b0s: merged with the EC constant), rotations."""
    ec, u = _G['ec'], _G['obs']
    xs = u.copy()
    xs[PE] += _field_inverse(Minv, c)
    xh = apply(Minv, xs)
    gf = np.stack([sample(_G['gfield'][k], xh) for k in range(3)])
    m = Minv[:3, 1]
    J = 1.0 / np.clip(1.0 + m @ gf + c @ ec.dphi_y(xs), 0.2, None)
    p = sample(_G['pred'][i], xh, order=3, prefiltered=True)
    s_p = s * p
    if not want_jac:
        return s_p * J, None
    gp = np.stack([sample(_G['gpred'][i][k], xh) for k in range(3)])
    gfy = np.stack([sample(_G['gfy'][k], xh) for k in range(3)])
    pm, fym = m @ gp, m @ gfy
    Phi, dPhi = ec.phi(xs), ec.dphi_y(xs)

    def col(dp, dfy, dfy_direct):                      # d(s p J) for a change dx_h: s J grad p.dx_h + s p dJ
        return s * J * dp - s_p * J ** 2 * (dfy + dfy_direct)

    cols = [p * J]
    dwi = _G['bval'][i] > B0_MAX
    if dwi:
        for k in range(ec.K):                          # dx_h = -J phi_k m
            cols.append(col(-J * Phi[k] * pm, -J * Phi[k] * fym, dPhi[k]))
    zi = 1.0 / _G['zooms']
    xmm = xh * _G['zooms'][:, None] - _G['c_mm'][:, None]
    for j in range(3):
        if j == PE and dwi:
            continue
        a = np.zeros(3)
        a[j] = -zi[j]
        t = -J * (a @ gf)                              # the head-attached field moves too
        cols.append(col(a @ gp + t * pm, a @ gfy + t * fym, 0.0))
    for j in range(3):
        a = np.cross(xmm.T, np.eye(3)[j]).T * zi[:, None]
        t = -J * (gf * a).sum(0)
        cols.append(col((gp * a).sum(0) + t * pm, (gfy * a).sum(0) + t * fym, 0.0))
    return s_p * J, np.stack(cols, 1)


def _update(i, Minv, c, s, d):
    k = 1
    s = s + d[0]
    dwi = _G['bval'][i] > B0_MAX
    if dwi:
        c = c + d[k:k + _G['ec'].K]
        k += _G['ec'].K
    dt = np.zeros(3)
    for j in range(3):
        if j == PE and dwi:
            continue
        dt[j] = d[k]
        k += 1
    Z, Zi, c_mm = np.diag(_G['zooms']), np.diag(1 / _G['zooms']), _G['c_mm']
    R = Rotation.from_rotvec(-d[k:k + 3]).as_matrix()
    D = np.eye(4)                                      # head voxel -> head voxel: R(x - c) + c - dt in mm
    D[:3, :3] = Zi @ R @ Z
    D[:3, 3] = Zi @ (c_mm - R @ c_mm - dt)
    return D @ Minv, c, s


def _register(args):
    """Levenberg-Marquardt / Gauss-Newton of one volume onto its prediction (line search by halving)."""
    i, Minv, c, s, iters = args
    y, w = _G['y'][i], _G['w']
    yhat, Jac = _warp(i, Minv, c, s, True)
    r = y - yhat
    cost = (w * r * r).sum()
    mu = 1e-3
    for it in range(iters):
        Jw = Jac * np.sqrt(w)[:, None]
        H = Jw.T @ Jw
        d = np.linalg.solve(H + mu * np.diag(np.diag(H)) + 1e-12 * np.eye(len(H)), Jac.T @ (w * r))
        for _ in range(4):
            Mn, cn, sn = _update(i, Minv, c, s, d)
            r2 = y - _warp(i, Mn, cn, sn, False)[0]
            c2 = (w * r2 * r2).sum()
            if c2 < cost:
                break
            d = d / 2
        else:
            break
        Minv, c, s, cost = Mn, cn, sn, c2
        if it < iters - 1:
            yhat, Jac = _warp(i, Minv, c, s, True)
            r = y - yhat
    return Minv, c, s


def _to_model(args):
    """Volume i in head space: y(u(x_h)) (1 + dF/dpe) / s on the full grid."""
    i, M, c, s = args
    ec, xh = _G['ec'], _G['grid']
    xs = apply(M, xh)
    Minv = np.linalg.inv(M)
    F = sample(_G['field'], xh) + c @ ec.phi(xs)
    gf = np.stack([sample(_G['gfield'][k], xh) for k in range(3)])
    Fy = Minv[:3, 1] @ gf + c @ ec.dphi_y(xs)
    u = xs.copy()
    u[PE] += F
    return (sample(_G['raw'][i], u, order=3, prefiltered=True) * (1 + Fy) / s).astype(np.float32)


def anchor(moving, fixed, mask, zooms, outer=6, iters=3, bins=48, ec=None, dilate=ANCHOR_DILATE):
    """Voxel map V (fixed voxel -> moving voxel) registering `moving` to `fixed` rigidly by the correlation ratio
    (fixed's intensity bins mapped to moving's mean, refitted each round; Gauss-Newton). With an EC basis, also a
    displacement e . phi along the phase axis (the constant left to the rigid part): returns (V, e)."""
    c_mm = zooms * np.argwhere(mask).mean(0)
    region = ndimage.binary_dilation(mask, iterations=dilate)
    x = np.stack(np.nonzero(region)).astype(np.float64)
    f = fixed[region]
    edges = np.quantile(f, np.linspace(0, 1, bins + 1))
    lab = np.clip(np.searchsorted(edges, f, side='right') - 1, 0, bins - 1)
    mc = ndimage.spline_filter(moving)
    gm = np.gradient(moving)
    V = np.eye(4)
    Phi = ec.phi(x)[1:] if ec is not None else np.zeros((0, x.shape[1]))
    e = np.zeros(len(Phi))

    def pos():
        xv = apply(V, x)
        xv[PE] += e @ Phi
        return xv

    for _ in range(outer):
        m = sample(mc, pos(), order=3, prefiltered=True)
        T = (np.bincount(lab, m, bins) / np.maximum(np.bincount(lab, None, bins), 1))[lab]
        for _ in range(iters):
            xv = pos()
            m = sample(mc, xv, order=3, prefiltered=True)
            g = np.stack([sample(gm[k], xv) for k in range(3)])
            xmm = xv * zooms[:, None] - c_mm[:, None]
            cols = [g[j] / zooms[j] for j in range(3)]
            cols += [(g * (np.cross(np.eye(3)[j], xmm.T).T / zooms[:, None])).sum(0) for j in range(3)]
            cols += [g[PE] * p for p in Phi]
            Jm = np.stack(cols, 1)
            d = np.linalg.solve(Jm.T @ Jm + 1e-9 * np.eye(Jm.shape[1]), Jm.T @ (T - m))
            V = rigid(d[:6], zooms, c_mm) @ V
            e = e + d[6:]
    return (V, np.r_[0.0, e]) if ec is not None else V


def _anchor_volume(model_img):
    lin = ECBasis(_G['shape'], 1)
    V, e = anchor(model_img.reshape(_G['shape']), _G['b0mean'], _G['mask'], _G['zooms'], outer=3, iters=2, ec=lin)
    return V, e


# ---------------------------------------------------------------- driver
class _Pool:
    """Fork workers sharing _G as it is when created (serial where fork is unavailable)."""

    def __init__(self, workers):
        self.workers = workers if 'fork' in mp.get_all_start_methods() else 1

    def map(self, fn, items):
        if self.workers <= 1:
            return [fn(a) for a in items]
        with mp.get_context('fork').Pool(min(self.workers, len(items))) as pool:
            return pool.map(fn, items)


def fill_holes(m):
    """Holes filled in 3D and then slice by slice along every axis (dark deep grey matter in a b0 can open a hole to
    the outside through one slice, which a 3D fill alone leaves)."""
    m = ndimage.binary_fill_holes(m)
    for ax in range(3):
        m = np.moveaxis(np.stack([ndimage.binary_fill_holes(s) for s in np.moveaxis(m, ax, 0)]), 0, ax)
    return m


def head_mask(b0):
    m = fill_holes(b0 > 0.15 * np.percentile(b0, 99))
    lab, n = ndimage.label(m)
    if n > 1:
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return fill_holes(ndimage.binary_erosion(m, iterations=2))


def estimate(data, bval, bvec, zooms, field=None, mask=None, order=2, outer=OUTER, fwhm=FWHM, gn_iters=GN_ITERS,
             workers=1, verbose=0, anchor_b0=True, history=None):
    """Motion + EC of every volume of data (x, y, z, n; the phase axis is axis 1) with bval (n,) and bvec (n, 3)
    in the same voxel axes. field: susceptibility displacement (voxels along axis 1, head = reference b0 frame) or
    None. anchor_b0: True (both anchors), 'rigid' (only the mean DWI -> mean b0 rigid alignment, as a
    shell-level alignment) or False (no anchor; ablations). history: a list that receives dict(M, c) after every outer
    iteration. Returns dict(M (n, 4, 4) head -> scanner voxel, c (n, K) EC coefficients, s (n,) scale, ref, order,
    mask, c_mm)."""
    t0 = time.time()
    data = np.asarray(data, np.float64)
    shape, N = data.shape[:3], data.shape[3]
    zooms = np.asarray(zooms, np.float64)
    bval = np.asarray(bval, np.float64)
    bvec = np.asarray(bvec, np.float64)
    dw = bval > B0_MAX
    b0 = np.nonzero(~dw)[0]
    if not len(b0):
        raise ValueError('dwimc needs at least one b0 volume (b < 50)')
    if dw.sum() < 6:
        raise ValueError('dwimc needs at least 6 diffusion-weighted volumes')
    ref = int(b0[0])
    field = np.zeros(shape) if field is None else np.asarray(field, np.float64)
    if mask is None:
        mask = head_mask(data[..., b0].mean(-1))
    ec = ECBasis(shape, order)
    gsc = bvec * np.sqrt(bval / bval.max())[:, None]   # EC follows the gradient amplitude (sqrt b)
    shells = np.round(bval / 100) * 100
    obs_mask = ndimage.binary_dilation(mask, iterations=2)
    c_mm = zooms * np.argwhere(mask).mean(0)
    gfield = np.gradient(field)
    _G.clear()
    _G.update(field=field, gfield=gfield, gfy=np.gradient(gfield[PE]), ec=ec, zooms=zooms, bval=bval, shape=shape,
              grid=grid(shape), c_mm=c_mm, obs=np.stack(np.nonzero(obs_mask)).astype(np.float64),
              obs_idx=np.nonzero(obs_mask), raw=[ndimage.spline_filter(data[..., i]) for i in range(N)],
              w=mask[obs_mask].astype(np.float64), mask=mask)
    Minv = [np.eye(4) for _ in range(N)]
    c = [np.zeros(ec.K) for _ in range(N)]
    s = [1.0] * N
    pool = _Pool(workers)
    idx = np.nonzero(dw)[0]
    for it in range(outer):
        fw = fwhm[min(it, len(fwhm) - 1)]
        Ms = pool.map(_to_model, [(i, np.linalg.inv(Minv[i]), c[i], s[i]) for i in range(N)])
        if it > 0 and anchor_b0:
            # the DWI frame: mean DWI -> mean b0 (rigid), then the smooth-in-g part of per-volume offsets
            V = anchor(np.mean([Ms[i] for i in idx], 0).reshape(shape),
                       np.mean([Ms[i] for i in b0], 0).reshape(shape), mask, zooms)
            for i in idx:
                Minv[i] = np.linalg.inv(np.linalg.inv(Minv[i]) @ V)
            Ms = pool.map(_to_model, [(i, np.linalg.inv(Minv[i]), c[i], s[i]) for i in range(N)])
        if it > 0 and anchor_b0 is True:
            _G['b0mean'] = np.mean([Ms[i] for i in b0], 0).reshape(shape).astype(np.float64)
            anc = pool.map(_anchor_volume, [Ms[i] for i in idx])
            P6 = np.array([params_of(a[0], zooms, c_mm) for a in anc])
            E = np.array([a[1] for a in anc])
            Lb = ANCHOR_L if len(idx) > 30 else 2 if len(idx) > 12 else 0
            X = np.c_[sh_basis(bvec[idx], Lb)[0], gsc[idx]]
            fitP = X @ np.linalg.lstsq(X, P6, rcond=None)[0]
            fitE = X @ np.linalg.lstsq(X, E, rcond=None)[0]
            fitP -= fitP.mean(0)
            fitE -= fitE.mean(0)
            for n, i in enumerate(idx):
                Minv[i] = np.linalg.inv(np.linalg.inv(Minv[i]) @ rigid(fitP[n], zooms, c_mm))
                c[i] = c[i] + np.r_[fitE[n], np.zeros(ec.K - 4)]
            Ms = pool.map(_to_model, [(i, np.linalg.inv(Minv[i]), c[i], s[i]) for i in range(N)])
        S = np.stack([m.ravel() for m in Ms])
        del Ms
        P = np.empty_like(S)
        P[b0] = S[ref]
        if len(b0) > 1:
            P[ref] = S[b0[1:]].mean(0)
        for sb in np.unique(shells[dw]):
            k = np.nonzero(dw & (shells == sb))[0]
            P[k] = sh_predictor(bvec[k], S[k][:, mask.ravel()], sh_order(len(k))) @ S[k]
        del S
        sig = fw / 2.3548 / zooms
        pred = [p.reshape(shape) for p in P]
        del P
        if fw > 0:
            pred = [ndimage.gaussian_filter(p, sig) for p in pred]
            _G['y'] = [ndimage.gaussian_filter(data[..., i], sig)[obs_mask] for i in range(N)]
        else:
            _G['y'] = [data[..., i][obs_mask] for i in range(N)]
        _G['pred'] = [ndimage.spline_filter(p) for p in pred]
        _G['gpred'] = [np.gradient(p) for p in pred]
        del pred
        res = pool.map(_register, [(i, Minv[i], c[i], s[i], gn_iters) for i in range(N) if i != ref])
        for i, r in zip([i for i in range(N) if i != ref], res):
            Minv[i], c[i], s[i] = r
        # second level: EC terms 1.. = B g (no intercept); the constant's remainder is the phase-axis translation
        C = np.array([c[i] for i in idx])
        Bm = np.linalg.lstsq(gsc[idx], C, rcond=None)[0]
        fit = gsc[idx] @ Bm
        for n, i in enumerate(idx):
            shift = C[n][0] - fit[n][0]
            c[i] = fit[n].copy()
            M = np.linalg.inv(Minv[i])
            M[PE, 3] += shift
            Minv[i] = np.linalg.inv(M)
        if history is not None:
            history.append(dict(M=np.array([np.linalg.inv(m) for m in Minv]), c=np.array(c)))
        if verbose:
            print(f'dwimc: outer {it + 1}/{outer} (fwhm {fw:g} mm) {time.time() - t0:.1f} s', flush=True)
    for k in ('pred', 'gpred', 'y', 'raw'):
        _G.pop(k, None)
    return dict(M=np.array([np.linalg.inv(m) for m in Minv]), c=np.array(c), s=np.array(s), ref=ref, order=order,
                mask=mask, c_mm=c_mm, seconds=time.time() - t0)


def displacement(M, c, field, order, shape):
    """Observed position minus head position (voxels, 3 x n_vox) of one volume, and dF/dpe at the head grid."""
    ec = ECBasis(shape, order)
    xh = grid(shape)
    xs = apply(M, xh)
    F = sample(field, xh) + c @ ec.phi(xs)
    gf = np.stack([sample(g, xh) for g in np.gradient(field)])
    Fy = np.linalg.inv(M)[:3, 1] @ gf + c @ ec.dphi_y(xs)
    u = xs.copy()
    u[PE] += F
    return u - xh, Fy


def correct(volume, M, c, field, order):
    """One volume resampled into the reference head frame: cubic, times the Jacobian (1 + dF/dpe)."""
    shape = volume.shape
    D, Fy = displacement(M, c, field, order, shape)
    v = ndimage.map_coordinates(np.asarray(volume, np.float64), grid(shape) + D, order=3, mode='nearest')
    return (v * (1 + Fy)).reshape(shape).astype(np.float32)


def _correct_one(i):
    e = _G['est']
    v = correct(_G['data'][..., i], e['M'][i], e['c'][i], _G['field'], e['order'])
    D, _ = displacement(e['M'][i], e['c'][i], np.zeros(v.shape), e['order'], v.shape)
    m = e['mask'].ravel()
    rms = float(np.sqrt(((D[:, m] * _G['zooms'][:, None]) ** 2).sum(0).mean()))
    return np.maximum(v, 0), rms


def correct_all(data, est, field, zooms, workers=1, chunk=None):
    """Every volume corrected (as `correct`, negative interpolation overshoot set to 0) and the RMS displacement
    (mm, brain) of its motion + eddy currents; yields (volume, rms) in order, `chunk` volumes per worker round."""
    _G.clear()
    _G.update(data=data, est=est, field=np.zeros(data.shape[:3]) if field is None else field,
              zooms=np.asarray(zooms, float))
    n = data.shape[3]
    chunk = chunk or max(1, workers) * 4
    pool = _Pool(workers)
    for a in range(0, n, chunk):
        yield from pool.map(_correct_one, list(range(a, min(n, a + chunk))))
    _G.clear()


def rotation(M, zooms):
    """The rotation (3 x 3, mm voxel axes) of head -> scanner map M."""
    Z, Zi = np.diag(zooms), np.diag(1.0 / np.asarray(zooms))
    R = Z @ M[:3, :3] @ Zi
    u, _, vt = np.linalg.svd(R)
    return u @ vt


def default_workers():
    n = os.cpu_count() or 1
    return max(1, int(n * 0.7))
