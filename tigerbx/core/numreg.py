# Pure numpy/scipy affine registration used by tigerbx.reg (affine_type="numreg"); developed in the TigerBx lab.
"""Pure numpy/scipy affine registration (no ANTs / SimpleITK / torch).

Conventions
-----------
All transforms are 4x4 matrices in nibabel world space (RAS, mm) that map a FIXED world
point to the MOVING world point it pulls from (same direction as ITK/ANTs "fwdtransforms").
    y_moving = M @ x_fixed

    M = register(moving_nii, fixed_nii, moving_mask=..., fixed_mask=...)
    warped = apply(moving_nii, fixed_nii, M, order=1)       # array on the fixed grid
    write_itk_mat(M, 'x.mat'); M = read_itk_mat('x.mat')     # ANTs-compatible .mat
"""
import warnings

import numpy as np
from scipy import ndimage, optimize, io as sio

_LPS = np.diag([-1., -1., 1., 1.])


# --------------------------------------------------------------------------
# ITK / ANTs .mat interchange
# --------------------------------------------------------------------------
def read_itk_mat(path):
    m = sio.loadmat(path)
    key = [k for k in m if k.startswith('AffineTransform')][0]
    p = m[key].ravel().astype(np.float64)
    c = m['fixed'].ravel().astype(np.float64)
    A, t = p[:9].reshape(3, 3), p[9:12]
    T = np.eye(4)
    T[:3, :3] = A
    T[:3, 3] = t + c - A @ c
    return _LPS @ T @ _LPS


def write_itk_mat(M, path, center=None):
    T = _LPS @ M @ _LPS
    c = np.zeros(3) if center is None else (_LPS[:3, :3] @ np.asarray(center, float))
    A = T[:3, :3]
    t = T[:3, 3] - c + A @ c
    sio.savemat(path, {'AffineTransform_float_3_3': np.r_[A.ravel(), t][:, None].astype(np.float32),
                       'fixed': c[:, None].astype(np.float32)}, format='4')


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------
def apply(moving, ref, M, order=1, data=None, cval=0.0):
    """Resample `moving` (nibabel image, or `data` on its grid) onto `ref`'s grid through M."""
    data = np.asarray(moving.dataobj, dtype=np.float32) if data is None else np.asarray(data, np.float32)
    V = np.linalg.inv(moving.affine) @ M @ ref.affine          # ref voxel -> moving voxel
    return ndimage.affine_transform(data, V[:3, :3], V[:3, 3], output_shape=ref.shape[:3],
                                    order=order, mode='constant', cval=cval)


def resample(moving, ref, M, warp=None, order=3, data=None, slab=16):
    """`moving` (or `data` on its grid, any orientation) on `ref`'s grid through M, optionally after the dense
    displacement `warp` (3, X, Y, Z; ref voxels, as the VFA warp): out(p) = moving(V (p + warp(p))), one
    interpolation from the moving grid however many transforms. order 3 (cubic B-spline, as affine_transform
    mode='constant') is clipped to [min(data, 0), max(data, 0)] against overshoot; 0 outside the moving grid."""
    data = np.asarray(moving.dataobj, dtype=np.float32) if data is None else np.asarray(data, np.float32)
    V = np.linalg.inv(moving.affine) @ M @ ref.affine          # ref voxel -> moving voxel
    shape = tuple(ref.shape[:3])
    coef = ndimage.spline_filter(data, 3, output=np.float64, mode='mirror') if order == 3 else data
    out = np.zeros(shape, np.float32)
    jk = np.stack(np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing='ij')).astype(np.float64)
    for i0 in range(0, shape[0], slab):
        i1 = min(i0 + slab, shape[0])
        p = np.empty((3, i1 - i0) + shape[1:])
        p[0] = np.arange(i0, i1, dtype=np.float64)[:, None, None]
        p[1:] = jk[:, None]
        if warp is not None:
            p += warp[:, i0:i1]
        c = np.einsum('ij,j...->i...', V[:3, :3], p) + V[:3, 3].reshape(3, 1, 1, 1)
        out[i0:i1] = ndimage.map_coordinates(coef, c, order=order, mode='constant', prefilter=False)
    if order == 3:
        np.clip(out, min(float(data.min()), 0.0), max(float(data.max()), 0.0), out=out)
    return out


def _downsample(data, aff, step):
    """Gaussian-smooth and resample to isotropic `step` mm. Returns (array, vox2world)."""
    vox = np.linalg.norm(aff[:3, :3], axis=0)
    f = step / vox
    sig = np.sqrt(np.maximum(f ** 2 - 1, 0)) * 0.5
    if np.any(sig > 0):
        data = ndimage.gaussian_filter(data, sig, mode='constant')
    shape = np.maximum(np.ceil(np.array(data.shape) / f).astype(int), 2)
    off = (f - 1) / 2
    with warnings.catch_warnings():          # 1-D matrix = diagonal (the fast zoom path); SciPy notes a 0.18 change
        warnings.filterwarnings('ignore', message='The behavior of affine_transform with a 1-D array')
        out = ndimage.affine_transform(data, f, off, output_shape=tuple(shape), order=1, mode='constant')
    S = np.eye(4)
    S[:3, :3] = np.diag(f)
    S[:3, 3] = off
    return out.astype(np.float32), aff @ S


# --------------------------------------------------------------------------
# Trilinear interpolation with exact gradient
# --------------------------------------------------------------------------
class _Vol:
    PAD = 2

    def __init__(self, data, aff):
        p = self.PAD
        self.v = np.pad(data.astype(np.float32), p).ravel()
        self.shape = np.array(data.shape) + 2 * p
        self.st = np.array([self.shape[1] * self.shape[2], self.shape[2], 1])
        self.w2v = np.linalg.inv(aff)
        self.w2v[:3, 3] += p
        self.hi = self.shape - 1.001

    def sample(self, y, grad=True):
        """y: (N,3) world points -> values (N,), d value / d y (N,3)."""
        u = y @ self.w2v[:3, :3].T + self.w2v[:3, 3]
        np.clip(u, 0, self.hi, out=u)
        i0 = u.astype(np.int64)
        fr = (u - i0).astype(np.float32)
        b = i0 @ self.st
        sx, sy, sz = self.st
        v = self.v
        c000, c001 = v[b], v[b + sz]
        c010, c011 = v[b + sy], v[b + sy + sz]
        c100, c101 = v[b + sx], v[b + sx + sz]
        c110, c111 = v[b + sx + sy], v[b + sx + sy + sz]
        fx, fy, fz = fr[:, 0], fr[:, 1], fr[:, 2]
        # interpolate along z first
        c00 = c000 + fz * (c001 - c000)
        c01 = c010 + fz * (c011 - c010)
        c10 = c100 + fz * (c101 - c100)
        c11 = c110 + fz * (c111 - c110)
        c0 = c00 + fy * (c01 - c00)
        c1 = c10 + fy * (c11 - c10)
        val = c0 + fx * (c1 - c0)
        if not grad:
            return val, None
        gx = c1 - c0
        gy = (c01 - c00) + fx * ((c11 - c10) - (c01 - c00))
        d0 = (c001 - c000) + fy * (c011 - c001 - c010 + c000)
        d1 = (c101 - c100) + fy * (c111 - c101 - c110 + c100)
        gz = d0 + fx * (d1 - d0)
        gv = np.stack([gx, gy, gz], 1)
        return val, gv @ self.w2v[:3, :3]          # chain rule: voxel -> world gradient


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------
def _params_to_M(p, c, R):
    A = np.eye(3) + p[:9].reshape(3, 3) / R
    M = np.eye(4)
    M[:3, :3] = A
    M[:3, 3] = c + p[9:] - A @ c
    return M


def _M_to_params(M, c, R):
    A = M[:3, :3]
    return np.r_[((A - np.eye(3)) * R).ravel(), M[:3, 3] - c + A @ c]


class _NCC:
    """Global normalised cross-correlation (maximised)."""

    def __init__(self, f, w):
        self.w = w
        f = f - w @ f
        self.f = f / (np.sqrt(w @ (f * f)) + 1e-8)

    def __call__(self, m):
        w, f = self.w, self.f
        mc = m - (w @ m)
        sm = np.sqrt(w @ (mc * mc)) + 1e-8
        ncc = w @ (f * mc) / sm
        return ncc, w * (f - ncc * mc / sm) / sm


class _MI:
    """Mattes mutual information (maximised): box Parzen on fixed, cubic B-spline Parzen on moving."""

    def __init__(self, f, w, mhi, bins=32):
        self.B, self.w = bins, w
        fhi = np.percentile(f[f > 0], 99.5) if np.any(f > 0) else 1.0
        self.kf = np.clip(f / fhi * (bins - 1), 0, bins - 1).astype(np.int64)
        self.mscale = (bins - 5) / mhi             # moving bins 2 .. B-3 (room for the cubic kernel)

    def __call__(self, m):
        B, w = self.B, self.w
        z = np.clip(m * self.mscale, 0, B - 5) + 2
        k0 = z.astype(np.int64)
        t = z - k0
        t2, t3 = t * t, t * t * t
        u = 1 - t
        # cubic B-spline weights of bins k0-1 .. k0+2 and their derivatives w.r.t. z
        bs = (u * u * u / 6, (3 * t3 - 6 * t2 + 4) / 6, (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6)
        dbs = (-u * u / 2, (3 * t2 - 4 * t) / 2, (-3 * t2 + 2 * t + 1) / 2, t2 / 2)
        idx = self.kf * B + k0 - 1
        joint = np.zeros(B * B)
        for o in range(4):
            joint += np.bincount(idx + o, weights=w * bs[o], minlength=B * B)
        P = joint.reshape(B, B) / joint.sum()
        pf, pm = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
        nz = P > 0
        mi = float((P[nz] * np.log(P[nz] / (pf @ pm)[nz])).sum())
        L = np.zeros_like(P)
        L[nz] = np.log(P[nz] / np.broadcast_to(pm, P.shape)[nz])
        L = L.ravel().astype(np.float32)
        g = dbs[0] * L[idx]
        for o in range(1, 4):
            g += dbs[o] * L[idx + o]
        inside = (m * self.mscale > 0) & (m * self.mscale < B - 5)
        return mi, (w * g * self.mscale * inside).astype(np.float32)


def _cost(p, X, vol, metric, c, R, inv=None):
    """-metric between fixed samples at X and moving sampled at M(p) X; analytic gradient.
    inv = (Y, fvol, metric_b): adds the backward term, moving samples Y vs fixed sampled at M^-1 Y
    (symmetric / inverse-consistent cost; both terms are averaged)."""
    M = _params_to_M(p, c, R)
    Y = X @ M[:3, :3].T + M[:3, 3]
    m, g = vol.sample(Y)
    val, dm = metric(m)
    G = g * dm[:, None]
    gA = G.T @ (X - c) / R
    gt = G.sum(0)
    if inv is not None:
        Yb, fvol, metb = inv
        Ai = np.linalg.inv(M[:3, :3])
        Z = (Yb - M[:3, 3]) @ Ai.T
        fv, gz = fvol.sample(Z)
        vb, dfb = metb(fv)
        H = -(gz * dfb[:, None]) @ Ai                 # d cost / d(A, b) via z = A^-1 (y - b)
        val = 0.5 * (val + vb)
        gA = 0.5 * (gA + H.T @ (Z - c) / R)
        gt = 0.5 * (gt + H.sum(0))
    return -float(val), -np.r_[gA.ravel(), gt].astype(np.float64)


def _pyramid(data, mask, aff, steps):
    """[(data, mask, vox2world)] per step; each level is smoothed/resampled from the next finer one."""
    out = {}
    cur_d, cur_m, cur_a = data, mask.astype(np.float32), aff
    for st in sorted(steps):
        cur_d, a = _downsample(cur_d, cur_a, st)
        cur_m, _ = _downsample(cur_m, cur_a, st)
        cur_a = a
        out[st] = (cur_d, cur_m > 0.5, a)
    return out


def _samples(lev, margin, frac, rng, edge_w=1.0):
    """Sample points inside the mask plus a `margin` mm band around it (band weighted by edge_w)."""
    d, msk, a = lev
    step = np.linalg.norm(a[:3, 0])
    sel = msk
    if margin > 0:
        sel = ndimage.binary_dilation(sel, iterations=max(1, int(round(margin / step))))
    idx = np.argwhere(sel)
    if frac < 1:
        idx = idx[rng.random(len(idx)) < frac]
    X = idx @ a[:3, :3].T + a[:3, 3]
    f = d[tuple(idx.T)].astype(np.float32)
    w = np.where(msk[tuple(idx.T)], 1.0, edge_w).astype(np.float32)
    w /= w.sum()
    return X.astype(np.float32), f, w


_FIXED_CACHE = {}


def _com(mask, aff):
    """Centre of mass (world) and volume (mm^3) of a mask, from its axis projections (no argwhere)."""
    proj = [mask.sum(tuple(a for a in range(3) if a != k), dtype=np.int64) for k in range(3)]
    n = proj[0].sum()
    mean = np.array([p @ np.arange(len(p)) for p in proj]) / n
    return mean @ aff[:3, :3].T + aff[:3, 3], n * abs(np.linalg.det(aff[:3, :3]))


def _principal_axes(mask, aff):
    idx = np.argwhere(mask)
    X = idx @ aff[:3, :3].T + aff[:3, 3]
    C = np.cov((X - X.mean(0)).T)
    lam, V = np.linalg.eigh(C)
    return lam, V


def _pa_init(mmask, maff, fmask, faff):
    """Linear map (fixed -> moving) matching second moments; among the 24 axis
    permutations/sign flips, the rotation part closest to identity is chosen."""
    lm, Vm = _principal_axes(mmask, maff)
    lf, Vf = _principal_axes(fmask, faff)
    import itertools
    best = None
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            P = np.zeros((3, 3))
            P[list(perm), range(3)] = sg               # fixed axis j -> moving axis perm[j]
            Q = Vm @ P @ Vf.T
            if np.linalg.det(Q) < 0:
                continue
            if best is None or np.trace(Q) > best[0]:
                best = (np.trace(Q), Q, perm)
    _, Q, perm = best
    S = np.diag(np.sqrt(lm[list(perm)] / lf))
    return Q @ Vf @ S @ Vf.T


def _rot(ax, deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    i, j = [(1, 2), (0, 2), (0, 1)][ax]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


# start rotations about each axis (degrees) on top of the centre-of-mass start; the best n_starts (5) go on.
# +-20/45 recovers inputs rotated 40-60 degrees (reg60: 0 failures, was 3 with +-15)
START_DEGREES = (-45, -20, 20, 45)

# (step mm, sample fraction, max L-BFGS iterations, band margin mm, band weight)
DEFAULT_LEVELS = ((6.0, 1.0, 60, 6.0, 1.0), (3.0, 1.0, 60, 6.0, 1.0), (1.5, 0.25, 40, 6.0, 0.2))


def register(*args, **kwargs):
    """See _register. BLAS is limited to 1 thread inside: the hot loop is tall-skinny (N x 3) matrix
    products, where multi-threaded OpenBLAS only adds overhead (4 threads measured ~1.6x slower)."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return _register(*args, **kwargs)
    with threadpool_limits(limits=1, user_api='blas'):
        return _register(*args, **kwargs)


def _rigid_param_maps(c, R):
    """6-dof parametrisation q = (rotation vector * R, translation) on top of the 12-dof affine one:
    returns (M -> q, q -> affine params p, cost wrapper with the chain-rule gradient)."""
    from scipy.spatial.transform import Rotation

    def rot(w):
        return Rotation.from_rotvec(w / R).as_matrix()

    def enc(M):
        A = M[:3, :3]
        return np.r_[Rotation.from_matrix(A).as_rotvec() * R, M[:3, 3] - c + A @ c]

    def dec(q):
        return np.r_[((rot(q[:3]) - np.eye(3)) * R).ravel(), q[3:]]

    def cost(q, *a):
        val, g = _cost(dec(q), *a)
        gA = g[:9].reshape(3, 3) * R                   # d cost / dA
        gq = np.empty(6)
        gq[3:] = g[9:]
        for i in range(3):                             # dA / dq_i by central differences (3x3 only, cheap)
            e = np.zeros(3)
            e[i] = 1e-4
            gq[i] = np.sum(gA * (rot(q[:3] + e) - rot(q[:3] - e)) / 2e-4)
        return val, gq
    return enc, dec, cost


def _register(moving, fixed, moving_mask=None, fixed_mask=None, moving_data=None, fixed_data=None,
              levels=DEFAULT_LEVELS, margin=6.0, n_starts=5, metric='mi', bins=32, symmetric=True,
              sym_from=0, sym_to=99, init='com', edge_w=1.0, seed=0, verbose=False, dof=12, ftol=1e-5):
    """Affine registration (Mattes MI or NCC), coarse-to-fine L-BFGS, optionally symmetric
    (forward + inverse cost on level indices `sym_from` .. `sym_to`). Returns 4x4 world M (fixed->moving).
    Symmetric on every level (default): dropping the inverse term on 3 / 1.5 mm is ~25% faster but loses
    template Dice on harder images. ftol: L-BFGS-B
    relative-decrease stop; scipy's default 2.2e-9 only adds iterations inside the sampling noise of MI.
    dof=12: affine (default); dof=6: rigid (rotation + translation, no scaling or shear)."""
    if dof not in (6, 12):
        raise ValueError(f'dof must be 6 (rigid) or 12 (affine), got {dof}')
    rng = np.random.default_rng(seed)
    mdata = np.asarray(moving.dataobj, np.float32) if moving_data is None else np.asarray(moving_data, np.float32)
    fdata = np.asarray(fixed.dataobj, np.float32) if fixed_data is None else np.asarray(fixed_data, np.float32)
    if moving_mask is None:
        # brain = above 1e-3 of the 99th percentile: a spline-resampled (reordered oblique) skull-stripped image
        # leaves millions of ~1e-45 values outside the brain, which would make the mask the whole field of view
        pos = mdata[mdata > 0]
        mmask = mdata > 1e-3 * (np.percentile(pos, 99) if pos.size else 0)
    else:
        mmask = np.asarray(moving_mask) > 0
    maff, faff = moving.affine, fixed.affine
    steps = [lv[0] for lv in levels]

    key = (fdata.shape, faff.tobytes(), float(fdata.sum()), None if fixed_mask is None else int(np.sum(fixed_mask)),
           tuple(steps))
    if key not in _FIXED_CACHE:
        fmask = fdata > fdata.min() if fixed_mask is None else np.asarray(fixed_mask) > 0
        fd = (fdata - fdata[fmask].min()) * fmask
        cf, vf = _com(fmask, faff)
        _FIXED_CACHE.clear()
        _FIXED_CACHE[key] = (fmask, cf, vf, _pyramid(fd, fmask, faff, steps))
    fmask, cf, vf, fpyr = _FIXED_CACHE[key]

    # crop moving to the mask bounding box (+ margin) for speed
    lo, hi = np.zeros(3, int), np.zeros(3, int)
    for k in range(3):
        nz = np.flatnonzero(mmask.any(tuple(a for a in range(3) if a != k)))
        lo[k], hi[k] = nz[0], nz[-1] + 1
    pad = np.ceil(10 / np.linalg.norm(maff[:3, :3], axis=0)).astype(int)
    lo, hi = np.maximum(lo - pad, 0), np.minimum(hi + pad, mdata.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    mmask_c = mmask[sl]
    mdata = mdata[sl] * mmask_c
    maff = maff.copy()
    maff[:3, 3] = maff[:3, :3] @ lo + maff[:3, 3]
    mpyr = _pyramid(mdata, mmask_c, maff, steps)

    # initialisation: centre of mass + isotropic volume scaling (and/or principal axes)
    cm, vm = _com(mmask_c, maff)
    s = (vm / vf) ** (1 / 3)
    R = 50.0
    c = cf
    if dof == 6:
        enc, dec, cost = _rigid_param_maps(c, R)
    else:
        enc, dec, cost = (lambda M: _M_to_params(M, c, R)), (lambda p: p), _cost
    bases = []
    if init in ('com', 'both'):
        bases.append((1.0 if dof == 6 else s) * np.eye(3))
    if init in ('pa', 'both'):
        B = _pa_init(mmask_c, maff, fmask, faff)
        if dof == 6:                                    # nearest rotation (polar decomposition)
            U, _, Vt = np.linalg.svd(B)
            B = U @ Vt
        bases.append(B)
    inits = list(bases)
    if n_starts > 1:
        inits += [B @ _rot(a, d) for B in bases for a in range(3) for d in START_DEGREES]
    M0s = []
    for Ai in inits:
        M = np.eye(4)
        M[:3, :3] = Ai
        M[:3, 3] = cm - M[:3, :3] @ cf
        M0s.append(M)

    def make_metric(f, w, vals):
        if metric == 'mi':
            return _MI(f, w, np.percentile(vals[vals > 0], 99.5), bins)
        return _NCC(f, w)

    best = None
    for li, lv in enumerate(levels):
        step, frac, iters = lv[:3]
        mg = lv[3] if len(lv) > 3 else margin
        ew = lv[4] if len(lv) > 4 else edge_w
        X, f, w = _samples(fpyr[step], mg, frac, rng, ew)
        md, _, ma = mpyr[step]
        vol = _Vol(md, ma)
        met = make_metric(f, w, md)
        inv = None
        if symmetric and sym_from <= li <= sym_to:
            Yb, fb, wb = _samples(mpyr[step], mg, frac, rng, ew)
            fd = fpyr[step][0]
            inv = (Yb, _Vol(fd, fpyr[step][2]), make_metric(fb, wb, fd))
        if li == 0:
            # multi-start: short optimisation from each init, keep the top n_starts
            cands = []
            for M in M0s:
                p0 = enc(M)
                r = optimize.minimize(cost, p0, args=(X, vol, met, c, R, inv), jac=True,
                                      method='L-BFGS-B', options={'maxiter': 15, 'ftol': ftol})
                cands.append((r.fun, r.x))
            cands.sort(key=lambda t: t[0])
            starts = [x for _, x in cands[:n_starts]]
        else:
            starts = [best]
        res = []
        for p0 in starts:
            r = optimize.minimize(cost, p0, args=(X, vol, met, c, R, inv), jac=True,
                                  method='L-BFGS-B', options={'maxiter': iters, 'ftol': ftol})
            res.append((r.fun, r.x, r.nit))
        res.sort(key=lambda t: t[0])
        best = res[0][1]
        if verbose:
            print(f'level {step}mm  n={len(X)}  cost={-res[0][0]:.4f}  it={res[0][2]}')
    return _params_to_M(dec(best), c, R)
