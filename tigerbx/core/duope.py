"""DuoPE: susceptibility distortion from a pair of spin-echo EPI images with opposite phase encoding.

Model (as topup): one displacement d (voxels along the phase-encoding axis) moves the forward image by +d and the
reverse one by -d, the signal modulated by the Jacobian 1 +- dd/dpe:

    A(d) = Ia(x + d)(1 + g),   P(d) = Ip(x - d)(1 - g),   g = dd/dpe
    E(d) = 1/2 |A - P|^2 + alpha/2 sum w (Laplacian d)^2

w is 1 in homogeneous tissue and eps on a shell at every signal boundary: the field is harmonic where the
susceptibility is constant and bends only at tissue / air interfaces. Steps:

1. start: d = D chi + H c on a coarse grid (>= 4 mm), D the dipole kernel 1/3 - (k.b)^2/|k|^2 (b = B0 in voxel
   axes), chi on the boundary shell and outside the head, H the solid harmonics of order <= 2 (shim); fitted to a
   closed-form per-line start (cumulative-signal matching) and then to the data by Gauss-Newton;
2. Gauss-Newton on d at the finest level (<= 3.5 mm), matrix-free CG with a DCT + Jacobi preconditioner;
3. rigid motion of the reverse scan (numreg, the two unwarped images); the translation along the phase axis is a
   constant field offset and is not estimated. A large motion (> 1 mm or 1 degree) is refined jointly with d
   (Gauss-Newton on field and motion, the motion block eliminated by its Schur complement); otherwise d is refined
   once more against the moved reverse scan;
4. output images by one cubic resampling (motion and field together).

Arrays here have the phase axis as axis 1; `estimate` transposes other phase axes. numpy / scipy only.
"""
import time

import nibabel as nib
import numpy as np
from scipy import fft, ndimage
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial.transform import Rotation

from tigerbx.core import numreg

ALPHA = 0.15            # bending weight (intensities scaled to p99 = 1)
SHELL_EPS = 0.05        # bending weight on the interface shell
MAX_MM = 3.5            # the finest level's grid
DIPOLE_MM = 4.0         # the dipole start's grid
JOINT_MM, JOINT_DEG = 1.0, 1.0
MOTION_LEVELS = ((6.0, 1.0, 60, 6.0, 1.0), (3.0, 1.0, 60, 6.0, 1.0))
_SWAP = {0: (1, 0, 2), 1: (0, 1, 2), 2: (0, 2, 1)}       # brings the phase axis to axis 1 (its own inverse)


# ---------------------------------------------------------------- operators along axis 1

def dy(v):
    """Central difference along axis 1, edges replicated."""
    p = np.concatenate([v[:, :1], v, v[:, -1:]], axis=1)
    return 0.5 * (p[:, 2:] - p[:, :-2])


def dy_t(w):
    """Adjoint of dy."""
    out = np.zeros_like(w)
    out[:, 1:-1] = 0.5 * (w[:, :-2] - w[:, 2:])
    out[:, 0] = -0.5 * (w[:, 0] + w[:, 1])
    out[:, -1] = 0.5 * (w[:, -1] + w[:, -2])
    return out


def lap(v):
    """-Laplacian, reflecting boundary (diagonalised by the DCT-II)."""
    return -ndimage.laplace(v, mode='reflect')


def shift(img, s):
    """img at y + s along axis 1 (linear, clamped): value and derivative d img / dy (0 where clamped)."""
    n = img.shape[1]
    y_raw = np.arange(n)[None, :, None] + s
    y = np.clip(y_raw, 0, n - 1)
    i0 = np.minimum(np.floor(y).astype(np.int64), n - 2)
    w = y - i0
    v0 = np.take_along_axis(img, i0, axis=1)
    v1 = np.take_along_axis(img, i0 + 1, axis=1)
    inside = (y_raw >= 0) & (y_raw <= n - 1)
    return v0 + w * (v1 - v0), np.where(inside, v1 - v0, 0.0)


def correct(img, d):
    """Unwarped image (linear): img(x + d)(1 + dd/dy)."""
    v, _ = shift(np.asarray(img, np.float64), d)
    return v * (1 + dy(d))


def down(a, f):
    """Block mean by f in every axis (edge padded)."""
    if f == 1:
        return a
    a = np.pad(a, [(0, (-s) % f) for s in a.shape], mode='edge')
    s = a.shape
    return a.reshape(s[0] // f, f, s[1] // f, f, s[2] // f, f).mean(axis=(1, 3, 5))


def upsample(d, shape, f):
    """d on a grid f times coarser (its voxels) -> `shape` (full-grid voxels), linear."""
    if f == 1:
        return d
    z = [t / s for t, s in zip(shape, d.shape)]
    return f * ndimage.zoom(d, z, order=1, mode='nearest', grid_mode=True)[:shape[0], :shape[1], :shape[2]]


def normalise(a, p):
    """Each image to mean 1 inside the head (the two scans can differ in gain), then a common p99 of 1."""
    a = np.asarray(a, np.float64)
    p = np.asarray(p, np.float64)
    a = a / a[a > 0.2 * np.percentile(a, 99)].mean()
    p = p / p[p > 0.2 * np.percentile(p, 99)].mean()
    s = np.percentile(0.5 * (a + p), 99)
    return a / s, p / s


# ---------------------------------------------------------------- the dense field at one level

def shell_weight(ia, ip, eps=SHELL_EPS, k=2, outer=4, thr=0.25):
    """eps on the tissue side of every signal boundary (k voxels) and outside the head (outer voxels), 1 inside."""
    s = ndimage.gaussian_filter(0.5 * (ia + ip), 1.0)
    m = s > thr * np.percentile(s, 99)
    inner = m & ~ndimage.binary_erosion(m, iterations=k)
    out = ndimage.binary_dilation(m, iterations=outer) & ~m
    return np.where(inner | out, eps, 1.0)


_NB = np.zeros((3, 3, 3))
_NB[1, 1, 0] = _NB[1, 1, 2] = _NB[1, 0, 1] = _NB[1, 2, 1] = _NB[0, 1, 1] = _NB[2, 1, 1] = 1


class Prior:
    """alpha/2 sum w (Laplacian d)^2: its Hessian, Jacobi diagonal and DCT surrogate."""

    def __init__(self, w, alpha):
        self.w, self.alpha = w, alpha
        self.diag = alpha * (36.0 * w + ndimage.convolve(w, _NB, mode='nearest'))
        lam = np.zeros(w.shape)
        for k, n in enumerate(w.shape):
            lam = lam + (2 - 2 * np.cos(np.pi * np.arange(n) / n)).reshape([n if i == k else 1 for i in range(3)])
        self.lam2 = lam ** 2

    def hess(self, v):
        return self.alpha * lap(self.w * lap(v))

    def energy(self, d):
        return 0.5 * float((d * self.hess(d)).sum())


def _cost(ia, ip, d, prior):
    a, _ = shift(ia, d)
    p, _ = shift(ip, -d)
    g = dy(d)
    r = a * (1 + g) - p * (1 - g)
    return 0.5 * float((r * r).sum()) + prior.energy(d)


def gauss_newton(ia, ip, d, prior, iters, cg_iters=15):
    """Gauss-Newton on d; CG preconditioned by 0.5 (DCT surrogate + Jacobi); Armijo backtracking."""
    shape, n = d.shape, d.size
    for _ in range(iters):
        a, ga = shift(ia, d)
        p, gp = shift(ip, -d)
        g = dy(d)
        r = a * (1 + g) - p * (1 - g)
        ja, jb = ga * (1 + g) + gp * (1 - g), a + p
        grad = ja * r + dy_t(jb * r) + prior.hess(d)

        def hv(v):
            v = v.reshape(shape)
            jv = ja * v + jb * dy(v)
            return (ja * jv + dy_t(jb * jv) + prior.hess(v)).ravel()
        data_diag = ja * ja + 0.5 * jb * jb
        diag = (data_diag + prior.diag + 1e-6).ravel()
        c0 = float(np.median(data_diag[data_diag > 1e-3 * data_diag.max()])) if data_diag.max() > 0 else 1.0
        denom = c0 + prior.alpha * float(prior.w.mean()) * prior.lam2

        def pre(v):
            z = fft.idctn(fft.dctn(v.reshape(shape), type=2, norm='ortho', workers=1) / denom, type=2, norm='ortho',
                          workers=1)
            return 0.5 * (z.ravel() + v / diag)
        step = cg(LinearOperator((n, n), matvec=hv, dtype=np.float64), -grad.ravel(),
                  M=LinearOperator((n, n), matvec=pre, dtype=np.float64), maxiter=cg_iters)[0].reshape(shape)
        e0, t, slope = _cost(ia, ip, d, prior), 1.0, float((grad * step).sum())
        while t > 1e-3:
            if _cost(ia, ip, d + t * step, prior) <= e0 + 1e-4 * t * slope:
                d = d + t * step
                break
            t *= 0.5
        else:
            break
    return d


def refine(a, p, d0, f, zooms, iters, alpha=ALPHA, bias_mm=40.0):
    """Gauss-Newton on d (full-grid voxels, normalised images a / p) at block-mean level f."""
    ia, ip = down(a, f), down(p, f)
    d = down(d0, f) / f
    if bias_mm:
        # smooth intensity ratio of the two unwarped images (coil / gain differences), split between the scans
        sb = [bias_mm / (z * f) for z in zooms]
        ca, cp = correct(ia, d), correct(ip, -d)
        w = (ca + cp) > 0.2 * np.percentile(ca + cp, 99)
        num = ndimage.gaussian_filter(np.where(w, ca, 0), sb)
        den = ndimage.gaussian_filter(np.where(w, cp, 0), sb)
        ratio = np.clip(np.where(den > 1e-6, num / np.maximum(den, 1e-6), 1.0), 0.5, 2.0)
        ia, ip = ia / np.sqrt(ratio), ip * np.sqrt(ratio)
    prior = Prior(shell_weight(correct(ia, d), correct(ip, -d)), alpha)
    return upsample(gauss_newton(ia, ip, d, prior, iters), a.shape, f)


# ---------------------------------------------------------------- the dipole start

def line_start(ia, ip, floor=0.05, smooth_vox=2.0):
    """Closed-form per-line d (cumulative-signal matching along axis 1: x = (y_a + y_p)/2, d = (y_a - y_p)/2 for
    every quantile), smoothed with signal weights."""
    a = np.clip(ia - floor * np.percentile(ia, 99), 0, None) + 1e-6
    p = np.clip(ip - floor * np.percentile(ip, 99), 0, None) + 1e-6
    nx, n, nz = a.shape
    A = np.moveaxis(a, 1, -1).reshape(-1, n)
    P = np.moveaxis(p, 1, -1).reshape(-1, n)
    Ca = np.concatenate([np.zeros((len(A), 1)), np.cumsum(A, 1)], 1)
    Cp = np.concatenate([np.zeros((len(P), 1)), np.cumsum(P, 1)], 1)
    Ca /= Ca[:, -1:]
    Cp /= Cp[:, -1:]
    edges = np.arange(n + 1) - 0.5
    q = np.linspace(0, 1, 2 * n + 1)[1:-1]
    d = np.zeros((len(A), n))
    for k in range(len(A)):
        ya, yp = np.interp(q, Ca[k], edges), np.interp(q, Cp[k], edges)
        d[k] = np.interp(np.arange(n), 0.5 * (ya + yp), 0.5 * (ya - yp))
    d = np.moveaxis(d.reshape(nx, nz, n), -1, 1)
    w = np.minimum(ia, ip)
    w = w / w.max()
    return ndimage.gaussian_filter(w * d, smooth_vox) / (ndimage.gaussian_filter(w, smooth_vox) + 1e-6)


class Dipole:
    """d = D chi + H c on one grid: chi on `source`, D by a 2x zero-padded rFFT, H 9 solid harmonics (unit norm)."""

    def __init__(self, shape, zooms, b_vox, source):
        self.shape, self.S, self.ns = shape, source, int(source.sum())
        self.pad = [n // 2 + 2 for n in shape]
        self.ps = [n + 2 * p for n, p in zip(shape, self.pad)]
        k = [fft.fftfreq(n, d=z) for n, z in zip(self.ps[:2], zooms[:2])] + [fft.rfftfreq(self.ps[2], d=zooms[2])]
        kx, ky, kz = np.meshgrid(*k, indexing='ij', sparse=True)
        k2 = kx ** 2 + ky ** 2 + kz ** 2
        kb = kx * b_vox[0] + ky * b_vox[1] + kz * b_vox[2]
        self.D = np.where(k2 > 0, 1.0 / 3 - kb ** 2 / np.where(k2 > 0, k2, 1), 0.0)
        g = np.indices(shape).astype(float)
        x, y, z = [(g[i] - (shape[i] - 1) / 2) * zooms[i] / 100.0 for i in range(3)]
        H = [np.ones(shape), x, y, z, x * y, y * z, x * z, x * x - y * y, 2 * z * z - x * x - y * y]
        self.H = np.stack([h / np.linalg.norm(h) for h in H], -1)

    def _D(self, chi):
        f = fft.rfftn(np.pad(chi, [(p, p) for p in self.pad]), workers=1)
        out = fft.irfftn(f * self.D, s=self.ps, workers=1)
        return out[tuple(slice(p, p + n) for p, n in zip(self.pad, self.shape))]

    def field(self, x):
        chi = np.zeros(self.shape)
        chi[self.S] = x[:self.ns]
        return self._D(chi) + self.H @ x[self.ns:]

    def field_t(self, w):
        return np.r_[self._D(w)[self.S], np.tensordot(self.H, w, axes=([0, 1, 2], [0, 1, 2]))]


def dipole_start(a, p, zooms, b_vox, lam=1e-5, min_mm=DIPOLE_MM, gn_iters=3, cg_iters=15, shell_k=2):
    """The start D chi + H c (full-grid voxels) of normalised images a / p."""
    f = 1
    while min(zooms) * f < min_mm:
        f *= 2
    ia = ndimage.gaussian_filter(down(a, f), 1.0)
    ip = ndimage.gaussian_filter(down(p, f), 1.0)
    sig = ndimage.gaussian_filter(0.5 * (ia + ip), 1.0)
    m = sig > 0.25 * np.percentile(sig, 99)
    model = Dipole(ia.shape, [z * f for z in zooms], b_vox, ~ndimage.binary_erosion(m, iterations=shell_k))
    n = model.ns + 9
    reg = np.r_[np.full(model.ns, lam), np.zeros(9)]
    w = np.minimum(ia, ip)
    w = w / w.max()
    A = LinearOperator((n, n), dtype=np.float64, matvec=lambda x: model.field_t(w * model.field(x)) + reg * x)
    x = cg(A, model.field_t(w * line_start(ia, ip)), maxiter=cg_iters)[0]

    def cost(x):
        d = model.field(x)
        r = shift(ia, d)[0] * (1 + dy(d)) - shift(ip, -d)[0] * (1 - dy(d))
        return 0.5 * float((r * r).sum()) + 0.5 * float((reg * x * x).sum())
    for _ in range(gn_iters):
        d = model.field(x)
        aa, ga = shift(ia, d)
        pp, gp = shift(ip, -d)
        g = dy(d)
        r = aa * (1 + g) - pp * (1 - g)
        ja, jb = ga * (1 + g) + gp * (1 - g), aa + pp
        jdt = lambda u: ja * u + dy_t(jb * u)
        grad = model.field_t(jdt(r)) + reg * x
        H = LinearOperator((n, n), dtype=np.float64, matvec=lambda v: model.field_t(
            jdt(ja * model.field(v) + jb * dy(model.field(v)))) + reg * v)
        step = cg(H, -grad, maxiter=cg_iters)[0]
        e0, t = cost(x), 1.0
        while t > 1e-3:
            if cost(x + t * step) <= e0 + 1e-4 * t * float(grad @ step):
                x = x + t * step
                break
            t *= 0.5
        else:
            break
    return upsample(model.field(x), a.shape, f)


# ---------------------------------------------------------------- motion

def _shift_of(M, c):
    """Translation of the rigid M at the point c."""
    return M[:3, :3] @ c + M[:3, 3] - c


def field_on(d, aff, grid_aff, shape, M=None):
    """The displacement d (on the aff grid) at the voxels of another grid; M: that grid's world -> aff world."""
    vox = np.linalg.inv(aff) @ (np.eye(4) if M is None else M) @ grid_aff
    return ndimage.affine_transform(d, vox[:3, :3], vox[:3, 3], output_shape=tuple(shape), order=1, mode='nearest')


def series_motion(ref, ref_aff, fixed, aff, d, pe_axis, sign, iters=2):
    """Rigid M (series world -> forward world) of a series' reference image `ref` (distorted, on its own grid with
    phase axis pe_axis and polarity sign there) to the corrected forward image `fixed`: ref unwarped with the field d
    (forward grid, scaled to the series' readout time) at the current M, registered, repeated."""
    M = np.eye(4)
    mov = nib.Nifti1Image(np.asarray(fixed, np.float32), aff)
    mov_mask = head_mask(fixed, fixed)
    for _ in range(iters):
        u = unwarp(ref, field_on(d, aff, ref_aff, ref.shape, M), ref_aff, pe_axis, sign=sign)
        M = numreg.register(mov, nib.Nifti1Image(u.astype(np.float32), ref_aff), moving_mask=mov_mask,
                            fixed_mask=head_mask(u, u), dof=6, metric='ncc', levels=MOTION_LEVELS, n_starts=1)
    return M


def motion_size(M, c):
    """(mm at the point c, degrees) of the rigid M."""
    return (float(np.linalg.norm(_shift_of(M, c))),
            float(np.degrees(np.linalg.norm(Rotation.from_matrix(M[:3, :3]).as_rotvec()))))


def rigid_motion(cf, cr, aff, mask, pe_world):
    """numreg rigid transform (forward world -> reverse world) of the unwarped reverse scan to the unwarped forward
    one, without the translation along the phase axis (a constant field offset)."""
    mov, fix = nib.Nifti1Image(cr.astype(np.float32), aff), nib.Nifti1Image(cf.astype(np.float32), aff)
    M = numreg.register(mov, fix, moving_mask=mask, fixed_mask=mask, dof=6, metric='ncc', levels=MOTION_LEVELS,
                        n_starts=1)
    c = (aff @ np.r_[np.argwhere(mask).mean(0), 1])[:3]
    s = _shift_of(M, c)
    s[pe_world] = 0.0
    M = M.copy()
    M[:3, 3] = c + s - M[:3, :3] @ c
    return M


def move(img, aff, M):
    """img (on the grid aff) moved by the rigid M (forward world -> its world): img(M x)."""
    ref = nib.Nifti1Image(np.asarray(img, np.float32), aff)
    return numreg.apply(ref, ref, M, order=1)


class Joint:
    """Gauss-Newton on (d, theta): P(x) = Ip(V_theta x - d e)(1 - g); theta = rotation vector and the two world
    translations across the phase axis, eliminated per step through the Schur complement of the d block."""

    def __init__(self, a, p, aff, d0, M0, f, prior_alpha=ALPHA):
        self.f = f
        self.ia, self.ip = down(a, f), down(p, f)
        s = np.eye(4)
        s[:3, :3] *= f
        s[:3, 3] = (f - 1) / 2.0
        self.aff = aff @ s
        self.inv = np.linalg.inv(self.aff)
        self.d = down(d0, f) / f
        grid = np.indices(self.ia.shape).reshape(3, -1).astype(float)
        self.xw = (self.aff @ np.r_[grid, np.ones((1, grid.shape[1]))])[:3]
        self.c = self.xw[:, ::7].mean(1)
        self.free = [0, 1, 2] + [3 + k for k in range(3) if k != int(np.argmax(np.abs(aff[:3, 1])))]
        self.theta = np.r_[Rotation.from_matrix(M0[:3, :3]).as_rotvec(), _shift_of(M0, self.c)]
        self.prior = Prior(shell_weight(correct(self.ia, self.d), correct(self.ia, self.d)), prior_alpha)

    def _q(self, theta):
        R = Rotation.from_rotvec(theta[:3]).as_matrix()
        qw = R @ (self.xw - self.c[:, None]) + self.c[:, None] + theta[3:, None]
        return (self.inv[:3, :3] @ qw + self.inv[:3, 3:4]).reshape((3,) + self.ia.shape)

    def _sample(self, d, theta, grad=False):
        a, ga = shift(self.ia, d)
        q = self._q(theta)
        flat = q.copy()
        flat[1] -= d
        flat = flat.reshape(3, -1)
        p = ndimage.map_coordinates(self.ip, flat, order=1, mode='nearest').reshape(d.shape)
        g = dy(d)
        r = a * (1 + g) - p * (1 - g)
        if not grad:
            return r
        gp = []                       # gradient of the trilinear interpolant itself (consistent with the cost)
        for k in range(3):
            hi, lo = flat.copy(), flat.copy()
            hi[k] += 1e-3
            lo[k] -= 1e-3
            gp.append(((ndimage.map_coordinates(self.ip, hi, order=1, mode='nearest')
                        - ndimage.map_coordinates(self.ip, lo, order=1, mode='nearest')) / 2e-3).reshape(d.shape))
        return r, a, ga, p, gp, g, q

    def _cost(self, d, theta):
        r = self._sample(d, theta)
        return 0.5 * float((r * r).sum()) + self.prior.energy(d)

    def step(self, cg_iters=30):
        d, th = self.d, self.theta
        r, a, ga, p, gp, g, q = self._sample(d, th, grad=True)
        ja, jb = ga * (1 + g) + gp[1] * (1 - g), a + p
        sh, n = d.shape, d.size
        cols = []
        for k in self.free:
            e = np.zeros(6)
            e[k] = 1e-4
            dq = (self._q(th + e) - q) / 1e-4
            cols.append((-(1 - g) * sum(gp[i] * dq[i] for i in range(3))).ravel())
        Jt = np.stack(cols, 1)
        jdt = lambda w: ja * w + dy_t(jb * w)
        g_d = (jdt(r) + self.prior.hess(d)).ravel()
        g_t = Jt.T @ r.ravel()

        def hv(v):
            v = v.reshape(sh)
            return (jdt(ja * v + jb * dy(v)) + self.prior.hess(v)).ravel()
        diag = (ja * ja + 0.5 * jb * jb + self.prior.diag + 1e-6).ravel()
        H = LinearOperator((n, n), matvec=hv, dtype=np.float64)
        M = LinearOperator((n, n), matvec=lambda v: v / diag, dtype=np.float64)
        solve = lambda b: cg(H, b, M=M, maxiter=cg_iters)[0]
        Hdt = np.stack([jdt(Jt[:, k].reshape(sh)).ravel() for k in range(Jt.shape[1])], 1)
        Z = np.stack([solve(Hdt[:, k]) for k in range(Hdt.shape[1])], 1)
        y = solve(g_d)
        S = Jt.T @ Jt - Hdt.T @ Z + 1e-3 * np.eye(Jt.shape[1]) * np.trace(Jt.T @ Jt) / Jt.shape[1]
        dth_free = np.linalg.solve(S, -g_t + Hdt.T @ y)
        dd = -(y + Z @ dth_free).reshape(sh)
        dth = np.zeros(6)
        dth[self.free] = dth_free
        e0, slope, t = self._cost(d, th), float(g_d @ dd.ravel() + g_t @ dth_free), 1.0
        while t > 1e-3:
            if self._cost(d + t * dd, th + t * dth) <= e0 + 1e-4 * t * slope:
                self.d, self.theta = d + t * dd, th + t * dth
                return True
            t *= 0.5
        return False

    def run(self, iters=8):
        for _ in range(iters):
            if not self.step():
                break
        R = Rotation.from_rotvec(self.theta[:3]).as_matrix()
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = self.c + self.theta[3:] - R @ self.c
        return self.d, M


# ---------------------------------------------------------------- the whole estimate

def unwarp_map(d, aff, pe_axis, M=None, sign=1.0, src_aff=None):
    """(sample coordinates, Jacobian) of unwarp: voxel x of the aff grid takes the image at M(x) moved by sign * d(x)
    along the phase axis, in the voxels of the image's own grid src_aff (None: the aff grid)."""
    idx = np.indices(d.shape).reshape(3, -1).astype(np.float64)
    if M is not None:
        vox = np.linalg.inv(aff) @ M @ aff
        idx = (vox @ np.r_[idx, np.ones((1, idx.shape[1]))])[:3]
    idx[pe_axis] += sign * d.ravel()
    if src_aff is not None and not np.allclose(src_aff, aff, atol=1e-6):
        vox = np.linalg.inv(src_aff) @ aff
        idx = vox[:3, :3] @ idx + vox[:3, 3:]
    return idx, 1 + sign * np.gradient(d, axis=pe_axis)


def unwarp(img, d, aff, pe_axis, M=None, sign=1.0, order=3):
    """Image on the forward grid in one resampling (cubic B-spline): voxel x takes img at M(x) moved by sign * d(x)
    along the phase axis, times the Jacobian 1 + sign * dd/dpe. M: rigid forward world -> img world (None: same)."""
    idx, jac = unwarp_map(d, aff, pe_axis, M, sign)
    return ndimage.map_coordinates(np.asarray(img, np.float64), idx, order=order, mode='nearest').reshape(d.shape) * jac


def head_mask(fwd, rev):
    mean = 0.5 * (fwd / fwd.mean() + rev / rev.mean())
    return ndimage.binary_dilation(ndimage.binary_fill_holes(mean > 0.15 * np.percentile(mean, 99)), iterations=2)


def _final_level(zooms):
    f = 1
    while min(zooms) * f * 2 <= MAX_MM:
        f *= 2
    return f


def estimate(fwd, rev, aff, pe_axis=1, joint='auto', mask=None, margin=6):
    """Displacement d (voxels along `pe_axis`; the forward scan, positive phase-encoding polarity, is unwarped by
    fwd(x + d e)) and the rigid motion M (forward world -> reverse world) of a forward / reverse pair on one grid.
    joint: 'auto' (field + motion jointly when the motion exceeds 1 mm / 1 degree), True or False.
    Returns (d, M, info)."""
    t0 = time.time()
    fwd = np.asarray(fwd, np.float64)
    rev = np.asarray(rev, np.float64)
    if mask is None:
        mask = head_mask(fwd, rev)
    order = _SWAP[pe_axis]
    tr = lambda x: np.transpose(x, order)
    aff_t = aff[:, list(order) + [3]]                                 # the same world, voxel axes permuted
    zooms = [float(z) for z in np.sqrt((aff_t[:3, :3] ** 2).sum(0))]
    pe_world = int(np.argmax(np.abs(aff_t[:3, 1])))
    b_vox = (aff_t[:3, :3] / np.asarray(zooms)).T @ np.array([0.0, 0.0, 1.0])  # B0 (world z) in voxel axes
    F, Rv, Mk = tr(fwd), tr(rev), tr(mask)
    idx = np.argwhere(Mk)
    lo, hi = np.maximum(idx.min(0) - margin, 0), np.minimum(idx.max(0) + 1 + margin, Mk.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    shift_aff = np.eye(4)
    shift_aff[:3, 3] = lo
    aff_c = aff_t @ shift_aff                                         # the cropped grid
    f = _final_level(zooms)

    def solve(rev_arr, start, iters):
        a, p = normalise(F[sl], rev_arr[sl])
        return refine(a, p, start, f, zooms, iters)

    a, p = normalise(F[sl], Rv[sl])
    d = solve(Rv, dipole_start(a, p, zooms, b_vox), 8)
    M = rigid_motion(correct(F[sl], d), correct(Rv[sl], -d), aff_c, Mk[sl], pe_world)
    c = (aff_c @ np.r_[np.argwhere(Mk[sl]).mean(0), 1])[:3]
    deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(M[:3, :3]).as_rotvec())))
    mm = float(np.linalg.norm(_shift_of(M, c)))
    use_joint = (deg > JOINT_DEG or mm > JOINT_MM) if joint == 'auto' else bool(joint)
    if use_joint:
        a, p = normalise(F[sl], Rv[sl])
        dj, M = Joint(a, p, aff_c, d, M, f).run()
        d = upsample(dj, a.shape, f)
    else:
        d = solve(move(Rv, aff_t, M), d, 5)
    out = np.zeros(F.shape)
    out[sl] = d
    out = np.pad(out[sl], [(s.start, n - s.stop) for s, n in zip(sl, F.shape)], mode='edge')
    c = (aff_t @ np.r_[np.argwhere(Mk).mean(0), 1])[:3]
    mm, deg = motion_size(M, c)
    info = dict(motion_mm=mm, motion_deg=deg,
                joint=bool(use_joint), final_level=f, seconds=time.time() - t0)
    return tr(out), M, info
