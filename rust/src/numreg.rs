//! Affine registration, a port of TigerBx `tigerbx/core/numreg.py` (`register`, dof = 12):
//! Mattes mutual information, symmetric (forward + inverse) cost, 6 / 3 / 1.5 mm pyramid, L-BFGS, multi-start.
//!
//! Transforms are 4x4 world (RAS, mm) matrices M mapping a FIXED world point to the MOVING point it pulls from:
//! y_moving = M x_fixed. Results agree with numreg closely but not bit for bit: the 1.5 mm level draws its sample
//! subset with another random generator and the L-BFGS line search differs from scipy's L-BFGS-B.

use ndarray::{Array3, Axis};
use rayon::prelude::*;

use crate::resample::{col_norm, linear_transform, mat_inv, mat_mul, Mat4};

/// (step mm, sample fraction, max L-BFGS iterations, band margin mm, band weight), numreg DEFAULT_LEVELS.
const LEVELS: [(f64, f64, usize, f64, f64); 3] = [(6.0, 1.0, 60, 6.0, 1.0), (3.0, 1.0, 60, 6.0, 1.0), (1.5, 0.25, 40, 6.0, 0.2)];
const BINS: usize = 32;
const R_SCALE: f64 = 50.0;
const N_STARTS: usize = 2;

// ── small linear algebra ──────────────────────────────────────────────────────

type V3 = [f64; 3];

fn mat3_inv(a: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut m = [[0.0; 4]; 4];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = a[i][j];
        }
    }
    m[3][3] = 1.0;
    let inv = mat_inv(&m);
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = inv[i][j];
        }
    }
    r
}

fn affine_apply(m: &Mat4, x: &V3) -> V3 {
    [
        m[0][0] * x[0] + m[0][1] * x[1] + m[0][2] * x[2] + m[0][3],
        m[1][0] * x[0] + m[1][1] * x[1] + m[1][2] * x[2] + m[1][3],
        m[2][0] * x[0] + m[2][1] * x[1] + m[2][2] * x[2] + m[2][3],
    ]
}

/// numpy.percentile(..., q) with linear interpolation.
fn percentile(mut v: Vec<f64>, q: f64) -> f64 {
    if v.is_empty() {
        return 1.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = (v.len() - 1) as f64 * q / 100.0;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(v.len() - 1);
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
}

/// Deterministic uniform [0, 1) generator (splitmix64) for the sample subsets.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ── pyramid ───────────────────────────────────────────────────────────────────

/// scipy.ndimage.gaussian_filter(data, sigma, mode='constant') (truncate 4; axes with sigma 0 skipped).
fn gaussian_filter(data: &Array3<f64>, sigma: [f64; 3]) -> Array3<f64> {
    let mut cur = data.clone();
    for ax in 0..3 {
        let sd = sigma[ax];
        if sd <= 1e-15 {
            continue;
        }
        let r = (4.0 * sd + 0.5) as isize;
        let mut w: Vec<f64> = (-r..=r).map(|x| (-0.5 / (sd * sd) * (x * x) as f64).exp()).collect();
        let sum: f64 = w.iter().sum();
        w.iter_mut().for_each(|v| *v /= sum);
        let mut out = Array3::<f64>::zeros(cur.dim());
        let n = cur.len_of(Axis(ax)) as isize;
        let src: Vec<_> = cur.lanes(Axis(ax)).into_iter().collect();
        let mut dst: Vec<_> = out.lanes_mut(Axis(ax)).into_iter().collect();
        dst.par_iter_mut().zip(src.par_iter()).for_each(|(o, l)| {
            let line: Vec<f64> = l.iter().cloned().collect();
            for (i, v) in o.iter_mut().enumerate() {
                let mut s = 0.0;
                for (k, wk) in w.iter().enumerate() {
                    let j = i as isize + k as isize - r;
                    if j >= 0 && j < n {
                        s += wk * line[j as usize];
                    }
                }
                *v = s;
            }
        });
        cur = out;
    }
    cur
}

/// numreg `_downsample`: Gaussian-smooth and resample to isotropic `step` mm. Returns (array, vox2world).
fn downsample(data: &Array3<f64>, aff: &Mat4, step: f64) -> (Array3<f64>, Mat4) {
    let f: [f64; 3] = [0, 1, 2].map(|j| step / col_norm(aff, j));
    let sig = f.map(|fj| (fj * fj - 1.0).max(0.0).sqrt() * 0.5);
    let smooth = if sig.iter().any(|&s| s > 0.0) { gaussian_filter(data, sig) } else { data.clone() };
    let d = data.dim();
    let dims = [d.0, d.1, d.2];
    let shape: [usize; 3] = [0, 1, 2].map(|a| ((dims[a] as f64 / f[a]).ceil() as usize).max(2));
    let mut s = [[0.0; 4]; 4];
    for a in 0..3 {
        s[a][a] = f[a];
        s[a][3] = (f[a] - 1.0) / 2.0;
    }
    s[3][3] = 1.0;
    let out = linear_transform(&smooth, &s, (shape[0], shape[1], shape[2]));
    (out.mapv(|v| v as f32 as f64), mat_mul(aff, &s))
}

struct Level {
    data: Array3<f64>,
    mask: Array3<bool>,
    aff: Mat4,
}

/// numreg `_pyramid`: one level per step (finest first), each from the next finer one.
fn pyramid(data: &Array3<f64>, mask: &Array3<bool>, aff: &Mat4) -> Vec<(f64, Level)> {
    let mut steps: Vec<f64> = LEVELS.iter().map(|l| l.0).collect();
    steps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut cur_d = data.clone();
    let mut cur_m = mask.mapv(|b| b as u8 as f64);
    let mut cur_a = *aff;
    let mut out = Vec::new();
    for st in steps {
        let (d, a) = downsample(&cur_d, &cur_a, st);
        let (m, _) = downsample(&cur_m, &cur_a, st);
        cur_d = d;
        cur_m = m;
        cur_a = a;
        out.push((st, Level { data: cur_d.clone(), mask: cur_m.mapv(|v| v > 0.5), aff: a }));
    }
    out
}

fn level(pyr: &[(f64, Level)], step: f64) -> &Level {
    &pyr.iter().find(|(s, _)| *s == step).unwrap().1
}

/// One 6-connected binary dilation step (scipy default structure, border 0).
fn dilate(m: &Array3<bool>) -> Array3<bool> {
    let (nx, ny, nz) = m.dim();
    let mut out = m.clone();
    for ((i, j, k), &v) in m.indexed_iter() {
        if !v {
            continue;
        }
        for (a, b, c) in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)] {
            let (x, y, z) = (i as isize + a, j as isize + b, k as isize + c);
            if x >= 0 && y >= 0 && z >= 0 && (x as usize) < nx && (y as usize) < ny && (z as usize) < nz {
                out[[x as usize, y as usize, z as usize]] = true;
            }
        }
    }
    out
}

/// numreg `_samples`: points in the mask plus a `margin` mm band (band weight `edge_w`), a random `frac` of them.
fn samples(lev: &Level, margin: f64, frac: f64, rng: &mut Rng, edge_w: f64) -> (Vec<V3>, Vec<f64>, Vec<f64>) {
    let step = col_norm(&lev.aff, 0);
    let mut sel = lev.mask.clone();
    if margin > 0.0 {
        for _ in 0..((margin / step).round() as usize).max(1) {
            sel = dilate(&sel);
        }
    }
    let (mut xs, mut fs, mut ws) = (Vec::new(), Vec::new(), Vec::new());
    for ((i, j, k), &s) in sel.indexed_iter() {
        if !s {
            continue;
        }
        if frac < 1.0 && rng.next() >= frac {
            continue;
        }
        xs.push(affine_apply(&lev.aff, &[i as f64, j as f64, k as f64]).map(|v| v as f32 as f64));
        fs.push(lev.data[[i, j, k]]);
        ws.push(if lev.mask[[i, j, k]] { 1.0 } else { edge_w });
    }
    let sum: f64 = ws.iter().sum();
    ws.iter_mut().for_each(|w| *w /= sum);
    (xs, fs, ws)
}

// ── trilinear sampling with gradient ─────────────────────────────────────────

/// numreg `_Vol`: zero-padded (2 voxels) volume, trilinear value and world-space gradient.
struct Vol {
    v: Vec<f64>,
    st: [usize; 3],
    w2v: Mat4,
    hi: V3,
}

impl Vol {
    fn new(data: &Array3<f64>, aff: &Mat4) -> Vol {
        const P: usize = 2;
        let (a, b, c) = data.dim();
        let shape = [a + 2 * P, b + 2 * P, c + 2 * P];
        let mut v = vec![0.0; shape[0] * shape[1] * shape[2]];
        for ((i, j, k), &x) in data.indexed_iter() {
            v[((i + P) * shape[1] + j + P) * shape[2] + k + P] = x;
        }
        let mut w2v = mat_inv(aff);
        for r in w2v.iter_mut().take(3) {
            r[3] += P as f64;
        }
        Vol { v, st: [shape[1] * shape[2], shape[2], 1], w2v, hi: shape.map(|n| n as f64 - 1.001) }
    }

    /// values and d value / d y at world points y.
    fn sample(&self, y: &[V3]) -> (Vec<f64>, Vec<V3>) {
        let w = &self.w2v;
        y.par_iter()
            .map(|p| {
                let mut u = affine_apply(w, p);
                for a in 0..3 {
                    u[a] = u[a].clamp(0.0, self.hi[a]);
                }
                let i0 = u.map(|x| x as usize);
                let (fx, fy, fz) = (u[0] - i0[0] as f64, u[1] - i0[1] as f64, u[2] - i0[2] as f64);
                let b = i0[0] * self.st[0] + i0[1] * self.st[1] + i0[2];
                let (sx, sy, sz) = (self.st[0], self.st[1], 1);
                let v = &self.v;
                let (c000, c001, c010, c011) = (v[b], v[b + sz], v[b + sy], v[b + sy + sz]);
                let (c100, c101, c110, c111) = (v[b + sx], v[b + sx + sz], v[b + sx + sy], v[b + sx + sy + sz]);
                let c00 = c000 + fz * (c001 - c000);
                let c01 = c010 + fz * (c011 - c010);
                let c10 = c100 + fz * (c101 - c100);
                let c11 = c110 + fz * (c111 - c110);
                let c0 = c00 + fy * (c01 - c00);
                let c1 = c10 + fy * (c11 - c10);
                let val = c0 + fx * (c1 - c0);
                let gx = c1 - c0;
                let gy = (c01 - c00) + fx * ((c11 - c10) - (c01 - c00));
                let d0 = (c001 - c000) + fy * (c011 - c001 - c010 + c000);
                let d1 = (c101 - c100) + fy * (c111 - c101 - c110 + c100);
                let gz = d0 + fx * (d1 - d0);
                // chain rule voxel -> world: g_world = g_vox @ w2v[:3, :3]
                let g = [0, 1, 2].map(|j| gx * w[0][j] + gy * w[1][j] + gz * w[2][j]);
                (val, g)
            })
            .unzip()
    }
}

// ── Mattes mutual information ────────────────────────────────────────────────

/// numreg `_MI`: box Parzen window on the fixed samples, cubic B-spline on the moving values (maximised).
struct Mi {
    w: Vec<f64>,
    kf: Vec<usize>,
    mscale: f64,
}

impl Mi {
    fn new(f: &[f64], w: &[f64], mhi: f64) -> Mi {
        let pos: Vec<f64> = f.iter().cloned().filter(|&v| v > 0.0).collect();
        let fhi = if pos.is_empty() { 1.0 } else { percentile(pos, 99.5) };
        let kf = f.iter().map(|&v| (v / fhi * (BINS - 1) as f64).clamp(0.0, (BINS - 1) as f64) as usize).collect();
        Mi { w: w.to_vec(), kf, mscale: (BINS - 5) as f64 / mhi }
    }

    fn eval(&self, m: &[f64]) -> (f64, Vec<f64>) {
        let b = BINS;
        let n = m.len();
        let mut idx = vec![0usize; n];
        let mut bs = vec![[0.0f64; 4]; n];
        let mut dbs = vec![[0.0f64; 4]; n];
        let mut joint = vec![0.0f64; b * b];
        for i in 0..n {
            let z = (m[i] * self.mscale).clamp(0.0, (b - 5) as f64) + 2.0;
            let k0 = z as usize;
            let t = z - k0 as f64;
            let (t2, t3, u) = (t * t, t * t * t, 1.0 - t);
            bs[i] = [u * u * u / 6.0, (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0, (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0, t3 / 6.0];
            dbs[i] = [-u * u / 2.0, (3.0 * t2 - 4.0 * t) / 2.0, (-3.0 * t2 + 2.0 * t + 1.0) / 2.0, t2 / 2.0];
            idx[i] = self.kf[i] * b + k0 - 1;
            for o in 0..4 {
                joint[idx[i] + o] += self.w[i] * bs[i][o];
            }
        }
        let total: f64 = joint.iter().sum();
        let p: Vec<f64> = joint.iter().map(|v| v / total).collect();
        let pf: Vec<f64> = (0..b).map(|r| (0..b).map(|c| p[r * b + c]).sum()).collect();
        let pm: Vec<f64> = (0..b).map(|c| (0..b).map(|r| p[r * b + c]).sum()).collect();
        let mut mi = 0.0;
        let mut l = vec![0.0f64; b * b];
        for r in 0..b {
            for c in 0..b {
                let v = p[r * b + c];
                if v > 0.0 {
                    mi += v * (v / (pf[r] * pm[c])).ln();
                    l[r * b + c] = (v / pm[c]).ln();
                }
            }
        }
        let grad = (0..n)
            .map(|i| {
                let g: f64 = (0..4).map(|o| dbs[i][o] * l[idx[i] + o]).sum();
                let s = m[i] * self.mscale;
                if s > 0.0 && s < (b - 5) as f64 { self.w[i] * g * self.mscale } else { 0.0 }
            })
            .collect();
        (mi, grad)
    }
}

// ── cost and optimiser ───────────────────────────────────────────────────────

type P12 = [f64; 12];

fn params_to_m(p: &P12, c: &V3) -> Mat4 {
    let mut m = [[0.0; 4]; 4];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = (i == j) as u8 as f64 + p[3 * i + j] / R_SCALE;
        }
    }
    for i in 0..3 {
        m[i][3] = c[i] + p[9 + i] - (0..3).map(|j| m[i][j] * c[j]).sum::<f64>();
    }
    m[3][3] = 1.0;
    m
}

fn m_to_params(m: &Mat4, c: &V3) -> P12 {
    let mut p = [0.0; 12];
    for i in 0..3 {
        for j in 0..3 {
            p[3 * i + j] = (m[i][j] - (i == j) as u8 as f64) * R_SCALE;
        }
        p[9 + i] = m[i][3] - c[i] + (0..3).map(|j| m[i][j] * c[j]).sum::<f64>();
    }
    p
}

struct Term<'a> {
    x: &'a [V3],
    vol: &'a Vol,
    mi: &'a Mi,
}

/// numreg `_cost`: -MI(fixed samples, moving at M x) averaged with the inverse term -MI(moving samples, fixed at
/// M^-1 y); analytic gradient.
fn cost(p: &P12, c: &V3, fwd: &Term, inv: &Term) -> (f64, P12) {
    let m = params_to_m(p, c);
    let y: Vec<V3> = fwd.x.par_iter().map(|x| affine_apply(&m, x)).collect();
    let (mv, g) = fwd.vol.sample(&y);
    let (val, dm) = fwd.mi.eval(&mv);
    let mut ga = [[0.0; 3]; 3];
    let mut gt = [0.0; 3];
    for n in 0..y.len() {
        let gn = [g[n][0] * dm[n], g[n][1] * dm[n], g[n][2] * dm[n]];
        for i in 0..3 {
            gt[i] += gn[i];
            for j in 0..3 {
                ga[i][j] += gn[i] * (fwd.x[n][j] - c[j]) / R_SCALE;
            }
        }
    }
    let a = [[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]];
    let ai = mat3_inv(&a);
    let t = [m[0][3], m[1][3], m[2][3]];
    let z: Vec<V3> = inv
        .x
        .par_iter()
        .map(|yb| {
            let d = [yb[0] - t[0], yb[1] - t[1], yb[2] - t[2]];
            [0, 1, 2].map(|i| ai[i][0] * d[0] + ai[i][1] * d[1] + ai[i][2] * d[2])
        })
        .collect();
    let (fv, gz) = inv.vol.sample(&z);
    let (vb, dfb) = inv.mi.eval(&fv);
    let mut ha = [[0.0; 3]; 3];
    let mut ht = [0.0; 3];
    for n in 0..z.len() {
        // H = -(gz * dfb) @ Ai
        let h = [0, 1, 2].map(|j| -(0..3).map(|i| gz[n][i] * dfb[n] * ai[i][j]).sum::<f64>());
        for i in 0..3 {
            ht[i] += h[i];
            for j in 0..3 {
                ha[i][j] += h[i] * (z[n][j] - c[j]) / R_SCALE;
            }
        }
    }
    let mut grad = [0.0; 12];
    for i in 0..3 {
        for j in 0..3 {
            grad[3 * i + j] = -0.5 * (ga[i][j] + ha[i][j]);
        }
        grad[9 + i] = -0.5 * (gt[i] + ht[i]);
    }
    (-0.5 * (val + vb), grad)
}

fn dot(a: &P12, b: &P12) -> f64 {
    (0..12).map(|i| a[i] * b[i]).sum()
}

/// L-BFGS (memory 10) with a strong-Wolfe line search (c1 = 1e-3, c2 = 0.9, <= 20 evaluations), stopping as
/// scipy's L-BFGS-B without bounds: max |g| <= 1e-5, relative decrease <= 2.2e-9, or `maxiter` iterations.
fn lbfgs<F: Fn(&P12) -> (f64, P12)>(f: &F, x0: P12, maxiter: usize) -> (f64, P12) {
    let mut x = x0;
    let (mut fx, mut g) = f(&x);
    let mut hist: Vec<(P12, P12)> = Vec::new();
    for it in 0..maxiter {
        if g.iter().fold(0.0f64, |a, v| a.max(v.abs())) <= 1e-5 {
            break;
        }
        // two-loop recursion
        let mut q = g;
        let mut alpha = vec![0.0; hist.len()];
        for (k, (s, y)) in hist.iter().enumerate().rev() {
            let rho = 1.0 / dot(y, s);
            alpha[k] = rho * dot(s, &q);
            for i in 0..12 {
                q[i] -= alpha[k] * y[i];
            }
        }
        if let Some((s, y)) = hist.last() {
            let gamma = dot(s, y) / dot(y, y);
            q.iter_mut().for_each(|v| *v *= gamma);
        }
        for (k, (s, y)) in hist.iter().enumerate() {
            let rho = 1.0 / dot(y, s);
            let beta = rho * dot(y, &q);
            for i in 0..12 {
                q[i] += s[i] * (alpha[k] - beta);
            }
        }
        let mut d = q.map(|v| -v);
        let mut dphi0 = dot(&g, &d);
        if dphi0 >= 0.0 {
            hist.clear();
            d = g.map(|v| -v);
            dphi0 = dot(&g, &d);
        }
        let a0 = if it == 0 { 1.0 / dot(&d, &d).sqrt() } else { 1.0 };
        let Some((xn, fnew, gn)) = line_search(f, &x, fx, &d, dphi0, a0) else { break };
        let s: P12 = std::array::from_fn(|i| xn[i] - x[i]);
        let y: P12 = std::array::from_fn(|i| gn[i] - g[i]);
        if dot(&s, &y) > f64::EPSILON * dot(&y, &y) {
            hist.push((s, y));
            if hist.len() > 10 {
                hist.remove(0);
            }
        }
        let rel = (fx - fnew) / fx.abs().max(fnew.abs()).max(1.0);
        x = xn;
        fx = fnew;
        g = gn;
        if rel <= 2.2e-9 {
            break;
        }
    }
    (fx, x)
}

fn line_search<F: Fn(&P12) -> (f64, P12)>(f: &F, x: &P12, f0: f64, d: &P12, dphi0: f64, a0: f64) -> Option<(P12, f64, P12)> {
    const C1: f64 = 1e-3;
    const C2: f64 = 0.9;
    const MAX_EVAL: usize = 20;
    let eval = |a: f64| {
        let xa: P12 = std::array::from_fn(|i| x[i] + a * d[i]);
        let (fa, ga) = f(&xa);
        (xa, fa, ga)
    };
    let mut n = 0;
    let (mut a_lo, mut f_lo, mut dphi_lo) = (0.0, f0, dphi0);
    let mut a = a0;
    let mut a_hi;
    let mut f_hi;
    let mut best: Option<(P12, f64, P12)> = None;
    loop {
        let (xa, fa, ga) = eval(a);
        n += 1;
        let dphi = dot(&ga, d);
        if fa > f0 + C1 * a * dphi0 || (n > 1 && fa >= f_lo) {
            a_hi = a;
            f_hi = fa;
            break;
        }
        if dphi.abs() <= -C2 * dphi0 {
            return Some((xa, fa, ga));
        }
        best = Some((xa, fa, ga));
        if dphi >= 0.0 {
            a_hi = a_lo;
            f_hi = f_lo;
            a_lo = a;
            f_lo = fa;
            dphi_lo = dphi;
            break;
        }
        a_lo = a;
        f_lo = fa;
        dphi_lo = dphi;
        a *= 2.0;
        if n >= MAX_EVAL {
            return best;
        }
    }
    // zoom between a_lo (satisfies sufficient decrease) and a_hi
    while n < MAX_EVAL {
        // quadratic interpolation from f_lo, dphi_lo, f_hi; safeguarded to the inner 80% of the interval
        let da = a_hi - a_lo;
        let denom = 2.0 * (f_hi - f_lo - dphi_lo * da);
        let mut aj = if denom.abs() > 1e-300 { a_lo - dphi_lo * da * da / denom } else { a_lo + 0.5 * da };
        let (lo, hi) = (a_lo.min(a_hi), a_lo.max(a_hi));
        let w = hi - lo;
        if !aj.is_finite() || aj < lo + 0.1 * w || aj > hi - 0.1 * w {
            aj = 0.5 * (a_lo + a_hi);
        }
        let (xa, fa, ga) = eval(aj);
        n += 1;
        let dphi = dot(&ga, d);
        if fa > f0 + C1 * aj * dphi0 || fa >= f_lo {
            a_hi = aj;
            f_hi = fa;
        } else {
            if dphi.abs() <= -C2 * dphi0 {
                return Some((xa, fa, ga));
            }
            if dphi * (a_hi - a_lo) >= 0.0 {
                a_hi = a_lo;
                f_hi = f_lo;
            }
            a_lo = aj;
            f_lo = fa;
            dphi_lo = dphi;
            best = Some((xa, fa, ga));
        }
        if (a_hi - a_lo).abs() < 1e-12 {
            break;
        }
    }
    best
}

fn rot(ax: usize, deg: f64) -> [[f64; 3]; 3] {
    let t = deg.to_radians();
    let (c, s) = (t.cos(), t.sin());
    let (i, j) = [(1, 2), (0, 2), (0, 1)][ax];
    let mut r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    r[i][i] = c;
    r[i][j] = -s;
    r[j][i] = s;
    r[j][j] = c;
    r
}

fn com(mask: &Array3<bool>, aff: &Mat4) -> (V3, f64) {
    let mut s = [0.0; 3];
    let mut n = 0usize;
    for ((i, j, k), &m) in mask.indexed_iter() {
        if m {
            s[0] += i as f64;
            s[1] += j as f64;
            s[2] += k as f64;
            n += 1;
        }
    }
    let mean = s.map(|v| v / n as f64);
    let a = [[aff[0][0], aff[0][1], aff[0][2]], [aff[1][0], aff[1][1], aff[1][2]], [aff[2][0], aff[2][1], aff[2][2]]];
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    (affine_apply(aff, &mean), n as f64 * det.abs())
}

/// Affine registration of `moving` (data, vox2world) to `fixed`; returns the world M (fixed -> moving).
/// As numreg.register(moving, fixed) with the moving mask = moving > 0 and the fixed mask = fixed > min.
pub fn register(mdata: &Array3<f64>, maff: &Mat4, fdata: &Array3<f64>, faff: &Mat4) -> Mat4 {
    let mut rng = Rng(0);
    // fixed: mask above the background value, shifted to 0 outside
    let fmin_all = fdata.iter().cloned().fold(f64::INFINITY, f64::min);
    let fmask = fdata.mapv(|v| v > fmin_all);
    let fmin = fdata.iter().zip(fmask.iter()).filter(|(_, &m)| m).map(|(&v, _)| v).fold(f64::INFINITY, f64::min);
    let fd = ndarray::Zip::from(fdata).and(&fmask).map_collect(|&v, &m| if m { v - fmin } else { 0.0 });
    let (cf, vf) = com(&fmask, faff);
    let fpyr = pyramid(&fd, &fmask, faff);

    // moving: crop to the mask bounding box + 10 mm
    let mmask = mdata.mapv(|v| v > 0.0);
    let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
    for ((i, j, k), &m) in mmask.indexed_iter() {
        if m {
            for (a, v) in [i, j, k].into_iter().enumerate() {
                lo[a] = lo[a].min(v);
                hi[a] = hi[a].max(v + 1);
            }
        }
    }
    let d = mdata.dim();
    let dims = [d.0, d.1, d.2];
    for a in 0..3 {
        let pad = (10.0 / col_norm(maff, a)).ceil() as usize;
        lo[a] = lo[a].saturating_sub(pad);
        hi[a] = (hi[a] + pad).min(dims[a]);
    }
    let sl = ndarray::s![lo[0]..hi[0], lo[1]..hi[1], lo[2]..hi[2]];
    let mmask_c = mmask.slice(sl).to_owned();
    let mdata_c = ndarray::Zip::from(mdata.slice(sl)).and(&mmask_c).map_collect(|&v, &m| if m { v } else { 0.0 });
    let mut maff_c = *maff;
    for i in 0..3 {
        maff_c[i][3] = (0..3).map(|j| maff[i][j] * lo[j] as f64).sum::<f64>() + maff[i][3];
    }
    let mpyr = pyramid(&mdata_c, &mmask_c, &maff_c);

    // initialisation: centre of mass + isotropic volume scaling, +-15 degree rotations about each axis
    let (cm, vm) = com(&mmask_c, &maff_c);
    let s = (vm / vf).powf(1.0 / 3.0);
    let c = cf;
    let base = [[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]];
    let mut inits = vec![base];
    for ax in 0..3 {
        for deg in [-15.0, 15.0] {
            let r = rot(ax, deg);
            inits.push(std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| base[i][k] * r[k][j]).sum())));
        }
    }
    let m0s: Vec<Mat4> = inits
        .iter()
        .map(|a| {
            let mut m = [[0.0; 4]; 4];
            for i in 0..3 {
                for j in 0..3 {
                    m[i][j] = a[i][j];
                }
                m[i][3] = cm[i] - (0..3).map(|j| a[i][j] * cf[j]).sum::<f64>();
            }
            m[3][3] = 1.0;
            m
        })
        .collect();

    let positive = |a: &Array3<f64>| a.iter().cloned().filter(|&v| v > 0.0).collect::<Vec<f64>>();
    let mut best = [0.0; 12];
    for (li, &(step, frac, iters, margin, edge_w)) in LEVELS.iter().enumerate() {
        let (fl, ml) = (level(&fpyr, step), level(&mpyr, step));
        let (x, f, w) = samples(fl, margin, frac, &mut rng, edge_w);
        let vol = Vol::new(&ml.data, &ml.aff);
        let mi = Mi::new(&f, &w, percentile(positive(&ml.data), 99.5));
        let (yb, fb, wb) = samples(ml, margin, frac, &mut rng, edge_w);
        let fvol = Vol::new(&fl.data, &fl.aff);
        let mib = Mi::new(&fb, &wb, percentile(positive(&fl.data), 99.5));
        let fwd = Term { x: &x, vol: &vol, mi: &mi };
        let inv = Term { x: &yb, vol: &fvol, mi: &mib };
        let fobj = |p: &P12| cost(p, &c, &fwd, &inv);
        let starts: Vec<P12> = if li == 0 {
            let mut cands: Vec<(f64, P12)> = m0s.iter().map(|m| lbfgs(&fobj, m_to_params(m, &c), 15)).collect();
            cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            cands.into_iter().take(N_STARTS).map(|(_, p)| p).collect()
        } else {
            vec![best]
        };
        let mut res: Vec<(f64, P12)> = starts.into_iter().map(|p0| lbfgs(&fobj, p0, iters)).collect();
        res.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        best = res[0].1;
    }
    params_to_m(&best, &c)
}
