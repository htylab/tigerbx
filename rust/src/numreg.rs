//! Affine registration, a port of TigerBx `tigerbx/core/numreg.py` (`register`, dof = 12):
//! Mattes mutual information, symmetric (forward + inverse) cost, 6 / 3 / 1.5 mm pyramid, L-BFGS, multi-start.
//! `register_with` takes numreg's other options (levels, starts, NCC, rigid dof = 6, given masks; duope).
//!
//! Transforms are 4x4 world (RAS, mm) matrices M mapping a FIXED world point to the MOVING point it pulls from:
//! y_moving = M x_fixed. Results agree with numreg closely but not bit for bit: the 1.5 mm level draws its sample
//! subset with another random generator and the L-BFGS line search differs from scipy's L-BFGS-B.

use ndarray::Array3;
use rayon::prelude::*;

use crate::resample::{col_norm, linear_transform, mat_inv, mat_mul, Mat4};

/// (step mm, sample fraction, max L-BFGS iterations, band margin mm, band weight), numreg DEFAULT_LEVELS.
const LEVELS: [(f64, f64, usize, f64, f64); 3] = [(6.0, 1.0, 60, 6.0, 1.0), (3.0, 1.0, 60, 6.0, 1.0), (1.5, 0.25, 40, 6.0, 0.2)];
/// Levels that add the inverse term to the cost (numreg sym_from .. sym_to: every level).
const SYM_LEVELS: [bool; 3] = [true, true, true];
/// L-BFGS stop on the relative decrease of the cost (numreg ftol).
const FTOL: f64 = 1e-5;
const BINS: usize = 32;
const R_SCALE: f64 = 50.0;
const N_STARTS: usize = 5;
/// Start rotations about each axis (degrees) on top of the centre-of-mass start, numreg START_DEGREES.
const START_DEGREES: [f64; 4] = [-45.0, -20.0, 20.0, 45.0];

/// One pyramid level: (step mm, sample fraction, max L-BFGS iterations, band margin mm, band weight).
pub type LevelSpec = (f64, f64, usize, f64, f64);

/// numreg.register options other than the defaults of `register`.
pub struct Options<'a> {
    pub levels: &'a [LevelSpec],
    pub n_starts: usize,
    /// NCC instead of Mattes MI
    pub ncc: bool,
    /// dof = 6 (rotation + translation) instead of 12
    pub rigid: bool,
    pub moving_mask: Option<&'a Array3<bool>>,
    pub fixed_mask: Option<&'a Array3<bool>>,
}

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

/// numpy.percentile(..., q) with linear interpolation (selection, no full sort).
fn percentile(mut v: Vec<f64>, q: f64) -> f64 {
    if v.is_empty() {
        return 1.0;
    }
    let pos = (v.len() - 1) as f64 * q / 100.0;
    let lo = pos.floor() as usize;
    let (_, &mut a, right) = v.select_nth_unstable_by(lo, |a, b| a.partial_cmp(b).unwrap());
    let b = right.iter().cloned().fold(f64::INFINITY, f64::min);
    if b.is_finite() { a + (b - a) * (pos - lo as f64) } else { a }
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
/// Each axis is a sum of shifted copies (axpy on contiguous rows / planes), parallel over the first axis.
fn gaussian_filter(data: &Array3<f64>, sigma: [f64; 3]) -> Array3<f64> {
    let (nx, ny, nz) = data.dim();
    let mut cur = data.as_standard_layout().into_owned();
    for ax in 0..3 {
        let sd = sigma[ax];
        if sd <= 1e-15 {
            continue;
        }
        let r = (4.0 * sd + 0.5) as isize;
        let mut w: Vec<f64> = (-r..=r).map(|x| (-0.5 / (sd * sd) * (x * x) as f64).exp()).collect();
        let sum: f64 = w.iter().sum();
        w.iter_mut().for_each(|v| *v /= sum);
        let src = cur.as_slice().unwrap();
        let mut out = Array3::<f64>::zeros((nx, ny, nz));
        let plane = ny * nz;
        out.as_slice_mut().unwrap().par_chunks_mut(plane).enumerate().for_each(|(i, o)| {
            // o_row = sum_k w_k src_row(k): all taps of a row in one pass (rows stay in cache)
            let conv_rows = |orow: &mut [f64], rows: &[(&[f64], f64)]| {
                for (z, v) in orow.iter_mut().enumerate() {
                    *v = rows.iter().map(|(r, wk)| wk * r[z]).sum();
                }
            };
            let mut rows: Vec<(&[f64], f64)> = Vec::with_capacity(w.len());
            for j in 0..ny {
                let orow = &mut o[j * nz..][..nz];
                if ax == 2 {
                    let row = &src[i * plane + j * nz..][..nz];
                    for (z, v) in orow.iter_mut().enumerate() {
                        let mut acc = 0.0;
                        for (k, &wk) in w.iter().enumerate() {
                            let zz = z as isize + k as isize - r;
                            if zz >= 0 && (zz as usize) < nz {
                                acc += wk * row[zz as usize];
                            }
                        }
                        *v = acc;
                    }
                    continue;
                }
                rows.clear();
                for (k, &wk) in w.iter().enumerate() {
                    let off = k as isize - r;
                    let (ii, jj) = if ax == 0 { (i as isize + off, j as isize) } else { (i as isize, j as isize + off) };
                    if ii >= 0 && (ii as usize) < nx && jj >= 0 && (jj as usize) < ny {
                        rows.push((&src[ii as usize * plane + jj as usize * nz..][..nz], wk));
                    }
                }
                conv_rows(orow, &rows);
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
fn pyramid(data: &Array3<f64>, mask: &Array3<bool>, aff: &Mat4, levels: &[LevelSpec]) -> Vec<(f64, Level)> {
    let mut steps: Vec<f64> = levels.iter().map(|l| l.0).collect();
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

    /// Joint-histogram bin of sample i (first of its 4 moving bins), the fractional position t in it.
    fn bin(&self, i: usize, m: f64) -> (usize, f64) {
        let z = (m * self.mscale).clamp(0.0, (BINS - 5) as f64) + 2.0;
        let k0 = z as usize;
        (self.kf[i] * BINS + k0 - 1, z - k0 as f64)
    }

    fn eval(&self, m: &[f64]) -> (f64, Vec<f64>) {
        let b = BINS;
        let n = m.len();
        // joint histogram, cubic B-spline Parzen window on the moving value; fixed chunks summed in order
        // (deterministic for any thread count)
        let parts: Vec<Vec<f64>> = (0..n.div_ceil(CHUNK))
            .into_par_iter()
            .map(|ch| {
                let mut h = vec![0.0f64; b * b];
                for i in ch * CHUNK..(ch * CHUNK + CHUNK).min(n) {
                    let (k, t) = self.bin(i, m[i]);
                    let (t2, t3, u) = (t * t, t * t * t, 1.0 - t);
                    let bs = [u * u * u / 6.0, (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0, (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0, t3 / 6.0];
                    for o in 0..4 {
                        h[k + o] += self.w[i] * bs[o];
                    }
                }
                h
            })
            .collect();
        let mut joint = vec![0.0f64; b * b];
        for h in parts {
            joint.iter_mut().zip(h).for_each(|(x, y)| *x += y);
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
            .into_par_iter()
            .with_min_len(8192)
            .map(|i| {
                let (k, t) = self.bin(i, m[i]);
                let (t2, u) = (t * t, 1.0 - t);
                let dbs = [-u * u / 2.0, (3.0 * t2 - 4.0 * t) / 2.0, (-3.0 * t2 + 2.0 * t + 1.0) / 2.0, t2 / 2.0];
                let g: f64 = (0..4).map(|o| dbs[o] * l[k + o]).sum();
                let s = m[i] * self.mscale;
                if s > 0.0 && s < (b - 5) as f64 { self.w[i] * g * self.mscale } else { 0.0 }
            })
            .collect();
        (mi, grad)
    }
}

/// numreg `_NCC`: global normalised cross-correlation (maximised).
struct Ncc {
    w: Vec<f64>,
    f: Vec<f64>,
}

impl Ncc {
    fn new(f: &[f64], w: &[f64]) -> Ncc {
        let mean: f64 = w.iter().zip(f).map(|(a, b)| a * b).sum();
        let fc: Vec<f64> = f.iter().map(|v| v - mean).collect();
        let sd = w.iter().zip(&fc).map(|(a, b)| a * b * b).sum::<f64>().sqrt() + 1e-8;
        Ncc { w: w.to_vec(), f: fc.iter().map(|v| v / sd).collect() }
    }

    fn eval(&self, m: &[f64]) -> (f64, Vec<f64>) {
        let (w, f) = (&self.w, &self.f);
        let mean: f64 = w.iter().zip(m).map(|(a, b)| a * b).sum();
        let mc: Vec<f64> = m.iter().map(|v| v - mean).collect();
        let sm = w.iter().zip(&mc).map(|(a, b)| a * b * b).sum::<f64>().sqrt() + 1e-8;
        let ncc = (0..m.len()).map(|i| w[i] * f[i] * mc[i]).sum::<f64>() / sm;
        (ncc, (0..m.len()).map(|i| w[i] * (f[i] - ncc * mc[i] / sm) / sm).collect())
    }
}

enum Metric {
    Mi(Mi),
    Ncc(Ncc),
}

impl Metric {
    fn eval(&self, m: &[f64]) -> (f64, Vec<f64>) {
        match self {
            Metric::Mi(x) => x.eval(m),
            Metric::Ncc(x) => x.eval(m),
        }
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
    mi: &'a Metric,
}

/// Points per parallel chunk of the sums; chunk partial sums are added in order (deterministic results).
const CHUNK: usize = 8192;

/// (sum_n h_n (x_n - c)^T / R, sum_n h_n), parallel over fixed chunks of n.
fn accumulate<H: Fn(usize) -> V3 + Sync>(len: usize, x: &[V3], c: &V3, h: H) -> ([[f64; 3]; 3], V3) {
    let parts: Vec<[f64; 12]> = (0..len.div_ceil(CHUNK))
        .into_par_iter()
        .map(|ch| {
            let mut s = [0.0f64; 12];
            for n in ch * CHUNK..(ch * CHUNK + CHUNK).min(len) {
                let hn = h(n);
                for i in 0..3 {
                    s[9 + i] += hn[i];
                    for j in 0..3 {
                        s[3 * i + j] += hn[i] * (x[n][j] - c[j]) / R_SCALE;
                    }
                }
            }
            s
        })
        .collect();
    let s = parts.iter().fold([0.0f64; 12], |a, b| std::array::from_fn(|k| a[k] + b[k]));
    (std::array::from_fn(|i| std::array::from_fn(|j| s[3 * i + j])), [s[9], s[10], s[11]])
}

/// numreg `_cost`: -MI(fixed samples, moving at M x), averaged with the inverse term -MI(moving samples, fixed at
/// M^-1 y) when given; analytic gradient.
fn cost(p: &P12, c: &V3, fwd: &Term, inv: Option<&Term>) -> (f64, P12) {
    let m = params_to_m(p, c);
    let y: Vec<V3> = fwd.x.par_iter().map(|x| affine_apply(&m, x)).collect();
    let (mv, g) = fwd.vol.sample(&y);
    let (val, dm) = fwd.mi.eval(&mv);
    // sum_n g_n dm_n [(x_n - c) / R, 1] -> (ga, gt)
    let (ga, gt) = accumulate(y.len(), fwd.x, c, |n| [g[n][0] * dm[n], g[n][1] * dm[n], g[n][2] * dm[n]]);
    let Some(inv) = inv else {
        let mut grad = [0.0; 12];
        for i in 0..3 {
            for j in 0..3 {
                grad[3 * i + j] = -ga[i][j];
            }
            grad[9 + i] = -gt[i];
        }
        return (-val, grad);
    };
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
    // H = -(gz * dfb) @ Ai
    let (ha, ht) = accumulate(z.len(), &z, c, |n| [0, 1, 2].map(|j| -(0..3).map(|i| gz[n][i] * dfb[n] * ai[i][j]).sum::<f64>()));
    let mut grad = [0.0; 12];
    for i in 0..3 {
        for j in 0..3 {
            grad[3 * i + j] = -0.5 * (ga[i][j] + ha[i][j]);
        }
        grad[9 + i] = -0.5 * (gt[i] + ht[i]);
    }
    (-0.5 * (val + vb), grad)
}

fn dot<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    (0..N).map(|i| a[i] * b[i]).sum()
}

/// L-BFGS (memory 10) with a strong-Wolfe line search (c1 = 1e-3, c2 = 0.9, <= 20 evaluations), stopping as
/// scipy's L-BFGS-B without bounds: max |g| <= 1e-5, relative decrease <= 2.2e-9, or `maxiter` iterations.
fn lbfgs<const N: usize, F: Fn(&[f64; N]) -> (f64, [f64; N])>(f: &F, x0: [f64; N], maxiter: usize) -> (f64, [f64; N]) {
    let mut x = x0;
    let (mut fx, mut g) = f(&x);
    let mut hist: Vec<([f64; N], [f64; N])> = Vec::new();
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
            for i in 0..N {
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
            for i in 0..N {
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
        let s: [f64; N] = std::array::from_fn(|i| xn[i] - x[i]);
        let y: [f64; N] = std::array::from_fn(|i| gn[i] - g[i]);
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
        if rel <= FTOL {
            break;
        }
    }
    (fx, x)
}

#[allow(clippy::type_complexity)]
fn line_search<const N: usize, F: Fn(&[f64; N]) -> (f64, [f64; N])>(f: &F, x: &[f64; N], f0: f64, d: &[f64; N], dphi0: f64, a0: f64) -> Option<([f64; N], f64, [f64; N])> {
    const C1: f64 = 1e-3;
    const C2: f64 = 0.9;
    const MAX_EVAL: usize = 20;
    let eval = |a: f64| {
        let xa: [f64; N] = std::array::from_fn(|i| x[i] + a * d[i]);
        let (fa, ga) = f(&xa);
        (xa, fa, ga)
    };
    let mut n = 0;
    let (mut a_lo, mut f_lo, mut dphi_lo) = (0.0, f0, dphi0);
    let mut a = a0;
    let mut a_hi;
    let mut f_hi;
    let mut best: Option<([f64; N], f64, [f64; N])> = None;
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

/// scipy Rotation.from_rotvec(w).as_matrix() (Rodrigues).
pub fn rotvec_to_matrix(w: &V3) -> [[f64; 3]; 3] {
    let t = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    let (a, b) = if t < 1e-6 { (1.0 - t * t / 6.0, 0.5 - t * t / 24.0) } else { (t.sin() / t, (1.0 - t.cos()) / (t * t)) };
    let k = [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]];
    std::array::from_fn(|i| {
        std::array::from_fn(|j| (i == j) as u8 as f64 + a * k[i][j] + b * (0..3).map(|l| k[i][l] * k[l][j]).sum::<f64>())
    })
}

/// scipy Rotation.from_matrix(r).as_rotvec() (through the quaternion, the scalar part kept >= 0).
pub fn matrix_to_rotvec(r: &[[f64; 3]; 3]) -> V3 {
    let tr = r[0][0] + r[1][1] + r[2][2];
    let q = if tr > r[0][0].max(r[1][1]).max(r[2][2]) {
        let s = (1.0 + tr).sqrt() * 2.0;
        [(r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s, 0.25 * s]
    } else if r[0][0] >= r[1][1] && r[0][0] >= r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
        [0.25 * s, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s, (r[2][1] - r[1][2]) / s]
    } else if r[1][1] >= r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
        [(r[0][1] + r[1][0]) / s, 0.25 * s, (r[1][2] + r[2][1]) / s, (r[0][2] - r[2][0]) / s]
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
        [(r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, 0.25 * s, (r[1][0] - r[0][1]) / s]
    };
    let sign = if q[3] < 0.0 { -1.0 } else { 1.0 };
    let v = [q[0] * sign, q[1] * sign, q[2] * sign];
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n < 1e-12 {
        return [2.0 * v[0], 2.0 * v[1], 2.0 * v[2]];
    }
    let angle = 2.0 * n.atan2(q[3] * sign);
    v.map(|x| x / n * angle)
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
/// Fixed-side preparation (centre of mass, volume, pyramid), kept for the next call with the same fixed image
/// (numreg `_FIXED_CACHE`): the template is the same for every input of a run.
struct Fixed {
    key: (usize, u64, [u64; 16]),
    cf: V3,
    vf: f64,
    pyr: Vec<(f64, Level)>,
}

static FIXED: std::sync::Mutex<Option<std::sync::Arc<Fixed>>> = std::sync::Mutex::new(None);

fn fixed_prep(fdata: &Array3<f64>, faff: &Mat4) -> std::sync::Arc<Fixed> {
    let sum: f64 = fdata.iter().sum();
    let key = (fdata.len(), sum.to_bits(), std::array::from_fn(|k| faff[k / 4][k % 4].to_bits()));
    let mut cache = FIXED.lock().unwrap();
    if let Some(f) = cache.as_ref().filter(|f| f.key == key) {
        return f.clone();
    }
    // mask above the background value, shifted to 0 outside
    let fmin_all = fdata.iter().cloned().fold(f64::INFINITY, f64::min);
    let fmask = fdata.mapv(|v| v > fmin_all);
    let fmin = fdata.iter().zip(fmask.iter()).filter(|(_, &m)| m).map(|(&v, _)| v).fold(f64::INFINITY, f64::min);
    let fd = ndarray::Zip::from(fdata).and(&fmask).map_collect(|&v, &m| if m { v - fmin } else { 0.0 });
    let (cf, vf) = com(&fmask, faff);
    let f = std::sync::Arc::new(Fixed { key, cf, vf, pyr: pyramid(&fd, &fmask, faff, &LEVELS) });
    *cache = Some(f.clone());
    f
}

pub fn register(mdata: &Array3<f64>, maff: &Mat4, fdata: &Array3<f64>, faff: &Mat4) -> Mat4 {
    let fx = fixed_prep(fdata, faff);
    let opts = Options { levels: &LEVELS, n_starts: N_STARTS, ncc: false, rigid: false, moving_mask: None, fixed_mask: None };
    register_core(mdata, maff, &fx, &opts)
}

/// numreg.register(moving, fixed, moving_mask, fixed_mask, levels, n_starts, metric, dof) for the options given;
/// symmetric on every level, centre-of-mass start (no fixed-side cache).
pub fn register_with(mdata: &Array3<f64>, maff: &Mat4, fdata: &Array3<f64>, faff: &Mat4, opts: &Options) -> Mat4 {
    let fmask = match opts.fixed_mask {
        Some(m) => m.clone(),
        None => {
            let fmin_all = fdata.iter().cloned().fold(f64::INFINITY, f64::min);
            fdata.mapv(|v| v > fmin_all)
        }
    };
    let fmin = fdata.iter().zip(fmask.iter()).filter(|(_, &m)| m).map(|(&v, _)| v).fold(f64::INFINITY, f64::min);
    let fd = ndarray::Zip::from(fdata).and(&fmask).map_collect(|&v, &m| if m { v - fmin } else { 0.0 });
    let (cf, vf) = com(&fmask, faff);
    let fx = Fixed { key: (0, 0, [0; 16]), cf, vf, pyr: pyramid(&fd, &fmask, faff, opts.levels) };
    register_core(mdata, maff, &fx, opts)
}

fn register_core(mdata: &Array3<f64>, maff: &Mat4, fx: &Fixed, opts: &Options) -> Mat4 {
    let mut rng = Rng(0);
    let (cf, vf, fpyr) = (fx.cf, fx.vf, &fx.pyr);

    // moving: crop to the mask bounding box + 10 mm
    // brain = above 1e-3 of the 99th percentile (numreg): a spline-resampled skull-stripped image leaves ~1e-45
    // values outside the brain, which would make the mask the whole field of view
    let mmask = match opts.moving_mask {
        Some(m) => m.clone(),
        None => {
            let thr = 1e-3 * percentile(mdata.iter().cloned().filter(|&v| v > 0.0).collect(), 99.0);
            mdata.mapv(|v| v > thr)
        }
    };
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
    let mpyr = pyramid(&mdata_c, &mmask_c, &maff_c, opts.levels);

    // initialisation: centre of mass + isotropic volume scaling (none for rigid), START_DEGREES rotations about each
    // axis when more than one start is kept
    let (cm, vm) = com(&mmask_c, &maff_c);
    let s = if opts.rigid { 1.0 } else { (vm / vf).powf(1.0 / 3.0) };
    let c = cf;
    let base = [[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]];
    let mut inits = vec![base];
    for ax in 0..3 {
        for deg in START_DEGREES {
            if opts.n_starts < 2 {
                break;
            }
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
    let metric = |f: &[f64], w: &[f64], vals: &Array3<f64>| {
        if opts.ncc { Metric::Ncc(Ncc::new(f, w)) } else { Metric::Mi(Mi::new(f, w, percentile(positive(vals), 99.5))) }
    };
    let mut best12 = [0.0; 12];
    let mut best6 = [0.0; 6];
    for (li, &(step, frac, iters, margin, edge_w)) in opts.levels.iter().enumerate() {
        let (fl, ml) = (level(fpyr, step), level(&mpyr, step));
        let (x, f, w) = samples(fl, margin, frac, &mut rng, edge_w);
        let vol = Vol::new(&ml.data, &ml.aff);
        let mi = metric(&f, &w, &ml.data);
        let fwd = Term { x: &x, vol: &vol, mi: &mi };
        let back = SYM_LEVELS.get(li).copied().unwrap_or(true).then(|| {
            let (yb, fb, wb) = samples(ml, margin, frac, &mut rng, edge_w);
            let fvol = Vol::new(&fl.data, &fl.aff);
            let mib = metric(&fb, &wb, &fl.data);
            (yb, fvol, mib)
        });
        let inv = back.as_ref().map(|(yb, fvol, mib)| Term { x: yb, vol: fvol, mi: mib });
        let fobj = |p: &P12| cost(p, &c, &fwd, inv.as_ref());
        if opts.rigid {
            let fq = |q: &[f64; 6]| rigid_cost(q, &fobj);
            best6 = optimise(&fq, li, &m0s.iter().map(|m| rigid_enc(m, &c)).collect::<Vec<_>>(), best6, iters, opts.n_starts);
        } else {
            best12 = optimise(&fobj, li, &m0s.iter().map(|m| m_to_params(m, &c)).collect::<Vec<_>>(), best12, iters, opts.n_starts);
        }
    }
    params_to_m(&if opts.rigid { rigid_dec(&best6) } else { best12 }, &c)
}

/// One level: on the first, 15 L-BFGS iterations from every start and the best `n_starts` go on; then `iters`.
fn optimise<const N: usize, F: Fn(&[f64; N]) -> (f64, [f64; N]) + Sync>(f: &F, li: usize, inits: &[[f64; N]], best: [f64; N], iters: usize, n_starts: usize) -> [f64; N] {
    let starts: Vec<[f64; N]> = if li == 0 {
        let mut cands: Vec<(f64, [f64; N])> = inits.par_iter().map(|p| lbfgs(f, *p, 15)).collect();
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        cands.into_iter().take(n_starts.max(1)).map(|(_, p)| p).collect()
    } else {
        vec![best]
    };
    let mut res: Vec<(f64, [f64; N])> = starts.into_par_iter().map(|p0| lbfgs(f, p0, iters)).collect();
    res.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    res[0].1
}

// numreg `_rigid_param_maps`: q = (rotation vector * R_SCALE, translation) on top of the affine parameters
fn rigid_rot(q: &[f64; 6], e: V3) -> [[f64; 3]; 3] {
    rotvec_to_matrix(&[0, 1, 2].map(|i| (q[i] + e[i]) / R_SCALE))
}

fn rigid_enc(m: &Mat4, c: &V3) -> [f64; 6] {
    let a = [[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]];
    let w = matrix_to_rotvec(&a);
    let t: V3 = std::array::from_fn(|i| m[i][3] - c[i] + (0..3).map(|j| a[i][j] * c[j]).sum::<f64>());
    [w[0] * R_SCALE, w[1] * R_SCALE, w[2] * R_SCALE, t[0], t[1], t[2]]
}

fn rigid_dec(q: &[f64; 6]) -> P12 {
    let r = rigid_rot(q, [0.0; 3]);
    let mut p = [0.0; 12];
    for i in 0..3 {
        for j in 0..3 {
            p[3 * i + j] = (r[i][j] - (i == j) as u8 as f64) * R_SCALE;
        }
        p[9 + i] = q[3 + i];
    }
    p
}

fn rigid_cost<F: Fn(&P12) -> (f64, P12)>(q: &[f64; 6], f: &F) -> (f64, [f64; 6]) {
    let (val, g) = f(&rigid_dec(q));
    let mut gq = [0.0; 6];
    gq[3..].copy_from_slice(&g[9..]);
    for i in 0..3 {
        // dA / dq_i by central differences (3 x 3 only)
        let mut e = [0.0; 3];
        e[i] = 1e-4;
        let (hi, lo) = (rigid_rot(q, e), rigid_rot(q, e.map(|v| -v)));
        gq[i] = (0..9).map(|k| g[k] * R_SCALE * (hi[k / 3][k % 3] - lo[k / 3][k % 3]) / 2e-4).sum();
    }
    (val, gq)
}

#[cfg(test)]
mod tests {
    use crate::resample::{self, Kind, Volume};
    use ndarray::Ix3;

    /// Timing harness, not run by default: NUMREG_BENCH_LIST = text file with one skull-stripped NIfTI per line,
    /// NUMREG_BENCH_OUT = directory for <stem>_M.txt (4x4 fixed -> moving world, as TIGERBX_DUMP_REG).
    /// cargo test --release numreg_bench -- --ignored --nocapture
    #[test]
    #[ignore]
    fn numreg_bench() {
        let list = std::env::var("NUMREG_BENCH_LIST").expect("NUMREG_BENCH_LIST");
        let out = std::path::PathBuf::from(std::env::var("NUMREG_BENCH_OUT").expect("NUMREG_BENCH_OUT"));
        std::fs::create_dir_all(&out).unwrap();
        let t = crate::reg::load_template(include_bytes!("../../tigerbx/template/mni_template.nii.gz")).unwrap();
        let mut total = 0.0;
        let files: Vec<String> = std::fs::read_to_string(list).unwrap().lines().map(String::from).collect();
        for f in &files {
            let (h, d) = crate::nifti_io::read_nifti(std::path::Path::new(f)).unwrap();
            let d = d.into_dimensionality::<Ix3>().unwrap();
            let mv = resample::reorder(Volume { data: d.mapv(|v| v as f32 as f64), affine: resample::best_affine(&h), kind: Kind::Float32 });
            let t0 = std::time::Instant::now();
            let m = super::register(&mv.data, &mv.affine, &t.vol.data, &t.vol.affine);
            let dt = t0.elapsed().as_secs_f64();
            total += dt;
            let stem = std::path::Path::new(f).file_name().unwrap().to_string_lossy().replace(".nii.gz", "");
            let txt: String = m.iter().map(|r| r.iter().map(|v| format!("{v:.10}")).collect::<Vec<_>>().join(" ") + "\n").collect();
            std::fs::write(out.join(format!("{stem}_M.txt")), txt).unwrap();
            println!("{stem} {dt:.3} s");
        }
        println!("mean {:.3} s over {}", total / files.len() as f64, files.len());
    }
}
