//! dwimc: head motion and eddy-current correction of a diffusion series (one phase-encoding direction), a port of
//! TigerBx `tigerbx/core/dwimc.py` and `tigerbx/pipelines/dwimc.py` (same model, steps and constants; see there).
//!
//! Per volume: x_s = M x_h (rigid, head = reference b0 -> scanner), u = x_s + F(x_s) e_pe with
//! F = f(M^-1 x_s) + c . phi(x_s) (susceptibility field attached to the head + EC polynomial), signal times
//! 1 / (1 + dF/dpe). Gauss-Newton per volume in observed space against a leave-one-out spherical-harmonic
//! prediction, EC terms c = B g, b0 anchors (correlation ratio), one cubic resampling. Arrays here have the phase
//! axis as axis 1; `run_one` transposes. Agreement with Python is close, not bit for bit (the prediction's
//! gradient comes from the spline here, from central differences there; floating-point order).

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use ndarray::{Array2, Array3, ArrayView2, Axis, Zip};
use rayon::prelude::*;

use crate::duope::{self, dilate, gaussian, linear_nearest, percentile, A3};
use crate::duope_pipe::{self, Nii, VolWriter};
use crate::numreg::{matrix_to_rotvec, rotvec_to_matrix};
use crate::resample::{linear_transform, mat_inv, mat_mul, spline_coefficients, Mat4};

type V3 = [f64; 3];
type Shape = (usize, usize, usize);

const PE: usize = 1;
pub const B0_MAX: f64 = 50.0;
const FWHM: [f64; 2] = [8.0, 0.0];
pub const OUTER: usize = 3;
const GN_ITERS: usize = 4;
const ANCHOR_L: usize = 6;
/// The anchors see the brain edge and some background.
const ANCHOR_DILATE: usize = 7;
const NPAD: usize = 12;

pub struct Settings {
    pub output: Option<PathBuf>,
    pub reverse: Vec<PathBuf>,
    pub field: Option<PathBuf>,
    pub pe: Option<String>,
    pub readout: Option<f64>,
    pub order: usize,
    pub iters: usize,
}

// ---------------------------------------------------------------- small linear algebra

fn solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a.iter().zip(b).map(|(r, &v)| r.iter().cloned().chain([v]).collect()).collect();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| m[i][c].abs().partial_cmp(&m[j][c].abs()).unwrap()).unwrap();
        m.swap(c, p);
        if m[c][c].abs() < 1e-300 {
            continue;
        }
        for r in c + 1..n {
            let f = m[r][c] / m[c][c];
            for k in c..=n {
                m[r][k] -= f * m[c][k];
            }
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s = m[r][n] - (r + 1..n).map(|k| m[r][k] * x[k]).sum::<f64>();
        x[r] = if m[r][r].abs() < 1e-300 { 0.0 } else { s / m[r][r] };
    }
    x
}

/// Least squares X b = Y column by column (normal equations with a tiny ridge); X: rows, returns (p, q).
fn lstsq(x: &[Vec<f64>], y: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let p = x[0].len();
    let q = y[0].len();
    let mut xtx = vec![vec![0.0; p]; p];
    for r in x {
        for i in 0..p {
            for j in 0..p {
                xtx[i][j] += r[i] * r[j];
            }
        }
    }
    let tr: f64 = (0..p).map(|i| xtx[i][i]).sum::<f64>() / p as f64;
    for (i, row) in xtx.iter_mut().enumerate() {
        row[i] += 1e-12 * tr.max(1e-300);
    }
    let mut out = vec![vec![0.0; q]; p];
    for c in 0..q {
        let xty: Vec<f64> = (0..p).map(|i| x.iter().zip(y).map(|(r, yr)| r[i] * yr[c]).sum()).collect();
        let b = solve(&xtx, &xty);
        for i in 0..p {
            out[i][c] = b[i];
        }
    }
    out
}

fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Some(l)
}

// ---------------------------------------------------------------- geometry

fn apply(m: &Mat4, x: &V3) -> V3 {
    std::array::from_fn(|r| m[r][0] * x[0] + m[r][1] * x[1] + m[r][2] * x[2] + m[r][3])
}

fn eye() -> Mat4 {
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
}

/// 6 parameters (translation mm, rotation vector rad about c_mm) -> head voxel -> scanner voxel.
fn rigid(p: &[f64], z: &V3, c: &V3) -> Mat4 {
    let r = rotvec_to_matrix(&[p[3], p[4], p[5]]);
    let mut m = eye();
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = r[i][j] * z[j] / z[i];
        }
        let rc: f64 = (0..3).map(|j| r[i][j] * c[j]).sum();
        m[i][3] = (c[i] - rc + p[i]) / z[i];
    }
    m
}

fn rot_mm(m: &Mat4, z: &V3) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| z[i] * m[i][j] / z[j]))
}

fn params_of(m: &Mat4, z: &V3, c: &V3) -> [f64; 6] {
    let r = rot_mm(m, z);
    let w = matrix_to_rotvec(&r);
    let mut p = [0.0; 6];
    for i in 0..3 {
        p[i] = z[i] * m[i][3] - c[i] + (0..3).map(|j| r[i][j] * c[j]).sum::<f64>();
    }
    p[3..].copy_from_slice(&w);
    p
}

/// The rotation (mm voxel axes) of head -> scanner map m (orthonormalised).
pub fn rotation(m: &Mat4, z: &V3) -> [[f64; 3]; 3] {
    rotvec_to_matrix(&matrix_to_rotvec(&rot_mm(m, z)))
}

#[derive(Clone)]
struct Ec {
    order: usize,
    k: usize,
    ctr: V3,
    sc: V3,
}

impl Ec {
    fn new(shape: Shape, order: usize) -> Ec {
        let s = [shape.0 as f64, shape.1 as f64, shape.2 as f64];
        Ec { order, k: if order == 1 { 4 } else { 10 }, ctr: s.map(|n| (n - 1.0) / 2.0), sc: s.map(|n| n / 2.0) }
    }

    #[inline]
    fn phi(&self, xs: &V3) -> [f64; 10] {
        let x = (xs[0] - self.ctr[0]) / self.sc[0];
        let y = (xs[1] - self.ctr[1]) / self.sc[1];
        let z = (xs[2] - self.ctr[2]) / self.sc[2];
        let mut o = [1.0, x, y, z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        if self.order == 2 {
            o[4..].copy_from_slice(&[x * x, y * y, z * z, x * y, x * z, y * z]);
        }
        o
    }

    #[inline]
    fn dphi_y(&self, xs: &V3) -> [f64; 10] {
        let x = (xs[0] - self.ctr[0]) / self.sc[0];
        let y = (xs[1] - self.ctr[1]) / self.sc[1];
        let z = (xs[2] - self.ctr[2]) / self.sc[2];
        let s = 1.0 / self.sc[1];
        let mut o = [0.0, 0.0, s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        if self.order == 2 {
            o[5] = 2.0 * y * s;
            o[7] = x * s;
            o[9] = z * s;
        }
        o
    }

    #[inline]
    fn dot(&self, c: &[f64], v: &[f64; 10]) -> f64 {
        (0..self.k).map(|i| c[i] * v[i]).sum()
    }
}

// ---------------------------------------------------------------- sampling

/// Cubic B-spline (scipy map_coordinates order 3, mode 'nearest') with its gradient.
struct SplineG {
    coef: A3,
    dims: [usize; 3],
}

/// An image for registration: its spline and its central-difference gradient (numpy.gradient, sampled linearly).
struct Target {
    sp: SplineG,
    grad: [A3; 3],
}

impl Target {
    fn new(img: &A3) -> Target {
        Target { sp: SplineG::new(img), grad: [gradient(img, 0), gradient(img, 1), gradient(img, 2)] }
    }

    #[inline]
    fn eval(&self, c: &V3, grad: bool) -> (f64, V3) {
        let v = self.sp.eval(c, false).0;
        if !grad {
            return (v, [0.0; 3]);
        }
        (v, std::array::from_fn(|a| linear_nearest(&self.grad[a], c)))
    }
}

impl SplineG {
    fn new(img: &A3) -> SplineG {
        let d = img.dim();
        let pd = (d.0 + 2 * NPAD, d.1 + 2 * NPAD, d.2 + 2 * NPAD);
        let cl = |i: usize, n: usize| (i as isize - NPAD as isize).clamp(0, n as isize - 1) as usize;
        let padded = A3::from_shape_fn(pd, |(i, j, k)| img[[cl(i, d.0), cl(j, d.1), cl(k, d.2)]]);
        SplineG { coef: spline_coefficients(&padded), dims: [pd.0, pd.1, pd.2] }
    }

    #[inline]
    fn eval(&self, c: &V3, grad: bool) -> (f64, V3) {
        let mut idx = [[0usize; 4]; 3];
        let mut w = [[0.0; 4]; 3];
        let mut dw = [[0.0; 4]; 3];
        for a in 0..3 {
            let x = (c[a] + NPAD as f64).clamp(0.0, (self.dims[a] - 1) as f64);
            let f = x.floor();
            let t = x - f;
            let (t2, t3) = (t * t, t * t * t);
            w[a] = [(1.0 - t).powi(3) / 6.0, (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0, (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0, t3 / 6.0];
            dw[a] = [-(1.0 - t) * (1.0 - t) / 2.0, -2.0 * t + 1.5 * t2, 0.5 + t - 1.5 * t2, t2 / 2.0];
            for (q, id) in idx[a].iter_mut().enumerate() {
                let i = f as isize - 1 + q as isize;
                let n = self.dims[a] as isize;
                let p = 2 * n - 2;
                let mut i = i.abs() % p.max(1);
                if i >= n {
                    i = p - i;
                }
                *id = i as usize;
            }
        }
        let (mut s, mut g) = (0.0, [0.0; 3]);
        for p in 0..4 {
            for q in 0..4 {
                for r in 0..4 {
                    let v = self.coef[[idx[0][p], idx[1][q], idx[2][r]]];
                    s += w[0][p] * w[1][q] * w[2][r] * v;
                    if grad {
                        g[0] += dw[0][p] * w[1][q] * w[2][r] * v;
                        g[1] += w[0][p] * dw[1][q] * w[2][r] * v;
                        g[2] += w[0][p] * w[1][q] * dw[2][r] * v;
                    }
                }
            }
        }
        (s, g)
    }
}

/// numpy.gradient along `axis` (central inside, one-sided at the edges).
fn gradient(d: &A3, axis: usize) -> A3 {
    let mut out = A3::zeros(d.dim());
    Zip::from(out.lanes_mut(Axis(axis))).and(d.lanes(Axis(axis))).for_each(|mut o, x| {
        let n = x.len();
        if n < 2 {
            return;
        }
        o[0] = x[1] - x[0];
        o[n - 1] = x[n - 1] - x[n - 2];
        for i in 1..n - 1 {
            o[i] = 0.5 * (x[i + 1] - x[i - 1]);
        }
    });
    out
}

/// F (nx, ny, nz): displacement along axis 1 at the scanner grid points, u = y + F. Returns g with
/// y_s = y_u + g(y_u): per line, u(y) made monotone (folds collapse) and inverted linearly (np.interp).
fn line_inverse(f: &A3) -> A3 {
    let (nx, ny, nz) = f.dim();
    let mut out = A3::zeros((nx, ny, nz));
    let mut u = vec![0.0; ny];
    for i in 0..nx {
        for k in 0..nz {
            let mut run = f64::NEG_INFINITY;
            for j in 0..ny {
                run = run.max(j as f64 + f[[i, j, k]]);
                u[j] = run + 1e-9 * j as f64;
            }
            let mut p = 0usize;
            for j in 0..ny {
                let t = j as f64;
                let ys = if t <= u[0] {
                    0.0
                } else if t >= u[ny - 1] {
                    (ny - 1) as f64
                } else {
                    while u[p + 1] < t {
                        p += 1;
                    }
                    p as f64 + (t - u[p]) / (u[p + 1] - u[p])
                };
                out[[i, j, k]] = ys - t;
            }
        }
    }
    out
}

// ---------------------------------------------------------------- spherical harmonics and the predictor

/// Real even spherical harmonics up to order L (rows: directions), and each column's order.
fn sh_basis(g: &[V3], l_max: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut ls = Vec::new();
    for l in (0..=l_max).step_by(2) {
        for _ in 0..(2 * l + 1) {
            ls.push(l);
        }
    }
    let fact = |n: usize| (1..=n).fold(1.0f64, |a, b| a * b as f64);
    let rows = g
        .iter()
        .map(|v| {
            let nrm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
            let x = (v[2] / nrm).clamp(-1.0, 1.0);
            let phi = v[1].atan2(v[0]);
            let sx = (1.0 - x * x).max(0.0).sqrt();
            // associated Legendre P_l^m(x) with the Condon-Shortley phase (as scipy)
            let mut p = vec![vec![0.0; l_max + 1]; l_max + 1];
            p[0][0] = 1.0;
            for m in 1..=l_max {
                p[m][m] = -(2.0 * m as f64 - 1.0) * sx * p[m - 1][m - 1];
            }
            for m in 0..l_max {
                p[m + 1][m] = x * (2.0 * m as f64 + 1.0) * p[m][m];
            }
            for m in 0..=l_max {
                for l in m + 2..=l_max {
                    p[l][m] = ((2.0 * l as f64 - 1.0) * x * p[l - 1][m] - (l + m - 1) as f64 * p[l - 2][m]) / (l - m) as f64;
                }
            }
            let mut row = Vec::new();
            for l in (0..=l_max).step_by(2) {
                for mm in -(l as isize)..=(l as isize) {
                    let m = mm.unsigned_abs();
                    let n = ((2 * l + 1) as f64 / (4.0 * std::f64::consts::PI) * fact(l - m) / fact(l + m)).sqrt();
                    let base = n * p[l][m];
                    row.push(match mm.signum() {
                        -1 => std::f64::consts::SQRT_2 * base * (m as f64 * phi).sin(),
                        0 => base,
                        _ => std::f64::consts::SQRT_2 * base * (m as f64 * phi).cos(),
                    });
                }
            }
            row
        })
        .collect();
    (rows, ls)
}

/// scipy.optimize.minimize(method='Nelder-Mead') with its defaults (initial simplex 5% / 0.00025, xatol, fatol).
fn nelder_mead(f: &dyn Fn(&[f64]) -> f64, x0: &[f64], maxiter: usize, xatol: f64, fatol: f64) -> Vec<f64> {
    let n = x0.len();
    let mut sim: Vec<Vec<f64>> = vec![x0.to_vec()];
    for k in 0..n {
        let mut y = x0.to_vec();
        y[k] = if y[k] != 0.0 { 1.05 * y[k] } else { 0.00025 };
        sim.push(y);
    }
    let mut fs: Vec<f64> = sim.iter().map(|x| f(x)).collect();
    let sort = |sim: &mut Vec<Vec<f64>>, fs: &mut Vec<f64>| {
        let mut ord: Vec<usize> = (0..fs.len()).collect();
        ord.sort_by(|&a, &b| fs[a].partial_cmp(&fs[b]).unwrap_or(std::cmp::Ordering::Equal));
        *sim = ord.iter().map(|&i| sim[i].clone()).collect();
        *fs = ord.iter().map(|&i| fs[i]).collect();
    };
    sort(&mut sim, &mut fs);
    let comb = |a: f64, x: &[f64], b: f64, y: &[f64]| -> Vec<f64> { x.iter().zip(y).map(|(p, q)| a * p + b * q).collect() };
    for _ in 1..maxiter {
        let dx = sim[1..].iter().flat_map(|s| s.iter().zip(&sim[0]).map(|(a, b)| (a - b).abs())).fold(0.0, f64::max);
        let df = fs[1..].iter().map(|v| (v - fs[0]).abs()).fold(0.0, f64::max);
        if dx <= xatol && df <= fatol {
            break;
        }
        let xbar: Vec<f64> = (0..n).map(|j| sim[..n].iter().map(|s| s[j]).sum::<f64>() / n as f64).collect();
        let xr = comb(2.0, &xbar, -1.0, &sim[n]);
        let fxr = f(&xr);
        let mut shrink = false;
        if fxr < fs[0] {
            let xe = comb(3.0, &xbar, -2.0, &sim[n]);
            let fxe = f(&xe);
            if fxe < fxr {
                sim[n] = xe;
                fs[n] = fxe;
            } else {
                sim[n] = xr;
                fs[n] = fxr;
            }
        } else if fxr < fs[n - 1] {
            sim[n] = xr;
            fs[n] = fxr;
        } else if fxr < fs[n] {
            let xc = comb(1.5, &xbar, -0.5, &sim[n]);
            let fxc = f(&xc);
            if fxc <= fxr {
                sim[n] = xc;
                fs[n] = fxc;
            } else {
                shrink = true;
            }
        } else {
            let xcc = comb(0.5, &xbar, 0.5, &sim[n]);
            let fxcc = f(&xcc);
            if fxcc < fs[n] {
                sim[n] = xcc;
                fs[n] = fxcc;
            } else {
                shrink = true;
            }
        }
        if shrink {
            for j in 1..=n {
                sim[j] = comb(0.5, &sim[0], 0.5, &sim[j]);
                fs[j] = f(&sim[j]);
            }
        }
        sort(&mut sim, &mut fs);
    }
    sim.swap_remove(0)
}

fn sh_order(n: usize) -> usize {
    if n <= 12 { 2 } else if n <= 28 { 4 } else { 6 }
}

/// Leave-one-out hat matrix of one shell (rows = volumes); gm: the empirical covariance S S' / V.
fn sh_predictor(g: &[V3], gm: &[Vec<f64>], l_max: usize) -> Vec<Vec<f64>> {
    let (b, ls) = sh_basis(g, l_max);
    let n = b.len();
    let nc = ls.len();
    let lo: Vec<usize> = ls.iter().map(|l| l / 2).collect();
    let scale = (0..n).map(|i| gm[i][i]).sum::<f64>() / n as f64;
    let nll = |th: &[f64]| -> f64 {
        let a: Vec<f64> = th[..th.len() - 1].iter().map(|t| t.exp() * scale).collect();
        let s2 = th[th.len() - 1].exp() * scale;
        let k: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| (0..nc).map(|c| b[i][c] * a[lo[c]] * b[j][c]).sum::<f64>() + if i == j { s2 } else { 0.0 }).collect())
            .collect();
        let Some(l) = cholesky(&k) else { return 1e30 };
        let logdet: f64 = 2.0 * (0..n).map(|i| l[i][i].ln()).sum::<f64>();
        // trace(K^-1 G) through the Cholesky factor, column by column
        let mut tr = 0.0;
        for c in 0..n {
            let mut y = vec![0.0; n];
            for i in 0..n {
                y[i] = (gm[i][c] - (0..i).map(|k| l[i][k] * y[k]).sum::<f64>()) / l[i][i];
            }
            let mut x = vec![0.0; n];
            for i in (0..n).rev() {
                x[i] = (y[i] - (i + 1..n).map(|k| l[k][i] * x[k]).sum::<f64>()) / l[i][i];
            }
            tr += x[c];
        }
        logdet + tr
    };
    let x0: Vec<f64> = [1.0, 0.3, 0.1, 0.03, 0.01][..l_max / 2 + 1].iter().chain([0.05].iter()).map(|v: &f64| v.ln()).collect();
    let th = nelder_mead(&nll, &x0, 2000, 1e-3, 1e-6);
    let a: Vec<f64> = th[..th.len() - 1].iter().map(|t| (t.exp() * scale).max(1e-12 * scale)).collect();
    let s2 = th[th.len() - 1].exp() * scale;
    // H = B (B'B + s2 diag(1/a))^-1 B'
    let mut btb: Vec<Vec<f64>> = (0..nc).map(|p| (0..nc).map(|q| (0..n).map(|i| b[i][p] * b[i][q]).sum()).collect()).collect();
    for (c, row) in btb.iter_mut().enumerate() {
        row[c] += s2 / a[lo[c]];
    }
    let mut h = vec![vec![0.0; n]; n];
    for j in 0..n {
        let rhs: Vec<f64> = (0..nc).map(|c| b[j][c]).collect();
        let col = solve(&btb, &rhs);
        for i in 0..n {
            h[i][j] = (0..nc).map(|c| b[i][c] * col[c]).sum();
        }
    }
    let d: Vec<f64> = (0..n).map(|i| h[i][i]).collect();
    (0..n).map(|i| (0..n).map(|j| if i == j { 0.0 } else { h[i][j] / (1.0 - d[i]) }).collect()).collect()
}

// ---------------------------------------------------------------- per volume

struct Ctx {
    shape: Shape,
    zooms: V3,
    c_mm: V3,
    field: A3,
    gfield: [A3; 3],
    gfy: [A3; 3],
    ec: Ec,
    bval: Vec<f64>,
    obs: Vec<[usize; 3]>,
    w: Vec<f64>,
}

struct Vol {
    minv: Mat4,
    c: Vec<f64>,
    s: f64,
}

impl Ctx {
    fn dwi(&self, i: usize) -> bool {
        self.bval[i] > B0_MAX
    }

    /// g at the observed voxels: u + g e_y = x_s, u = x_s + F(x_s) e_y (per scanner line).
    fn inverse(&self, minv: &Mat4, c: &[f64]) -> Vec<f64> {
        let mut f = A3::zeros(self.shape);
        Zip::indexed(&mut f).for_each(|(i, j, k), o| {
            let xs = [i as f64, j as f64, k as f64];
            *o = linear_nearest(&self.field, &apply(minv, &xs)) + self.ec.dot(c, &self.ec.phi(&xs));
        });
        let g = line_inverse(&f);
        self.obs.iter().map(|&[i, j, k]| g[[i, j, k]]).collect()
    }

    /// Weighted cost and (with `normal`) the normal equations (J'WJ, J'W r) of volume i against its prediction.
    fn warp(&self, i: usize, pred: &Target, v: &Vol, y: &[f64], normal: bool) -> (f64, Vec<Vec<f64>>, Vec<f64>) {
        let gi = self.inverse(&v.minv, &v.c);
        let dwi = self.dwi(i);
        let k = self.ec.k;
        let np = 1 + if dwi { k + 5 } else { 6 };
        let m = [v.minv[0][1], v.minv[1][1], v.minv[2][1]];
        let zi = self.zooms.map(|z| 1.0 / z);
        let mut h = vec![vec![0.0; np]; np];
        let mut rhs = vec![0.0; np];
        let mut cost = 0.0;
        let mut col = vec![0.0; np];
        for (o, (&[a, b, cc], &g)) in self.obs.iter().zip(&gi).enumerate() {
            let w = self.w[o];
            if w == 0.0 {
                continue;
            }
            let xs = [a as f64, b as f64 + g, cc as f64];
            let xh = apply(&v.minv, &xs);
            let gf: V3 = std::array::from_fn(|q| linear_nearest(&self.gfield[q], &xh));
            let dphi = self.ec.dphi_y(&xs);
            let jac = 1.0 / (1.0 + m[0] * gf[0] + m[1] * gf[1] + m[2] * gf[2] + self.ec.dot(&v.c, &dphi)).max(0.2);
            let (p, gp) = pred.eval(&xh, normal);
            let yhat = v.s * p * jac;
            let r = y[o] - yhat;
            cost += w * r * r;
            if !normal {
                continue;
            }
            let gfy: V3 = std::array::from_fn(|q| linear_nearest(&self.gfy[q], &xh));
            let pm = m[0] * gp[0] + m[1] * gp[1] + m[2] * gp[2];
            let fym = m[0] * gfy[0] + m[1] * gfy[1] + m[2] * gfy[2];
            let sp = v.s * p;
            let colf = |dp: f64, dfy: f64, direct: f64| v.s * jac * dp - sp * jac * jac * (dfy + direct);
            let mut n = 0;
            col[n] = p * jac;
            n += 1;
            if dwi {
                let phi = self.ec.phi(&xs);
                for q in 0..k {
                    col[n] = colf(-jac * phi[q] * pm, -jac * phi[q] * fym, dphi[q]);
                    n += 1;
                }
            }
            for j in 0..3 {
                if j == PE && dwi {
                    continue;
                }
                let aj = -zi[j];
                let t = -jac * aj * gf[j];
                col[n] = colf(aj * gp[j] + t * pm, aj * gfy[j] + t * fym, 0.0);
                n += 1;
            }
            let xmm: V3 = std::array::from_fn(|q| xh[q] * self.zooms[q] - self.c_mm[q]);
            for j in 0..3 {
                // (x - c) x e_j, per head voxel
                let mut e = [0.0; 3];
                e[j] = 1.0;
                let cr = [xmm[1] * e[2] - xmm[2] * e[1], xmm[2] * e[0] - xmm[0] * e[2], xmm[0] * e[1] - xmm[1] * e[0]];
                let a3: V3 = std::array::from_fn(|q| cr[q] * zi[q]);
                let t = -jac * (gf[0] * a3[0] + gf[1] * a3[1] + gf[2] * a3[2]);
                col[n] = colf(gp[0] * a3[0] + gp[1] * a3[1] + gp[2] * a3[2] + t * pm, gfy[0] * a3[0] + gfy[1] * a3[1] + gfy[2] * a3[2] + t * fym, 0.0);
                n += 1;
            }
            for p1 in 0..np {
                let wc = w * col[p1];
                rhs[p1] += wc * r;
                for p2 in p1..np {
                    h[p1][p2] += wc * col[p2];
                }
            }
        }
        for p1 in 0..np {
            for p2 in 0..p1 {
                h[p1][p2] = h[p2][p1];
            }
        }
        (cost, h, rhs)
    }

    fn update(&self, i: usize, v: &Vol, d: &[f64]) -> Vol {
        let dwi = self.dwi(i);
        let mut k = 1;
        let s = v.s + d[0];
        let mut c = v.c.clone();
        if dwi {
            for q in 0..self.ec.k {
                c[q] += d[k + q];
            }
            k += self.ec.k;
        }
        let mut dt = [0.0; 3];
        for (j, t) in dt.iter_mut().enumerate() {
            if j == PE && dwi {
                continue;
            }
            *t = d[k];
            k += 1;
        }
        let r = rotvec_to_matrix(&[-d[k], -d[k + 1], -d[k + 2]]);
        let (z, cm) = (&self.zooms, &self.c_mm);
        let mut dm = eye();
        for a in 0..3 {
            for b in 0..3 {
                dm[a][b] = r[a][b] * z[b] / z[a];
            }
            let rc: f64 = (0..3).map(|b| r[a][b] * cm[b]).sum();
            dm[a][3] = (cm[a] - rc - dt[a]) / z[a];
        }
        Vol { minv: mat_mul(&dm, &v.minv), c, s }
    }

    fn register(&self, i: usize, pred: &Target, y: &[f64], mut v: Vol, iters: usize) -> Vol {
        let (mut cost, mut h, mut rhs) = self.warp(i, pred, &v, y, true);
        let mu = 1e-3;
        for it in 0..iters {
            let n = rhs.len();
            let a: Vec<Vec<f64>> = (0..n).map(|p| (0..n).map(|q| h[p][q] + if p == q { mu * h[p][p] + 1e-12 } else { 0.0 }).collect()).collect();
            let mut d = solve(&a, &rhs);
            let mut accepted = None;
            for _ in 0..4 {
                let nv = self.update(i, &v, &d);
                let (c2, _, _) = self.warp(i, pred, &nv, y, false);
                if c2 < cost {
                    accepted = Some((nv, c2));
                    break;
                }
                d.iter_mut().for_each(|x| *x /= 2.0);
            }
            let Some((nv, c2)) = accepted else { break };
            v = nv;
            cost = c2;
            if it + 1 < iters {
                let r = self.warp(i, pred, &v, y, true);
                h = r.1;
                rhs = r.2;
            }
        }
        v
    }

    /// Volume `data` in head space: y(u(x_h)) (1 + dF/dpe) / s on the full grid.
    fn to_model(&self, raw: &SplineG, v: &Vol) -> A3 {
        let m = mat_inv(&v.minv);
        let mc = [v.minv[0][1], v.minv[1][1], v.minv[2][1]];
        let mut out = A3::zeros(self.shape);
        Zip::indexed(&mut out).for_each(|(i, j, k), o| {
            let xh = [i as f64, j as f64, k as f64];
            let xs = apply(&m, &xh);
            let f = self.field[[i, j, k]] + self.ec.dot(&v.c, &self.ec.phi(&xs));
            let fy = mc[0] * self.gfield[0][[i, j, k]] + mc[1] * self.gfield[1][[i, j, k]] + mc[2] * self.gfield[2][[i, j, k]]
                + self.ec.dot(&v.c, &self.ec.dphi_y(&xs));
            let u = [xs[0], xs[1] + f, xs[2]];
            *o = raw.eval(&u, false).0 * (1.0 + fy) / v.s;
        });
        out
    }
}

/// Rigid (and with `ec`, linear EC along axis 1) correlation-ratio registration of `moving` to `fixed`:
/// V (fixed voxel -> moving voxel) and e (the EC coefficients; e[0] = 0, the constant left to the rigid part).
fn anchor(moving: &A3, fixed: &A3, mask: &Array3<bool>, zooms: &V3, outer: usize, iters: usize, ec: bool, par: bool) -> (Mat4, [f64; 4]) {
    let chunk = if par { 4096 } else { usize::MAX };
    let bins = 48usize;
    let idx: Vec<V3> = mask.indexed_iter().filter(|(_, &m)| m).map(|((i, j, k), _)| [i as f64, j as f64, k as f64]).collect();
    let mut cm = [0.0; 3];
    for x in &idx {
        for a in 0..3 {
            cm[a] += x[a];
        }
    }
    let c_mm: V3 = std::array::from_fn(|a| zooms[a] * cm[a] / idx.len() as f64);
    let region = dilate(mask, ANCHOR_DILATE);
    let x: Vec<V3> = region.indexed_iter().filter(|(_, &m)| m).map(|((i, j, k), _)| [i as f64, j as f64, k as f64]).collect();
    let f: Vec<f64> = region.indexed_iter().filter(|(_, &m)| m).map(|((i, j, k), _)| fixed[[i, j, k]]).collect();
    let mut sorted = f.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| {
        let pos = p * (sorted.len() - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(sorted.len() - 1);
        sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo])
    };
    let edges: Vec<f64> = (0..=bins).map(|b| q(b as f64 / bins as f64)).collect();
    let lab: Vec<usize> = f.iter().map(|v| (edges.partition_point(|e| e <= v) as isize - 1).clamp(0, bins as isize - 1) as usize).collect();
    let sp = Target::new(moving);
    let ecb = Ec::new(moving.dim(), 1);
    let ne = if ec { 3 } else { 0 };
    let mut v = eye();
    let mut e = [0.0; 3];
    let pos = |v: &Mat4, e: &[f64; 3], p: &V3| {
        let mut xv = apply(v, p);
        if ec {
            let ph = ecb.phi(p);
            xv[PE] += e[0] * ph[1] + e[1] * ph[2] + e[2] * ph[3];
        }
        xv
    };
    for _ in 0..outer {
        let m: Vec<f64> = x.par_iter().with_min_len(chunk).map(|p| sp.eval(&pos(&v, &e, p), false).0).collect();
        let mut sum = vec![0.0; bins];
        let mut cnt = vec![0.0f64; bins];
        for (l, val) in lab.iter().zip(&m) {
            sum[*l] += val;
            cnt[*l] += 1.0;
        }
        let t: Vec<f64> = lab.iter().map(|&l| sum[l] / cnt[l].max(1.0)).collect();
        for _ in 0..iters {
            let np = 6 + ne;
            let (h, rhs) = x
                .par_iter()
                .zip(&t)
                .with_min_len(chunk)
                .fold(
                    || (vec![vec![0.0; np]; np], vec![0.0; np]),
                    |(mut h, mut rhs), (p, &tv)| {
                        let xv = pos(&v, &e, p);
                        let (mv, g) = sp.eval(&xv, true);
                        let xmm: V3 = std::array::from_fn(|a| xv[a] * zooms[a] - c_mm[a]);
                        let mut col = vec![0.0; np];
                        for j in 0..3 {
                            col[j] = g[j] / zooms[j];
                        }
                        for j in 0..3 {
                            let mut ej = [0.0; 3];
                            ej[j] = 1.0;
                            let cr = [ej[1] * xmm[2] - ej[2] * xmm[1], ej[2] * xmm[0] - ej[0] * xmm[2], ej[0] * xmm[1] - ej[1] * xmm[0]];
                            col[3 + j] = (0..3).map(|a| g[a] * cr[a] / zooms[a]).sum();
                        }
                        if ec {
                            let ph = ecb.phi(p);
                            for q in 0..3 {
                                col[6 + q] = g[PE] * ph[q + 1];
                            }
                        }
                        let r = tv - mv;
                        for a in 0..np {
                            rhs[a] += col[a] * r;
                            for b in 0..np {
                                h[a][b] += col[a] * col[b];
                            }
                        }
                        (h, rhs)
                    },
                )
                .reduce(
                    || (vec![vec![0.0; np]; np], vec![0.0; np]),
                    |(mut h1, mut r1), (h2, r2)| {
                        for a in 0..np {
                            r1[a] += r2[a];
                            for b in 0..np {
                                h1[a][b] += h2[a][b];
                            }
                        }
                        (h1, r1)
                    },
                );
            let a: Vec<Vec<f64>> = (0..np).map(|p| (0..np).map(|q| h[p][q] + if p == q { 1e-9 } else { 0.0 }).collect()).collect();
            let d = solve(&a, &rhs);
            v = mat_mul(&rigid(&d[..6], zooms, &c_mm), &v);
            for q in 0..ne {
                e[q] += d[6 + q];
            }
        }
    }
    (v, [0.0, e[0], e[1], e[2]])
}

// ---------------------------------------------------------------- driver

/// One 2D slice with its holes filled (4-connected background reached from the slice border stays background).
fn fill_holes_2d(s: ArrayView2<bool>) -> Array2<bool> {
    let d = s.dim();
    let mut reached = Array2::from_elem(d, false);
    let mut stack = Vec::new();
    for ((i, j), &v) in s.indexed_iter() {
        if !v && (i == 0 || j == 0 || i == d.0 - 1 || j == d.1 - 1) {
            reached[[i, j]] = true;
            stack.push((i, j));
        }
    }
    while let Some((i, j)) = stack.pop() {
        for (a, b) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
            let (x, y) = (i as isize + a, j as isize + b);
            if x < 0 || y < 0 || x >= d.0 as isize || y >= d.1 as isize {
                continue;
            }
            let p = (x as usize, y as usize);
            if !s[p] && !reached[p] {
                reached[p] = true;
                stack.push(p);
            }
        }
    }
    reached.mapv(|r| !r)
}

/// Holes filled in 3D and then slice by slice along every axis (as tigerbx.core.dwimc.fill_holes: dark deep grey
/// matter in a b0 can open a hole to the outside through one slice, which a 3D fill alone leaves).
pub fn fill_holes(m: &Array3<bool>) -> Array3<bool> {
    let mut m = duope::fill_holes(m);
    for ax in 0..3 {
        for k in 0..m.len_of(Axis(ax)) {
            let f = fill_holes_2d(m.index_axis(Axis(ax), k));
            m.index_axis_mut(Axis(ax), k).assign(&f);
        }
    }
    m
}

pub fn head_mask(b0: &A3) -> Array3<bool> {
    let t = 0.15 * percentile(b0, 99.0);
    let m = fill_holes(&b0.mapv(|v| v > t));
    // the largest 6-connected component
    let d = m.dim();
    let mut lab = Array3::<u32>::zeros(d);
    let mut sizes = vec![0usize];
    let nb: [(isize, isize, isize); 6] = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)];
    for ((i, j, k), &v) in m.indexed_iter() {
        if !v || lab[[i, j, k]] != 0 {
            continue;
        }
        let id = sizes.len() as u32;
        let mut n = 0;
        let mut stack = vec![(i, j, k)];
        lab[[i, j, k]] = id;
        while let Some((a, b, c)) = stack.pop() {
            n += 1;
            for &(oa, ob, oc) in &nb {
                let (x, y, z) = (a as isize + oa, b as isize + ob, c as isize + oc);
                if x < 0 || y < 0 || z < 0 || x >= d.0 as isize || y >= d.1 as isize || z >= d.2 as isize {
                    continue;
                }
                let p = (x as usize, y as usize, z as usize);
                if m[[p.0, p.1, p.2]] && lab[[p.0, p.1, p.2]] == 0 {
                    lab[[p.0, p.1, p.2]] = id;
                    stack.push(p);
                }
            }
        }
        sizes.push(n);
    }
    let best = (1..sizes.len()).max_by_key(|&s| sizes[s]).unwrap_or(0) as u32;
    fill_holes(&duope::erode(&lab.mapv(|l| best > 0 && l == best), 2))
}

pub struct Estimate {
    pub m: Vec<Mat4>,
    pub c: Vec<Vec<f64>>,
    pub s: Vec<f64>,
    pub mask: Array3<bool>,
    pub c_mm: V3,
}

/// Motion + EC of every volume (phase axis = axis 1); bvec in the same voxel axes; field: voxels along axis 1.
pub fn estimate(data: &[A3], bval: &[f64], bvec: &[V3], zooms: V3, field: Option<&A3>, order: usize, outer: usize, verbose: bool) -> Result<Estimate> {
    let t0 = std::time::Instant::now();
    let n = data.len();
    let shape = data[0].dim();
    let dw: Vec<bool> = bval.iter().map(|&b| b > B0_MAX).collect();
    let b0: Vec<usize> = (0..n).filter(|&i| !dw[i]).collect();
    let idx: Vec<usize> = (0..n).filter(|&i| dw[i]).collect();
    if b0.is_empty() {
        bail!("dwimc needs at least one b0 volume (b < 50)");
    }
    if idx.len() < 6 {
        bail!("dwimc needs at least 6 diffusion-weighted volumes");
    }
    let rf = b0[0];
    let field = field.cloned().unwrap_or_else(|| A3::zeros(shape));
    let mean_b0 = b0.iter().fold(A3::zeros(shape), |a, &i| a + &data[i]) / b0.len() as f64;
    let mask = head_mask(&mean_b0);
    let bmax = bval.iter().cloned().fold(0.0, f64::max);
    let gsc: Vec<V3> = (0..n).map(|i| bvec[i].map(|g| g * (bval[i] / bmax).max(0.0).sqrt())).collect();
    let shells: Vec<i64> = bval.iter().map(|b| (b / 100.0).round() as i64).collect();
    let obs_mask = dilate(&mask, 2);
    let obs: Vec<[usize; 3]> = obs_mask.indexed_iter().filter(|(_, &m)| m).map(|((i, j, k), _)| [i, j, k]).collect();
    let w: Vec<f64> = obs.iter().map(|&[i, j, k]| if mask[[i, j, k]] { 1.0 } else { 0.0 }).collect();
    let mut cm = [0.0; 3];
    let mut nm = 0.0f64;
    for ((i, j, k), &m) in mask.indexed_iter() {
        if m {
            cm[0] += i as f64;
            cm[1] += j as f64;
            cm[2] += k as f64;
            nm += 1.0;
        }
    }
    let c_mm: V3 = std::array::from_fn(|a| zooms[a] * cm[a] / nm.max(1.0));
    let gfield = [gradient(&field, 0), gradient(&field, 1), gradient(&field, 2)];
    let gfy = [gradient(&gfield[PE], 0), gradient(&gfield[PE], 1), gradient(&gfield[PE], 2)];
    let ec = Ec::new(shape, order);
    let ctx = Ctx { shape, zooms, c_mm, field, gfield, gfy, ec: ec.clone(), bval: bval.to_vec(), obs, w };
    let raw: Vec<SplineG> = data.par_iter().map(SplineG::new).collect();
    let mut vols: Vec<Vol> = (0..n).map(|_| Vol { minv: eye(), c: vec![0.0; ec.k], s: 1.0 }).collect();
    let model = |vols: &[Vol]| -> Vec<A3> { (0..n).into_par_iter().map(|i| ctx.to_model(&raw[i], &vols[i])).collect() };
    let mean_of = |ms: &[A3], which: &[usize]| which.iter().fold(A3::zeros(shape), |a, &i| a + &ms[i]) / which.len() as f64;
    for it in 0..outer {
        let fw = FWHM[it.min(FWHM.len() - 1)];
        let mut ms = model(&vols);
        if it > 0 {
            let (v, _) = anchor(&mean_of(&ms, &idx), &mean_of(&ms, &b0), &mask, &zooms, 6, 3, false, true);
            for &i in &idx {
                vols[i].minv = mat_inv(&mat_mul(&mat_inv(&vols[i].minv), &v));
            }
            ms = model(&vols);
            let b0mean = mean_of(&ms, &b0);
            let anc: Vec<(Mat4, [f64; 4])> = idx.par_iter().map(|&i| anchor(&ms[i], &b0mean, &mask, &zooms, 3, 2, true, false)).collect();
            let p6: Vec<Vec<f64>> = anc.iter().map(|(v, _)| params_of(v, &zooms, &c_mm).to_vec()).collect();
            let e4: Vec<Vec<f64>> = anc.iter().map(|(_, e)| e.to_vec()).collect();
            let lb = if idx.len() > 30 { ANCHOR_L } else if idx.len() > 12 { 2 } else { 0 };
            let gi: Vec<V3> = idx.iter().map(|&i| bvec[i]).collect();
            let (sh, _) = sh_basis(&gi, lb);
            let x: Vec<Vec<f64>> = idx.iter().enumerate().map(|(r, &i)| sh[r].iter().cloned().chain(gsc[i]).collect()).collect();
            let fit = |y: &[Vec<f64>]| -> Vec<Vec<f64>> {
                let b = lstsq(&x, y);
                let mut f: Vec<Vec<f64>> = x.iter().map(|r| (0..y[0].len()).map(|c| r.iter().zip(&b).map(|(xv, br)| xv * br[c]).sum()).collect()).collect();
                let q = y[0].len();
                for c in 0..q {
                    let mean = f.iter().map(|r| r[c]).sum::<f64>() / f.len() as f64;
                    f.iter_mut().for_each(|r| r[c] -= mean);
                }
                f
            };
            let (fp, fe) = (fit(&p6), fit(&e4));
            for (r, &i) in idx.iter().enumerate() {
                vols[i].minv = mat_inv(&mat_mul(&mat_inv(&vols[i].minv), &rigid(&fp[r], &zooms, &c_mm)));
                for q in 0..4 {
                    vols[i].c[q] += fe[r][q];
                }
            }
            ms = model(&vols);
        }
        // predictions
        let mut pred: Vec<A3> = vec![A3::zeros((0, 0, 0)); n];
        for &i in &b0 {
            pred[i] = ms[rf].clone();
        }
        if b0.len() > 1 {
            pred[rf] = mean_of(&ms, &b0[1..]);
        }
        let mut uniq: Vec<i64> = idx.iter().map(|&i| shells[i]).collect();
        uniq.sort();
        uniq.dedup();
        let mvox: Vec<(usize, usize, usize)> = mask.indexed_iter().filter(|(_, &m)| m).map(|(p, _)| p).collect();
        for sb in uniq {
            let k: Vec<usize> = idx.iter().cloned().filter(|&i| shells[i] == sb).collect();
            let nk = k.len();
            let gm: Vec<Vec<f64>> = (0..nk)
                .into_par_iter()
                .map(|a| (0..nk).map(|b| mvox.iter().map(|&p| ms[k[a]][p] * ms[k[b]][p]).sum::<f64>() / mvox.len() as f64).collect())
                .collect();
            let gk: Vec<V3> = k.iter().map(|&i| bvec[i]).collect();
            let h = sh_predictor(&gk, &gm, sh_order(nk));
            let ps: Vec<A3> = (0..nk).into_par_iter().map(|a| k.iter().enumerate().fold(A3::zeros(shape), |acc, (b, &j)| acc + &(&ms[j] * h[a][b]))).collect();
            for (a, p) in k.iter().zip(ps) {
                pred[*a] = p;
            }
        }
        drop(ms);
        let sig = zooms.map(|z| fw / 2.3548 / z);
        let preds: Vec<Target> = pred.par_iter().map(|p| Target::new(&if fw > 0.0 { gaussian(p, sig) } else { p.clone() })).collect();
        drop(pred);
        let ys: Vec<Vec<f64>> = data
            .par_iter()
            .map(|d| {
                let s = if fw > 0.0 { gaussian(d, sig) } else { d.clone() };
                ctx.obs.iter().map(|&[i, j, k]| s[[i, j, k]]).collect()
            })
            .collect();
        let others: Vec<usize> = (0..n).filter(|&i| i != rf).collect();
        let new: Vec<(usize, Vol)> = others
            .par_iter()
            .map(|&i| {
                let v = Vol { minv: vols[i].minv, c: vols[i].c.clone(), s: vols[i].s };
                (i, ctx.register(i, &preds[i], &ys[i], v, GN_ITERS))
            })
            .collect();
        for (i, v) in new {
            vols[i] = v;
        }
        // second level: EC terms 1.. = B g (no intercept); the constant's remainder is the phase-axis translation
        let xg: Vec<Vec<f64>> = idx.iter().map(|&i| gsc[i].to_vec()).collect();
        let cy: Vec<Vec<f64>> = idx.iter().map(|&i| vols[i].c.clone()).collect();
        let b = lstsq(&xg, &cy);
        for (r, &i) in idx.iter().enumerate() {
            let fit: Vec<f64> = (0..ec.k).map(|q| (0..3).map(|a| xg[r][a] * b[a][q]).sum()).collect();
            let shift = cy[r][0] - fit[0];
            vols[i].c = fit;
            let mut m = mat_inv(&vols[i].minv);
            m[PE][3] += shift;
            vols[i].minv = mat_inv(&m);
        }
        if verbose {
            eprintln!("dwimc: outer {}/{} (fwhm {} mm) {:.1} s", it + 1, outer, fw, t0.elapsed().as_secs_f64());
        }
    }
    Ok(Estimate {
        m: vols.iter().map(|v| mat_inv(&v.minv)).collect(),
        s: vols.iter().map(|v| v.s).collect(),
        c: vols.into_iter().map(|v| v.c).collect(),
        mask,
        c_mm,
    })
}

/// One volume in the reference head frame (cubic, times the Jacobian, negative overshoot 0) and the RMS
/// displacement (mm, brain) of its motion + eddy currents.
fn correct(vol: &A3, m: &Mat4, c: &[f64], field: &A3, order: usize, mask: &Array3<bool>, zooms: &V3) -> (A3, f64) {
    let shape = vol.dim();
    let ec = Ec::new(shape, order);
    let sp = SplineG::new(vol);
    let minv = mat_inv(m);
    let mc = [minv[0][1], minv[1][1], minv[2][1]];
    let g = [gradient(field, 0), gradient(field, 1), gradient(field, 2)];
    let mut out = A3::zeros(shape);
    Zip::indexed(&mut out).par_for_each(|(i, j, k), o| {
        let xh = [i as f64, j as f64, k as f64];
        let xs = apply(m, &xh);
        let f = field[[i, j, k]] + ec.dot(c, &ec.phi(&xs));
        let fy = mc[0] * g[0][[i, j, k]] + mc[1] * g[1][[i, j, k]] + mc[2] * g[2][[i, j, k]] + ec.dot(c, &ec.dphi_y(&xs));
        *o = (sp.eval(&[xs[0], xs[1] + f, xs[2]], false).0 * (1.0 + fy)).max(0.0);
    });
    let (mut ss, mut nn) = (0.0f64, 0.0f64);
    for ((i, j, k), &mk) in mask.indexed_iter() {
        if mk {
            let xh = [i as f64, j as f64, k as f64];
            let xs = apply(m, &xh);
            let d = [xs[0] - xh[0], xs[1] + ec.dot(c, &ec.phi(&xs)) - xh[1], xs[2] - xh[2]];
            ss += (0..3).map(|a| (d[a] * zooms[a]).powi(2)).sum::<f64>();
            nn += 1.0;
        }
    }
    (out, (ss / nn.max(1.0)).sqrt())
}

// ---------------------------------------------------------------- pipeline

fn swap(ax: usize) -> [usize; 3] {
    [[1, 0, 2], [0, 1, 2], [0, 2, 1]][ax]
}

fn transpose(a: &A3, o: [usize; 3]) -> A3 {
    a.clone().permuted_axes(o).as_standard_layout().into_owned()
}

fn read_table(p: &Path) -> Result<Vec<Vec<f64>>> {
    let txt = std::fs::read_to_string(p).with_context(|| format!("Failed to read {}", p.display()))?;
    Ok(txt.lines().map(|l| l.split_whitespace().filter_map(|x| x.parse().ok()).collect::<Vec<f64>>()).filter(|r: &Vec<f64>| !r.is_empty()).collect())
}

fn ec_names(o: [usize; 3], k: usize) -> Vec<String> {
    let l = |a: usize| ["i", "j", "k"][o[a]];
    let (a, b, c) = (l(0), l(1), l(2));
    let all = ["const".to_string(), a.into(), b.into(), c.into(), format!("{a}{a}"), format!("{b}{b}"), format!("{c}{c}"), format!("{a}{b}"), format!("{a}{c}"), format!("{b}{c}")];
    all[..k].to_vec()
}

/// Python's str(round(v, 5)) for the CSV.
fn num(v: f64) -> String {
    let r = (v * 1e5).round() / 1e5;
    let s = format!("{r}");
    if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("NaN") { s } else { format!("{s}.0") }
}

fn run_one(f: &Path, reverse: Option<&Path>, s: &Settings) -> Result<()> {
    let t0 = std::time::Instant::now();
    let img = Nii::open(f)?;
    if img.extra.is_empty() {
        bail!("{}: a 4D DWI series is needed", duope_pipe::name(f));
    }
    let meta = duope_pipe::sidecar(f)?;
    let st = duope_pipe::stem(f);
    let b: Vec<f64> = read_table(Path::new(&(st.clone() + ".bval")))?.concat();
    let tbl = read_table(Path::new(&(st.clone() + ".bvec")))?;
    let n = img.nvol();
    let g: Vec<V3> = if tbl.len() == 3 && tbl[0].len() == n {
        (0..n).map(|i| [tbl[0][i], tbl[1][i], tbl[2][i]]).collect()
    } else if tbl.len() == n && tbl.iter().all(|r| r.len() == 3) {
        tbl.iter().map(|r| [r[0], r[1], r[2]]).collect()
    } else {
        bail!("{}: {} volumes but the .bvec is not 3 x {}", duope_pipe::name(f), n, n);
    };
    if b.len() != n {
        bail!("{}: {} volumes but {} b-values", duope_pipe::name(f), n, b.len());
    }
    let aff = img.aff;
    let zooms: V3 = std::array::from_fn(|a| (0..3).map(|r| aff[r][a] * aff[r][a]).sum::<f64>().sqrt());
    let data: Vec<A3> = img.volumes()?.collect::<Result<_>>()?;
    let (ax, pol) = duope_pipe::phase(&img, &meta, s.pe.as_deref(), None)?;
    let det = aff[0][0] * (aff[1][1] * aff[2][2] - aff[1][2] * aff[2][1]) - aff[0][1] * (aff[1][0] * aff[2][2] - aff[1][2] * aff[2][0])
        + aff[0][2] * (aff[1][0] * aff[2][1] - aff[1][1] * aff[2][0]);
    let fx = if det > 0.0 { -1.0 } else { 1.0 };
    let g_vox: Vec<V3> = g.iter().map(|v| [fx * v[0], v[1], v[2]]).collect();

    let disp: Option<A3> = if let Some(r) = reverse {
        let ri = Nii::open(r)?;
        let mut rev = duope_pipe::b0(&ri)?;
        if !duope_pipe::same_grid(&ri, &img) {
            rev = linear_transform(&rev, &mat_mul(&mat_inv(&ri.aff), &aff), img.dim3());
        }
        let use_v: Vec<usize> = (0..n).filter(|&i| b[i] < B0_MAX).collect();
        let fwd = use_v.iter().fold(A3::zeros(img.dim3()), |a, &i| a + &data[i]) / use_v.len().max(1) as f64;
        let (pos, neg) = if pol > 0.0 { (&fwd, &rev) } else { (&rev, &fwd) };
        let small = rayon::ThreadPoolBuilder::new().num_threads(rayon::current_num_threads().min(8)).build()?;
        let (d, _, _) = small.install(|| duope::estimate(pos, neg, &aff, ax, None, duope::ALPHA, duope::MAX_MM));
        Some(d * pol)
    } else if let Some(fp) = &s.field {
        let trt = duope_pipe::readout_of(&meta, img.shape[ax]).or(s.readout).with_context(|| format!("{}: --field needs the readout time (JSON TotalReadoutTime or --readout)", duope_pipe::name(f)))?;
        let fi = Nii::open(fp)?;
        let mut fh = fi.volumes()?.next().context("empty field")??;
        if !duope_pipe::same_grid(&fi, &img) {
            fh = linear_transform(&fh, &mat_mul(&mat_inv(&fi.aff), &aff), img.dim3());
        }
        Some(fh * (pol * trt))
    } else {
        None
    };

    let o = swap(ax);
    let data_t: Vec<A3> = data.iter().map(|v| transpose(v, o)).collect();
    drop(data);
    let zt: V3 = std::array::from_fn(|a| zooms[o[a]]);
    let g_t: Vec<V3> = g_vox.iter().map(|v| std::array::from_fn(|a| v[o[a]])).collect();
    let disp_t = disp.as_ref().map(|d| transpose(d, o));
    let est = estimate(&data_t, &b, &g_t, zt, disp_t.as_ref(), s.order, s.iters, true)?;
    let field_t = disp_t.unwrap_or_else(|| A3::zeros(data_t[0].dim()));

    let out = duope_pipe::out_path(f, s.output.as_deref(), "dwimc");
    let mut dims = img.shape.to_vec();
    dims.extend(&img.extra);
    let mut w = VolWriter::create(&out, &img.header, &dims)?;
    let mut rms = vec![0.0; n];
    for chunk in (0..n).collect::<Vec<_>>().chunks(16) {
        let res: Vec<(A3, f64)> = chunk.iter().map(|&i| correct(&data_t[i], &est.m[i], &est.c[i], &field_t, s.order, &est.mask, &zt)).collect();
        for (&i, (v, r)) in chunk.iter().zip(res) {
            rms[i] = r;
            w.put(&transpose(&v, o))?;
        }
    }
    w.finish()?;
    let ost = duope_pipe::stem(&out);

    // rotated b-vectors (FSL convention) and the per-volume table, in the input's voxel axes
    let perm = |m: &Mat4| -> Mat4 { std::array::from_fn(|r| std::array::from_fn(|c| if r < 3 && c < 3 { m[o[r]][o[c]] } else if r < 3 { m[o[r]][3] } else { m[3][c] })) };
    let c_orig: V3 = std::array::from_fn(|a| est.c_mm[o[a]]);
    let names = ec_names(o, est.c[0].len());
    let mut csv = String::from("volume,b,tx_mm,ty_mm,tz_mm,rx_deg,ry_deg,rz_deg");
    for nm in &names {
        csv += &format!(",ec_{nm}_vox");
    }
    csv += ",scale,rms_mm\r\n";
    let mut bv = [String::new(), String::new(), String::new()];
    for i in 0..n {
        let m = perm(&est.m[i]);
        let r = rotation(&m, &zooms);
        let gi = if b[i] > B0_MAX {
            let gv = g_vox[i];
            let rt: V3 = std::array::from_fn(|a| (0..3).map(|q| r[q][a] * gv[q]).sum());
            [fx * rt[0], rt[1], rt[2]]
        } else {
            g[i]
        };
        for a in 0..3 {
            bv[a] += &format!("{}{:.6}", if i > 0 { " " } else { "" }, gi[a]);
        }
        let p = params_of(&m, &zooms, &c_orig);
        csv += &format!("{i},{},{},{},{},{},{},{}", num(b[i]), num(p[0]), num(p[1]), num(p[2]), num(p[3].to_degrees()), num(p[4].to_degrees()), num(p[5].to_degrees()));
        for v in &est.c[i] {
            csv += &format!(",{}", num(*v));
        }
        csv += &format!(",{},{}\r\n", num(est.s[i]), num(rms[i]));
    }
    std::fs::write(ost.clone() + ".bval", b.iter().map(|v| format!("{v}")).collect::<Vec<_>>().join(" ") + "\n")?;
    let mut fh = std::fs::File::create(ost.clone() + ".bvec")?;
    for line in &bv {
        writeln!(fh, "{line}")?;
    }
    std::fs::write(ost + ".csv", csv)?;
    eprintln!("  {}: {} volumes, {:.1} s", duope_pipe::name(f), n, t0.elapsed().as_secs_f64());
    Ok(())
}

/// Every input series; --reverse: one for all or one per input.
pub fn run(inputs: &[String], s: &Settings) -> Result<()> {
    if inputs.is_empty() {
        bail!("--dwimc needs DWI series: tigerbx --dwimc dwi.nii.gz [--reverse pa.nii.gz]");
    }
    if !s.reverse.is_empty() && s.reverse.len() != 1 && s.reverse.len() != inputs.len() {
        bail!("--reverse: one for all inputs or one per input");
    }
    if !s.reverse.is_empty() && s.field.is_some() {
        bail!("--reverse and --field: give one of them");
    }
    let mut failed = Vec::new();
    for (k, f) in inputs.iter().enumerate() {
        let f = PathBuf::from(f);
        eprintln!("dwimc: {}", duope_pipe::name(&f));
        let rev = s.reverse.get(if s.reverse.len() == 1 { 0 } else { k }).map(PathBuf::as_path);
        match run_one(&f, rev, s) {
            Ok(()) => {}
            Err(e) if inputs.len() > 1 => {
                eprintln!("  FAILED {}: {e:#}", duope_pipe::name(&f));
                failed.push(duope_pipe::name(&f));
            }
            Err(e) => return Err(e),
        }
    }
    if !failed.is_empty() {
        eprintln!("Warning: {} of {} inputs failed and were skipped: {}", failed.len(), inputs.len(), failed.join(", "));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_mask_has_no_holes() {
        // a bright head with a dark deep region joined to the outside by a one-voxel channel (a 3D fill keeps it out)
        let (nx, ny, nz) = (40, 40, 30);
        let inside = |i: usize, j: usize, k: usize, rx: f64, rz: f64| {
            let (a, b, c) = ((i as f64 - 20.0) / rx, (j as f64 - 20.0) / rx, (k as f64 - 15.0) / rz);
            a * a + b * b + c * c < 1.0
        };
        let b0 = A3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            if i == 20 && j == 20 && k >= 15 {
                0.0
            } else if inside(i, j, k, 5.0, 4.0) {
                50.0
            } else if inside(i, j, k, 16.0, 12.0) {
                1000.0
            } else {
                0.0
            }
        });
        assert!(!duope::fill_holes(&b0.mapv(|v| v > 150.0))[[20, 20, 15]]);
        let m = head_mask(&b0);
        for ((i, j, k), &v) in m.indexed_iter() {
            if inside(i, j, k, 5.0, 4.0) {
                assert!(v, "hole at {i} {j} {k}");
            }
        }
    }

    #[test]
    fn line_inverse_inverts_a_monotone_map() {
        let (nx, ny, nz) = (2, 30, 2);
        let f = A3::from_shape_fn((nx, ny, nz), |(i, j, k)| 1.5 * ((j as f64) / 5.0 + i as f64 + k as f64).sin());
        let g = line_inverse(&f);
        for i in 0..nx {
            for k in 0..nz {
                for j in 3..ny - 3 {
                    let ys = j as f64 + g[[i, j, k]];
                    let fl = ys.floor() as usize;
                    let t = ys - fl as f64;
                    let fv = f[[i, fl, k]] * (1.0 - t) + f[[i, (fl + 1).min(ny - 1), k]] * t;
                    assert!((ys + fv - j as f64).abs() < 0.05, "{} {}", ys + fv, j);
                }
            }
        }
    }

    #[test]
    fn sh_basis_is_orthonormal_on_many_directions() {
        // Fibonacci sphere: (1/N) sum Y_a Y_b ~ delta_ab / (4 pi)
        let n = 4000;
        let g: Vec<V3> = (0..n)
            .map(|i| {
                let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
                let r = (1.0 - z * z).sqrt();
                let p = i as f64 * 2.399963229728653;
                [r * p.cos(), r * p.sin(), z]
            })
            .collect();
        let (b, _) = sh_basis(&g, 8);
        let nc = b[0].len();
        for a in 0..nc {
            for c in 0..nc {
                let v: f64 = b.iter().map(|r| r[a] * r[c]).sum::<f64>() * 4.0 * std::f64::consts::PI / n as f64;
                assert!((v - if a == c { 1.0 } else { 0.0 }).abs() < 0.02, "{a} {c} {v}");
            }
        }
    }

    #[test]
    fn nelder_mead_finds_a_quadratic_minimum() {
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + 3.0 * (x[1] + 0.5).powi(2);
        let x = nelder_mead(&f, &[0.2, 0.3], 2000, 1e-6, 1e-12);
        assert!((x[0] - 1.0).abs() < 1e-4 && (x[1] + 0.5).abs() < 1e-4);
    }

    #[test]
    fn rigid_and_params_round_trip() {
        let p = [0.5, -1.0, 0.3, 0.02, -0.01, 0.03];
        let z = [2.0, 2.5, 2.2];
        let c = [40.0, 50.0, 30.0];
        let q = params_of(&rigid(&p, &z, &c), &z, &c);
        for a in 0..6 {
            assert!((p[a] - q[a]).abs() < 1e-9);
        }
    }
}
