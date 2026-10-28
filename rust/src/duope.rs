//! DuoPE: susceptibility distortion from a pair of spin-echo EPI images with opposite phase encoding, a port of
//! TigerBx `tigerbx/core/duope.py` (same model, steps and constants; see there):
//!
//!   A(d) = Ia(x + d)(1 + g),  P(d) = Ip(x - d)(1 - g),  g = dd/dpe
//!   E(d) = 1/2 |A - P|^2 + alpha/2 sum w (Laplacian d)^2
//!
//! dipole start on a coarse grid, Gauss-Newton (DCT + Jacobi preconditioned CG) at the finest level, rigid motion of
//! the reverse scan (numreg; jointly with the field above 1 mm / 1 degree), one cubic resampling for the output.
//! Arrays here have the phase axis as axis 1; `estimate` transposes. Agreement with Python is close, not bit for bit
//! (floating-point order, the FFT library, numreg's line search).

use std::sync::Arc;

use ndarray::{Array3, Axis, Zip};
use rustdct::{DctPlanner, TransformType2And3};
use rustfft::{num_complex::Complex, Fft, FftPlanner};

use crate::numreg::{self, matrix_to_rotvec, rotvec_to_matrix, LevelSpec, Options};
use crate::resample::{cubic_at, linear_transform, mat_inv, mat_mul, spline_coefficients, Mat4};

pub const ALPHA: f64 = 0.15;
pub const SHELL_EPS: f64 = 0.05;
const MAX_MM: f64 = 3.5;
const DIPOLE_MM: f64 = 4.0;
const JOINT_MM: f64 = 1.0;
const JOINT_DEG: f64 = 1.0;
pub const MOTION_LEVELS: [LevelSpec; 2] = [(6.0, 1.0, 60, 6.0, 1.0), (3.0, 1.0, 60, 6.0, 1.0)];

pub type A3 = Array3<f64>;
type V3 = [f64; 3];
type Shape = (usize, usize, usize);

// ---------------------------------------------------------------- numpy / scipy.ndimage helpers

/// numpy.percentile(v, q), linear.
pub fn percentile_of(mut v: Vec<f64>, q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let pos = (v.len() - 1) as f64 * q / 100.0;
    let lo = pos.floor() as usize;
    let (_, &mut a, right) = v.select_nth_unstable_by(lo, |a, b| a.partial_cmp(b).unwrap());
    let b = right.iter().cloned().fold(f64::INFINITY, f64::min);
    if b.is_finite() { a + (b - a) * (pos - lo as f64) } else { a }
}

pub fn percentile(a: &A3, q: f64) -> f64 {
    percentile_of(a.iter().cloned().collect(), q)
}

fn median(mut v: Vec<f64>) -> f64 {
    let n = v.len();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn mean(a: &A3) -> f64 {
    a.iter().sum::<f64>() / a.len() as f64
}

fn sum(a: &A3) -> f64 {
    a.iter().sum()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// scipy 'reflect' boundary (d c b a | a b c d | d c b a).
#[inline]
fn refl(i: isize, n: usize) -> usize {
    let n = n as isize;
    let mut i = i;
    loop {
        if i < 0 {
            i = -i - 1;
        } else if i >= n {
            i = 2 * n - i - 1;
        } else {
            return i as usize;
        }
    }
}

/// scipy.ndimage.gaussian_filter(a, sigma) (mode 'reflect', truncate 4; axes with sigma 0 skipped).
pub fn gaussian(a: &A3, sigma: [f64; 3]) -> A3 {
    let mut cur = a.clone();
    for ax in 0..3 {
        let sd = sigma[ax];
        if sd <= 1e-15 {
            continue;
        }
        let r = (4.0 * sd + 0.5) as isize;
        let mut w: Vec<f64> = (-r..=r).map(|x| (-0.5 / (sd * sd) * (x * x) as f64).exp()).collect();
        let s: f64 = w.iter().sum();
        w.iter_mut().for_each(|v| *v /= s);
        let mut out = A3::zeros(cur.dim());
        Zip::from(out.lanes_mut(Axis(ax))).and(cur.lanes(Axis(ax))).par_for_each(|mut o, x| {
            let n = x.len();
            for i in 0..n {
                o[i] = w.iter().enumerate().map(|(k, wk)| wk * x[refl(i as isize + k as isize - r, n)]).sum();
            }
        });
        cur = out;
    }
    cur
}

const NB6: [(isize, isize, isize); 6] = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)];

#[inline]
fn at(dims: Shape, i: usize, j: usize, k: usize, o: (isize, isize, isize)) -> Option<(usize, usize, usize)> {
    let (x, y, z) = (i as isize + o.0, j as isize + o.1, k as isize + o.2);
    (x >= 0 && y >= 0 && z >= 0 && (x as usize) < dims.0 && (y as usize) < dims.1 && (z as usize) < dims.2)
        .then_some((x as usize, y as usize, z as usize))
}

/// scipy binary_erosion / binary_dilation (6-connected structure, border 0), `iters` times.
fn erode(m: &Array3<bool>, iters: usize) -> Array3<bool> {
    let mut cur = m.clone();
    for _ in 0..iters {
        let d = cur.dim();
        cur = Array3::from_shape_fn(d, |(i, j, k)| cur[[i, j, k]] && NB6.iter().all(|&o| at(d, i, j, k, o).is_some_and(|p| cur[p])));
    }
    cur
}

pub fn dilate(m: &Array3<bool>, iters: usize) -> Array3<bool> {
    let mut cur = m.clone();
    for _ in 0..iters {
        let d = cur.dim();
        cur = Array3::from_shape_fn(d, |(i, j, k)| cur[[i, j, k]] || NB6.iter().any(|&o| at(d, i, j, k, o).is_some_and(|p| cur[p])));
    }
    cur
}

/// scipy binary_fill_holes (6-connected background reached from the border stays background).
fn fill_holes(m: &Array3<bool>) -> Array3<bool> {
    let d = m.dim();
    let mut reached = Array3::from_elem(d, false);
    let mut stack = Vec::new();
    for ((i, j, k), &v) in m.indexed_iter() {
        if !v && (i == 0 || j == 0 || k == 0 || i == d.0 - 1 || j == d.1 - 1 || k == d.2 - 1) {
            reached[[i, j, k]] = true;
            stack.push((i, j, k));
        }
    }
    while let Some((i, j, k)) = stack.pop() {
        for &o in &NB6 {
            if let Some(p) = at(d, i, j, k, o) {
                if !m[p] && !reached[p] {
                    reached[p] = true;
                    stack.push(p);
                }
            }
        }
    }
    reached.mapv(|r| !r)
}

/// numpy.gradient(d, axis) (central inside, one-sided at the edges).
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

/// Conjugate gradients as scipy.sparse.linalg.cg (x0 = 0, rtol 1e-5, `maxiter` iterations, optional preconditioner).
fn cg(b: &[f64], maxiter: usize, a: &dyn Fn(&[f64]) -> Vec<f64>, m: Option<&dyn Fn(&[f64]) -> Vec<f64>>) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    let bn = norm(b);
    if bn == 0.0 {
        return b.to_vec();
    }
    let atol = 1e-5 * bn;
    let mut r = b.to_vec();
    let mut p = vec![0.0; n];
    let mut rho_prev = 0.0;
    for it in 0..maxiter {
        if norm(&r) < atol {
            break;
        }
        let z = match m {
            Some(f) => f(&r),
            None => r.clone(),
        };
        let rho = dot(&r, &z);
        if it > 0 {
            let beta = rho / rho_prev;
            for i in 0..n {
                p[i] = p[i] * beta + z[i];
            }
        } else {
            p.copy_from_slice(&z);
        }
        let q = a(&p);
        let alpha = rho / dot(&p, &q);
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        rho_prev = rho;
    }
    x
}

fn flat(a: &A3) -> Vec<f64> {
    a.iter().cloned().collect()
}

fn arr(shape: Shape, v: Vec<f64>) -> A3 {
    A3::from_shape_vec(shape, v).unwrap()
}

fn solve_small(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a.iter().zip(b).map(|(r, &v)| r.iter().cloned().chain([v]).collect()).collect();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| m[i][c].abs().partial_cmp(&m[j][c].abs()).unwrap()).unwrap();
        m.swap(c, p);
        for r in c + 1..n {
            let f = m[r][c] / m[c][c];
            for k in c..=n {
                m[r][k] -= f * m[c][k];
            }
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        x[r] = (m[r][n] - (r + 1..n).map(|k| m[r][k] * x[k]).sum::<f64>()) / m[r][r];
    }
    x
}

// ---------------------------------------------------------------- operators along axis 1

/// Central difference along axis 1, edges replicated.
fn dy(v: &A3) -> A3 {
    let (_, n, _) = v.dim();
    let mut out = A3::zeros(v.dim());
    Zip::indexed(&mut out).par_for_each(|(i, j, k), o| *o = 0.5 * (v[[i, (j + 1).min(n - 1), k]] - v[[i, j.saturating_sub(1), k]]));
    out
}

/// Adjoint of dy.
fn dy_t(w: &A3) -> A3 {
    let (_, n, _) = w.dim();
    let mut out = A3::zeros(w.dim());
    Zip::indexed(&mut out).par_for_each(|(i, j, k), o| {
        *o = if j == 0 {
            -0.5 * (w[[i, 0, k]] + w[[i, 1, k]])
        } else if j == n - 1 {
            0.5 * (w[[i, n - 1, k]] + w[[i, n - 2, k]])
        } else {
            0.5 * (w[[i, j - 1, k]] - w[[i, j + 1, k]])
        }
    });
    out
}

/// -Laplacian, reflecting boundary (diagonalised by the DCT-II).
fn lap(v: &A3) -> A3 {
    let d = v.dim();
    let mut out = A3::zeros(d);
    Zip::indexed(&mut out).par_for_each(|(i, j, k), o| {
        let (x, y, z) = (i as isize, j as isize, k as isize);
        *o = 6.0 * v[[i, j, k]]
            - v[[refl(x - 1, d.0), j, k]] - v[[refl(x + 1, d.0), j, k]]
            - v[[i, refl(y - 1, d.1), k]] - v[[i, refl(y + 1, d.1), k]]
            - v[[i, j, refl(z - 1, d.2)]] - v[[i, j, refl(z + 1, d.2)]];
    });
    out
}

/// img at y + s along axis 1 (linear, clamped): value and derivative d img / dy (0 where clamped).
fn shift(img: &A3, s: &A3) -> (A3, A3) {
    let (_, n, _) = img.dim();
    let mut val = A3::zeros(img.dim());
    let mut der = A3::zeros(img.dim());
    Zip::indexed(&mut val).and(&mut der).and(s).par_for_each(|(i, j, k), v, g, &sv| {
        let y_raw = j as f64 + sv;
        let y = y_raw.clamp(0.0, (n - 1) as f64);
        let i0 = (y.floor() as usize).min(n - 2);
        let w = y - i0 as f64;
        let (v0, v1) = (img[[i, i0, k]], img[[i, i0 + 1, k]]);
        *v = v0 + w * (v1 - v0);
        *g = if y_raw >= 0.0 && y_raw <= (n - 1) as f64 { v1 - v0 } else { 0.0 };
    });
    (val, der)
}

/// Unwarped image (linear): img(x + d)(1 + dd/dy).
pub fn correct(img: &A3, d: &A3) -> A3 {
    let (v, _) = shift(img, d);
    v * &dy(d).mapv(|g| 1.0 + g)
}

/// Block mean by f in every axis (edge padded).
fn down(a: &A3, f: usize) -> A3 {
    if f == 1 {
        return a.clone();
    }
    let d = a.dim();
    let o = (d.0.div_ceil(f), d.1.div_ceil(f), d.2.div_ceil(f));
    let mut out = A3::zeros(o);
    Zip::indexed(&mut out).par_for_each(|(i, j, k), v| {
        let mut s = 0.0;
        for a0 in 0..f {
            for a1 in 0..f {
                for a2 in 0..f {
                    s += a[[(i * f + a0).min(d.0 - 1), (j * f + a1).min(d.1 - 1), (k * f + a2).min(d.2 - 1)]];
                }
            }
        }
        *v = s / (f * f * f) as f64;
    });
    out
}

/// d on a grid f times coarser (its voxels) -> `shape` (full-grid voxels), linear (scipy zoom, grid_mode, nearest).
fn upsample(d: &A3, shape: Shape, f: usize) -> A3 {
    if f == 1 {
        return d.clone();
    }
    let s = d.dim();
    let (sv, tv) = ([s.0, s.1, s.2], [shape.0, shape.1, shape.2]);
    // per axis: (lower index, weight of the upper one) of every output index
    let taps: Vec<Vec<(usize, usize, f64)>> = (0..3)
        .map(|a| {
            (0..tv[a])
                .map(|o| {
                    let c = ((o as f64 + 0.5) * sv[a] as f64 / tv[a] as f64 - 0.5).clamp(0.0, (sv[a] - 1) as f64);
                    let i0 = c.floor() as usize;
                    (i0, (i0 + 1).min(sv[a] - 1), c - i0 as f64)
                })
                .collect()
        })
        .collect();
    let mut out = A3::zeros(shape);
    Zip::indexed(&mut out).par_for_each(|(i, j, k), v| {
        let (x, y, z) = (taps[0][i], taps[1][j], taps[2][k]);
        let mut acc = 0.0;
        for (xi, wx) in [(x.0, 1.0 - x.2), (x.1, x.2)] {
            for (yi, wy) in [(y.0, 1.0 - y.2), (y.1, y.2)] {
                for (zi, wz) in [(z.0, 1.0 - z.2), (z.1, z.2)] {
                    acc += wx * wy * wz * d[[xi, yi, zi]];
                }
            }
        }
        *v = f as f64 * acc;
    });
    out
}

/// Each image to mean 1 inside the head (the two scans can differ in gain), then a common p99 of 1.
fn normalise(a: &A3, p: &A3) -> (A3, A3) {
    let unit = |x: &A3| {
        let t = 0.2 * percentile(x, 99.0);
        let v: Vec<f64> = x.iter().cloned().filter(|&v| v > t).collect();
        x / (v.iter().sum::<f64>() / v.len() as f64)
    };
    let (a, p) = (unit(a), unit(p));
    let s = percentile(&((&a + &p) * 0.5), 99.0);
    (a / s, p / s)
}

// ---------------------------------------------------------------- the dense field at one level

/// eps on the tissue side of every signal boundary (2 voxels) and outside the head (4 voxels), 1 inside.
fn shell_weight(ia: &A3, ip: &A3) -> A3 {
    let s = gaussian(&((ia + ip) * 0.5), [1.0; 3]);
    let t = 0.25 * percentile(&s, 99.0);
    let m = s.mapv(|v| v > t);
    let er = erode(&m, 2);
    let dl = dilate(&m, 4);
    Zip::from(&m).and(&er).and(&dl).map_collect(|&m, &e, &d| if (m && !e) || (d && !m) { SHELL_EPS } else { 1.0 })
}

/// The DCT-II (orthonormal) on a 3D grid and its inverse, through rustdct.
struct Dct3 {
    plans: [Arc<dyn TransformType2And3<f64>>; 3],
    scale: f64,
}

impl Dct3 {
    fn new(shape: Shape) -> Dct3 {
        let mut p = DctPlanner::new();
        let n = [shape.0, shape.1, shape.2];
        Dct3 { plans: n.map(|m| p.plan_dct2(m)), scale: n.iter().map(|&m| 2.0 / m as f64).product() }
    }

    /// idctn(dctn(v) / denom) (both orthonormal; the unnormalised pair DCT-III . DCT-II scaled by prod 2 / n).
    fn solve(&self, v: &mut A3, denom: &A3) {
        for ax in 0..3 {
            let plan = &self.plans[ax];
            Zip::from(v.lanes_mut(Axis(ax))).par_for_each(|mut lane| {
                let mut buf: Vec<f64> = lane.iter().cloned().collect();
                plan.process_dct2(&mut buf);
                lane.iter_mut().zip(buf).for_each(|(o, x)| *o = x);
            });
        }
        Zip::from(&mut *v).and(denom).par_for_each(|x, &d| *x /= d);
        for ax in 0..3 {
            let plan = &self.plans[ax];
            Zip::from(v.lanes_mut(Axis(ax))).par_for_each(|mut lane| {
                let mut buf: Vec<f64> = lane.iter().cloned().collect();
                plan.process_dct3(&mut buf);
                lane.iter_mut().zip(buf).for_each(|(o, x)| *o = x);
            });
        }
        let s = self.scale;
        v.par_mapv_inplace(|x| x * s);
    }
}

/// alpha/2 sum w (Laplacian d)^2: its Hessian, Jacobi diagonal and DCT surrogate.
struct Prior {
    w: A3,
    alpha: f64,
    diag: A3,
    lam2: A3,
}

impl Prior {
    fn new(w: A3, alpha: f64) -> Prior {
        let d = w.dim();
        let mut diag = A3::zeros(d);
        Zip::indexed(&mut diag).par_for_each(|(i, j, k), o| {
            let nb: f64 = NB6
                .iter()
                .map(|&(a, b, c)| {
                    let cl = |x: usize, o: isize, n: usize| (x as isize + o).clamp(0, n as isize - 1) as usize;
                    w[[cl(i, a, d.0), cl(j, b, d.1), cl(k, c, d.2)]]
                })
                .sum();
            *o = alpha * (36.0 * w[[i, j, k]] + nb);
        });
        let n = [d.0, d.1, d.2];
        let l = |a: usize, x: usize| 2.0 - 2.0 * (std::f64::consts::PI * x as f64 / n[a] as f64).cos();
        let lam2 = A3::from_shape_fn(d, |(i, j, k)| (l(0, i) + l(1, j) + l(2, k)).powi(2));
        Prior { w, alpha, diag, lam2 }
    }

    fn hess(&self, v: &A3) -> A3 {
        lap(&(&self.w * &lap(v))) * self.alpha
    }

    fn energy(&self, d: &A3) -> f64 {
        0.5 * sum(&(d * &self.hess(d)))
    }
}

/// (r, a, ga, p, gp, g) of the residual a(1 + g) - p(1 - g).
fn residual(ia: &A3, ip: &A3, d: &A3) -> (A3, A3, A3, A3, A3, A3) {
    let (a, ga) = shift(ia, d);
    let (p, gp) = shift(ip, &d.mapv(|v| -v));
    let g = dy(d);
    let r = Zip::from(&a).and(&p).and(&g).map_collect(|&a, &p, &g| a * (1.0 + g) - p * (1.0 - g));
    (r, a, ga, p, gp, g)
}

fn cost(ia: &A3, ip: &A3, d: &A3, prior: &Prior) -> f64 {
    let r = residual(ia, ip, d).0;
    0.5 * r.iter().map(|v| v * v).sum::<f64>() + prior.energy(d)
}

/// Gauss-Newton on d; CG preconditioned by 0.5 (DCT surrogate + Jacobi); Armijo backtracking.
fn gauss_newton(ia: &A3, ip: &A3, mut d: A3, prior: &Prior, iters: usize, cg_iters: usize) -> A3 {
    let sh = d.dim();
    let dct = Dct3::new(sh);
    let wmean = mean(&prior.w);
    for _ in 0..iters {
        let (r, a, ga, p, gp, g) = residual(ia, ip, &d);
        let ja = Zip::from(&ga).and(&gp).and(&g).map_collect(|&ga, &gp, &g| ga * (1.0 + g) + gp * (1.0 - g));
        let jb = &a + &p;
        let grad = &ja * &r + dy_t(&(&jb * &r)) + prior.hess(&d);
        let hv = |v: &[f64]| {
            let v = arr(sh, v.to_vec());
            let jv = &ja * &v + &jb * &dy(&v);
            flat(&(&ja * &jv + dy_t(&(&jb * &jv)) + prior.hess(&v)))
        };
        let data_diag = &ja * &ja + &jb * &jb * 0.5;
        let diag = flat(&(&data_diag + &prior.diag + 1e-6));
        let dmax = data_diag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let c0 = if dmax > 0.0 { median(data_diag.iter().cloned().filter(|&v| v > 1e-3 * dmax).collect()) } else { 1.0 };
        let denom = prior.lam2.mapv(|l| c0 + prior.alpha * wmean * l);
        let pre = |v: &[f64]| {
            let mut z = arr(sh, v.to_vec());
            dct.solve(&mut z, &denom);
            z.iter().zip(v).zip(&diag).map(|((z, v), q)| 0.5 * (z + v / q)).collect()
        };
        let step = arr(sh, cg(&flat(&grad).iter().map(|v| -v).collect::<Vec<_>>(), cg_iters, &hv, Some(&pre)));
        let (e0, slope) = (cost(ia, ip, &d, prior), sum(&(&grad * &step)));
        let mut t = 1.0;
        let mut moved = false;
        while t > 1e-3 {
            let cand = &d + &(&step * t);
            if cost(ia, ip, &cand, prior) <= e0 + 1e-4 * t * slope {
                d = cand;
                moved = true;
                break;
            }
            t *= 0.5;
        }
        if !moved {
            break;
        }
    }
    d
}

/// Gauss-Newton on d (full-grid voxels, normalised images a / p) at block-mean level f.
fn refine(a: &A3, p: &A3, d0: &A3, f: usize, zooms: V3, iters: usize) -> A3 {
    let (mut ia, mut ip) = (down(a, f), down(p, f));
    let d = down(d0, f) / f as f64;
    // smooth intensity ratio of the two unwarped images (coil / gain differences), split between the scans
    let sb = zooms.map(|z| 40.0 / (z * f as f64));
    let (ca, cp) = (correct(&ia, &d), correct(&ip, &d.mapv(|v| -v)));
    let sumc = &ca + &cp;
    let t = 0.2 * percentile(&sumc, 99.0);
    let w = sumc.mapv(|v| v > t);
    let num = gaussian(&Zip::from(&w).and(&ca).map_collect(|&w, &c| if w { c } else { 0.0 }), sb);
    let den = gaussian(&Zip::from(&w).and(&cp).map_collect(|&w, &c| if w { c } else { 0.0 }), sb);
    let ratio = Zip::from(&num).and(&den).map_collect(|&n, &d| (if d > 1e-6 { n / d.max(1e-6) } else { 1.0 }).clamp(0.5, 2.0));
    Zip::from(&mut ia).and(&mut ip).and(&ratio).for_each(|a, p, &r| {
        *a /= r.sqrt();
        *p *= r.sqrt();
    });
    let prior = Prior::new(shell_weight(&correct(&ia, &d), &correct(&ip, &d.mapv(|v| -v))), ALPHA);
    upsample(&gauss_newton(&ia, &ip, d, &prior, iters, 15), a.dim(), f)
}

// ---------------------------------------------------------------- the dipole start

/// numpy.interp(x, xp, fp) for increasing x.
fn interp_sorted(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    let n = xp.len();
    let mut j = 0;
    x.iter()
        .map(|&v| {
            if v < xp[0] {
                return fp[0];
            }
            while j + 1 < n && xp[j + 1] <= v {
                j += 1;
            }
            if j == n - 1 {
                return fp[n - 1];
            }
            fp[j] + (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]) * (v - xp[j])
        })
        .collect()
}

/// Closed-form per-line d (cumulative-signal matching along axis 1), smoothed with signal weights.
fn line_start(ia: &A3, ip: &A3) -> A3 {
    let (fa, fp) = (0.05 * percentile(ia, 99.0), 0.05 * percentile(ip, 99.0));
    let a = ia.mapv(|v| (v - fa).max(0.0) + 1e-6);
    let p = ip.mapv(|v| (v - fp).max(0.0) + 1e-6);
    let (_, n, _) = a.dim();
    let edges: Vec<f64> = (0..=n).map(|j| j as f64 - 0.5).collect();
    let q: Vec<f64> = (1..2 * n).map(|m| m as f64 / (2 * n) as f64).collect();
    let xs: Vec<f64> = (0..n).map(|j| j as f64).collect();
    let cum = |l: ndarray::ArrayView1<f64>| {
        let mut c = vec![0.0; n + 1];
        for j in 0..n {
            c[j + 1] = c[j] + l[j];
        }
        let last = c[n];
        c.iter_mut().for_each(|v| *v /= last);
        c
    };
    let mut d = A3::zeros(a.dim());
    Zip::from(d.lanes_mut(Axis(1))).and(a.lanes(Axis(1))).and(p.lanes(Axis(1))).par_for_each(|mut o, la, lp| {
        let ya = interp_sorted(&q, &cum(la), &edges);
        let yp = interp_sorted(&q, &cum(lp), &edges);
        let mid: Vec<f64> = ya.iter().zip(&yp).map(|(a, b)| 0.5 * (a + b)).collect();
        let half: Vec<f64> = ya.iter().zip(&yp).map(|(a, b)| 0.5 * (a - b)).collect();
        for (o, v) in o.iter_mut().zip(interp_sorted(&xs, &mid, &half)) {
            *o = v;
        }
    });
    let w = Zip::from(ia).and(ip).map_collect(|&a, &b| a.min(b));
    let wmax = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let w = w / wmax;
    let num = gaussian(&(&w * &d), [2.0; 3]);
    let den = gaussian(&w, [2.0; 3]);
    num / den.mapv(|v| v + 1e-6)
}

/// In-place 3D FFT (rustfft along each axis).
struct Fft3 {
    shape: [usize; 3],
    fwd: [Arc<dyn Fft<f64>>; 3],
    inv: [Arc<dyn Fft<f64>>; 3],
}

impl Fft3 {
    fn new(shape: [usize; 3]) -> Fft3 {
        let mut p = FftPlanner::new();
        Fft3 { shape, fwd: shape.map(|n| p.plan_fft_forward(n)), inv: shape.map(|n| p.plan_fft_inverse(n)) }
    }

    fn run(&self, buf: &mut Array3<Complex<f64>>, inverse: bool) {
        for ax in 0..3 {
            let plan = if inverse { &self.inv[ax] } else { &self.fwd[ax] };
            Zip::from(buf.lanes_mut(Axis(ax))).par_for_each(|mut lane| {
                let mut v: Vec<Complex<f64>> = lane.iter().cloned().collect();
                plan.process(&mut v);
                lane.iter_mut().zip(v).for_each(|(o, x)| *o = x);
            });
        }
        if inverse {
            let s = 1.0 / self.shape.iter().product::<usize>() as f64;
            buf.par_mapv_inplace(|x| x * s);
        }
    }
}

/// d = D chi + H c on one grid: chi on `source`, D by a 2x zero-padded FFT, H 9 solid harmonics (unit norm).
struct Dipole {
    shape: Shape,
    src: Vec<(usize, usize, usize)>,
    pad: [usize; 3],
    kernel: A3,
    fft: Fft3,
    h: Vec<A3>,
}

impl Dipole {
    fn new(shape: Shape, zooms: V3, b: V3, source: &Array3<bool>) -> Dipole {
        let n = [shape.0, shape.1, shape.2];
        let pad = n.map(|m| m / 2 + 2);
        let ps: [usize; 3] = std::array::from_fn(|a| n[a] + 2 * pad[a]);
        // numpy fftfreq (cycles / mm) on the first two axes, rfftfreq (Nyquist positive) on the last
        let freq = |a: usize, i: usize| {
            let m = ps[a];
            let pos = if a == 2 { i <= m / 2 } else { i <= (m - 1) / 2 };
            (if pos { i as f64 } else { i as f64 - m as f64 }) / (m as f64 * zooms[a])
        };
        let kernel = A3::from_shape_fn((ps[0], ps[1], ps[2]), |(i, j, k)| {
            let kv = [freq(0, i), freq(1, j), freq(2, k)];
            let k2 = kv[0] * kv[0] + kv[1] * kv[1] + kv[2] * kv[2];
            let kb = kv[0] * b[0] + kv[1] * b[1] + kv[2] * b[2];
            if k2 > 0.0 { 1.0 / 3.0 - kb * kb / k2 } else { 0.0 }
        });
        let c = |a: usize, i: usize| (i as f64 - (n[a] as f64 - 1.0) / 2.0) * zooms[a] / 100.0;
        type Basis = fn(f64, f64, f64) -> f64;
        let basis: [Basis; 9] = [
            |_, _, _| 1.0,
            |x, _, _| x,
            |_, y, _| y,
            |_, _, z| z,
            |x, y, _| x * y,
            |_, y, z| y * z,
            |x, _, z| x * z,
            |x, y, _| x * x - y * y,
            |x, y, z| 2.0 * z * z - x * x - y * y,
        ];
        let h = basis
            .iter()
            .map(|f| {
                let a = A3::from_shape_fn(shape, |(i, j, k)| f(c(0, i), c(1, j), c(2, k)));
                let nrm = a.iter().map(|v| v * v).sum::<f64>().sqrt();
                a / nrm
            })
            .collect();
        let src = source.indexed_iter().filter(|(_, &s)| s).map(|(p, _)| p).collect();
        Dipole { shape, src, pad, kernel, fft: Fft3::new(ps), h }
    }

    fn ns(&self) -> usize {
        self.src.len()
    }

    fn conv(&self, chi: &A3) -> A3 {
        let k = self.kernel.dim();
        let mut buf = Array3::from_elem(k, Complex::new(0.0, 0.0));
        for ((i, j, l), &v) in chi.indexed_iter() {
            buf[[i + self.pad[0], j + self.pad[1], l + self.pad[2]]] = Complex::new(v, 0.0);
        }
        self.fft.run(&mut buf, false);
        Zip::from(&mut buf).and(&self.kernel).par_for_each(|b, &d| *b *= d);
        self.fft.run(&mut buf, true);
        A3::from_shape_fn(self.shape, |(i, j, l)| buf[[i + self.pad[0], j + self.pad[1], l + self.pad[2]]].re)
    }

    fn field(&self, x: &[f64]) -> A3 {
        let mut chi = A3::zeros(self.shape);
        for (p, &v) in self.src.iter().zip(x) {
            chi[*p] = v;
        }
        let mut out = self.conv(&chi);
        for (hk, &c) in self.h.iter().zip(&x[self.ns()..]) {
            out.scaled_add(c, hk);
        }
        out
    }

    fn field_t(&self, w: &A3) -> Vec<f64> {
        let dw = self.conv(w);
        let mut out: Vec<f64> = self.src.iter().map(|p| dw[*p]).collect();
        out.extend(self.h.iter().map(|hk| sum(&(hk * w))));
        out
    }
}

/// The start D chi + H c (full-grid voxels) of normalised images a / p.
fn dipole_start(a: &A3, p: &A3, zooms: V3, b: V3) -> A3 {
    let (lam, gn_iters, cg_iters) = (1e-5, 3, 15);
    let mut f = 1;
    while zooms.iter().cloned().fold(f64::INFINITY, f64::min) * (f as f64) < DIPOLE_MM {
        f *= 2;
    }
    let ia = gaussian(&down(a, f), [1.0; 3]);
    let ip = gaussian(&down(p, f), [1.0; 3]);
    let sig = gaussian(&((&ia + &ip) * 0.5), [1.0; 3]);
    let t = 0.25 * percentile(&sig, 99.0);
    let source = erode(&sig.mapv(|v| v > t), 2).mapv(|v| !v);
    let model = Dipole::new(ia.dim(), zooms.map(|z| z * f as f64), b, &source);
    let ns = model.ns();
    let n = ns + 9;
    let reg: Vec<f64> = (0..n).map(|i| if i < ns { lam } else { 0.0 }).collect();
    let w = Zip::from(&ia).and(&ip).map_collect(|&a, &b| a.min(b));
    let wmax = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let w = w / wmax;
    let addreg = |mut y: Vec<f64>, x: &[f64]| {
        y.iter_mut().zip(x).zip(&reg).for_each(|((y, x), r)| *y += r * x);
        y
    };
    let amat = |x: &[f64]| addreg(model.field_t(&(&w * &model.field(x))), x);
    let mut x = cg(&model.field_t(&(&w * &line_start(&ia, &ip))), cg_iters, &amat, None);
    let costx = |x: &[f64]| {
        let r = residual(&ia, &ip, &model.field(x)).0;
        0.5 * r.iter().map(|v| v * v).sum::<f64>() + 0.5 * x.iter().zip(&reg).map(|(x, r)| r * x * x).sum::<f64>()
    };
    for _ in 0..gn_iters {
        let d = model.field(&x);
        let (r, aa, ga, pp, gp, g) = residual(&ia, &ip, &d);
        let ja = Zip::from(&ga).and(&gp).and(&g).map_collect(|&ga, &gp, &g| ga * (1.0 + g) + gp * (1.0 - g));
        let jb = &aa + &pp;
        let jdt = |u: &A3| &ja * u + dy_t(&(&jb * u));
        let grad = addreg(model.field_t(&jdt(&r)), &x);
        let hmat = |v: &[f64]| {
            let fv = model.field(v);
            addreg(model.field_t(&jdt(&(&ja * &fv + &jb * &dy(&fv)))), v)
        };
        let step = cg(&grad.iter().map(|v| -v).collect::<Vec<_>>(), cg_iters, &hmat, None);
        let (e0, slope) = (costx(&x), dot(&grad, &step));
        let mut t = 1.0;
        let mut moved = false;
        while t > 1e-3 {
            let cand: Vec<f64> = x.iter().zip(&step).map(|(x, s)| x + t * s).collect();
            if costx(&cand) <= e0 + 1e-4 * t * slope {
                x = cand;
                moved = true;
                break;
            }
            t *= 0.5;
        }
        if !moved {
            break;
        }
    }
    upsample(&model.field(&x), a.dim(), f)
}

// ---------------------------------------------------------------- motion

fn rot3(m: &Mat4) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| m[i][j]))
}

fn apply(m: &Mat4, x: &V3) -> V3 {
    std::array::from_fn(|i| m[i][0] * x[0] + m[i][1] * x[1] + m[i][2] * x[2] + m[i][3])
}

/// Translation of the rigid M at the point c.
fn shift_of(m: &Mat4, c: &V3) -> V3 {
    let y = apply(m, c);
    std::array::from_fn(|i| y[i] - c[i])
}

/// (mm at the point c, degrees) of the rigid M.
pub fn motion_size(m: &Mat4, c: &V3) -> (f64, f64) {
    let s = shift_of(m, c);
    let w = matrix_to_rotvec(&rot3(m));
    ((s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt(), (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt().to_degrees())
}

/// World centre of a mask (mean voxel index through aff).
pub fn mask_centre(mask: &Array3<bool>, aff: &Mat4) -> V3 {
    let mut s = [0.0; 3];
    let mut n = 0.0;
    for ((i, j, k), &m) in mask.indexed_iter() {
        if m {
            s[0] += i as f64;
            s[1] += j as f64;
            s[2] += k as f64;
            n += 1.0;
        }
    }
    apply(aff, &s.map(|v| v / n))
}

fn f32r(a: &A3) -> A3 {
    a.mapv(|v| v as f32 as f64)
}

fn rigid_options<'a>(mm: &'a Array3<bool>, fm: &'a Array3<bool>) -> Options<'a> {
    Options { levels: &MOTION_LEVELS, n_starts: 1, ncc: true, rigid: true, moving_mask: Some(mm), fixed_mask: Some(fm) }
}

/// numreg rigid transform (forward world -> reverse world) of the unwarped reverse scan to the unwarped forward one,
/// without the translation along the phase axis (a constant field offset).
fn rigid_motion(cf: &A3, cr: &A3, aff: &Mat4, mask: &Array3<bool>, pe_world: usize) -> Mat4 {
    let mut m = numreg::register_with(&f32r(cr), aff, &f32r(cf), aff, &rigid_options(mask, mask));
    let c = mask_centre(mask, aff);
    let mut s = shift_of(&m, &c);
    s[pe_world] = 0.0;
    for i in 0..3 {
        m[i][3] = c[i] + s[i] - (0..3).map(|j| m[i][j] * c[j]).sum::<f64>();
    }
    m
}

/// img (on the grid aff) moved by the rigid M (forward world -> its world): img(M x), linear, 0 outside.
fn move_img(img: &A3, aff: &Mat4, m: &Mat4) -> A3 {
    let v = mat_mul(&mat_inv(aff), &mat_mul(m, aff));
    linear_transform(&f32r(img), &v, img.dim())
}

/// scipy map_coordinates(data, c, order=1, mode='nearest') at one point.
#[inline]
fn linear_nearest(data: &A3, c: &V3) -> f64 {
    let d = data.dim();
    let n = [d.0, d.1, d.2];
    let mut i0 = [0usize; 3];
    let mut i1 = [0usize; 3];
    let mut t = [0.0; 3];
    for a in 0..3 {
        let x = c[a].clamp(0.0, (n[a] - 1) as f64);
        let f = x.floor();
        i0[a] = f as usize;
        i1[a] = (i0[a] + 1).min(n[a] - 1);
        t[a] = x - f;
    }
    let mut s = 0.0;
    for (x, wx) in [(i0[0], 1.0 - t[0]), (i1[0], t[0])] {
        for (y, wy) in [(i0[1], 1.0 - t[1]), (i1[1], t[1])] {
            for (z, wz) in [(i0[2], 1.0 - t[2]), (i1[2], t[2])] {
                s += wx * wy * wz * data[[x, y, z]];
            }
        }
    }
    s
}

/// Gauss-Newton on (d, theta): P(x) = Ip(V_theta x - d e)(1 - g); theta = rotation vector and the two world
/// translations across the phase axis, eliminated per step through the Schur complement of the d block.
struct Joint {
    ia: A3,
    ip: A3,
    inv: Mat4,
    d: A3,
    xw: Vec<V3>,
    c: V3,
    free: Vec<usize>,
    theta: [f64; 6],
    prior: Prior,
}

impl Joint {
    fn new(a: &A3, p: &A3, aff: &Mat4, d0: &A3, m0: &Mat4, f: usize) -> Joint {
        let (ia, ip) = (down(a, f), down(p, f));
        let mut s = [[0.0; 4]; 4];
        for i in 0..3 {
            s[i][i] = f as f64;
            s[i][3] = (f as f64 - 1.0) / 2.0;
        }
        s[3][3] = 1.0;
        let affl = mat_mul(aff, &s);
        let d = down(d0, f) / f as f64;
        let xw: Vec<V3> = ia.indexed_iter().map(|((i, j, k), _)| apply(&affl, &[i as f64, j as f64, k as f64])).collect();
        let picks: Vec<&V3> = xw.iter().step_by(7).collect();
        let c: V3 = std::array::from_fn(|a| picks.iter().map(|x| x[a]).sum::<f64>() / picks.len() as f64);
        let pe_world = (0..3).max_by(|&x, &y| aff[x][1].abs().partial_cmp(&aff[y][1].abs()).unwrap()).unwrap();
        let free: Vec<usize> = [0, 1, 2].into_iter().chain((0..3).filter(|&k| k != pe_world).map(|k| 3 + k)).collect();
        let w = matrix_to_rotvec(&rot3(m0));
        let t = shift_of(m0, &c);
        let ci = correct(&ia, &d);
        let prior = Prior::new(shell_weight(&ci, &ci), ALPHA);
        Joint { inv: mat_inv(&affl), ia, ip, d, xw, c, free, theta: [w[0], w[1], w[2], t[0], t[1], t[2]], prior }
    }

    fn q(&self, th: &[f64; 6]) -> Vec<V3> {
        let r = rotvec_to_matrix(&[th[0], th[1], th[2]]);
        let c = self.c;
        let inv = &self.inv;
        use rayon::prelude::*;
        self.xw
            .par_iter()
            .map(|x| {
                let u = [x[0] - c[0], x[1] - c[1], x[2] - c[2]];
                let qw: V3 = std::array::from_fn(|i| r[i][0] * u[0] + r[i][1] * u[1] + r[i][2] * u[2] + c[i] + th[3 + i]);
                apply(inv, &qw)
            })
            .collect()
    }

    fn sample_p(&self, d: &A3, q: &[V3], off: Option<(usize, f64)>) -> A3 {
        let sh = d.dim();
        let df: Vec<f64> = d.iter().cloned().collect();
        use rayon::prelude::*;
        let v: Vec<f64> = q
            .par_iter()
            .zip(df.par_iter())
            .map(|(qq, dv)| {
                let mut c = *qq;
                c[1] -= dv;
                if let Some((k, e)) = off {
                    c[k] += e;
                }
                linear_nearest(&self.ip, &c)
            })
            .collect();
        arr(sh, v)
    }

    fn residual(&self, d: &A3, th: &[f64; 6]) -> A3 {
        let (a, _) = shift(&self.ia, d);
        let p = self.sample_p(d, &self.q(th), None);
        let g = dy(d);
        Zip::from(&a).and(&p).and(&g).map_collect(|&a, &p, &g| a * (1.0 + g) - p * (1.0 - g))
    }

    fn cost(&self, d: &A3, th: &[f64; 6]) -> f64 {
        let r = self.residual(d, th);
        0.5 * r.iter().map(|v| v * v).sum::<f64>() + self.prior.energy(d)
    }

    fn step(&mut self) -> bool {
        let (d, th) = (self.d.clone(), self.theta);
        let sh = d.dim();
        let (a, ga) = shift(&self.ia, &d);
        let q = self.q(&th);
        let p = self.sample_p(&d, &q, None);
        let g = dy(&d);
        let r = Zip::from(&a).and(&p).and(&g).map_collect(|&a, &p, &g| a * (1.0 + g) - p * (1.0 - g));
        // gradient of the trilinear interpolant itself (consistent with the cost)
        let gp: Vec<A3> = (0..3).map(|k| (self.sample_p(&d, &q, Some((k, 1e-3))) - self.sample_p(&d, &q, Some((k, -1e-3)))) / 2e-3).collect();
        let ja = Zip::from(&ga).and(&gp[1]).and(&g).map_collect(|&ga, &gp, &g| ga * (1.0 + g) + gp * (1.0 - g));
        let jb = &a + &p;
        use rayon::prelude::*;
        let (gf, gpf): (Vec<f64>, Vec<Vec<f64>>) = (flat(&g), gp.iter().map(flat).collect());
        let cols: Vec<Vec<f64>> = self
            .free
            .iter()
            .map(|&k| {
                let mut e = th;
                e[k] += 1e-4;
                let qe = self.q(&e);
                (0..gf.len())
                    .into_par_iter()
                    .map(|n| {
                        let s: f64 = (0..3).map(|i| gpf[i][n] * (qe[n][i] - q[n][i]) / 1e-4).sum();
                        -(1.0 - gf[n]) * s
                    })
                    .collect()
            })
            .collect();
        let nf = cols.len();
        let jdt = |w: &A3| &ja * w + dy_t(&(&jb * w));
        let g_d = flat(&(jdt(&r) + self.prior.hess(&d)));
        let rf = flat(&r);
        let g_t: Vec<f64> = cols.iter().map(|c| dot(c, &rf)).collect();
        let prior = &self.prior;
        let hv = |v: &[f64]| {
            let v = arr(sh, v.to_vec());
            flat(&(jdt(&(&ja * &v + &jb * &dy(&v))) + prior.hess(&v)))
        };
        let diag = flat(&(&ja * &ja + &jb * &jb * 0.5 + &prior.diag + 1e-6));
        let jac = |v: &[f64]| v.iter().zip(&diag).map(|(v, q)| v / q).collect::<Vec<f64>>();
        let solve = |b: &[f64]| cg(b, 30, &hv, Some(&jac));
        let hdt: Vec<Vec<f64>> = cols.iter().map(|c| flat(&jdt(&arr(sh, c.clone())))).collect();
        // the nf + 1 solves are independent
        let mut zs: Vec<Vec<f64>> = hdt.iter().chain([&g_d]).collect::<Vec<_>>().par_iter().map(|b| solve(b)).collect();
        let y = zs.pop().unwrap();
        let z = zs;
        let jtj: Vec<Vec<f64>> = (0..nf).map(|i| (0..nf).map(|j| dot(&cols[i], &cols[j])).collect()).collect();
        let tr: f64 = (0..nf).map(|i| jtj[i][i]).sum::<f64>() / nf as f64;
        let smat: Vec<Vec<f64>> = (0..nf)
            .map(|i| (0..nf).map(|j| jtj[i][j] - dot(&hdt[i], &z[j]) + if i == j { 1e-3 * tr } else { 0.0 }).collect())
            .collect();
        let rhs: Vec<f64> = (0..nf).map(|i| -g_t[i] + dot(&hdt[i], &y)).collect();
        let dth_free = solve_small(&smat, &rhs);
        let dd: Vec<f64> = (0..y.len()).map(|n| -(y[n] + (0..nf).map(|k| z[k][n] * dth_free[k]).sum::<f64>())).collect();
        let mut dth = [0.0; 6];
        for (k, &fk) in self.free.iter().enumerate() {
            dth[fk] = dth_free[k];
        }
        let e0 = self.cost(&d, &th);
        let slope = dot(&g_d, &dd) + dot(&g_t, &dth_free);
        let dd = arr(sh, dd);
        let mut t = 1.0;
        while t > 1e-3 {
            let dn = &d + &(&dd * t);
            let tn: [f64; 6] = std::array::from_fn(|i| th[i] + t * dth[i]);
            if self.cost(&dn, &tn) <= e0 + 1e-4 * t * slope {
                self.d = dn;
                self.theta = tn;
                return true;
            }
            t *= 0.5;
        }
        false
    }

    fn run(mut self, iters: usize) -> (A3, Mat4) {
        for _ in 0..iters {
            if !self.step() {
                break;
            }
        }
        let th = self.theta;
        let r = rotvec_to_matrix(&[th[0], th[1], th[2]]);
        let mut m = [[0.0; 4]; 4];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = r[i][j];
            }
            m[i][3] = self.c[i] + th[3 + i] - (0..3).map(|j| r[i][j] * self.c[j]).sum::<f64>();
        }
        m[3][3] = 1.0;
        (self.d, m)
    }
}

// ---------------------------------------------------------------- resampling of the outputs

/// scipy map_coordinates(img, c, order=3, mode='nearest') prepared once: the input edge-padded by 12 voxels (as
/// scipy does for 'nearest'), its spline coefficients; `at` clamps the coordinate into the padded grid.
pub struct Spline {
    coef: A3,
    dims: [usize; 3],
}

const NPAD: usize = 12;

impl Spline {
    pub fn new(img: &A3) -> Spline {
        let d = img.dim();
        let pd = (d.0 + 2 * NPAD, d.1 + 2 * NPAD, d.2 + 2 * NPAD);
        let cl = |i: usize, n: usize| (i as isize - NPAD as isize).clamp(0, n as isize - 1) as usize;
        let padded = A3::from_shape_fn(pd, |(i, j, k)| img[[cl(i, d.0), cl(j, d.1), cl(k, d.2)]]);
        Spline { coef: spline_coefficients(&padded), dims: [pd.0, pd.1, pd.2] }
    }

    #[inline]
    pub fn at(&self, c: &V3) -> f64 {
        let p: V3 = std::array::from_fn(|a| (c[a] + NPAD as f64).clamp(0.0, (self.dims[a] - 1) as f64));
        cubic_at(&self.coef, self.dims, &p)
    }
}

/// (sample coordinates, Jacobian) of unwarp: voxel x of the aff grid takes the image at M(x) moved by sign * d(x)
/// along the phase axis, in the voxels of the image's own grid src_aff (None: the aff grid).
pub fn unwarp_map(d: &A3, aff: &Mat4, pe_axis: usize, m: Option<&Mat4>, sign: f64, src_aff: Option<&Mat4>) -> (Vec<V3>, A3) {
    let vm = m.map(|m| mat_mul(&mat_inv(aff), &mat_mul(m, aff)));
    let sv = src_aff.filter(|s| (0..16).any(|k| (s[k / 4][k % 4] - aff[k / 4][k % 4]).abs() > 1e-6)).map(|s| mat_mul(&mat_inv(s), aff));
    let idx = d
        .indexed_iter()
        .map(|((i, j, k), &dv)| {
            let x = [i as f64, j as f64, k as f64];
            let mut u = match &vm {
                Some(v) => apply(v, &x),
                None => x,
            };
            u[pe_axis] += sign * dv;
            match &sv {
                Some(v) => apply(v, &u),
                None => u,
            }
        })
        .collect();
    (idx, gradient(d, pe_axis).mapv(|g| 1.0 + sign * g))
}

/// Image on the forward grid in one cubic resampling: voxel x takes img at M(x) moved by sign * d(x) along the phase
/// axis, times the Jacobian.
pub fn unwarp(img: &A3, d: &A3, aff: &Mat4, pe_axis: usize, m: Option<&Mat4>, sign: f64) -> A3 {
    let (idx, jac) = unwarp_map(d, aff, pe_axis, m, sign, None);
    sample_map(&Spline::new(img), &idx, &jac)
}

/// Spline values at the coordinates (C order of jac's grid) times the Jacobian.
pub fn sample_map(sp: &Spline, idx: &[V3], jac: &A3) -> A3 {
    let mut out = jac.clone();
    let sl = out.as_slice_mut().unwrap();
    use rayon::prelude::*;
    sl.par_iter_mut().zip(idx.par_iter()).for_each(|(o, c)| *o *= sp.at(c));
    out
}

/// The displacement d (on the aff grid) at the voxels of another grid; M: that grid's world -> aff world (linear,
/// nearest outside).
pub fn field_on(d: &A3, aff: &Mat4, grid_aff: &Mat4, shape: Shape, m: Option<&Mat4>) -> A3 {
    let id = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
    let v = mat_mul(&mat_inv(aff), &mat_mul(m.unwrap_or(&id), grid_aff));
    let mut out = A3::zeros(shape);
    Zip::indexed(&mut out).par_for_each(|(i, j, k), o| *o = linear_nearest(d, &apply(&v, &[i as f64, j as f64, k as f64])));
    out
}

pub fn head_mask(fwd: &A3, rev: &A3) -> Array3<bool> {
    let (mf, mr) = (mean(fwd), mean(rev));
    let m = Zip::from(fwd).and(rev).map_collect(|&a, &b| 0.5 * (a / mf + b / mr));
    let t = 0.15 * percentile(&m, 99.0);
    dilate(&fill_holes(&m.mapv(|v| v > t)), 2)
}

/// Rigid M (series world -> forward world) of a series' reference image (distorted, on its own grid with phase axis
/// pe_axis and polarity sign there) to the corrected forward image: unwarped with the field d (forward grid, scaled
/// to the series' readout time) at the current M, registered, repeated twice.
pub fn series_motion(reference: &A3, ref_aff: &Mat4, fixed: &A3, aff: &Mat4, d: &A3, pe_axis: usize, sign: f64) -> Mat4 {
    let mut m = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
    let mov_mask = head_mask(fixed, fixed);
    let mov = f32r(fixed);
    for _ in 0..2 {
        let u = unwarp(reference, &field_on(d, aff, ref_aff, reference.dim(), Some(&m)), ref_aff, pe_axis, None, sign);
        let fix_mask = head_mask(&u, &u);
        m = numreg::register_with(&mov, aff, &f32r(&u), ref_aff, &rigid_options(&mov_mask, &fix_mask));
    }
    m
}

// ---------------------------------------------------------------- the whole estimate

pub struct Info {
    pub motion_mm: f64,
    pub motion_deg: f64,
    pub joint: bool,
}

fn final_level(zooms: V3) -> usize {
    let zmin = zooms.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut f = 1;
    while zmin * (f * 2) as f64 <= MAX_MM {
        f *= 2;
    }
    f
}

/// The axis order bringing the phase axis to axis 1 (its own inverse).
fn swap(pe_axis: usize) -> [usize; 3] {
    [[1, 0, 2], [0, 1, 2], [0, 2, 1]][pe_axis]
}

fn transpose(a: &A3, o: [usize; 3]) -> A3 {
    a.clone().permuted_axes(o).as_standard_layout().into_owned()
}

/// Displacement d (voxels along `pe_axis`; the forward scan, positive polarity, is unwarped by fwd(x + d e)) and the
/// rigid motion M (forward world -> reverse world) of a forward / reverse pair on one grid. joint: None = auto
/// (field + motion jointly when the motion exceeds 1 mm / 1 degree).
pub fn estimate(fwd: &A3, rev: &A3, aff: &Mat4, pe_axis: usize, joint: Option<bool>) -> (A3, Mat4, Info) {
    let t0 = std::time::Instant::now();
    let mask = head_mask(fwd, rev);
    let o = swap(pe_axis);
    let mut aff_t = *aff;
    for i in 0..4 {
        for (j, &oj) in o.iter().enumerate() {
            aff_t[i][j] = aff[i][oj];
        }
    }
    let zooms: V3 = std::array::from_fn(|j| (0..3).map(|i| aff_t[i][j] * aff_t[i][j]).sum::<f64>().sqrt());
    let pe_world = (0..3).max_by(|&x, &y| aff_t[x][1].abs().partial_cmp(&aff_t[y][1].abs()).unwrap()).unwrap();
    let b: V3 = std::array::from_fn(|j| aff_t[2][j] / zooms[j]);
    let (fa, ra) = (transpose(fwd, o), transpose(rev, o));
    let mk = mask.clone().permuted_axes(o).as_standard_layout().into_owned();
    let dm = mk.dim();
    let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
    for ((i, j, k), &m) in mk.indexed_iter() {
        if m {
            for (a, v) in [i, j, k].into_iter().enumerate() {
                lo[a] = lo[a].min(v);
                hi[a] = hi[a].max(v + 1);
            }
        }
    }
    let n = [dm.0, dm.1, dm.2];
    for a in 0..3 {
        lo[a] = lo[a].saturating_sub(6);
        hi[a] = (hi[a] + 6).min(n[a]);
    }
    let sl = ndarray::s![lo[0]..hi[0], lo[1]..hi[1], lo[2]..hi[2]];
    let mut aff_c = aff_t;
    for i in 0..3 {
        aff_c[i][3] = (0..3).map(|j| aff_t[i][j] * lo[j] as f64).sum::<f64>() + aff_t[i][3];
    }
    let f = final_level(zooms);
    let fc = fa.slice(sl).to_owned();
    let rc = ra.slice(sl).to_owned();
    let mkc = mk.slice(sl).to_owned();

    let timing = std::env::var_os("TIGERBX_DUOPE_TIMING").is_some();
    let lap_t = |what: &str| {
        if timing {
            eprintln!("    {what}: {:.2} s", t0.elapsed().as_secs_f64());
        }
    };
    let (a, p) = normalise(&fc, &rc);
    let start = dipole_start(&a, &p, zooms, b);
    lap_t("dipole start");
    let mut d = refine(&a, &p, &start, f, zooms, 8);
    lap_t("refine");
    let mut m = rigid_motion(&correct(&fc, &d), &correct(&rc, &d.mapv(|v| -v)), &aff_c, &mkc, pe_world);
    lap_t("rigid motion");
    let (mm, deg) = motion_size(&m, &mask_centre(&mkc, &aff_c));
    let use_joint = joint.unwrap_or(deg > JOINT_DEG || mm > JOINT_MM);
    if use_joint {
        let (dj, mj) = Joint::new(&a, &p, &aff_c, &d, &m, f).run(8);
        d = upsample(&dj, a.dim(), f);
        m = mj;
    } else {
        let moved = move_img(&ra, &aff_t, &m).slice(sl).to_owned();
        let (a, p) = normalise(&fc, &moved);
        d = refine(&a, &p, &d, f, zooms, 5);
    }
    lap_t(if use_joint { "joint" } else { "second refine" });
    // back onto the whole grid, edge padded
    let full = A3::from_shape_fn(dm, |(i, j, k)| {
        let cl = |x: usize, a: usize| x.clamp(lo[a], hi[a] - 1) - lo[a];
        d[[cl(i, 0), cl(j, 1), cl(k, 2)]]
    });
    let (motion_mm, motion_deg) = motion_size(&m, &mask_centre(&mk, &aff_t));
    let info = Info { motion_mm, motion_deg, joint: use_joint };
    (transpose(&full, o), m, info)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// a[i, j, k] = ((37 i + 17 j + 7 k) mod 23) / 7 on a 5 x 6 x 7 grid; the reference values are scipy's.
    fn grid() -> A3 {
        A3::from_shape_fn((5, 6, 7), |(i, j, k)| ((i * 37 + j * 17 + k * 7) % 23) as f64 / 7.0)
    }

    fn close(got: &[f64], want: &[f64], tol: f64) {
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < tol, "{got:?} != {want:?}");
        }
    }

    #[test]
    fn helpers_match_scipy() {
        let a = grid();
        let g = gaussian(&a, [1.0, 0.7, 0.0]);
        close(&[g[[0, 0, 0]], g[[2, 3, 4]], g[[4, 5, 6]]], &[0.9617053193092767, 1.4740363613299592, 2.1811518235478666], 1e-12);
        let l = lap(&a);
        close(&[l[[0, 0, 0]], l[[2, 3, 4]], l[[4, 5, 6]]], &[-5.428571428571429, 3.2857142857142856, 5.428571428571429], 1e-12);
        let small = a.slice(ndarray::s![..3, ..3, ..4]).to_owned();
        let z = upsample(&small, (6, 6, 8), 2) / 2.0;
        close(&[z[[0, 0, 0]], z[[1, 2, 3]], z[[5, 5, 7]], z[[3, 4, 2]]], &[0.0, 1.0558035714285714, 2.0, 1.1339285714285716], 1e-12);
        let sp = Spline::new(&a);
        let pts = [[1.3, 2.7, 0.5], [-0.4, 3.0, 2.0], [4.6, 6.2, 7.5]];
        let v: Vec<f64> = pts.iter().map(|c| sp.at(c)).collect();
        close(&v, &[1.4243427599497966, 2.7974477399706434, 3.2721550106457906], 1e-6);
        let mut m = Array3::from_elem((7, 7, 7), false);
        m.slice_mut(ndarray::s![1..6, 1..6, 1..6]).fill(true);
        m[[3, 3, 3]] = false;
        m[[2, 2, 2]] = false;
        let count = |x: &Array3<bool>| x.iter().filter(|&&v| v).count();
        assert_eq!((count(&fill_holes(&m)), count(&erode(&m, 1)), count(&dilate(&m, 2))), (125, 16, 335));
    }

    #[test]
    fn dct_solve_with_unit_denominator_is_the_identity() {
        let a = grid();
        let mut z = a.clone();
        Dct3::new(a.dim()).solve(&mut z, &A3::ones(a.dim()));
        assert!(z.iter().zip(a.iter()).all(|(x, y)| (x - y).abs() < 1e-12));
    }

    #[test]
    fn cg_solves_a_small_spd_system() {
        let a = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let mv = |x: &[f64]| (0..3).map(|i| (0..3).map(|j| a[i][j] * x[j]).sum()).collect::<Vec<f64>>();
        let x = cg(&[1.0, 2.0, 3.0], 10, &mv, None);
        close(&mv(&x), &[1.0, 2.0, 3.0], 1e-8);
    }

    #[test]
    fn rotvec_round_trip() {
        for w in [[0.1, -0.2, 0.3], [0.0, 0.0, 0.0], [1e-8, 0.0, 0.0], [2.5, 0.5, -1.0]] {
            close(&matrix_to_rotvec(&rotvec_to_matrix(&w)), &w, 1e-10);
        }
    }

    // ---------------------------------------------------------------- the estimate on a phantom

    const SHAPE: Shape = (30, 38, 24);

    fn aff() -> Mat4 {
        [[-2.5, 0.0, 0.0, 0.0], [0.0, 2.5, 0.0, 0.0], [0.0, 0.0, 2.5, 0.0], [0.0, 0.0, 0.0, 1.0]]
    }

    /// (image, displacement, inside): a textured ellipsoid and a smooth 2-voxel bump of d.
    pub(crate) fn phantom() -> (A3, A3, Array3<bool>) {
        let n = [SHAPE.0, SHAPE.1, SHAPE.2].map(|n| n as f64);
        let c = n.map(|n| (n - 1.0) / 2.0);
        let body = gaussian(
            &A3::from_shape_fn(SHAPE, |(i, j, k)| {
                let r = [(i as f64 - c[0]) / (0.38 * n[0]), (j as f64 - c[1]) / (0.38 * n[1]), (k as f64 - c[2]) / (0.38 * n[2])];
                ((r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) < 1.0) as u8 as f64
            }),
            [1.0; 3],
        );
        let mut seed = 1u64;
        let noise = A3::from_shape_fn(SHAPE, |_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        });
        let tex = gaussian(&noise, [1.5; 3]);
        let sd = (tex.iter().map(|v| v * v).sum::<f64>() / tex.len() as f64).sqrt();
        let u = Zip::from(&body).and(&tex).map_collect(|&b, &t| (b * (1.0 + 0.6 * t / sd)).max(0.0) * 100.0);
        let d = A3::from_shape_fn(SHAPE, |(i, j, k)| {
            let x = [(i as f64 - c[0]) / 6.0, (j as f64 - c[1] - 4.0) / 7.0, (k as f64 - c[2]) / 5.0];
            2.0 * (-(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])).exp()
        });
        (u, d, body.mapv(|v| v > 0.5))
    }

    /// Signal-conserving distortion along axis 1: the scan whose unwarping is u(x) = obs(x + d)(1 + dd/dy).
    pub(crate) fn distort(u: &A3, d: &A3) -> A3 {
        let n = u.dim().1;
        let mut out = A3::zeros(u.dim());
        let edges: Vec<f64> = (0..=n).map(|j| j as f64 - 0.5).collect();
        for i in 0..u.dim().0 {
            for k in 0..u.dim().2 {
                let mut y = vec![0.0; n];
                for j in 0..n {
                    y[j] = j as f64 + d[[i, j, k]] + 1e-6 * j as f64;
                    if j > 0 {
                        y[j] = y[j].max(y[j - 1]);
                    }
                }
                let mut c = vec![0.0; n + 1];
                for j in 0..n {
                    c[j + 1] = c[j] + u[[i, j, k]];
                }
                let yp: Vec<f64> = [y[0] - 0.5].into_iter().chain(y.iter().cloned()).chain([y[n - 1] + 0.5]).collect();
                let xp: Vec<f64> = [-0.5].into_iter().chain((0..n).map(|j| j as f64)).chain([n as f64 - 0.5]).collect();
                let xe = interp_sorted(&edges, &yp, &xp);
                let ce = interp_sorted(&xe, &edges, &c);
                for j in 0..n {
                    out[[i, j, k]] = ce[j + 1] - ce[j];
                }
            }
        }
        out
    }

    pub(crate) fn mae(a: &A3, b: &A3, m: &Array3<bool>) -> f64 {
        let e: Vec<f64> = a.iter().zip(b).zip(m).filter(|(_, &m)| m).map(|((x, y), _)| x - y).collect();
        let med = median(e.clone());
        e.iter().map(|v| (v - med).abs()).sum::<f64>() / e.len() as f64
    }

    #[test]
    fn estimate_recovers_the_displacement() {
        let (u, d, inside) = phantom();
        let (fwd, rev) = (distort(&u, &d), distort(&u, &d.mapv(|v| -v)));
        let (est, _, info) = estimate(&fwd, &rev, &aff(), 1, None);
        let (e, e0) = (mae(&est, &d, &inside), mae(&A3::zeros(SHAPE), &d, &inside));
        assert!(e < 0.15 && e0 > 0.15, "error {e:.3}, uncorrected {e0:.3}");
        assert!(info.motion_mm < 0.5 && !info.joint);
        // phase axis 0: the transposed problem on the same world
        let t = |a: &A3| a.clone().permuted_axes([1, 0, 2]).as_standard_layout().into_owned();
        let mut at = aff();
        for r in at.iter_mut() {
            r.swap(0, 1);
        }
        let (est_t, _, _) = estimate(&t(&fwd), &t(&rev), &at, 0, None);
        let ins_t = inside.clone().permuted_axes([1, 0, 2]).as_standard_layout().into_owned();
        assert!(mae(&est_t, &t(&d), &ins_t) < 0.15);
    }

    #[test]
    fn rigid_ncc_recovers_a_shift() {
        let (u, _, _) = phantom();
        let a = aff();
        // moving = the phantom shifted by 1.5 voxels (3.75 mm) along k: M maps fixed world to moving world
        let mv = Array3::from_shape_fn(SHAPE, |(i, j, k)| linear_nearest(&u, &[i as f64, j as f64, k as f64 - 1.5]));
        let mask = dilate(&u.mapv(|v| v > 5.0), 2);
        let m = numreg::register_with(&mv, &a, &u, &a, &rigid_options(&mask, &mask));
        let c = mask_centre(&mask, &a);
        let s = shift_of(&m, &c);
        assert!((s[2] - 3.75).abs() < 0.3 && s[0].abs() < 0.3 && s[1].abs() < 0.3, "{s:?}");
        assert!(motion_size(&m, &c).1 < 1.0);
    }
}
