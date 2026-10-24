//! Tissue-anchored intensity normalisation (gnorm), a port of `tigerbx.core.tissuenorm.tissue_norm` in prior
//! mode: background 0, CSF 1/3, GM 2/3, WM 1 (piecewise linear, above WM continued to 4/3, 5/3).
//!
//! Tissue probabilities: 3-class GMM whose mixing weights are the template's tissue priors (smoothed template
//! aseg), alternated with a multiplicative bias-field estimate on a 2 mm grid. Peaks = medians of
//! high-probability, low-gradient voxels of each class. T1 only: if the peaks are not ordered CSF < GM < WM
//! the result falls back to img / p99 of the brain and the peaks are NaN (never an error, batches keep going).

use anyhow::{bail, Result};
use ndarray::{Array3, Array4, Axis, Zip};
use rayon::prelude::*;

const Y_NODES: [f64; 6] = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 4.0 / 3.0, 5.0 / 3.0];
const LAB: [&[u8]; 3] = [&[4, 43, 14, 15, 24], &[3, 42, 8, 47], &[2, 41, 7, 46]];

/// (3, X, Y, Z) prior probabilities of [CSF, GM, WM] from the template aseg (`template_prior`, sigma 2, truncate 3).
pub fn template_prior(aseg: &Array3<u8>) -> Array4<f64> {
    let (x, y, z) = aseg.dim();
    let mut p = Array4::<f64>::zeros((3, x, y, z));
    for (k, labels) in LAB.iter().enumerate() {
        let b = aseg.mapv(|v| if labels.contains(&v) { 1.0 } else { 0.0 });
        p.index_axis_mut(Axis(0), k).assign(&gaussian_reflect(&b, 2.0, 3.0));
    }
    p
}

/// scipy `ndimage.gaussian_filter(a, sigma, truncate)` with the default mode 'reflect'.
fn gaussian_reflect(data: &Array3<f64>, sd: f64, truncate: f64) -> Array3<f64> {
    let r = (truncate * sd + 0.5) as isize;
    let mut w: Vec<f64> = (-r..=r).map(|x| (-0.5 / (sd * sd) * (x * x) as f64).exp()).collect();
    let sum: f64 = w.iter().sum();
    w.iter_mut().for_each(|v| *v /= sum);
    let mut cur = data.clone();
    for ax in 0..3 {
        let mut out = Array3::<f64>::zeros(cur.dim());
        let n = cur.len_of(Axis(ax)) as isize;
        let reflect = |mut j: isize| -> usize {
            loop {
                if j < 0 {
                    j = -j - 1;
                } else if j >= n {
                    j = 2 * n - j - 1;
                } else {
                    return j as usize;
                }
            }
        };
        let src: Vec<_> = cur.lanes(Axis(ax)).into_iter().collect();
        let mut dst: Vec<_> = out.lanes_mut(Axis(ax)).into_iter().collect();
        dst.par_iter_mut().zip(src.par_iter()).for_each(|(o, l)| {
            let line: Vec<f64> = l.iter().cloned().collect();
            for (i, v) in o.iter_mut().enumerate() {
                *v = w.iter().enumerate().map(|(k, wk)| wk * line[reflect(i as isize + k as isize - r)]).sum();
            }
        });
        cur = out;
    }
    cur
}

/// `ndimage.binary_erosion(m, iterations=1)`: 6-connected cross, outside = false.
fn erode(m: &Array3<bool>) -> Array3<bool> {
    let (x, y, z) = m.dim();
    let at = |i: isize, j: isize, k: isize| {
        i >= 0 && j >= 0 && k >= 0 && (i as usize) < x && (j as usize) < y && (k as usize) < z && m[[i as usize, j as usize, k as usize]]
    };
    Array3::from_shape_fn((x, y, z), |(i, j, k)| {
        let (i, j, k) = (i as isize, j as isize, k as isize);
        at(i, j, k) && at(i - 1, j, k) && at(i + 1, j, k) && at(i, j - 1, k) && at(i, j + 1, k) && at(i, j, k - 1) && at(i, j, k + 1)
    })
}

/// `_down`: 2x2x2 block mean (trailing odd planes dropped).
fn down(a: &Array3<f64>) -> Array3<f64> {
    let (x, y, z) = a.dim();
    Array3::from_shape_fn((x / 2, y / 2, z / 2), |(i, j, k)| {
        let mut s = 0.0;
        for d in 0..8 {
            s += a[[2 * i + (d >> 2), 2 * j + ((d >> 1) & 1), 2 * k + (d & 1)]];
        }
        s / 8.0
    })
}

/// `_up`: `ndimage.zoom(a, 2, order=1)` (corner-aligned linear), cut or edge-padded to `shape`.
fn up(a: &Array3<f64>, shape: (usize, usize, usize)) -> Array3<f64> {
    let n = [a.dim().0, a.dim().1, a.dim().2];
    let zn = n.map(|v| 2 * v);
    let s = [shape.0, shape.1, shape.2];
    let coord = |o: usize, ax: usize| -> (usize, usize, f64) {
        let o = o.min(zn[ax] - 1); // edge pad beyond the zoomed size
        let c = if zn[ax] > 1 { o as f64 * (n[ax] - 1) as f64 / (zn[ax] - 1) as f64 } else { 0.0 };
        let f = (c.floor() as usize).min(n[ax] - 1);
        let g = (f + 1).min(n[ax] - 1);
        (f, g, c - f as f64)
    };
    let cx: Vec<_> = (0..s[0]).map(|o| coord(o, 0)).collect();
    let cy: Vec<_> = (0..s[1]).map(|o| coord(o, 1)).collect();
    let cz: Vec<_> = (0..s[2]).map(|o| coord(o, 2)).collect();
    Array3::from_shape_fn(shape, |(i, j, k)| {
        let (x0, x1, tx) = cx[i];
        let (y0, y1, ty) = cy[j];
        let (z0, z1, tz) = cz[k];
        let mut v = 0.0;
        for (xi, wx) in [(x0, 1.0 - tx), (x1, tx)] {
            for (yi, wy) in [(y0, 1.0 - ty), (y1, ty)] {
                for (zi, wz) in [(z0, 1.0 - tz), (z1, tz)] {
                    v += wx * wy * wz * a[[xi, yi, zi]];
                }
            }
        }
        v as f32 as f64
    })
}

/// numpy median (NaN when empty).
fn median(mut v: Vec<f64>) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let (_, &mut hi, _) = v.select_nth_unstable_by(n / 2, |a, b| a.total_cmp(b));
    if n % 2 == 1 {
        return hi;
    }
    let lo = v[..n / 2].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (lo + hi) / 2.0
}

fn masked(a: &Array3<f64>, m: impl Fn(usize) -> bool) -> Vec<f64> {
    a.iter().enumerate().filter(|(i, _)| m(*i)).map(|(_, &v)| v).collect()
}

/// `_map`: piecewise-linear intensity map through the tissue peaks (np.interp, float32).
fn map(x: &Array3<f64>, peaks: &[f64; 3], xmax: f64) -> Array3<f64> {
    let [csf, gm, wm] = *peaks;
    let hi = wm + (wm - csf) / 2.0;
    let xn = [0.0, csf, gm, wm, hi, xmax.max(hi * 1.001)];
    x.mapv(|v| {
        let y = if v.is_nan() {
            f64::NAN
        } else if v <= xn[0] {
            Y_NODES[0]
        } else if v >= xn[5] {
            Y_NODES[5]
        } else {
            let j = (0..5).rev().find(|&j| xn[j] <= v).unwrap_or(0);
            let d = xn[j + 1] - xn[j];
            if d == 0.0 { Y_NODES[j + 1] } else { Y_NODES[j] + (v - xn[j]) / d * (Y_NODES[j + 1] - Y_NODES[j]) }
        };
        y as f32 as f64
    })
}

/// `_low_grad`: |np.gradient(y)| / max(y, 1e-3) inside `core`, inf outside.
fn low_grad(y: &Array3<f64>, core: &Array3<bool>) -> Array3<f64> {
    let (nx, ny, nz) = y.dim();
    let n = [nx, ny, nz];
    let d = |idx: [usize; 3], ax: usize| -> f64 {
        let i = idx[ax];
        let mut a = idx;
        let mut b = idx;
        if n[ax] < 2 {
            return 0.0;
        }
        let h = if i == 0 {
            b[ax] = 1;
            1.0
        } else if i == n[ax] - 1 {
            a[ax] = i - 1;
            1.0
        } else {
            a[ax] = i - 1;
            b[ax] = i + 1;
            2.0
        };
        (y[b] - y[a]) / h
    };
    Array3::from_shape_fn(y.dim(), |(i, j, k)| {
        if !core[[i, j, k]] {
            return f64::INFINITY;
        }
        let g = (0..3).map(|ax| d([i, j, k], ax).powi(2)).sum::<f64>().sqrt();
        g / y[[i, j, k]].max(1e-3)
    })
}

/// `_select`: peak voxels per class (probability > 0.75, low relative gradient; CSF also dark).
fn select(img: &Array3<f64>, core: &Array3<bool>, prob: &Array4<f64>, peaks: &[f64; 3]) -> [Array3<bool>; 3] {
    let xmax = img.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y0 = map(img, peaks, xmax);
    let yg = low_grad(&y0, core);
    std::array::from_fn(|k| {
        let mut s = Array3::from_elem(img.dim(), false);
        Zip::from(&mut s).and(prob.index_axis(Axis(0), k)).and(core).and(&yg).and(&y0).for_each(|s, &p, &c, &g, &y| {
            *s = p > 0.75 && c && g < if k == 2 { 0.10 } else { 0.20 } && (k != 0 || y < 0.45);
        });
        s
    })
}

/// `_estep`: posteriors (3, N) for intensities x and log-priors lp.
fn estep(x: &[f64], lp: &[Vec<f64>; 3], mu: &[f64; 3], sd: &[f64; 3]) -> [Vec<f64>; 3] {
    let n = x.len();
    let mut r: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; n]);
    let (r0, rest) = r.split_at_mut(1);
    let (r1, r2) = rest.split_at_mut(1);
    r0[0].par_iter_mut().zip(r1[0].par_iter_mut()).zip(r2[0].par_iter_mut()).enumerate().for_each(|(i, ((a, b), c))| {
        let ll: [f64; 3] = std::array::from_fn(|j| lp[j][i] - 0.5 * ((x[i] - mu[j]) / sd[j]).powi(2) - sd[j].ln());
        let m = ll[0].max(ll[1]).max(ll[2]);
        let e = ll.map(|l| (l - m).exp());
        let s = e[0] + e[1] + e[2];
        *a = e[0] / s;
        *b = e[1] / s;
        *c = e[2] / s;
    });
    r
}

/// `_em`: GMM with per-voxel mixing weights, fitted on a regular subsample of <= 150000 voxels.
/// Returns (mu, responsibilities for all voxels).
fn em(x: &[f64], prior: &[Vec<f64>; 3], mu0: Option<[f64; 3]>) -> ([f64; 3], [Vec<f64>; 3]) {
    let iters = if mu0.is_some() { 8 } else { 30 };
    let x: Vec<f64> = x.iter().map(|&v| v as f32 as f64).collect();
    let lp: [Vec<f64>; 3] = std::array::from_fn(|j| prior[j].iter().map(|&p| ((p + 1e-4) as f32).ln() as f64).collect());
    let k = (x.len() / 150000).max(1);
    let xs: Vec<f64> = x.iter().step_by(k).cloned().collect();
    let lps: [Vec<f64>; 3] = std::array::from_fn(|j| lp[j].iter().step_by(k).cloned().collect());
    let mut mu = mu0.unwrap_or_else(|| {
        std::array::from_fn(|j| {
            let (s, w) = xs.iter().zip(&lps[j]).fold((0.0, 0.0), |(s, w), (&v, &l)| {
                let wt = (l as f32).exp() as f64 + 1e-6;
                (s + wt * v, w + wt)
            });
            s / w
        })
    });
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let std = (xs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / xs.len() as f64).sqrt();
    let mut sd = [std / 3.0; 3];
    for _ in 0..iters {
        let r = estep(&xs, &lps, &mu, &sd);
        for j in 0..3 {
            let n = r[j].iter().sum::<f64>() + 1e-9;
            mu[j] = r[j].iter().zip(&xs).map(|(a, b)| a * b).sum::<f64>() / n;
            sd[j] = (r[j].iter().zip(&xs).map(|(a, b)| a * (b - mu[j]).powi(2)).sum::<f64>() / n).sqrt() + 1e-6;
        }
    }
    (mu, estep(&x, &lp, &mu, &sd))
}

/// `_probs`: (mu, probabilities (3, X, Y, Z), zero outside `core`).
fn probs(img: &Array3<f64>, core: &Array3<bool>, prior: &Array4<f64>, mu: Option<[f64; 3]>) -> ([f64; 3], Array4<f64>) {
    let idx: Vec<usize> = core.iter().enumerate().filter(|(_, &c)| c).map(|(i, _)| i).collect();
    let flat = |a: ndarray::ArrayView3<f64>| -> Vec<f64> {
        let s = a.as_slice().expect("standard layout");
        idx.iter().map(|&i| s[i]).collect()
    };
    let x = flat(img.view());
    let pr: [Vec<f64>; 3] = std::array::from_fn(|k| flat(prior.index_axis(Axis(0), k)));
    let (mu, r) = em(&x, &pr, mu);
    let (a, b, c) = img.dim();
    let mut prob = Array4::<f64>::zeros((3, a, b, c));
    for k in 0..3 {
        let mut pk = prob.index_axis_mut(Axis(0), k);
        let s = pk.as_slice_mut().expect("standard layout");
        for (&i, &v) in idx.iter().zip(&r[k]) {
            s[i] = v;
        }
    }
    (mu, prob)
}

fn class_peaks(img: &Array3<f64>, core: &Array3<bool>, prob: &Array4<f64>) -> [f64; 3] {
    let c = core.as_slice().expect("standard layout");
    std::array::from_fn(|k| {
        let p = prob.index_axis(Axis(0), k);
        let p = p.as_slice().expect("standard layout");
        median(masked(img, |i| p[i] > 0.75 && c[i]))
    })
}

/// Normalise a skull-stripped T1 on the template grid; brain = `mask`, voxels > 0 when none is given.
/// Returns (y, peaks); on failure y = img / p99 of the brain and peaks = NaN, with a warning.
pub fn tissue_norm(img: &Array3<f32>, mask: Option<&Array3<bool>>, prior: &Array4<f64>) -> (Array3<f32>, [f64; 3]) {
    let mut x = img.mapv(|v| if v.is_finite() { v as f64 } else { 0.0 });
    let mask = match mask {
        Some(m) if m.dim() == x.dim() => m.clone(),
        _ => x.mapv(|v| v > 0.0),
    };
    Zip::from(&mut x).and(&mask).for_each(|v, &m| {
        if !m {
            *v = 0.0;
        }
    });
    let res = if prior.shape()[1..] != *x.shape() {
        Err(anyhow::anyhow!("prior {:?} and image {:?} differ in shape", &prior.shape()[1..], x.shape()))
    } else {
        run(&x, &mask, prior)
    };
    match res {
        Ok((y, pk)) => (y.mapv(|v| v as f32), pk),
        Err(e) => {
            eprintln!("  Warning: tissue_norm failed ({e}); falling back to img / p99");
            let mut v: Vec<f64> = x.iter().cloned().filter(|&v| v > 0.0).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            let scale = if v.is_empty() {
                1.0
            } else {
                // np.percentile(v, 99), linear
                let h = 0.99 * (v.len() - 1) as f64;
                let (f, t) = (h.floor() as usize, h - h.floor());
                v[f] + t * (v[(f + 1).min(v.len() - 1)] - v[f])
            };
            (x.mapv(|v| (v.max(0.0) / scale.max(1e-6)) as f32), [f64::NAN; 3])
        }
    }
}

fn run(x: &Array3<f64>, mask: &Array3<bool>, prior: &Array4<f64>) -> Result<(Array3<f64>, [f64; 3])> {
    let mut img = x.clone();
    let core = erode(mask);

    // bias field on a 2 mm grid (low-gradient GM/WM voxels / class peak == 1, smooth, divide)
    let d_img = down(&img);
    let d_core = down(&core.mapv(|c| if c { 1.0 } else { 0.0 })).mapv(|v| v > 0.99);
    let d_p = {
        let (a, b, c) = d_img.dim();
        let mut p = Array4::<f64>::zeros((3, a, b, c));
        for k in 0..3 {
            p.index_axis_mut(Axis(0), k).assign(&down(&prior.index_axis(Axis(0), k).to_owned()));
        }
        p
    };
    let mut d_field = Array3::<f64>::ones(d_img.dim());
    let mut mu = None;
    for _ in 0..3 {
        let cur = &d_img / &d_field;
        let (m, prob) = probs(&cur, &d_core, &d_p, mu);
        mu = Some(m);
        let pk = class_peaks(&cur, &d_core, &prob);
        let sel = select(&cur, &d_core, &prob, &pk);
        let mut r = Array3::<f64>::zeros(cur.dim());
        let mut w = Array3::<f64>::zeros(cur.dim());
        for (k, wk) in [(1usize, 0.5), (2, 1.0)] {
            Zip::from(&mut r).and(&mut w).and(&sel[k]).and(&cur).for_each(|r, w, &s, &c| {
                if s {
                    *r = c / pk[k];
                    *w = wk;
                }
            });
        }
        let num = gaussian_reflect(&(&r * &w), 6.0, 4.0);
        let den = gaussian_reflect(&w, 6.0, 4.0);
        let mut upd = Zip::from(&num).and(&den).map_collect(|&n, &d| if d > 1e-3 { n / d.max(1e-6) } else { 1.0 });
        let dc = d_core.as_slice().expect("standard layout");
        let m = median(masked(&upd, |i| dc[i]));
        upd.mapv_inplace(|v| v / m);
        d_field = d_field * upd;
    }
    let field = up(&d_field, img.dim());
    Zip::from(&mut img).and(&field).and(mask).for_each(|v, &f, &m| {
        *v = if m { (*v / f.max(1e-3)) as f32 as f64 } else { 0.0 };
    });

    // final peaks on the full-resolution corrected image
    let dc = d_core.as_slice().expect("standard layout");
    let fm = median(masked(&d_field, |i| dc[i]));
    let (_, prob) = probs(&img, &core, prior, mu.map(|m| m.map(|v| v / fm)));
    let pk0 = class_peaks(&img, &core, &prob);
    let sel = select(&img, &core, &prob, &pk0);
    let pk: [f64; 3] = std::array::from_fn(|k| {
        let s = sel[k].as_slice().expect("standard layout");
        let v = masked(&img, |i| s[i]);
        if v.len() > 50 { median(v) } else { pk0[k] }
    });
    if !(0.0 < pk[0] && pk[0] < pk[1] && pk[1] < pk[2]) {
        bail!("unexpected tissue order {pk:?}");
    }
    let xmax = img.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut y = map(&img, &pk, xmax);
    Zip::from(&mut y).and(mask).for_each(|v, &m| {
        if !m {
            *v = 0.0;
        }
    });
    Ok((y, pk))
}
