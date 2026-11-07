//! tigerbx.inspect (tigerbx/pipelines/inspect.py) as a native port: the same network input, header features, model
//! outputs, rule-based checks and inspect_summary.csv. The input follows scipy: uniform_filter / gaussian_filter
//! (mode 'reflect', truncate 4) and map_coordinates (order 1, mode 'nearest'), numpy percentile (linear).
use anyhow::{Context, Result};
use ndarray::{Array3, ArrayD};
use nifti::NiftiHeader;
use ort::session::Session;
use ort::value::TensorRef;
use rayon::prelude::*;
use std::io::Write;
use std::path::Path;

use crate::resample::{best_affine, Mat4};

const N: usize = 96;
const STEP: f64 = 256.0 / 96.0;
/// Block average target (mm) before the anti-alias Gaussian.
const BLOCK: f64 = 2.0;
pub const CONTRASTS: [&str; 6] = ["T1w", "T2w", "FLAIR", "b0", "DWI", "other"];
const MOTION_CONTRASTS: [&str; 3] = ["T1w", "T2w", "FLAIR"];
pub const SUMMARY_NAME: &str = "inspect_summary.csv";
/// The columns of tigerbx.inspect (SUMMARY_COLUMNS) and of detail=True (DETAIL_COLUMNS).
pub const SUMMARY_COLUMNS: [&str; 14] = [
    "file", "status", "contrast", "p_contrast", "skull_stripped", "epi", "orientation", "truncated", "motion", "ndim",
    "matrix", "voxel_mm", "warnings", "error",
];
pub const DETAIL_COLUMNS: [&str; 31] = [
    "file", "status", "contrast", "p_contrast", "p_T1w", "p_T2w", "p_FLAIR", "p_b0", "p_DWI", "p_other",
    "skull_stripped", "p_skull_stripped", "epi", "p_epi", "orientation", "p_orientation", "truncated", "p_truncated",
    "motion", "motion_flag", "n_volumes", "ndim", "matrix", "voxel_mm", "thick_slices", "oblique_deg", "fov_mm",
    "nonfinite_frac", "label_map", "warnings", "error",
];

/// One row: (column, value as Python's csv module writes it).
pub type Row = Vec<(&'static str, String)>;

// ── Python-compatible formatting ─────────────────────────────────────────────────────────────────────────────────

/// repr(float) of Python: shortest round-trip digits, scientific below 1e-4 and from 1e16, '.0' for integers.
fn py_float(x: f64) -> String {
    if !x.is_finite() {
        return if x.is_nan() { "nan".into() } else if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let a = x.abs();
    if !(1e-4..1e16).contains(&a) {
        let s = format!("{:e}", x); // shortest digits, e.g. 1e-6, 2.5e-5
        let (m, e) = s.split_once('e').unwrap();
        let e: i32 = e.parse().unwrap();
        return format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
    }
    let s = format!("{}", x);
    if s.contains('.') { s } else { format!("{s}.0") }
}

/// Python round(x, n) (round half to even on the decimal value, close enough for probabilities).
fn py_round(x: f64, n: i32) -> f64 {
    let p = 10f64.powi(n);
    let y = x * p;
    let r = y.round();
    let r = if (y - y.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - y.signum() } else { r };
    r / p
}

fn py_bool(b: bool) -> String {
    if b { "True".into() } else { "False".into() }
}

/// Python f'{x:.2g}' (2 significant digits, scientific when the exponent is < -4 or >= 2).
fn py_g2(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    let e = x.abs().log10().floor() as i32;
    let s = format!("{:.1e}", x);
    let (m, ee) = s.split_once('e').unwrap();
    let ee: i32 = ee.parse().unwrap();
    if ee < -4 || ee >= 2 {
        let m = m.trim_end_matches('0').trim_end_matches('.');
        return format!("{m}e{}{:02}", if ee < 0 { '-' } else { '+' }, ee.abs());
    }
    let _ = e;
    let dec = (1 - ee).max(0) as usize;
    let s = format!("{:.*}", dec, x);
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

// ── geometry ─────────────────────────────────────────────────────────────────────────────────────────────────────

fn inv_affine(a: &Mat4) -> Mat4 {
    let m = [[a[0][0], a[0][1], a[0][2]], [a[1][0], a[1][1], a[1][2]], [a[2][0], a[2][1], a[2][2]]];
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let (i1, i2, j1, j2) = ((j + 1) % 3, (j + 2) % 3, (i + 1) % 3, (i + 2) % 3);
            r[i][j] = (m[i1][j1] * m[i2][j2] - m[i1][j2] * m[i2][j1]) / det;
        }
    }
    let mut out = [[0.0; 4]; 4];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = r[i][j];
        }
        out[i][3] = -(0..3).map(|j| r[i][j] * a[j][3]).sum::<f64>();
    }
    out[3][3] = 1.0;
    out
}

fn apply(m: &Mat4, x: [f64; 3]) -> [f64; 3] {
    let mut o = [0.0; 3];
    for i in 0..3 {
        o[i] = m[i][0] * x[0] + m[i][1] * x[1] + m[i][2] * x[2] + m[i][3];
    }
    o
}

/// Voxel sizes: the norms of the affine's columns.
fn zooms(a: &Mat4) -> [f64; 3] {
    let mut z = [0.0; 3];
    for j in 0..3 {
        z[j] = (a[0][j] * a[0][j] + a[1][j] * a[1][j] + a[2][j] * a[2][j]).sqrt();
    }
    z
}

// ── filters (scipy.ndimage, mode 'reflect') ──────────────────────────────────────────────────────────────────────

/// scipy 'reflect' index (half-sample symmetric: -1 -> 0, n -> n - 1), any distance.
fn reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    if n == 1 {
        return 0;
    }
    let p = 2 * n;
    let mut j = i.rem_euclid(p);
    if j >= n {
        j = p - 1 - j;
    }
    j as usize
}

/// 1D correlation of every line along `axis` with `w` (centred), mode 'reflect'.
fn correlate1d(a: &Array3<f64>, axis: usize, w: &[f64]) -> Array3<f64> {
    let dims = a.dim();
    let shape = [dims.0, dims.1, dims.2];
    let n = shape[axis];
    let r = (w.len() / 2) as isize;
    let src = a.as_standard_layout();
    let src = src.as_slice().unwrap();
    let strides = [shape[1] * shape[2], shape[2], 1];
    let mut out = Array3::<f64>::zeros(dims);
    let st = strides[axis];
    out.as_slice_mut().unwrap().par_iter_mut().enumerate().for_each(|(idx, o)| {
        let pos = (idx / strides[axis]) % n;
        let base = idx - pos * st;
        let mut acc = 0.0;
        for (k, &wk) in w.iter().enumerate() {
            let p = reflect(pos as isize + k as isize - r, n);
            acc += wk * src[base + p * st];
        }
        *o = acc;
    });
    out
}

/// scipy.ndimage.uniform_filter(a, 3).
fn uniform3(a: &Array3<f64>) -> Array3<f64> {
    let w = [1.0 / 3.0; 3];
    let mut cur = a.clone();
    for ax in 0..3 {
        cur = correlate1d(&cur, ax, &w);
    }
    cur
}

/// scipy.ndimage.gaussian_filter(a, sigma) (truncate 4; axes with sigma <= 1e-15 skipped).
fn gaussian(a: &Array3<f64>, sigma: [f64; 3]) -> Array3<f64> {
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
        cur = correlate1d(&cur, ax, &w);
    }
    cur
}

/// scipy.ndimage.map_coordinates(a, c, order=1, mode='nearest') at one point.
fn linear(a: &Array3<f64>, c: [f64; 3]) -> f64 {
    let (nx, ny, nz) = a.dim();
    let dims = [nx, ny, nz];
    let mut i0 = [0usize; 3];
    let mut f = [0.0; 3];
    let mut i1 = [0usize; 3];
    for d in 0..3 {
        let x = c[d].clamp(0.0, (dims[d] - 1) as f64);
        let fl = x.floor();
        i0[d] = fl as usize;
        i1[d] = (i0[d] + 1).min(dims[d] - 1);
        f[d] = x - fl;
    }
    let mut v = 0.0;
    for (a0, wa) in [(i0[0], 1.0 - f[0]), (i1[0], f[0])] {
        for (b0, wb) in [(i0[1], 1.0 - f[1]), (i1[1], f[1])] {
            for (c0, wc) in [(i0[2], 1.0 - f[2]), (i1[2], f[2])] {
                let w = wa * wb * wc;
                if w != 0.0 {
                    v += w * a[[a0, b0, c0]];
                }
            }
        }
    }
    v
}

/// numpy.percentile(v, q) with linear interpolation.
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

// ── image ────────────────────────────────────────────────────────────────────────────────────────────────────────

/// The first volume of a 3D / 4D / ... array (trailing dimensions at index 0), as nibabel `_volume`.
fn first_volume(data: &ArrayD<f64>) -> Result<Array3<f64>> {
    let sh = data.shape();
    if sh.len() < 3 {
        anyhow::bail!("Expected at least 3 dimensions, got shape {:?}", sh);
    }
    let mut v = data.view();
    while v.ndim() > 3 {
        let last = ndarray::Axis(v.ndim() - 1);
        v = v.index_axis_move(last, 0);
    }
    Ok(v.into_dimensionality::<ndarray::Ix3>()?.as_standard_layout().into_owned())
}

/// The (4, 96, 96, 96) network input (inspect.make_input).
fn make_input(d0: &Array3<f64>, a0: &Mat4) -> Vec<f32> {
    let dims = d0.dim();
    let shape0 = [dims.0, dims.1, dims.2];
    let z0 = zooms(a0);
    let k: Vec<usize> = z0.iter().map(|z| ((BLOCK / z).floor() as usize).max(1)).collect();
    let d = d0.mapv(|v| if v.is_finite() { v } else { 0.0 });
    let empty = d0.mapv(|v| if !v.is_finite() || v == 0.0 { 1.0 } else { 0.0 });
    // p99 and the centre of mass from every k-th voxel
    let mut nz = Vec::new();
    for i in (0..shape0[0]).step_by(k[0]) {
        for j in (0..shape0[1]).step_by(k[1]) {
            for l in (0..shape0[2]).step_by(k[2]) {
                if empty[[i, j, l]] == 0.0 {
                    nz.push(d[[i, j, l]]);
                }
            }
        }
    }
    let p99 = if nz.is_empty() { 1.0 } else { percentile(nz, 99.0) };
    let (mut acc, mut ws) = ([0.0f64; 3], 0.0f64);
    let mut any_fg = false;
    for i in (0..shape0[0]).step_by(k[0]) {
        for j in (0..shape0[1]).step_by(k[1]) {
            for l in (0..shape0[2]).step_by(k[2]) {
                let v = d[[i, j, l]];
                if v > 0.05 * p99 {
                    any_fg = true;
                    acc[0] += i as f64 * v;
                    acc[1] += j as f64 * v;
                    acc[2] += l as f64 * v;
                    ws += v;
                }
            }
        }
    }
    if !any_fg {
        for i in (0..shape0[0]).step_by(k[0]) {
            for j in (0..shape0[1]).step_by(k[1]) {
                for l in (0..shape0[2]).step_by(k[2]) {
                    acc[0] += i as f64;
                    acc[1] += j as f64;
                    acc[2] += l as f64;
                    ws += 1.0;
                }
            }
        }
    }
    let vox = [acc[0] / ws, acc[1] / ws, acc[2] / ws];
    let mut c = [0.0; 3];
    for r in 0..3 {
        c[r] = a0[r][0] * vox[0] + a0[r][1] * vox[1] + a0[r][2] * vox[2] + a0[r][3];
    }
    let u = uniform3(&d);
    let hf = ndarray::Zip::from(&d).and(&u).map_collect(|a, b| (a - b).abs());
    let mut vols = [d, empty, hf];
    let mut a = *a0;
    if k.iter().any(|&v| v > 1) {
        let n = [shape0[0] / k[0], shape0[1] / k[1], shape0[2] / k[2]];
        let cnt = (k[0] * k[1] * k[2]) as f64;
        vols = vols.map(|v| {
            Array3::from_shape_fn((n[0], n[1], n[2]), |(i, j, l)| {
                let mut s = 0.0;
                for di in 0..k[0] {
                    for dj in 0..k[1] {
                        for dl in 0..k[2] {
                            s += v[[i * k[0] + di, j * k[1] + dj, l * k[2] + dl]];
                        }
                    }
                }
                s / cnt
            })
        });
        for r in 0..3 {
            for cc in 0..3 {
                a[r][cc] = a0[r][cc] * k[cc] as f64;
            }
            a[r][3] = (0..3).map(|cc| a0[r][cc] * (k[cc] as f64 - 1.0) / 2.0).sum::<f64>() + a0[r][3];
        }
    }
    // Gaussian of sigma0 = sqrt((STEP / z)^2 - 1) / 2 voxels at the full resolution, minus the box's variance
    let mut sig = [0.0; 3];
    for i in 0..3 {
        let s0 = ((STEP / z0[i]).powi(2) - 1.0).max(0.0).sqrt() / 2.0;
        let kk = k[i] as f64;
        sig[i] = (s0 * s0 - (kk * kk - 1.0) / 12.0).max(0.0).sqrt() / kk;
    }
    if sig.iter().any(|&s| s > 0.0) {
        vols = vols.map(|v| gaussian(&v, sig));
    }
    let ia = inv_affine(&a);
    let ia0 = inv_affine(a0);
    let h = (N as f64 - 1.0) / 2.0;
    let mut x = vec![0f32; 4 * N * N * N];
    let (x0, rest) = x.split_at_mut(N * N * N);
    let (x1, rest) = rest.split_at_mut(N * N * N);
    let (x2, x3) = rest.split_at_mut(N * N * N);
    x0.par_iter_mut().zip(x1.par_iter_mut()).zip(x2.par_iter_mut()).zip(x3.par_iter_mut()).enumerate().for_each(
        |(idx, (((o0, o1), o2), o3))| {
            let (i, j, l) = (idx / (N * N), (idx / N) % N, idx % N);
            let g = [(i as f64 - h) * STEP + c[0], (j as f64 - h) * STEP + c[1], (l as f64 - h) * STEP + c[2]];
            let v = apply(&ia, g);
            let v0 = apply(&ia0, g);
            let inside = (0..3).all(|r| v0[r] >= -0.5 && v0[r] <= shape0[r] as f64 - 0.5);
            let fi = if inside { 1.0 } else { 0.0 };
            let ds = linear(&vols[0], v);
            let es = linear(&vols[1], v);
            let hs = linear(&vols[2], v);
            *o0 = (ds * fi / p99.max(1e-6)).clamp(0.0, 1.5) as f32;
            *o1 = if inside { es as f32 } else { 1.0 };
            *o2 = fi as f32;
            *o3 = (hs * fi / p99.max(1e-6)).clamp(0.0, 1.5) as f32;
        },
    );
    x
}

fn is_float(h: &NiftiHeader) -> bool {
    matches!(h.datatype, 16 | 64 | 1536)
}

/// inspect.header_features: log voxel size (3), matrix / 256 (3), obliqueness, sform 4, qform 4, 0, float data.
fn header_features(h: &NiftiHeader, a: &Mat4, shape: [usize; 3]) -> [f32; 11] {
    let z = zooms(a);
    let mut obl: f64 = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let v = (a[i][j] / z[j]).abs();
            obl = obl.max((v - v.round()).abs());
        }
    }
    let mut f = [0f32; 11];
    for i in 0..3 {
        f[i] = z[i].max(0.1).ln() as f32;
        f[3 + i] = (shape[i] as f64 / 256.0) as f32;
    }
    f[6] = obl as f32;
    f[7] = (h.sform_code == 4) as u8 as f32;
    f[8] = (h.qform_code == 4) as u8 as f32;
    f[10] = is_float(h) as u8 as f32;
    f
}

const PERMS: [[usize; 3]; 6] = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];

/// Orientation class -> anatomical direction of the header's +x, +y, +z ('RAS' = class 0).
fn orientation_code(k: usize) -> String {
    let p = PERMS[k / 4];
    let (f_ap, f_si) = ((k / 2) % 2, k % 2);
    let letters = ["R", if f_ap == 1 { "P" } else { "A" }, if f_si == 1 { "I" } else { "S" }];
    p.iter().map(|&a| letters[a]).collect()
}

fn softmax(z: &[f32]) -> Vec<f64> {
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let e: Vec<f64> = z.iter().map(|&v| (v as f64 - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

fn sigmoid(z: f32) -> f64 {
    1.0 / (1.0 + (-(z as f64)).exp())
}

/// inspect.rule_checks: values of the check columns and the rule warnings. `d`: the first volume with NaN / Inf
/// set to 0, `nonfinite`: their fraction (only reported).
fn rule_checks(h: &NiftiHeader, a: &Mat4, full_shape: &[usize], d: &Array3<f64>, nonfinite: f64) -> (Row, Vec<String>) {
    let n_vol: usize = if full_shape.len() > 3 { full_shape[3..].iter().product() } else { 1 };
    let z = zooms(a);
    let mut oblique: f64 = 0.0;
    for j in 0..3 {
        let m = (0..3).map(|i| (a[i][j] / z[j]).abs()).fold(0.0f64, f64::max).clamp(0.0, 1.0);
        oblique = oblique.max(m.acos().to_degrees());
    }
    let fov: Vec<f64> = (0..3).map(|i| z[i] * full_shape[i] as f64).collect();
    let (nx, ny, nz) = d.dim();
    let q = [(nx / 64).max(1), (ny / 64).max(1), (nz / 64).max(1)];
    let mut sub = Vec::new();
    for i in (0..nx).step_by(q[0]) {
        for j in (0..ny).step_by(q[1]) {
            for l in (0..nz).step_by(q[2]) {
                sub.push(d[[i, j, l]] as f32 as f64); // the Python checks see float32 data
            }
        }
    }
    let integer = !sub.is_empty() && sub.iter().all(|v| *v == v.round());
    let mut uniq: Vec<f64> = Vec::new();
    if integer {
        uniq = sub.clone();
        uniq.sort_by(|x, y| x.partial_cmp(y).unwrap());
        uniq.dedup();
    }
    let span = if uniq.is_empty() { 0.0 } else { uniq[uniq.len() - 1] - uniq[0] + 1.0 };
    let label_map = integer && uniq.len() > 1 && (uniq.len() <= 20 || (uniq.len() <= 300 && uniq.len() as f64 / span < 0.5));
    let zmax = z.iter().cloned().fold(f64::MIN, f64::max);
    let zmin = z.iter().cloned().fold(f64::MAX, f64::min);
    let thick = zmax >= 3.0 || zmax / zmin.max(1e-6) >= 3.0;
    let no_space = h.sform_code == 0 && h.qform_code == 0;
    let fov_min = fov.iter().cloned().fold(f64::MAX, f64::min);
    let mut w = Vec::new();
    if n_vol > 1 {
        w.push(format!("{n_vol} volumes, only the first is used"));
    }
    if thick {
        w.push("thick slices / low resolution".into());
    }
    if oblique > 20.0 {
        w.push(format!("oblique header ({:.0} deg)", oblique));
    }
    if fov_min < 80.0 {
        w.push(format!("slab: field of view {:.0} mm along one axis", fov_min));
    }
    if nonfinite > 0.0 {
        w.push(format!("non-finite voxels ({}%)", py_g2(100.0 * nonfinite)));
    }
    if label_map {
        w.push("a label map / mask, not an image".into());
    }
    let constant = sub.is_empty() || {
        let lo = sub.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = sub.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        lo == hi
    };
    if constant {
        w.push("empty or constant image".into());
    }
    if no_space {
        w.push("no spatial header (sform and qform codes 0): the orientation is a guess".into());
    }
    let row: Row = vec![
        ("n_volumes", n_vol.to_string()),
        ("ndim", full_shape.len().to_string()),
        ("matrix", full_shape.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("x")),
        ("voxel_mm", z.iter().map(|v| format!("{:.2}", v)).collect::<Vec<_>>().join("x")),
        ("thick_slices", py_bool(thick)),
        ("oblique_deg", py_float(py_round(oblique, 1))),
        ("fov_mm", fov.iter().map(|v| format!("{:.0}", v)).collect::<Vec<_>>().join("x")),
        ("nonfinite_frac", py_float(py_round(nonfinite, 6))),
        ("label_map", py_bool(label_map)),
    ];
    (row, w)
}

/// Inspect one image (header + raw data, NaN kept): every column of DETAIL_COLUMNS but file / status / error.
pub fn inspect_one(header: &NiftiHeader, data: &ArrayD<f64>, session: &mut Session) -> Result<Row> {
    let a = best_affine(header);
    let raw = first_volume(data)?;
    // NaN / Inf are only reported (nonfinite_frac); the input and every check see them as 0
    let nonfinite = raw.iter().filter(|v| !v.is_finite()).count() as f64 / raw.len() as f64;
    let d = raw.mapv(|v| if v.is_finite() { v } else { 0.0 });
    let (nx, ny, nz) = d.dim();
    let x = make_input(&d, &a);
    let hf = header_features(header, &a, [nx, ny, nz]);
    let xt = TensorRef::from_array_view(([1usize, 4, N, N, N], &x[..])).context("Failed to create input tensor")?;
    let ht = TensorRef::from_array_view(([1usize, 11], &hf[..])).context("Failed to create header tensor")?;
    let outputs = session.run(ort::inputs!["x" => xt, "h" => ht]).context("ONNX inference failed")?;
    let get = |name: &str| -> Result<Vec<f32>> {
        let (_, v) = outputs[name].try_extract_tensor::<f32>().with_context(|| format!("Failed to extract {name}"))?;
        Ok(v.to_vec())
    };
    let pc = softmax(&get("contrast")?);
    let po = softmax(&get("orientation")?);
    let (stripped, epi, trunc, motion) = (get("stripped")?[0], get("epi")?[0], get("truncated")?[0], get("motion")?[0]);
    let ic = (0..6).max_by(|&i, &j| pc[i].partial_cmp(&pc[j]).unwrap().then(j.cmp(&i))).unwrap();
    let ko = (0..24).max_by(|&i, &j| po[i].partial_cmp(&po[j]).unwrap().then(j.cmp(&i))).unwrap();
    let contrast = CONTRASTS[ic];
    let mut row: Row = vec![("contrast", contrast.to_string()), ("p_contrast", py_float(py_round(pc[ic], 4)))];
    for (c, p) in CONTRASTS.iter().zip(&pc) {
        row.push((match *c {
            "T1w" => "p_T1w", "T2w" => "p_T2w", "FLAIR" => "p_FLAIR", "b0" => "p_b0", "DWI" => "p_DWI", _ => "p_other",
        }, py_float(py_round(*p, 4))));
    }
    let ps = sigmoid(stripped);
    let pe = sigmoid(epi);
    let pt = sigmoid(trunc);
    row.push(("skull_stripped", py_bool(ps > 0.5)));
    row.push(("p_skull_stripped", py_float(py_round(ps, 4))));
    row.push(("epi", py_bool(pe > 0.5)));
    row.push(("p_epi", py_float(py_round(pe, 4))));
    let code = orientation_code(ko);
    row.push(("orientation", code.clone()));
    row.push(("p_orientation", py_float(py_round(po[ko], 4))));
    row.push(("truncated", py_bool(pt > 0.5)));
    row.push(("p_truncated", py_float(py_round(pt, 4))));
    let mflag = if MOTION_CONTRASTS.contains(&contrast) {
        let pm = sigmoid(motion);
        row.push(("motion", py_float(py_round(pm, 4))));
        row.push(("motion_flag", py_bool(pm > 0.5)));
        pm > 0.5
    } else {
        row.push(("motion", String::new()));
        row.push(("motion_flag", String::new()));
        false
    };
    let full_shape: Vec<usize> = data.shape().to_vec();
    let (checks, rule_w) = rule_checks(header, &a, &full_shape, &d, nonfinite);
    let mut warnings = Vec::new();
    if code != "RAS" {
        warnings.push(format!("orientation {code}: the data do not match the header"));
    }
    if pt > 0.5 {
        warnings.push("head cut by the field of view".to_string());
    }
    if mflag {
        warnings.push("motion artifacts".to_string());
    }
    warnings.extend(rule_w);
    row.extend(checks);
    row.push(("warnings", warnings.join("; ")));
    Ok(row)
}

/// CSV field as Python's csv module (minimal quoting: , " or newline).
fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Write inspect_summary.csv with `columns` (missing values empty), lines ending in \r\n as Python's csv.
pub fn write_summary(path: &Path, rows: &[Row], columns: &[&str]) -> Result<()> {
    let mut f = std::fs::File::create(path).with_context(|| format!("Failed to write {}", path.display()))?;
    write!(f, "{}\r\n", columns.join(","))?;
    for r in rows {
        let vals: Vec<String> = columns
            .iter()
            .map(|c| r.iter().find(|(k, _)| k == c).map(|(_, v)| csv_field(v)).unwrap_or_default())
            .collect();
        write!(f, "{}\r\n", vals.join(","))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a[i, j, k] = ((37 i + 17 j + 7 k) mod 23) / 7 on a 5 x 6 x 7 grid; the reference values are scipy's / numpy's.
    fn grid() -> Array3<f64> {
        Array3::from_shape_fn((5, 6, 7), |(i, j, k)| ((i * 37 + j * 17 + k * 7) % 23) as f64 / 7.0)
    }

    const PICK: [[usize; 3]; 3] = [[0, 0, 0], [2, 3, 6], [4, 5, 2]];

    fn close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-12, "{got:?} != {want:?}");
        }
    }

    #[test]
    fn reflect_is_half_sample_symmetric() {
        let got: Vec<usize> = (-4..9).map(|i| reflect(i, 4)).collect();
        assert_eq!(got, vec![3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0]);
        assert_eq!(reflect(-3, 1), 0);
    }

    #[test]
    fn uniform3_matches_scipy_uniform_filter() {
        let u = uniform3(&grid());
        close(&PICK.map(|p| u[p]), &[1.20105820105820094e+00, 1.61904761904761885e+00, 1.31746031746031789e+00]);
    }

    #[test]
    fn gaussian_matches_scipy_gaussian_filter() {
        let g = gaussian(&grid(), [1.2, 0.7, 0.0]);
        close(&PICK.map(|p| g[p]), &[1.04049688433557930e+00, 1.53829710339211401e+00, 1.49754737377575831e+00]);
    }

    #[test]
    fn linear_matches_scipy_map_coordinates_nearest() {
        let a = grid();
        let pts = [[1.3, 2.7, 0.5], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0], [2.25, 1.5, 3.75], [-0.5, 6.2, 3.3], [1.5, 2.5, 7.0]];
        close(&pts.map(|c| linear(&a, c)), &[1.44285714285714306e+00, 0.0, 3.14285714285714279e+00,
                                             1.93303571428571441e+00, 2.29999999999999982e+00, 1.92857142857142838e+00]);
    }

    #[test]
    fn percentile_matches_numpy() {
        let v: Vec<f64> = grid().iter().step_by(3).cloned().collect();
        let got: Vec<f64> = [99.0, 50.0, 0.0, 100.0].iter().map(|&q| percentile(v.clone(), q)).collect();
        close(&got, &[3.14285714285714279e+00, 1.57142857142857140e+00, 0.0, 3.14285714285714279e+00]);
        assert_eq!(percentile(vec![], 99.0), 1.0);
    }
}
