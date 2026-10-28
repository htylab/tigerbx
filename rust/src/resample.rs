//! Resampling that reproduces TigerBx's model-space preprocessing (`tigerbx/core/resample.py`,
//! itself a port of nilearn + `scipy.ndimage.affine_transform`):
//!
//! - `best_affine`: the voxel-to-world affine as nibabel picks it (sform, else qform, else base).
//! - `resample_voxel`: resample to a new voxel size, keeping orientation (cubic B-spline).
//! - `reorder`: make the affine diagonal and positive (axis swaps / flips, or a cubic resample
//!   when the affine has rotations).
//! - `nearest_to_grid`: bring a label volume back onto another grid (nearest neighbour).
//!
//! Interpolation follows scipy's `mode='constant'`: a sample is taken only when its coordinate lies in
//! [0, n-1] on every axis (otherwise 0); the cubic spline is prefiltered with mirror boundaries.

use ndarray::{Array3, Axis};
use nifti::NiftiHeader;
use rayon::prelude::*;

pub type Mat4 = [[f64; 4]; 4];

/// How the voxel values are stored; scipy keeps the input dtype for the output, which rounds and
/// saturates unsigned integers (signed integers are promoted to float64 by TigerBx).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Unsigned(f64), // maximum value of the type
    Float32,
    Float64,
}

impl Kind {
    pub fn from_header(h: &NiftiHeader) -> Kind {
        let slope = h.scl_slope as f64;
        let inter = if h.scl_inter.is_finite() { h.scl_inter as f64 } else { 0.0 };
        let scaled = slope.is_finite() && slope != 0.0 && !(slope == 1.0 && inter == 0.0);
        if scaled {
            return Kind::Float64;
        }
        match h.datatype {
            2 => Kind::Unsigned(u8::MAX as f64),
            512 => Kind::Unsigned(u16::MAX as f64),
            768 => Kind::Unsigned(u32::MAX as f64),
            16 => Kind::Float32,
            _ => Kind::Float64, // signed integers become float64 on the first resample; floats stay
        }
    }

    fn cast(self, v: f64) -> f64 {
        match self {
            Kind::Unsigned(max) => {
                if v <= 0.0 {
                    0.0
                } else {
                    (v + 0.5).floor().min(max)
                }
            }
            Kind::Float32 => v as f32 as f64,
            Kind::Float64 => v,
        }
    }
}

/// A volume with its voxel-to-world affine.
pub struct Volume {
    pub data: Array3<f64>,
    pub affine: Mat4,
    pub kind: Kind,
}

// ---------------------------------------------------------------- affine helpers

pub(crate) fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut c = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            c[i][j] = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

pub(crate) fn mat_inv(m: &Mat4) -> Mat4 {
    // Gauss-Jordan with partial pivoting
    let mut a = *m;
    let mut inv = [[0.0; 4]; 4];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for c in 0..4 {
        let p = (c..4).max_by(|&x, &y| a[x][c].abs().partial_cmp(&a[y][c].abs()).unwrap()).unwrap();
        a.swap(c, p);
        inv.swap(c, p);
        let d = a[c][c];
        for j in 0..4 {
            a[c][j] /= d;
            inv[c][j] /= d;
        }
        for r in 0..4 {
            if r != c {
                let f = a[r][c];
                for j in 0..4 {
                    a[r][j] -= f * a[c][j];
                    inv[r][j] -= f * inv[c][j];
                }
            }
        }
    }
    inv
}

fn allclose(a: &Mat4, b: &Mat4) -> bool {
    // numpy.allclose defaults: |a - b| <= 1e-8 + 1e-5 |b|
    (0..4).all(|i| (0..4).all(|j| (a[i][j] - b[i][j]).abs() <= 1e-8 + 1e-5 * b[i][j].abs()))
}

pub(crate) fn col_norm(m: &Mat4, j: usize) -> f64 {
    (0..3).map(|i| m[i][j] * m[i][j]).sum::<f64>().sqrt()
}

/// Voxel-to-world affine as nibabel's `img.affine`: sform if sform_code != 0, else qform if
/// qform_code != 0, else the base (centred, x-flipped) affine.
pub fn best_affine(h: &NiftiHeader) -> Mat4 {
    if h.sform_code != 0 {
        let r = [h.srow_x, h.srow_y, h.srow_z];
        let mut a = [[0.0; 4]; 4];
        for i in 0..3 {
            for j in 0..4 {
                a[i][j] = r[i][j] as f64;
            }
        }
        a[3][3] = 1.0;
        return a;
    }
    let zooms = [h.pixdim[1] as f64, h.pixdim[2] as f64, h.pixdim[3] as f64];
    if h.qform_code != 0 {
        let (b, c, d) = (h.quatern_b as f64, h.quatern_c as f64, h.quatern_d as f64);
        let a = (1.0 - (b * b + c * c + d * d)).max(0.0).sqrt();
        let n = a * a + b * b + c * c + d * d;
        let r = if n < f64::EPSILON {
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        } else {
            let s = 2.0 / n;
            let (x, y, z, w) = (b, c, d, a);
            let (xs, ys, zs) = (x * s, y * s, z * s);
            let (wx, wy, wz) = (w * xs, w * ys, w * zs);
            let (xx, xy, xz) = (x * xs, x * ys, x * zs);
            let (yy, yz, zz) = (y * ys, y * zs, z * zs);
            [
                [1.0 - (yy + zz), xy - wz, xz + wy],
                [xy + wz, 1.0 - (xx + zz), yz - wx],
                [xz - wy, yz + wx, 1.0 - (xx + yy)],
            ]
        };
        let qfac = if h.pixdim[0] < 0.0 { -1.0 } else { 1.0 };
        let z = [zooms[0], zooms[1], zooms[2] * qfac];
        let off = [h.quatern_x as f64, h.quatern_y as f64, h.quatern_z as f64];
        let mut m = [[0.0; 4]; 4];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = r[i][j] * z[j];
            }
            m[i][3] = off[i];
        }
        m[3][3] = 1.0;
        return m;
    }
    let dims = [h.dim[1] as f64, h.dim[2] as f64, h.dim[3] as f64];
    let o = [(dims[0] - 1.0) / 2.0, (dims[1] - 1.0) / 2.0, (dims[2] - 1.0) / 2.0];
    [
        [-zooms[0], 0.0, 0.0, zooms[0] * o[0]],
        [0.0, zooms[1], 0.0, -zooms[1] * o[1]],
        [0.0, 0.0, zooms[2], -zooms[2] * o[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// TigerBx `core.resample._snap`: entries within 1e-9 of an integer set to it, so that a voxel-to-voxel map whose
/// grid starts exactly on an input plane gives that plane exactly (an offset of -1e-14 would put the plane outside).
fn snap(m: Mat4) -> Mat4 {
    m.map(|row| row.map(|v| if (v - v.round()).abs() < 1e-9 { v.round() } else { v }))
}

/// World-space bounds of a grid of `shape` under `affine` (min, max per axis).
fn bounds(shape: (usize, usize, usize), affine: &Mat4) -> [(f64, f64); 3] {
    let (a, b, c) = ((shape.0 - 1) as f64, (shape.1 - 1) as f64, (shape.2 - 1) as f64);
    let mut out = [(f64::INFINITY, f64::NEG_INFINITY); 3];
    for p in [[0.0, 0.0, 0.0], [a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c], [a, b, 0.0], [a, 0.0, c], [0.0, b, c], [a, b, c]] {
        for i in 0..3 {
            let v = affine[i][0] * p[0] + affine[i][1] * p[1] + affine[i][2] * p[2] + affine[i][3];
            out[i].0 = out[i].0.min(v);
            out[i].1 = out[i].1.max(v);
        }
    }
    out
}

// ---------------------------------------------------------------- cubic B-spline (scipy order 3)

const POLE: f64 = -0.267_949_192_431_122_7; // sqrt(3) - 2

/// In-place cubic B-spline prefilter of one line, mirror boundaries (scipy `spline_filter1d`).
fn prefilter_line(c: &mut [f64]) {
    let n = c.len();
    if n < 2 {
        return;
    }
    let z = POLE;
    for v in c.iter_mut() {
        *v *= 6.0; // (1 - z)(1 - 1/z)
    }
    // causal initialisation (mirror)
    let z_n_1 = z.powi(n as i32 - 1);
    let mut z_i = z;
    let mut c0 = c[0] + z_n_1 * c[n - 1];
    for i in 1..n - 1 {
        c0 += z_i * (c[i] + z_n_1 * c[n - 1 - i]);
        z_i *= z;
    }
    c[0] = c0 / (1.0 - z_n_1 * z_n_1);
    for i in 1..n {
        c[i] += z * c[i - 1];
    }
    // anti-causal initialisation (mirror)
    c[n - 1] = (z / (z * z - 1.0)) * (z * c[n - 2] + c[n - 1]);
    for i in (0..n - 1).rev() {
        c[i] = z * (c[i + 1] - c[i]);
    }
}

/// Spline coefficients of a volume: the prefilter applied along each axis.
pub(crate) fn spline_coefficients(data: &Array3<f64>) -> Array3<f64> {
    let mut c = data.clone();
    for ax in 0..3 {
        let mut lanes: Vec<_> = c.lanes_mut(Axis(ax)).into_iter().collect();
        lanes.par_iter_mut().for_each(|lane| {
            let mut buf: Vec<f64> = lane.iter().cloned().collect();
            prefilter_line(&mut buf);
            for (d, s) in lane.iter_mut().zip(buf) {
                *d = s;
            }
        });
    }
    c
}

#[inline]
fn mirror(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let p = 2 * n as isize - 2;
    let mut i = i.abs() % p;
    if i >= n as isize {
        i = p - i;
    }
    i as usize
}

#[inline]
fn cubic_weights(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0,
        (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0,
        (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0,
        t3 / 6.0,
    ]
}

/// Input coordinate of output voxel (i, j, k) under `m` (= inverse(src) · target).
#[inline]
pub(crate) fn coord(m: &Mat4, i: usize, j: usize, k: usize) -> [f64; 3] {
    let (x, y, z) = (i as f64, j as f64, k as f64);
    [
        m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
        m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
        m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3],
    ]
}

#[inline]
pub(crate) fn inside(c: &[f64; 3], dims: [usize; 3]) -> bool {
    (0..3).all(|a| c[a] >= 0.0 && c[a] <= (dims[a] - 1) as f64)
}

/// Cubic B-spline value at input voxel coordinate `c` (inside the grid) from the coefficients `coef`.
#[inline]
pub(crate) fn cubic_at(coef: &Array3<f64>, dims: [usize; 3], c: &[f64; 3]) -> f64 {
    let mut idx = [[0usize; 4]; 3];
    let mut w = [[0.0; 4]; 3];
    for a in 0..3 {
        let f = c[a].floor();
        w[a] = cubic_weights(c[a] - f);
        for t in 0..4 {
            idx[a][t] = mirror(f as isize - 1 + t as isize, dims[a]);
        }
    }
    let mut s = 0.0;
    for p in 0..4 {
        for q in 0..4 {
            let wpq = w[0][p] * w[1][q];
            for r in 0..4 {
                s += wpq * w[2][r] * coef[[idx[0][p], idx[1][q], idx[2][r]]];
            }
        }
    }
    s
}

/// Trilinear value at input voxel coordinate `c` (inside the grid).
#[inline]
fn linear_at(data: &Array3<f64>, dims: [usize; 3], c: &[f64; 3]) -> f64 {
    let mut i0 = [0usize; 3];
    let mut t = [0.0; 3];
    for a in 0..3 {
        let f = c[a].floor();
        i0[a] = f as usize;
        t[a] = c[a] - f;
    }
    let mut s = 0.0;
    for (dx, wx) in [(0usize, 1.0 - t[0]), (1, t[0])] {
        let x = (i0[0] + dx).min(dims[0] - 1);
        for (dy, wy) in [(0usize, 1.0 - t[1]), (1, t[1])] {
            let y = (i0[1] + dy).min(dims[1] - 1);
            for (dz, wz) in [(0usize, 1.0 - t[2]), (1, t[2])] {
                let z = (i0[2] + dz).min(dims[2] - 1);
                s += wx * wy * wz * data[[x, y, z]];
            }
        }
    }
    s
}

/// `scipy.ndimage.map_coordinates(data, at, order=1 or 3, mode='constant')` on a grid of `shape`: output voxel
/// (i, j, k) takes `data` at input voxel coordinate `at(i, j, k)`, 0 outside the input grid.
pub fn sample(data: &Array3<f64>, shape: (usize, usize, usize), cubic: bool, at: impl Fn(usize, usize, usize) -> [f64; 3] + Sync) -> Array3<f64> {
    let coef = if cubic { Some(spline_coefficients(data)) } else { None };
    let d = data.dim();
    let dims = [d.0, d.1, d.2];
    let mut out = Array3::<f64>::zeros(shape);
    out.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut plane)| {
        for ((j, k), o) in plane.indexed_iter_mut() {
            let c = at(i, j, k);
            if inside(&c, dims) {
                *o = match coef.as_ref() {
                    Some(coef) => cubic_at(coef, dims, &c),
                    None => linear_at(data, dims, &c),
                };
            }
        }
    });
    out
}

/// `sample` with nearest-neighbour interpolation (scipy order 0, mode 'constant': 0 outside the grid).
pub fn sample_nearest(data: &Array3<f64>, shape: (usize, usize, usize), at: impl Fn(usize, usize, usize) -> [f64; 3] + Sync) -> Array3<f64> {
    let d = data.dim();
    let dims = [d.0, d.1, d.2];
    let mut out = Array3::<f64>::zeros(shape);
    out.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut plane)| {
        for ((j, k), o) in plane.indexed_iter_mut() {
            let c = at(i, j, k);
            if inside(&c, dims) {
                *o = data[[(c[0] + 0.5).floor() as usize, (c[1] + 0.5).floor() as usize, (c[2] + 0.5).floor() as usize]];
            }
        }
    });
    out
}

/// `scipy.ndimage.affine_transform(data, m, order=3, mode='constant')` onto `shape`.
fn cubic_transform(data: &Array3<f64>, m: &Mat4, shape: (usize, usize, usize)) -> Array3<f64> {
    sample(data, shape, true, |i, j, k| coord(m, i, j, k))
}

/// `resample_img(..., interpolation='continuous')` with a full target affine and shape, including the
/// dtype cast and the clip to [min(data, 0), max(data, 0)].
fn resample_cubic(vol: &Volume, target: Mat4, shape: (usize, usize, usize)) -> Volume {
    let m = if allclose(&target, &vol.affine) {
        let mut e = [[0.0; 4]; 4];
        for (i, row) in e.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        e
    } else {
        snap(mat_mul(&mat_inv(&vol.affine), &target))
    };
    let mut out = cubic_transform(&vol.data, &m, shape);
    let lo = vol.data.iter().cloned().fold(0.0f64, f64::min);
    let hi = vol.data.iter().cloned().fold(0.0f64, f64::max);
    let kind = vol.kind;
    out.par_mapv_inplace(|v| kind.cast(v).clamp(lo, hi));
    Volume { data: out, affine: target, kind }
}

/// `resample_voxel(img, voxelsize)`: same orientation and origin, new voxel size, grid covering the
/// input.
pub fn resample_voxel(vol: &Volume, voxel: [f64; 3]) -> Volume {
    let mut target = vol.affine;
    for j in 0..3 {
        let s = voxel[j] / col_norm(&vol.affine, j);
        for row in target.iter_mut().take(3) {
            row[j] *= s;
        }
    }
    let t = snap(mat_mul(&mat_inv(&target), &vol.affine));
    let b = bounds(vol.data.dim(), &t);
    let shape = (b[0].1.ceil() as usize + 1, b[1].1.ceil() as usize + 1, b[2].1.ceil() as usize + 1);
    resample_cubic(vol, target, shape)
}

/// `reorder_img(img, resample='continuous')`: a diagonal, positive affine.
pub fn reorder(vol: Volume) -> Volume {
    let a = vol.affine;
    let rotated = (0..3).any(|j| (0..3).filter(|&i| a[i][j].abs() > 0.001).count() != 1);
    if rotated {
        // Target voxel sizes from the QR decomposition of the 3x3 part (Gram-Schmidt: |R_jj| and
        // |Q| do not depend on the sign convention).
        let cols: Vec<[f64; 3]> = (0..3).map(|j| [a[0][j], a[1][j], a[2][j]]).collect();
        let mut q = [[0.0f64; 3]; 3]; // q[j] = j-th column of Q
        let mut rdiag = [0.0f64; 3];
        for j in 0..3 {
            let mut v = cols[j];
            for qk in q.iter().take(j) {
                let dot: f64 = (0..3).map(|i| qk[i] * cols[j][i]).sum();
                for i in 0..3 {
                    v[i] -= dot * qk[i];
                }
            }
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            rdiag[j] = n;
            q[j] = [v[0] / n, v[1] / n, v[2] / n];
        }
        // target = diag(|diag(R)|[argmax_j |Q[i, j]|]) for each row i
        let mut diag = [0.0; 3];
        for (i, d) in diag.iter_mut().enumerate() {
            let jmax = (0..3).max_by(|&x, &y| q[x][i].abs().partial_cmp(&q[y][i].abs()).unwrap()).unwrap();
            *d = rdiag[jmax];
        }
        // resample_img with a 3x3 target affine: offset from the bounds, shape from their extent
        let mut target = [[0.0; 4]; 4];
        for i in 0..3 {
            target[i][i] = diag[i];
        }
        target[3][3] = 1.0;
        let t = snap(mat_mul(&mat_inv(&target), &vol.affine));
        let b = bounds(vol.data.dim(), &t);
        for i in 0..3 {
            target[i][3] = diag[i] * b[i].0;
        }
        let shape = (
            (b[0].1 - b[0].0).ceil() as usize + 1,
            (b[1].1 - b[1].0).ceil() as usize + 1,
            (b[2].1 - b[2].0).ceil() as usize + 1,
        );
        return resample_cubic(&vol, target, shape);
    }

    // No rotation: permute axes so that voxel axis j maps to world axis j, then flip negative axes.
    let world_of: Vec<usize> = (0..3)
        .map(|j| (0..3).max_by(|&x, &y| a[x][j].abs().partial_cmp(&a[y][j].abs()).unwrap()).unwrap())
        .collect();
    let mut perm = [0usize; 3]; // perm[w] = voxel axis that maps to world axis w
    for (j, &w) in world_of.iter().enumerate() {
        perm[w] = j;
    }
    let mut data = vol.data.permuted_axes(perm);
    let mut b = [a[0][3], a[1][3], a[2][3]];
    let mut pix = [0.0; 3];
    for w in 0..3 {
        pix[w] = a[w][perm[w]];
        if pix[w] < 0.0 {
            b[w] += pix[w] * (data.len_of(Axis(w)) - 1) as f64;
            pix[w] = -pix[w];
            data.invert_axis(Axis(w));
        }
    }
    let mut affine = [[0.0; 4]; 4];
    for w in 0..3 {
        affine[w][w] = pix[w];
        affine[w][3] = b[w];
    }
    affine[3][3] = 1.0;
    Volume { data: data.as_standard_layout().to_owned(), affine, kind: vol.kind }
}

/// `resample_to_img(labels, target, interpolation='nearest')`: labels on the grid of `target_shape` /
/// `target_affine`.
pub fn nearest_to_grid(labels: &Array3<u8>, src_affine: &Mat4, target_affine: &Mat4, target_shape: (usize, usize, usize)) -> Array3<u8> {
    let m = if allclose(target_affine, src_affine) {
        let mut e = [[0.0; 4]; 4];
        for (i, row) in e.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        e
    } else {
        snap(mat_mul(&mat_inv(src_affine), target_affine))
    };
    let d = labels.dim();
    let dims = [d.0, d.1, d.2];
    let mut out = Array3::<u8>::zeros(target_shape);
    out.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut plane)| {
        for ((j, k), o) in plane.indexed_iter_mut() {
            let c = coord(&m, i, j, k);
            if inside(&c, dims) {
                *o = labels[[(c[0] + 0.5).floor() as usize, (c[1] + 0.5).floor() as usize, (c[2] + 0.5).floor() as usize]];
            }
        }
    });
    out
}

/// TigerBx `read_tbx_input`: resample to 1 mm when any voxel size is outside [0.9, 1.1] mm, then reorder to a
/// diagonal affine; the values are stored as `kind` (TigerBx keeps the tBX brain = input x mask in float64, so its
/// resampling neither rounds nor saturates).
pub fn to_model_space_as(data: Array3<f64>, header: &NiftiHeader, kind: Kind) -> Volume {
    model_space(Volume { data, affine: best_affine(header), kind }, header_zooms(header))
}

/// Voxel sizes (mm) of the first three axes of a header.
pub fn header_zooms(header: &NiftiHeader) -> [f64; 3] {
    [header.pixdim[1] as f64, header.pixdim[2] as f64, header.pixdim[3] as f64]
}

/// `to_model_space` of a volume (e.g. a crop of the input) whose voxel sizes are `zooms`.
pub fn model_space(vol: Volume, zooms: [f64; 3]) -> Volume {
    let zmax = zooms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let zmin = zooms.iter().cloned().fold(f64::INFINITY, f64::min);
    let vol = if zmax > 1.1 || zmin < 0.9 { resample_voxel(&vol, [1.0, 1.0, 1.0]) } else { vol };
    reorder(vol)
}

/// TigerBx `crop_cube(x, keep > 0, padding)` + `_crop_nib`: the bounding box of the non-zero voxels of `keep` (same
/// grid as `vol`), `padding` voxels wider (clipped), with the affine origin moved to the first voxel of the box.
/// None if `keep` is all zero.
pub fn crop_to_mask(vol: &Volume, keep: &Array3<u8>, padding: usize) -> Option<Volume> {
    bbox(keep, [padding; 3], |&v| v > 0).map(|r| crop_box(vol, r))
}

/// `crop_to_mask` for the voxels of `vol` where `keep` holds, with a padding per axis, the box grown (centred) to at
/// least `min_size` voxels per axis and slid back inside the volume before clipping.
pub fn crop_where_min(vol: &Volume, padding: [usize; 3], min_size: [usize; 3], keep: impl Fn(f64) -> bool) -> Option<Volume> {
    let dims = vol.data.dim();
    let dims = [dims.0, dims.1, dims.2];
    let r = bbox(&vol.data, [0; 3], |&v| keep(v))?;
    Some(crop_box(
        vol,
        std::array::from_fn(|a| {
            // as Python: lo = first - pad, hi = last + 1 + pad, grown by max(min - (hi - lo), 0) split lo / hi
            let (lo, hi) = (r[a].0 as i64 - padding[a] as i64, r[a].1 as i64 + 1 + padding[a] as i64);
            let grow = (min_size[a] as i64 - (hi - lo)).max(0);
            let (lo, hi) = (lo - grow / 2, hi + grow - grow / 2);
            // slide a grown window back inside the volume
            let shift = if grow > 0 { (-lo).max(0) - (hi - dims[a] as i64).max(0) } else { 0 };
            ((lo + shift).max(0) as usize, ((hi + shift).min(dims[a] as i64) - 1) as usize)
        }),
    ))
}

/// Inclusive bounding box of the voxels where `keep` holds, `padding` voxels wider per axis (clipped).
fn bbox<T>(data: &Array3<T>, padding: [usize; 3], keep: impl Fn(&T) -> bool) -> Option<[(usize, usize); 3]> {
    let (h, w, d) = data.dim();
    let dims = [h, w, d];
    let mut lo = [usize::MAX; 3];
    let mut hi = [0usize; 3];
    for ((i, j, k), v) in data.indexed_iter() {
        if keep(v) {
            for (a, c) in [i, j, k].into_iter().enumerate() {
                lo[a] = lo[a].min(c);
                hi[a] = hi[a].max(c);
            }
        }
    }
    if lo[0] == usize::MAX {
        return None;
    }
    Some(std::array::from_fn(|a| (lo[a].saturating_sub(padding[a]), (hi[a] + padding[a]).min(dims[a] - 1))))
}

/// The box `r` of `vol`, with the affine origin moved to its first voxel.
fn crop_box(vol: &Volume, r: [(usize, usize); 3]) -> Volume {
    let data = vol.data.slice(ndarray::s![r[0].0..=r[0].1, r[1].0..=r[1].1, r[2].0..=r[2].1]).to_owned();
    let mut affine = vol.affine;
    for i in 0..3 {
        affine[i][3] = (0..3).map(|j| vol.affine[i][j] * r[j].0 as f64).sum::<f64>() + vol.affine[i][3];
    }
    Volume { data, affine, kind: vol.kind }
}

/// `scipy.ndimage.affine_transform(data, m, order=1, mode='constant')` onto `shape`: trilinear, 0 where the
/// input coordinate is outside [0, n-1] on any axis.
pub fn linear_transform(data: &Array3<f64>, m: &Mat4, shape: (usize, usize, usize)) -> Array3<f64> {
    sample(data, shape, false, |i, j, k| coord(m, i, j, k))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a[i, j, k] = ((37 i + 17 j + 7 k) mod 23) / 7 on a 5 x 6 x 7 grid; the reference values are scipy's.
    fn grid() -> Array3<f64> {
        Array3::from_shape_fn((5, 6, 7), |(i, j, k)| ((i * 37 + j * 17 + k * 7) % 23) as f64 / 7.0)
    }

    const PTS: [[f64; 3]; 4] = [[1.3, 2.7, 0.5], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0], [2.25, 1.5, 3.75]];

    fn at_points(cubic: bool) -> Vec<f64> {
        let a = grid();
        sample(&a, (PTS.len(), 1, 1), cubic, |i, _, _| PTS[i]).iter().cloned().collect()
    }

    fn close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-12, "{got:?} != {want:?}");
        }
    }

    #[test]
    fn cubic_matches_scipy_map_coordinates_order3() {
        close(&at_points(true), &[1.60593540548768843e+00, 7.64879361376485103e-16, 3.14285714285714457e+00,
                                  2.46865198980747991e+00]);
    }

    #[test]
    fn linear_matches_scipy_map_coordinates_order1() {
        close(&at_points(false), &[1.44285714285714306e+00, 0.0, 3.14285714285714279e+00, 1.93303571428571441e+00]);
    }

    #[test]
    fn outside_the_grid_is_zero() {
        let a = grid();
        let out = sample(&a, (2, 1, 1), true, |i, _, _| [[-0.01, 1.0, 1.0], [1.0, 5.01, 1.0]][i]);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn snap_sets_near_integers_only() {
        let mut m = [[1.0, 0.0, 0.0, -1.42e-14], [0.0, 1.0, 0.0, 2.0 + 5e-10], [0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 0.0, 1.0]];
        m = snap(m);
        assert_eq!(m[0][3], 0.0);
        assert_eq!(m[1][3], 2.0);
        assert_eq!(m[2][3], 0.5);
        assert_eq!(snap([[1.0 + 2e-9, 0.0, 0.0, 0.0], [0.0; 4], [0.0; 4], [0.0; 4]])[0][0], 1.0 + 2e-9);
    }

    #[test]
    fn mat_inv_inverts() {
        let m = [[0.9, 0.1, 0.0, 3.0], [-0.1, 1.1, 0.2, -2.0], [0.0, 0.3, 1.2, 5.0], [0.0, 0.0, 0.0, 1.0]];
        let p = mat_mul(&m, &mat_inv(&m));
        for i in 0..4 {
            for j in 0..4 {
                assert!((p[i][j] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
    }
}
