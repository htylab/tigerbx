//! Registration to the TigerBx template, as `tigerbx.reg('AD')` / the VBM `_norm`:
//! numreg affine (`_affine`, min-max normalised tBET brain on the template grid), then VFA (`_norm`).

use anyhow::{Context, Result};
use ndarray::{Array3, Axis, Ix3};
use nifti::{IntoNdArray, NiftiHeader, NiftiObject};
use ort::session::Session;
use ort::session::run_options::OutputSelector;
use ort::value::TensorRef;
use rayon::prelude::*;

use crate::numreg;
use crate::resample::{self, linear_transform, mat_inv, mat_mul, Kind, Mat4, Volume};

/// The template on its reordered (diagonal) grid, with the header used for template-space outputs.
pub struct Template {
    pub vol: Volume,
    pub header: NiftiHeader,
}

/// Load a gzipped NIfTI template from memory (TigerBx `lib_reg.get_template` + `reorder_img`).
pub fn load_template(gz_bytes: &[u8]) -> Result<Template> {
    let obj = nifti::InMemNiftiObject::from_reader(flate2::read::GzDecoder::new(gz_bytes))
        .context("Failed to read the embedded template")?;
    let header = obj.header().clone();
    let data = obj
        .into_volume()
        .into_ndarray::<f64>()
        .context("Failed to convert the template")?
        .into_dimensionality::<Ix3>()?;
    // float32 on disk: reorder keeps it float32
    let vol = resample::reorder(Volume { data: data.mapv(|v| v as f32 as f64), affine: resample::best_affine(&header), kind: Kind::Float32 });
    let mut header = header;
    set_affine(&mut header, &vol.affine);
    Ok(Template { vol, header })
}

/// Load a gzipped uint8 label image on the template grid (the template aseg), reordered as the template.
pub fn load_labels(gz_bytes: &[u8]) -> Result<Array3<u8>> {
    let obj = nifti::InMemNiftiObject::from_reader(flate2::read::GzDecoder::new(gz_bytes))
        .context("Failed to read the embedded template aseg")?;
    let affine = resample::best_affine(obj.header());
    let data = obj.into_volume().into_ndarray::<f64>().context("Failed to convert the template aseg")?.into_dimensionality::<Ix3>()?;
    let vol = resample::reorder(Volume { data, affine, kind: Kind::Unsigned(255.0) });
    Ok(vol.data.mapv(|v| v as u8))
}

/// Header for an image on `affine` (diagonal, positive): sform and qform both set to it.
pub fn set_affine(h: &mut NiftiHeader, a: &Mat4) {
    h.srow_x = [a[0][0] as f32, a[0][1] as f32, a[0][2] as f32, a[0][3] as f32];
    h.srow_y = [a[1][0] as f32, a[1][1] as f32, a[1][2] as f32, a[1][3] as f32];
    h.srow_z = [a[2][0] as f32, a[2][1] as f32, a[2][2] as f32, a[2][3] as f32];
    h.sform_code = 2;
    h.qform_code = 2;
    h.quatern_b = 0.0;
    h.quatern_c = 0.0;
    h.quatern_d = 0.0;
    h.quatern_x = a[0][3] as f32;
    h.quatern_y = a[1][3] as f32;
    h.quatern_z = a[2][3] as f32;
    h.pixdim[0] = 1.0;
    for i in 0..3 {
        h.pixdim[i + 1] = a[i][i] as f32;
    }
    h.scl_slope = 1.0;
    h.scl_inter = 0.0;
}

/// TigerBx `lib_reg.min_max_norm` in float32.
fn min_max_norm(x: &Array3<f64>) -> Array3<f64> {
    let (lo, hi) = x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    if hi == lo {
        return Array3::zeros(x.dim());
    }
    let (lo, d) = (lo as f32, (hi - lo) as f32);
    x.mapv(|v| ((v as f32 - lo) / d) as f64)
}

/// Results on the template grid.
pub struct RegResult {
    /// numreg affine: tBET brain resampled onto the template, min-max normalised
    pub affine: Array3<f32>,
    /// VFA: the affine image warped onto the template (when a VFA session is given)
    pub norm: Option<Array3<f32>>,
    /// world matrix template -> input (fixed -> moving)
    pub matrix: Mat4,
    /// VFA voxel sampling grid [3, X, Y, Z] on the template grid (when VFA ran)
    pub grid: Option<Vec<f32>>,
    /// the tBET mask carried through the same affine + VFA warp (> 0.5), when a mask was given and VFA ran
    pub norm_mask: Option<Array3<bool>>,
}

/// `brain` = tBET brain (input x mask) on the input grid, `header` = its header; `mask` = the tBET mask, warped
/// along when given (the brain mask of _norm: oblique inputs leave cubic-resampling ringing outside the brain).
pub fn run_reg(
    brain: &Array3<f64>,
    mask: Option<&Array3<u8>>,
    header: &NiftiHeader,
    template: &Template,
    vfa: Option<&mut Session>,
) -> Result<RegResult> {
    // TigerBx reg: bet = (input x mask).astype(float32), reorder_img(continuous)
    let moving = resample::reorder(Volume {
        data: brain.mapv(|v| v as f32 as f64),
        affine: resample::best_affine(header),
        kind: Kind::Float32,
    });
    let t = &template.vol;
    let m = numreg::register(&moving.data, &moving.affine, &t.data, &t.affine);

    // numreg.apply: template voxel -> moving voxel, trilinear, 0 outside; then min-max
    let v = mat_mul(&mat_mul(&mat_inv(&moving.affine), &m), &t.affine);
    let affined = min_max_norm(&linear_transform(&moving.data, &v, t.data.dim()).mapv(|x| x as f32 as f64));

    let (norm, grid) = match vfa {
        None => (None, None),
        Some(session) => {
            let grid = run_vfa(session, &affined, &min_max_norm(&t.data))?;
            let mv: Vec<f32> = affined.iter().map(|&v| v as f32).collect();
            (Some(warp(&mv, t.data.dim(), &grid)), Some(grid))
        }
    };
    let norm_mask = match (mask, grid.as_ref()) {
        (Some(mask), Some(grid)) => {
            let mv = resample::reorder(Volume {
                data: mask.mapv(|v| if v > 0 { 1.0 } else { 0.0 }),
                affine: resample::best_affine(header),
                kind: Kind::Float32,
            });
            let am: Vec<f32> = linear_transform(&mv.data, &v, t.data.dim()).iter().map(|&x| x as f32).collect();
            Some(warp(&am, t.data.dim(), grid).mapv(|x| x > 0.5))
        }
        _ => None,
    };
    Ok(RegResult { affine: affined.mapv(|x| x as f32), norm, matrix: m, grid, norm_mask })
}

/// VFA step 'D': the model maps (moving, fixed) to a voxel sampling grid [3, X, Y, Z].
fn run_vfa(session: &mut Session, moving: &Array3<f64>, fixed: &Array3<f64>) -> Result<Vec<f32>> {
    let (x, y, z) = moving.dim();
    let mv: Vec<f32> = moving.iter().map(|&v| v as f32).collect();
    let fx: Vec<f32> = fixed.iter().map(|&v| v as f32).collect();
    let shape = [1usize, 1, x, y, z];
    let inputs = ort::inputs![
        "moving" => TensorRef::from_array_view((shape, mv.as_slice()))?,
        "fixed" => TensorRef::from_array_view((shape, fx.as_slice()))?,
    ];
    let mut opts = ort::session::RunOptions::new()?.with_outputs(OutputSelector::no_default().with("grid"));
    opts.add_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")?;
    let outputs = session.run_with_options(inputs, &opts).context("VFA inference failed")?;
    let (_, grid) = outputs["grid"].try_extract_tensor::<f32>().context("Failed to extract the VFA grid")?;
    Ok(grid.to_vec())
}

/// Sample `vol` (C order, `dim`) at the VFA grid trilinearly with zeros outside (scipy map_coordinates order 1,
/// mode 'grid-constant').
fn warp(vol: &[f32], dim: (usize, usize, usize), grid: &[f32]) -> Array3<f32> {
    let (x, y, z) = dim;
    let n = x * y * z;
    let dims = [x, y, z];
    let at = |i: isize, j: isize, k: isize| -> f32 {
        if i < 0 || j < 0 || k < 0 || i >= x as isize || j >= y as isize || k >= z as isize {
            0.0
        } else {
            vol[(i as usize * y + j as usize) * z + k as usize]
        }
    };
    let mut out = Array3::<f32>::zeros((x, y, z));
    out.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut plane)| {
        for ((j, k), o) in plane.indexed_iter_mut() {
            let p = (i * y + j) * z + k;
            // coords = identity + (grid - identity), in float32 as TigerBx
            let id = [i as f32, j as f32, k as f32];
            let c: [f64; 3] = std::array::from_fn(|a| (id[a] + (grid[a * n + p] - id[a])) as f64);
            if (0..3).any(|a| c[a] <= -1.0 || c[a] >= dims[a] as f64) {
                continue;
            }
            let f = c.map(|v| v.floor());
            let t: [f64; 3] = std::array::from_fn(|a| c[a] - f[a]);
            let b = f.map(|v| v as isize);
            let mut s = 0.0f64;
            for (dx, wx) in [(0, 1.0 - t[0]), (1, t[0])] {
                for (dy, wy) in [(0, 1.0 - t[1]), (1, t[1])] {
                    for (dz, wz) in [(0, 1.0 - t[2]), (1, t[2])] {
                        s += wx * wy * wz * at(b[0] + dx, b[1] + dy, b[2] + dz) as f64;
                    }
                }
            }
            *o = s as f32;
        }
    });
    out
}
