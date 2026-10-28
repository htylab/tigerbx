//! Registration to the TigerBx template, as `tigerbx.reg('AD')` / the VBM `_norm`:
//! numreg affine (`_affine`, tBX brain on the template grid), then VFA (`_norm`).

use anyhow::{Context, Result};
use ndarray::{Array3, Ix3};
use nifti::{IntoNdArray, NiftiHeader, NiftiObject};
use ort::session::Session;
use ort::session::run_options::OutputSelector;
use ort::value::TensorRef;

use crate::numreg;
use crate::resample::{self, mat_inv, mat_mul, Kind, Mat4, Volume};

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

/// Load a gzipped uint8 label image on the template grid (the template tseg), reordered as the template.
pub fn load_labels(gz_bytes: &[u8]) -> Result<Array3<u8>> {
    let obj = nifti::InMemNiftiObject::from_reader(flate2::read::GzDecoder::new(gz_bytes))
        .context("Failed to read the embedded template tseg")?;
    let affine = resample::best_affine(obj.header());
    let data = obj.into_volume().into_ndarray::<f64>().context("Failed to convert the template tseg")?.into_dimensionality::<Ix3>()?;
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

/// TigerBx `lib_reg.min_max_norm` of a float32 image.
pub fn min_max_f32(x: &Array3<f32>) -> Array3<f32> {
    min_max_norm(&x.mapv(|v| v as f64)).mapv(|v| v as f32)
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
    /// numreg affine: tBX brain sampled onto the template (cubic, clipped, from the input grid; the input's units)
    pub affine: Array3<f32>,
    /// VFA: tBX brain sampled at the affine + VFA warp (cubic, clipped, from the input grid; when VFA ran)
    pub norm: Option<Array3<f32>>,
    /// world matrix template -> input (fixed -> moving)
    pub matrix: Mat4,
    /// VFA voxel sampling grid [3, X, Y, Z] on the template grid (when VFA ran)
    pub grid: Option<Vec<f32>>,
    /// the tBX mask carried through the same affine + VFA warp (> 0.5), when a mask was given and VFA ran
    pub norm_mask: Option<Array3<bool>>,
    /// the tBX mask through the affine only (> 0.5), when a mask was given: the brain mask of the gnorm _affine
    pub affine_mask: Option<Array3<bool>>,
}

/// `brain` = tBX brain (input x mask) on the input grid, `header` = its header; `mask` = the tBX mask, carried
/// along when given (the brain masks of _affine / _norm: resampling rings outside the brain).
pub fn run_reg(
    brain: &Array3<f64>,
    mask: Option<&Array3<u8>>,
    header: &NiftiHeader,
    template: &Template,
    vfa: Option<&mut Session>,
) -> Result<RegResult> {
    // TigerBx reg: tbx = (input x mask).astype(float32); numreg on its reorder_img(continuous)
    let native = Volume { data: brain.mapv(|v| v as f32 as f64), affine: resample::best_affine(header), kind: Kind::Float32 };
    let moving = resample::reorder(Volume { data: native.data.clone(), affine: native.affine, kind: native.kind });
    let t = &template.vol;
    let dim = t.data.dim();
    let m = numreg::register(&moving.data, &moving.affine, &t.data, &t.affine);

    // numreg.resample: every output sampled once from the input grid at template voxel -> input voxel
    let v = mat_mul(&mat_mul(&mat_inv(&native.affine), &m), &t.affine);
    let affine_at = |i: usize, j: usize, k: usize| resample::coord(&v, i, j, k);
    // _affine cubic (clipped; the input's units), the VFA input trilinear (as VFA expects) and min-max
    let affined = clip_f32(resample::sample(&native.data, dim, true, affine_at), &native.data);
    let vfa_input = min_max_norm(&resample::sample(&native.data, dim, false, affine_at).mapv(|x| x as f32 as f64));

    let grid = match vfa {
        None => None,
        Some(session) => Some(run_vfa(session, &vfa_input, &min_max_norm(&t.data))?),
    };
    // VFA: affine + warp composed (coordinates = identity + (grid - identity), float32 as TigerBx), one sample
    let n = dim.0 * dim.1 * dim.2;
    let warped_at = |g: &[f32]| {
        let g = g.to_vec();
        move |i: usize, j: usize, k: usize| {
            let p = (i * dim.1 + j) * dim.2 + k;
            let id = [i as f32, j as f32, k as f32];
            let q: [f64; 3] = std::array::from_fn(|a| id[a] as f64 + (g[a * n + p] - id[a]) as f64);
            let mut c = [0.0; 3];
            for (r, cr) in c.iter_mut().enumerate() {
                *cr = v[r][0] * q[0] + v[r][1] * q[1] + v[r][2] * q[2] + v[r][3];
            }
            c
        }
    };
    let norm = grid.as_ref().map(|g| {
        clip_f32(resample::sample(&native.data, dim, true, warped_at(g)), &native.data).mapv(|x| x as f32)
    });
    // the tBX mask through the affine (and the VFA warp when it ran), trilinear from the input grid, > 0.5
    let mask_data = mask.map(|mask| mask.mapv(|v| if v > 0 { 1.0 } else { 0.0 }));
    let affine_mask = mask_data.as_ref().map(|md| resample::sample(md, dim, false, affine_at).mapv(|x| x as f32 > 0.5));
    let norm_mask = match (mask_data.as_ref(), grid.as_ref()) {
        (Some(md), Some(g)) => Some(resample::sample(md, dim, false, warped_at(g)).mapv(|x| x as f32 > 0.5)),
        _ => None,
    };
    Ok(RegResult { affine: affined.mapv(|x| x as f32), norm, matrix: m, grid, norm_mask, affine_mask })
}

/// `numreg.resample` order 3: float32, clipped to [min(data, 0), max(data, 0)].
fn clip_f32(x: Array3<f64>, data: &Array3<f64>) -> Array3<f64> {
    let lo = data.iter().cloned().fold(0.0f64, f64::min) as f32;
    let hi = data.iter().cloned().fold(0.0f64, f64::max) as f32;
    x.mapv(|v| (v as f32).clamp(lo, hi) as f64)
}

/// VFA step 'D': the model maps (moving, fixed) to a voxel sampling grid [3, X, Y, Z].
pub fn run_vfa(session: &mut Session, moving: &Array3<f64>, fixed: &Array3<f64>) -> Result<Vec<f32>> {
    let (x, y, z) = moving.dim();
    let mv: Vec<f32> = moving.iter().map(|&v| v as f32).collect();
    let fx: Vec<f32> = fixed.iter().map(|&v| v as f32).collect();
    let shape = [1usize, 1, x, y, z];
    let inputs = ort::inputs![
        "moving" => TensorRef::from_array_view((shape, mv.as_slice()))?,
        "fixed" => TensorRef::from_array_view((shape, fx.as_slice()))?,
    ];
    let opts = crate::inference::run_options()?.with_outputs(OutputSelector::no_default().with("grid"));
    let outputs = session.run_with_options(inputs, &opts).context("VFA inference failed")?;
    let (_, grid) = outputs["grid"].try_extract_tensor::<f32>().context("Failed to extract the VFA grid")?;
    Ok(grid.to_vec())
}
