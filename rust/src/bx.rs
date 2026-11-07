use anyhow::{bail, Context, Result};
use ndarray::{s, Array3, ArrayD, Ix3};
use nifti::NiftiHeader;
use ort::session::Session;

use crate::inference;
use crate::postprocess;
use crate::resample;

/// FreeSurfer label number of TSEG output channel k, TigerBx `label_all['tseg_v1']` (the last channel, dura / venous
/// sinuses / vessels, is written as 0).
const TSEG_LABELS: [u8; 34] = [
    0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53,
    54, 58, 60, 0,
];
/// TSEG / DGM input crop: brain > 0 + 16 voxels. DGM: when that box exceeds 256^3 voxels, the tBX mask + 32 voxels
/// instead (TigerBx `_prepare_tbx111_crop`).
const TSEG_CROP_PADDING: usize = 16;
const DGM_CROP_MAX_VOXELS: usize = 256 * 256 * 256;
const DGM_MASK_CROP_PADDING: usize = 32;
/// Padding (mm) of the tBX input crop around the non-zero voxels and its minimum extent (mm) per axis (TigerBx
/// `_TBX_CROP_MM`, `_TBX_CROP_MIN_MM`).
const TBX_CROP_MM: f64 = 16.0;
const TBX_CROP_MIN_MM: f64 = 256.0;

/// Results of one image, all on the input grid.
pub struct BxResult {
    /// tBX brain mask, u8 0 / 1 (dual mask model: its inner mask, the brain following its surface)
    pub mask: ArrayD<u8>,
    /// dual mask model (TigerBx mode `tbxdual`): the outer mask (the brain with the surrounding CSF, up to the inner
    /// surface of the skull; the TSEG input)
    pub outer: Option<ArrayD<u8>>,
    /// tBX brain = input x mask (scaled input values)
    pub brain: ArrayD<f64>,
    /// TSEG labels (FreeSurfer numbers), as TigerBx `_tseg`
    pub tseg: Option<ArrayD<u8>>,
    /// DGM12 labels 1..12 (TigerBx `_dgm`), when a DGM session was given
    pub dgm: Option<ArrayD<u8>>,
}

/// Run tBX (and TSEG / DGM12 when their sessions are given) on one image, as TigerBx `run('bmtd')`.
///
/// `data`: 3D f64 array from NIfTI (H, W, D), scaled; trailing singleton dimensions are dropped
/// `header`: its NIfTI header (affine, voxel size, data type)
///
/// tBX runs in model space (1 mm when a voxel size is outside [0.9, 1.1] mm, diagonal affine) on the input cropped
/// to its non-zero voxels + 16 mm; its masks are brought back onto the input grid. TSEG gets the brain inside the
/// outer mask in the same model space, cropped to it + 16 voxels, / max clipped to [0, 1] and padded to a multiple of
/// 16; its labels are brought back onto the input grid and kept inside the outer mask (TigerBx `lib_bx.tseg`). DGM12
/// gets the tBX brain cropped the same way (the tBX mask + 32 when that box exceeds 256^3 voxels), its labels masked
/// by tBX.
pub fn run_bx(
    data: &ArrayD<f64>,
    header: &NiftiHeader,
    tbx_session: &mut Session,
    tseg_session: Option<&mut Session>,
    dgm_session: Option<&mut Session>,
) -> Result<BxResult> {
    let shape: Vec<usize> = data.shape().iter().cloned().filter(|&n| n != 1).collect();
    if shape.len() != 3 {
        bail!("Expected a 3D image, got shape {:?}", data.shape());
    }
    let native = data
        .to_shape((shape[0], shape[1], shape[2]))
        .context("Expected 3D input")?
        .into_dimensionality::<Ix3>()?
        .to_owned();
    let input_affine = resample::best_affine(header);

    // ── tBX ────────────────────────────────────────────────────────────────
    // input: the bounding box of the voxels exactly != 0 (+ 16 mm, at least 256 mm per axis) on the input grid
    // (TigerBx _crop_nonzero_native); no thresholds, no head detection. On the input grid because resampling to 1 mm
    // spreads tiny non-zero values over a zero background. Images resampled onto a large grid (e.g. a PET FOV) are
    // mostly zeros, and without the crop their 1 mm model input can exceed what cuDNN supports; the 256 mm minimum
    // leaves usual fields of view whole (a tight crop of a skull-stripped image shrinks the tBX mask).
    let zooms = resample::header_zooms(header);
    let full = resample::Volume { data: native.clone(), affine: input_affine, kind: resample::Kind::from_header(header) };
    let pad = zooms.map(|z| (TBX_CROP_MM / z).ceil() as usize);
    let min_n = zooms.map(|z| (TBX_CROP_MIN_MM / z).ceil() as usize);
    let cropped = resample::crop_where_min(&full, pad, min_n, |v| v != 0.0).unwrap_or(full);
    let model = resample::model_space(cropped, zooms);
    let (outer, inner) = if inference::output_channels(tbx_session) == Some(2) {
        let (outer, inner) = tbx_dual(tbx_session, &model.data).context("tBX inference failed")?;
        (outer, Some(inner))
    } else {
        let mask = inference::run_onnx(tbx_session, normalized(&model.data).view()).context("tBX inference failed")?;
        (largest_component(mask)?, None)
    };
    if !outer.iter().any(|&m| m > 0) {
        bail!("tBX produced an empty brain mask");
    }
    let outer = resample::nearest_to_grid(&outer, &model.affine, &input_affine, native.dim());
    let inner = inner.map(|m| resample::nearest_to_grid(&m, &model.affine, &input_affine, native.dim()));
    let dual = inner.is_some();
    let mask = inner.unwrap_or_else(|| outer.clone());          // the tBX mask: inner (a single mask model: its mask)
    let masked = |m: &Array3<u8>| ndarray::Zip::from(&native).and(m).map_collect(|&d, &m| if m > 0 { d } else { 0.0 });
    let brain = masked(&mask);

    // ── TSEG: the brain inside the outer mask in model space, cropped to it + 16 voxels ──────────────────
    let tseg = match tseg_session {
        None => None,
        Some(session) => {
            let model = resample::to_model_space_as(masked(&outer), header, resample::Kind::Float64);
            let crop = resample::crop_where_min(&model, [TSEG_CROP_PADDING; 3], [0; 3], |v| v > 0.0).context("Empty tBX brain")?;
            let idx = run_padded(session, &unit_clipped(&crop.data)).context("TSEG inference failed")?;
            let lab = idx.mapv(|k| TSEG_LABELS.get(k as usize).copied().unwrap_or(0));
            let lab = resample::nearest_to_grid(&lab, &crop.affine, &input_affine, native.dim());
            Some(ndarray::Zip::from(&lab).and(&outer).map_collect(|&l, &m| if m > 0 { l } else { 0 }))
        }
    };
    // ── DGM12: the tBX brain in model space, cropped to the brain + 16 voxels ──────────────────────────────
    let dgm = match dgm_session {
        None => None,
        Some(session) => {
            let model = resample::to_model_space_as(brain.clone(), header, resample::Kind::Float64);
            let crop = resample::crop_where_min(&model, [TSEG_CROP_PADDING; 3], [0; 3], |v| v > 0.0).context("Empty tBX brain")?;
            // when the box exceeds 256^3 voxels (reordering an oblique image leaves tiny positive ringing far outside
            // the brain, making it the whole grid: e.g. 34 M voxels, too large for the CUDA Concat), the tBX mask
            // (nearest onto the model grid) + 32 (TigerBx _prepare_tbx111_crop)
            let model = &model;
            let crop = if crop.data.len() > DGM_CROP_MAX_VOXELS {
                let keep = resample::nearest_to_grid(&mask, &input_affine, &model.affine, model.data.dim());
                resample::crop_to_mask(model, &keep, DGM_MASK_CROP_PADDING).context("Empty tBX brain")?
            } else {
                crop
            };
            let idx = inference::run_onnx(session, normalized(&crop.data).view()).context("DGM12 inference failed")?;
            let labels = resample::nearest_to_grid(&idx, &crop.affine, &input_affine, native.dim());
            Some(ndarray::Zip::from(&labels).and(&mask).map_collect(|&l, &m| if m > 0 { l } else { 0 }))
        }
    };

    Ok(BxResult {
        mask: mask.into_dyn(),
        outer: dual.then(|| outer.into_dyn()),
        brain: brain.into_dyn(),
        tseg: tseg.map(|a| a.into_dyn()),
        dgm: dgm.map(|d| d.into_dyn()),
    })
}

fn largest_component(mask: Array3<u8>) -> Result<Array3<u8>> {
    Ok(postprocess::largest_connected_component_3d(&mask.into_dyn()).into_dimensionality::<Ix3>()?)
}

/// Dual brain mask model (TigerBx mode `tbxdual`, `lib_bx._tbxdual_labels`) -> (outer, inner) on the model grid. The
/// input is scaled in f32 (as the model's reference inference) and padded at the end of each axis to a multiple of 16;
/// the `labels` output is 0 background, 1 outer only, 2 inner. Outer and inner each keep their largest component, the
/// inner one inside the outer.
fn tbx_dual(session: &mut Session, x: &Array3<f64>) -> Result<(Array3<u8>, Array3<u8>)> {
    if !inference::has_labels_output(session) {
        bail!("dual mask model without a labels output");
    }
    let (h, w, d) = x.dim();
    let xf = x.mapv(|v| v as f32);
    let max_val = xf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let up = |n: usize| n.div_ceil(16) * 16;
    let mut input = Array3::<f32>::zeros((up(h), up(w), up(d)));
    input.slice_mut(s![..h, ..w, ..d]).assign(&if max_val > 0.0 { xf.mapv(|v| v / max_val) } else { xf });
    let lab = inference::run_onnx(session, input.view())?.slice(s![..h, ..w, ..d]).to_owned();
    let outer = largest_component(lab.mapv(|l| (l >= 1) as u8))?;
    let inner = ndarray::Zip::from(&lab).and(&outer).map_collect(|&l, &o| (l == 2 && o > 0) as u8);
    Ok((outer, largest_component(inner)?))
}

/// TSEG model input: the volume divided by its maximum, clipped to [0, 1], as f32 (TigerBx `lib_bx.tseg`).
fn unit_clipped(x: &Array3<f64>) -> Array3<f32> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    x.mapv(|v| (v / max_val).clamp(0.0, 1.0) as f32)
}

/// Labels of `x` padded at the end of each axis to a multiple of 16 (the padding cut off again).
fn run_padded(session: &mut Session, x: &Array3<f32>) -> Result<Array3<u8>> {
    let (h, w, d) = x.dim();
    let up = |n: usize| n.div_ceil(16) * 16;
    let mut input = Array3::<f32>::zeros((up(h), up(w), up(d)));
    input.slice_mut(s![..h, ..w, ..d]).assign(x);
    Ok(inference::run_onnx(session, input.view())?.slice(s![..h, ..w, ..d]).to_owned())
}

/// Model input: the volume divided by its maximum (TigerBx `/ max`), as f32.
fn normalized(x: &Array3<f64>) -> Array3<f32> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val > 0.0 {
        x.mapv(|v| (v / max_val) as f32)
    } else {
        x.mapv(|v| v as f32)
    }
}
