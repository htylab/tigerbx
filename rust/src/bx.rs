use anyhow::{bail, Context, Result};
use ndarray::{Array3, ArrayD, Ix3};
use nifti::NiftiHeader;
use ort::session::Session;

use crate::inference;
use crate::postprocess;
use crate::resample;

/// FreeSurfer label of ASEG output channel k (channel 0 = background), TigerBx `label_all['aseg43']`.
const ASEG43_LABELS: [u8; 43] = [
    2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51,
    52, 53, 54, 58, 60, 62, 63, 77, 85, 251, 252, 253, 254, 255,
];
/// tBET-mask voxels that no ASEG structure claims (TigerBx `ASEG_UNLABELED_INTRACRANIAL`).
const ASEG_UNLABELED_INTRACRANIAL: u8 = 99;
/// Padding (voxels) of the ASEG input crop around the brain (TigerBx `crop_cube` default).
const ASEG_CROP_PADDING: usize = 16;

/// Results of one image, all on the input grid.
pub struct BxResult {
    /// tBET brain mask, u8 0 / 1
    pub mask: ArrayD<u8>,
    /// tBET brain = input x mask (scaled input values)
    pub brain: ArrayD<f64>,
    /// ASEG labels (FreeSurfer numbers, 99 = Unlabeled-Intracranial), when an ASEG session was given
    pub aseg: Option<ArrayD<u8>>,
}

/// Run tBET (and ASEG when `aseg_session` is given) on one image, as TigerBx `run('bma')`.
///
/// `data`: 3D f64 array from NIfTI (H, W, D), scaled; trailing singleton dimensions are dropped
/// `header`: its NIfTI header (affine, voxel size, data type)
///
/// tBET runs in model space (1 mm when a voxel size is outside [0.9, 1.1] mm, diagonal affine); its mask is
/// brought back onto the input grid. ASEG gets the tBET brain in the same model space, cropped to the brain
/// + 16 voxels; its labels are brought back onto the input grid, masked by tBET, and mask voxels it leaves at 0
/// become 99.
pub fn run_bx(
    data: &ArrayD<f64>,
    header: &NiftiHeader,
    bet_session: &mut Session,
    aseg_session: Option<&mut Session>,
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

    // ── tBET ────────────────────────────────────────────────────────────────
    let model = resample::to_model_space(native.clone(), header);
    let mask = inference::run_onnx(bet_session, normalized(&model.data).view()).context("tBET inference failed")?;
    let mask = postprocess::largest_connected_component_3d(&mask.into_dyn()).into_dimensionality::<Ix3>()?;
    if !mask.iter().any(|&m| m > 0) {
        bail!("tBET produced an empty brain mask");
    }
    let mask = resample::nearest_to_grid(&mask, &model.affine, &input_affine, native.dim());
    let brain = ndarray::Zip::from(&native).and(&mask).map_collect(|&d, &m| if m > 0 { d } else { 0.0 });

    // ── ASEG ────────────────────────────────────────────────────────────────
    let aseg = match aseg_session {
        None => None,
        Some(session) => {
            let model = resample::to_model_space_as(brain.clone(), header, resample::Kind::Float64);
            let crop = resample::crop_nonzero(&model, ASEG_CROP_PADDING).context("Empty tBET brain")?;
            let idx = inference::run_onnx(session, normalized(&crop.data).view()).context("ASEG inference failed")?;
            let labels = idx.mapv(|k| if k == 0 { 0 } else { ASEG43_LABELS.get(k as usize - 1).copied().unwrap_or(0) });
            let labels = resample::nearest_to_grid(&labels, &crop.affine, &input_affine, native.dim());
            let aseg = ndarray::Zip::from(&labels).and(&mask).map_collect(|&l, &m| match (m > 0, l) {
                (false, _) => 0,
                (true, 0) => ASEG_UNLABELED_INTRACRANIAL,
                (true, l) => l,
            });
            Some(aseg.into_dyn())
        }
    };

    Ok(BxResult {
        mask: mask.into_dyn(),
        brain: brain.into_dyn(),
        aseg,
    })
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
