use anyhow::{bail, Context, Result};
use ndarray::{Array3, ArrayD, Ix3};
use nifti::NiftiHeader;
use ort::session::Session;

use crate::inference;
use crate::postprocess;
use crate::resample;

/// Run the full brain-extraction pipeline on input data.
///
/// `data`: 3D f32 array from NIfTI (H, W, D); a trailing singleton dimension is dropped
/// `header`: its NIfTI header (affine, voxel size, data type)
/// `session`: pre-built ONNX inference session
///
/// As in TigerBx, the model runs in model space (1 mm when the voxel size is outside [0.9, 1.1] mm,
/// diagonal affine) and the mask is brought back onto the input grid.
///
/// Returns `(brain_mask, brain_extracted)` on the input grid:
/// - `brain_mask`: u8 binary mask (0 or 1)
/// - `brain_extracted`: f32 input × mask
pub fn run_bx(data: &ArrayD<f32>, header: &NiftiHeader, session: &mut Session) -> Result<(ArrayD<u8>, ArrayD<f32>)> {
    let shape: Vec<usize> = data.shape().iter().cloned().filter(|&n| n != 1).collect();
    if shape.len() != 3 {
        bail!("Expected a 3D image, got shape {:?}", data.shape());
    }
    let native = data
        .to_shape((shape[0], shape[1], shape[2]))
        .context("Expected 3D input")?
        .into_dimensionality::<Ix3>()?
        .mapv(|v| v as f64);

    // 1. Model space (TigerBx read_bet_input)
    let model = resample::to_model_space(native.clone(), header);

    // 2. Normalize to [0, 1]
    let max_val = model.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let normalized: Array3<f32> = if max_val > 0.0 {
        model.data.mapv(|v| (v / max_val) as f32)
    } else {
        model.data.mapv(|v| v as f32)
    };

    // 3. ONNX inference → sigmoid → threshold at 0.5
    let mask = inference::run_onnx(session, normalized.view())
        .context("ONNX inference failed")?
        .into_dyn();

    // 4. Keep largest connected component (in model space, as TigerBx)
    let mask = postprocess::largest_connected_component_3d(&mask)
        .into_dimensionality::<Ix3>()?;

    // 5. Back onto the input grid (nearest neighbour)
    let mask = resample::nearest_to_grid(&mask, &model.affine, &resample::best_affine(header), native.dim());

    // 6. Brain extracted = input × mask
    let brain = ndarray::Zip::from(&native)
        .and(&mask)
        .map_collect(|&d, &m| if m > 0 { d as f32 } else { 0.0 });

    Ok((mask.into_dyn(), brain.into_dyn()))
}
