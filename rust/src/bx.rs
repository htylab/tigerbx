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
/// Padding (voxels) of the ASEG / DGM input crop around the brain (TigerBx `crop_cube` default).
const ASEG_CROP_PADDING: usize = 16;
/// FreeSurfer label of DGM12 output k (1..12): L/R thalamus, caudate, putamen, pallidum, hippocampus, amygdala.
const DGM12_ASEG: [u8; 12] = [10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54];

/// Results of one image, all on the input grid.
pub struct BxResult {
    /// tBET brain mask, u8 0 / 1
    pub mask: ArrayD<u8>,
    /// tBET brain = input x mask (scaled input values)
    pub brain: ArrayD<f64>,
    /// ASEG labels (FreeSurfer numbers, 99 = Unlabeled-Intracranial); with a DGM session its 12 deep-GM
    /// structures come from DGM12 (`merge_dgm`)
    pub aseg: Option<ArrayD<u8>>,
    /// DGM12 labels 1..12 (TigerBx `_dgm`), when a DGM session was given
    pub dgm: Option<ArrayD<u8>>,
}

/// Run tBET (and ASEG / DGM12 when their sessions are given) on one image, as TigerBx `run('bmad')`.
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

    // ── tBET ────────────────────────────────────────────────────────────────
    let model = resample::to_model_space(native.clone(), header);
    let mask = inference::run_onnx(bet_session, normalized(&model.data).view()).context("tBET inference failed")?;
    let mask = postprocess::largest_connected_component_3d(&mask.into_dyn()).into_dimensionality::<Ix3>()?;
    if !mask.iter().any(|&m| m > 0) {
        bail!("tBET produced an empty brain mask");
    }
    let mask = resample::nearest_to_grid(&mask, &model.affine, &input_affine, native.dim());
    let brain = ndarray::Zip::from(&native).and(&mask).map_collect(|&d, &m| if m > 0 { d } else { 0.0 });

    // ── ASEG and DGM12: the tBET brain in model space, cropped to the brain + 16 voxels ───────────────
    let crop = if aseg_session.is_some() || dgm_session.is_some() {
        let model = resample::to_model_space_as(brain.clone(), header, resample::Kind::Float64);
        Some(resample::crop_nonzero(&model, ASEG_CROP_PADDING).context("Empty tBET brain")?)
    } else {
        None
    };
    let segment = |session: &mut Session, lut: &dyn Fn(u8) -> u8, what: &str| -> Result<Array3<u8>> {
        let crop = crop.as_ref().unwrap();
        let idx = inference::run_onnx(session, normalized(&crop.data).view()).with_context(|| format!("{what} inference failed"))?;
        let labels = resample::nearest_to_grid(&idx.mapv(lut), &crop.affine, &input_affine, native.dim());
        Ok(ndarray::Zip::from(&labels).and(&mask).map_collect(|&l, &m| if m > 0 { l } else { 0 }))
    };
    let dgm = match dgm_session {
        None => None,
        Some(session) => Some(segment(session, &|k| k, "DGM12")?),
    };
    let aseg = match aseg_session {
        None => None,
        Some(session) => {
            let lut = |k: u8| if k == 0 { 0 } else { ASEG43_LABELS.get(k as usize - 1).copied().unwrap_or(0) };
            let mut aseg = segment(session, &lut, "ASEG")?;
            if let Some(d) = dgm.as_ref() {
                aseg = merge_dgm(&aseg, d);
            }
            Some(ndarray::Zip::from(&aseg).and(&mask).map_collect(|&l, &m| match (m > 0, l) {
                (false, _) => 0,
                (true, 0) => ASEG_UNLABELED_INTRACRANIAL,
                (true, l) => l,
            }))
        }
    };

    Ok(BxResult {
        mask: mask.into_dyn(),
        brain: brain.into_dyn(),
        aseg: aseg.map(|a| a.into_dyn()),
        dgm: dgm.map(|d| d.into_dyn()),
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

/// Replace the 12 deep-GM structures of `aseg` by DGM12 (more accurate on CANDI / OASIS / IBSR18 manual labels):
/// a voxel DGM12 labels gets its FreeSurfer number; a voxel ASEG gave a deep-GM label that DGM12 does not keep
/// gets the label of the nearest (6-connected) voxel outside the deep-GM structures.
fn merge_dgm(aseg: &Array3<u8>, dgm: &Array3<u8>) -> Array3<u8> {
    const HOLE: u8 = u8::MAX;
    let is_dgm = |l: u8| DGM12_ASEG.contains(&l);
    let mut out = ndarray::Zip::from(aseg).and(dgm).map_collect(|&a, &d| {
        if d > 0 && (d as usize) <= DGM12_ASEG.len() {
            DGM12_ASEG[d as usize - 1]
        } else if is_dgm(a) {
            HOLE
        } else {
            a
        }
    });
    // multi-source breadth-first fill of the holes from the non-deep-GM labels
    let (nx, ny, nz) = out.dim();
    let mut queue: std::collections::VecDeque<(usize, usize, usize)> = std::collections::VecDeque::new();
    for ((i, j, k), &l) in out.indexed_iter() {
        if l != HOLE && !is_dgm(l) {
            queue.push_back((i, j, k));
        }
    }
    while let Some((i, j, k)) = queue.pop_front() {
        let l = out[[i, j, k]];
        for (a, b, c) in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)] {
            let (x, y, z) = (i as isize + a, j as isize + b, k as isize + c);
            if x < 0 || y < 0 || z < 0 || x as usize >= nx || y as usize >= ny || z as usize >= nz {
                continue;
            }
            let p = [x as usize, y as usize, z as usize];
            if out[p] == HOLE {
                out[p] = l;
                queue.push_back((p[0], p[1], p[2]));
            }
        }
    }
    out.mapv(|l| if l == HOLE { 0 } else { l })
}
