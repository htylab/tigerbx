use anyhow::{bail, Context, Result};
use ndarray::{Array3, ArrayD, Ix3};
use nifti::NiftiHeader;
use ort::session::Session;

use crate::inference;
use crate::postprocess;
use crate::resample;

/// FreeSurfer label number of TSEG output channel k (channel 0 = background), TigerBx `label_all['tseg']`.
const TSEG_LABELS: [u8; 43] = [
    2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51,
    52, 53, 54, 58, 60, 62, 63, 77, 85, 251, 252, 253, 254, 255,
];
/// tBX-mask voxels that no TSEG structure claims (TigerBx `TSEG_UNLABELED_INTRACRANIAL`).
const TSEG_UNLABELED_INTRACRANIAL: u8 = 99;
/// TSEG / DGM input crop: brain > 0 + 16 voxels, unless that box exceeds 256^3 voxels; then the tBX mask + 32
/// voxels (TigerBx `_prepare_tbx111_crop`).
const TSEG_CROP_PADDING: usize = 16;
const TSEG_CROP_MAX_VOXELS: usize = 256 * 256 * 256;
const TSEG_MASK_CROP_PADDING: usize = 32;
/// Padding (mm) of the tBX input crop around the non-zero voxels and its minimum extent (mm) per axis (TigerBx
/// `_TBX_CROP_MM`, `_TBX_CROP_MIN_MM`).
const TBX_CROP_MM: f64 = 16.0;
const TBX_CROP_MIN_MM: f64 = 256.0;

/// Results of one image, all on the input grid.
pub struct BxResult {
    /// tBX brain mask, u8 0 / 1
    pub mask: ArrayD<u8>,
    /// tBX brain = input x mask (scaled input values)
    pub brain: ArrayD<f64>,
    /// TSEG labels (FreeSurfer numbers, 99 = Unlabeled-Intracranial): the TSEG model output, as TigerBx `_tseg`;
    /// with `merge` (the seg pipeline) its 12 deep-GM structures come from DGM12 (`merge_dgm`)
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
/// to its non-zero voxels + 16 mm; its mask is brought back onto the input grid. TSEG gets the tBX brain in the same
/// model space, cropped to the brain + 16 voxels (to the tBX mask + 32 when that box exceeds 256^3 voxels); its
/// labels are brought back onto the input grid, masked by tBX, and mask voxels it leaves at 0 become 99.
pub fn run_bx(
    data: &ArrayD<f64>,
    header: &NiftiHeader,
    tbx_session: &mut Session,
    tseg_session: Option<&mut Session>,
    dgm_session: Option<&mut Session>,
    merge: bool,
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
    let mask = inference::run_onnx(tbx_session, normalized(&model.data).view()).context("tBX inference failed")?;
    let mask = postprocess::largest_connected_component_3d(&mask.into_dyn()).into_dimensionality::<Ix3>()?;
    if !mask.iter().any(|&m| m > 0) {
        bail!("tBX produced an empty brain mask");
    }
    let mask = resample::nearest_to_grid(&mask, &model.affine, &input_affine, native.dim());
    let brain = ndarray::Zip::from(&native).and(&mask).map_collect(|&d, &m| if m > 0 { d } else { 0.0 });

    // ── TSEG and DGM12: the tBX brain in model space, cropped to the brain + 16 voxels ───────────────
    let crop = if tseg_session.is_some() || dgm_session.is_some() {
        // brain > 0 + 16; when that box exceeds 256^3 voxels (reordering an oblique image leaves tiny positive
        // ringing far outside the brain, making it the whole grid: e.g. 34 M voxels, too large for the CUDA Concat
        // of TSEG), the tBX mask (nearest onto the model grid) + 32 (TigerBx _prepare_tbx111_crop)
        let model = resample::to_model_space_as(brain.clone(), header, resample::Kind::Float64);
        let crop = resample::crop_where_min(&model, [TSEG_CROP_PADDING; 3], [0; 3], |v| v > 0.0).context("Empty tBX brain")?;
        if crop.data.len() > TSEG_CROP_MAX_VOXELS {
            let keep = resample::nearest_to_grid(&mask, &input_affine, &model.affine, model.data.dim());
            Some(resample::crop_to_mask(&model, &keep, TSEG_MASK_CROP_PADDING).context("Empty tBX brain")?)
        } else {
            Some(crop)
        }
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
    let tseg = match tseg_session {
        None => None,
        Some(session) => {
            let lut = |k: u8| if k == 0 { 0 } else { TSEG_LABELS.get(k as usize - 1).copied().unwrap_or(0) };
            let mut tseg = segment(session, &lut, "TSEG")?;
            if let (true, Some(d)) = (merge, dgm.as_ref()) {
                tseg = merge_dgm(&tseg, d);
            }
            Some(ndarray::Zip::from(&tseg).and(&mask).map_collect(|&l, &m| match (m > 0, l) {
                (false, _) => 0,
                (true, 0) => TSEG_UNLABELED_INTRACRANIAL,
                (true, l) => l,
            }))
        }
    };

    Ok(BxResult {
        mask: mask.into_dyn(),
        brain: brain.into_dyn(),
        tseg: tseg.map(|a| a.into_dyn()),
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

/// FreeSurfer number of DGM12 label k (1..12): L/R thalamus, caudate, putamen, pallidum, hippocampus, amygdala.
const DGM12_FS: [u8; 12] = [10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54];

/// The seg pipeline's post-processing (TigerBx `pipelines.seg.merge_dgm`, same rule): TSEG with its 12 deep-GM
/// structures from DGM12 (more accurate there). A voxel DGM12 labels gets its FreeSurfer number; a voxel TSEG gave a
/// deep-GM label that DGM12 does not keep is a hole, filled layer by layer from the 6-connected neighbours outside the
/// deep-GM structures (ties: the first of -x, +x, -y, +y, -z, +z). Holes left over get 0.
pub fn merge_dgm(tseg: &Array3<u8>, dgm: &Array3<u8>) -> Array3<u8> {
    let is_dgm = |l: u8| DGM12_FS.contains(&l);
    let (nx, ny, nz) = tseg.dim();
    let mut out = ndarray::Zip::from(tseg).and(dgm).map_collect(|&a, &d| if d > 0 && (d as usize) <= DGM12_FS.len() { DGM12_FS[d as usize - 1] } else { a });
    let mut hole = ndarray::Zip::from(tseg).and(dgm).map_collect(|&a, &d| !(d > 0 && (d as usize) <= DGM12_FS.len()) && is_dgm(a));
    let mut source = ndarray::Zip::from(&hole).and(&out).map_collect(|&h, &l| !h && !is_dgm(l));
    let mut holes: Vec<[usize; 3]> = hole.indexed_iter().filter(|(_, &h)| h).map(|((i, j, k), _)| [i, j, k]).collect();
    while !holes.is_empty() {
        // one layer: every hole takes the label of its first source neighbour, all at once
        let fills: Vec<([usize; 3], u8)> = holes
            .iter()
            .filter_map(|&[i, j, k]| {
                for (a, b, c) in [(-1isize, 0isize, 0isize), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)] {
                    let (x, y, z) = (i as isize + a, j as isize + b, k as isize + c);
                    if x < 0 || y < 0 || z < 0 || x as usize >= nx || y as usize >= ny || z as usize >= nz {
                        continue;
                    }
                    let p = [x as usize, y as usize, z as usize];
                    if source[p] {
                        return Some(([i, j, k], out[p]));
                    }
                }
                None
            })
            .collect();
        if fills.is_empty() {
            break;
        }
        for (p, l) in &fills {
            out[*p] = *l;
            hole[*p] = false;
            source[*p] = true;
        }
        holes.retain(|&p| hole[p]);
    }
    for p in holes {
        out[p] = 0;
    }
    out
}
