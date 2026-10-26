use anyhow::{bail, Context, Result};
use ndarray::ArrayD;
use nifti::volume::inmem::InMemNiftiVolume;
use nifti::{writer::WriterOptions, IntoNdArray, NiftiHeader, NiftiObject, ReaderOptions};
use std::io::Read;
use std::path::Path;

/// Header and raw voxel bytes of a NIfTI file. A single-file .nii[.gz] is read from vox_offset, as nibabel and the
/// NIfTI-1 standard do: the nifti crate skips to vox_offset only when the extension flag (byte 348) is set, so a
/// file with vox_offset > 352 and the flag 0 (e.g. some Philips conversions) was read from byte 352, shifted.
fn read_raw(path: &Path) -> Result<(NiftiHeader, Vec<u8>)> {
    let name = path.to_string_lossy().to_lowercase();
    if !(name.ends_with(".nii") || name.ends_with(".nii.gz")) {
        let obj = ReaderOptions::new().read_file(path)?;
        let header = obj.header().clone();
        return Ok((header, obj.into_volume().into_raw_data()));
    }
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.starts_with(&[0x1f, 0x8b]) {
        let mut out = Vec::new();
        flate2::read::MultiGzDecoder::new(&bytes[..]).read_to_end(&mut out)?;
        bytes = out;
    }
    let header = NiftiHeader::from_reader(&bytes[..])?;
    let start = (header.vox_offset.max(352.0)) as usize;
    let n = header.dim()?.iter().map(|&d| d as usize).product::<usize>() * (header.bitpix as usize / 8);
    if bytes.len() < start + n {
        bail!("{} bytes of voxel data expected from offset {}, the file has {}", n, start, bytes.len());
    }
    bytes.truncate(start + n);
    Ok((header, bytes.split_off(start)))
}

/// Read a NIfTI file and return (header, data as f64), scaled as nibabel does: a scl_slope of 0 or NaN means no
/// scaling (the nifti crate would multiply by NaN), a NaN scl_inter means 0. Non-finite voxels (NaN / Inf, e.g.
/// outside the head of template-space images) are set to 0 with a warning, as TigerBx `core.io.load_nifti`: they
/// would spread through the intensity scaling and the resampling.
pub fn read_nifti(path: &Path) -> Result<(NiftiHeader, ArrayD<f64>)> {
    let (header, raw) = read_raw(path).with_context(|| format!("Failed to read NIfTI file: {}", path.display()))?;
    let mut scaled = header.clone();
    if !scaled.scl_slope.is_finite() || scaled.scl_slope == 0.0 {
        scaled.scl_slope = 1.0;
        scaled.scl_inter = 0.0;
    } else if !scaled.scl_inter.is_finite() {
        scaled.scl_inter = 0.0;
    }
    let data = InMemNiftiVolume::from_raw_data(&scaled, raw)
        .context("Failed to rebuild NIfTI volume")?
        .into_ndarray::<f64>()
        .context("Failed to convert NIfTI volume to ndarray")?;
    let mut data = data;
    let bad = data.iter().filter(|v| !v.is_finite()).count();
    if bad > 0 {
        eprintln!("  Warning: {} non-finite voxels (NaN / Inf) in {} set to 0", bad, path.display());
        data.mapv_inplace(|v| if v.is_finite() { v } else { 0.0 });
    }

    Ok((header, data))
}

/// Write a u8 ndarray (mask, labels) to a NIfTI file, using a reference header (the writer sets the data type
/// from the array and scl_slope / scl_inter to 1 / 0).
pub fn write_nifti_u8(path: &Path, header: &NiftiHeader, data: &ArrayD<u8>) -> Result<()> {
    WriterOptions::new(path)
        .reference_header(header)
        .write_nifti(data)
        .with_context(|| format!("Failed to write NIfTI file: {}", path.display()))?;
    Ok(())
}

/// Write an f32 ndarray (template-space images) to a NIfTI file, using a reference header.
pub fn write_nifti_f32(path: &Path, header: &NiftiHeader, data: &ArrayD<f32>) -> Result<()> {
    WriterOptions::new(path)
        .reference_header(header)
        .write_nifti(data)
        .with_context(|| format!("Failed to write NIfTI file: {}", path.display()))?;
    Ok(())
}

/// Write the tBX brain as TigerBx does: in the input's data type when the values fit it (integers truncated
/// toward zero, no scaling), float32 when an integer type cannot hold them.
pub fn write_tbx(path: &Path, header: &NiftiHeader, data: &ArrayD<f64>) -> Result<()> {
    let (lo, hi) = data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let fits = |min: f64, max: f64| lo >= min && hi <= max;
    let w = WriterOptions::new(path).reference_header(header);
    let res = match header.datatype {
        2 if fits(u8::MIN as f64, u8::MAX as f64) => w.write_nifti(&data.mapv(|v| v as u8)),
        256 if fits(i8::MIN as f64, i8::MAX as f64) => w.write_nifti(&data.mapv(|v| v as i8)),
        4 if fits(i16::MIN as f64, i16::MAX as f64) => w.write_nifti(&data.mapv(|v| v as i16)),
        512 if fits(u16::MIN as f64, u16::MAX as f64) => w.write_nifti(&data.mapv(|v| v as u16)),
        8 if fits(i32::MIN as f64, i32::MAX as f64) => w.write_nifti(&data.mapv(|v| v as i32)),
        768 if fits(u32::MIN as f64, u32::MAX as f64) => w.write_nifti(&data.mapv(|v| v as u32)),
        64 => w.write_nifti(data),
        _ => w.write_nifti(&data.mapv(|v| v as f32)),
    };
    res.with_context(|| format!("Failed to write NIfTI file: {}", path.display()))
}
