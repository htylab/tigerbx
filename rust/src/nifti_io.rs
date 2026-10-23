use anyhow::{Context, Result};
use ndarray::ArrayD;
use nifti::{writer::WriterOptions, IntoNdArray, NiftiHeader, NiftiObject, ReaderOptions};
use std::path::Path;

/// Read a NIfTI file and return (header, data as f32).
pub fn read_nifti(path: &Path) -> Result<(NiftiHeader, ArrayD<f32>)> {
    let obj = ReaderOptions::new()
        .read_file(path)
        .with_context(|| format!("Failed to read NIfTI file: {}", path.display()))?;

    let header = obj.header().clone();
    let data = obj
        .into_volume()
        .into_ndarray::<f32>()
        .context("Failed to convert NIfTI volume to ndarray")?;

    Ok((header, data))
}

/// Write an f32 ndarray to a NIfTI file, using a reference header.
pub fn write_nifti(path: &Path, header: &NiftiHeader, data: &ArrayD<f32>) -> Result<()> {
    WriterOptions::new(path)
        .reference_header(header)
        .write_nifti(data)
        .with_context(|| format!("Failed to write NIfTI file: {}", path.display()))?;
    Ok(())
}

/// Write a u8 mask ndarray to a NIfTI file.
pub fn write_nifti_u8(path: &Path, header: &NiftiHeader, data: &ArrayD<u8>) -> Result<()> {
    WriterOptions::new(path)
        .reference_header(header)
        .write_nifti(data)
        .with_context(|| format!("Failed to write NIfTI file: {}", path.display()))?;
    Ok(())
}
