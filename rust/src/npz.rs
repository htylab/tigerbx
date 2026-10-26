//! Minimal NumPy .npz writer (uncompressed zip of .npy files, as numpy.savez), for the plain-array registration
//! file `_warp.npz` (TigerBx `core.warpfile`; numpy reads it with np.load, no pickle).

use anyhow::{bail, Result};
use std::io::Write;

/// One array of an .npz file.
pub enum Npy<'a> {
    F64(Vec<usize>, &'a [f64]),
    F32(Vec<usize>, &'a [f32]),
    I64(Vec<usize>, &'a [i64]),
    /// 0-d unicode string array (numpy '<U{n}')
    Str(&'a str),
}

impl Npy<'_> {
    /// .npy v1.0 bytes: magic, header dict padded to a multiple of 64 bytes, C-order little-endian data.
    fn bytes(&self) -> Vec<u8> {
        let (descr, shape, data): (String, Vec<usize>, Vec<u8>) = match self {
            Npy::F64(s, v) => ("<f8".into(), s.clone(), v.iter().flat_map(|x| x.to_le_bytes()).collect()),
            Npy::F32(s, v) => ("<f4".into(), s.clone(), v.iter().flat_map(|x| x.to_le_bytes()).collect()),
            Npy::I64(s, v) => ("<i8".into(), s.clone(), v.iter().flat_map(|x| x.to_le_bytes()).collect()),
            Npy::Str(t) => {
                let chars: Vec<char> = t.chars().collect();
                (format!("<U{}", chars.len()), vec![], chars.iter().flat_map(|&c| (c as u32).to_le_bytes()).collect())
            }
        };
        let shape = match shape.len() {
            0 => "()".to_string(),
            1 => format!("({},)", shape[0]),
            _ => format!("({})", shape.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")),
        };
        let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&data);
        out
    }
}

/// Write `arrays` (name -> array) as an uncompressed .npz (zip, method 0 "stored").
pub fn write_npz(path: &std::path::Path, arrays: &[(&str, Npy)]) -> Result<()> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, arr) in arrays {
        let data = arr.bytes();
        if data.len() > u32::MAX as usize || out.len() > u32::MAX as usize {
            bail!("{name}: npz entries over 4 GB are not supported");
        }
        let mut crc = flate2::Crc::new();
        crc.update(&data);
        let fname = format!("{name}.npy");
        let offset = out.len() as u32;
        // local file header
        let mut local = Vec::new();
        local.extend_from_slice(&0x04034b50u32.to_le_bytes());
        local.extend_from_slice(&20u16.to_le_bytes()); // version needed
        local.extend_from_slice(&0u16.to_le_bytes()); // flags
        local.extend_from_slice(&0u16.to_le_bytes()); // method: stored
        local.extend_from_slice(&0u16.to_le_bytes()); // time
        local.extend_from_slice(&0x21u16.to_le_bytes()); // date 1980-01-01
        local.extend_from_slice(&crc.sum().to_le_bytes());
        local.extend_from_slice(&(data.len() as u32).to_le_bytes());
        local.extend_from_slice(&(data.len() as u32).to_le_bytes());
        local.extend_from_slice(&(fname.len() as u16).to_le_bytes());
        local.extend_from_slice(&0u16.to_le_bytes()); // extra length
        local.extend_from_slice(fname.as_bytes());
        out.extend_from_slice(&local);
        out.extend_from_slice(&data);
        // central directory entry
        central.extend_from_slice(&0x02014b50u32.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes()); // version made by
        central.extend_from_slice(&local[4..28]); // version needed .. file name length (same fields)
        central.extend_from_slice(&0u16.to_le_bytes()); // extra length
        central.extend_from_slice(&0u16.to_le_bytes()); // comment length
        central.extend_from_slice(&0u16.to_le_bytes()); // disk number
        central.extend_from_slice(&0u16.to_le_bytes()); // internal attributes
        central.extend_from_slice(&0u32.to_le_bytes()); // external attributes
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(fname.as_bytes());
    }
    let cd_offset = out.len() as u32;
    out.extend_from_slice(&central);
    // end of central directory
    out.extend_from_slice(&0x06054b50u32.to_le_bytes());
    out.extend_from_slice(&[0u8; 4]); // disk numbers
    out.extend_from_slice(&(arrays.len() as u16).to_le_bytes());
    out.extend_from_slice(&(arrays.len() as u16).to_le_bytes());
    out.extend_from_slice(&(central.len() as u32).to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // comment length
    std::fs::File::create(path)?.write_all(&out)?;
    Ok(())
}
