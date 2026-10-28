//! The DuoPE pipeline (`--duope`), as TigerBx `tigerbx.duope` (`tigerbx/pipelines/duope.py`): phase encoding and
//! readout time from the BIDS JSON sidecars (else --pe / --readout, else the header's dim_info), b0 = mean of the
//! b < 50 volumes (.bval) or of all volumes, every input corrected as <input>_duope (4D read and written volume by
//! volume), series with their own grid / polarity / head position (--series), duope_summary.csv per output folder.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use ndarray::{Array3, ShapeBuilder};
use nifti::{Endianness, NiftiHeader};

use crate::duope::{self, A3};
use crate::resample::{best_affine, linear_transform, mat_inv, mat_mul, Mat4};

pub const SUMMARY_NAME: &str = "duope_summary.csv";
const AXES: [char; 3] = ['i', 'j', 'k'];
/// Anatomical phase-encoding directions (from -> to) as RAS+ world vectors.
const ANAT: [(&str, [f64; 3]); 6] =
    [("LR", [1.0, 0.0, 0.0]), ("RL", [-1.0, 0.0, 0.0]), ("PA", [0.0, 1.0, 0.0]), ("AP", [0.0, -1.0, 0.0]), ("IS", [0.0, 0.0, 1.0]), ("SI", [0.0, 0.0, -1.0])];
/// An anatomical direction must lie within ~40 degrees of a voxel axis.
const MIN_COS: f64 = 0.75;

pub struct Settings {
    pub output: Option<PathBuf>,
    pub series: Vec<PathBuf>,
    pub pe: Option<String>,
    pub readout: Option<f64>,
    pub series_pe: Vec<String>,
    pub series_align: bool,
    pub joint: Option<bool>,
    pub save_field: bool,
    pub save_disp: bool,
}

// ---------------------------------------------------------------- NIfTI, volume by volume

/// A NIfTI file: its header and affine; the voxels are read volume by volume (`volumes`).
pub struct Nii {
    pub path: PathBuf,
    pub header: NiftiHeader,
    pub aff: Mat4,
    pub shape: [usize; 3],
    /// the dimensions after the third (a 4D series: [n])
    pub extra: Vec<usize>,
}

fn reader(path: &Path) -> Result<Box<dyn Read>> {
    let mut f = BufReader::new(File::open(path).with_context(|| format!("Failed to open {}", path.display()))?);
    let mut magic = [0u8; 2];
    f.read_exact(&mut magic)?;
    let chained = std::io::Cursor::new(magic.to_vec()).chain(f);
    Ok(if magic == [0x1f, 0x8b] { Box::new(BufReader::new(flate2::read::MultiGzDecoder::new(chained))) } else { Box::new(chained) })
}

impl Nii {
    pub fn open(path: &Path) -> Result<Nii> {
        let mut r = reader(path)?;
        let mut hb = vec![0u8; 348];
        r.read_exact(&mut hb).with_context(|| format!("Failed to read the NIfTI header of {}", path.display()))?;
        let header = NiftiHeader::from_reader(&hb[..]).with_context(|| format!("Not a NIfTI file: {}", path.display()))?;
        let dim = header.dim()?.iter().map(|&d| d as usize).collect::<Vec<_>>();
        let mut shape = [1usize; 3];
        for (a, s) in shape.iter_mut().enumerate() {
            *s = dim.get(a).copied().unwrap_or(1);
        }
        let extra = if dim.len() > 3 { dim[3..].to_vec() } else { vec![] };
        Ok(Nii { path: path.to_path_buf(), aff: best_affine(&header), header, shape, extra })
    }

    pub fn nvol(&self) -> usize {
        self.extra.iter().product()
    }

    pub fn dim3(&self) -> (usize, usize, usize) {
        (self.shape[0], self.shape[1], self.shape[2])
    }

    /// Every volume in order (float64, scaled as nibabel, non-finite voxels set to 0), one in memory at a time.
    pub fn volumes(&self) -> Result<impl Iterator<Item = Result<A3>> + '_> {
        let mut r = reader(&self.path)?;
        let start = (self.header.vox_offset.max(352.0)) as u64;
        std::io::copy(&mut r.by_ref().take(start), &mut std::io::sink())?;
        let h = &self.header;
        let bpp = (h.bitpix as usize) / 8;
        let nvox = self.shape.iter().product::<usize>();
        let (slope, inter) = if !h.scl_slope.is_finite() || h.scl_slope == 0.0 {
            (1.0, 0.0)
        } else {
            (h.scl_slope as f64, if h.scl_inter.is_finite() { h.scl_inter as f64 } else { 0.0 })
        };
        let big = h.endianness == Endianness::Big;
        let dt = h.datatype;
        let shape = self.dim3();
        let mut buf = vec![0u8; nvox * bpp];
        Ok((0..self.nvol()).map(move |_| {
            r.read_exact(&mut buf).context("NIfTI file shorter than its header says")?;
            let v: Vec<f64> = buf
                .chunks_exact(bpp)
                .map(|c| {
                    let mut b = [0u8; 8];
                    b[..bpp].copy_from_slice(c);
                    if big {
                        b[..bpp].reverse();
                    }
                    let x = match dt {
                        2 => b[0] as f64,
                        256 => b[0] as i8 as f64,
                        4 => i16::from_le_bytes([b[0], b[1]]) as f64,
                        512 => u16::from_le_bytes([b[0], b[1]]) as f64,
                        8 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
                        768 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
                        16 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
                        64 => f64::from_le_bytes(b),
                        1024 => i64::from_le_bytes(b) as f64,
                        1280 => u64::from_le_bytes(b) as f64,
                        _ => f64::NAN,
                    };
                    let y = x * slope + inter;
                    if y.is_finite() { y } else { 0.0 }
                })
                .collect();
            if !matches!(dt, 2 | 256 | 4 | 512 | 8 | 768 | 16 | 64 | 1024 | 1280) {
                bail!("NIfTI data type {} not supported", dt);
            }
            Ok(Array3::from_shape_vec(shape.f(), v).unwrap().as_standard_layout().into_owned())
        }))
    }
}

/// The 348-byte NIfTI-1 header (little endian) of float32 data of `dims`, from `h` (+ 4 bytes: no extensions).
fn header_bytes(h: &NiftiHeader, dims: &[usize]) -> Vec<u8> {
    let mut b = Vec::with_capacity(352);
    let i16b = |b: &mut Vec<u8>, v: i16| b.extend_from_slice(&v.to_le_bytes());
    let f32b = |b: &mut Vec<u8>, v: f32| b.extend_from_slice(&v.to_le_bytes());
    b.extend_from_slice(&348i32.to_le_bytes());
    b.extend_from_slice(&h.data_type);
    b.extend_from_slice(&h.db_name);
    b.extend_from_slice(&h.extents.to_le_bytes());
    i16b(&mut b, h.session_error);
    b.push(h.regular);
    b.push(h.dim_info);
    let mut dim = [1i16; 8];
    dim[0] = dims.len() as i16;
    for (a, &d) in dims.iter().enumerate() {
        dim[a + 1] = d as i16;
    }
    dim.iter().for_each(|&d| i16b(&mut b, d));
    f32b(&mut b, h.intent_p1);
    f32b(&mut b, h.intent_p2);
    f32b(&mut b, h.intent_p3);
    i16b(&mut b, h.intent_code);
    i16b(&mut b, 16);
    i16b(&mut b, 32);
    i16b(&mut b, h.slice_start);
    let mut pixdim = h.pixdim;
    for p in pixdim.iter_mut().skip(dims.len() + 1) {
        *p = if *p == 0.0 { 1.0 } else { *p };
    }
    pixdim.iter().for_each(|&p| f32b(&mut b, p));
    f32b(&mut b, 352.0);
    f32b(&mut b, 1.0);
    f32b(&mut b, 0.0);
    i16b(&mut b, h.slice_end);
    b.push(h.slice_code);
    b.push(h.xyzt_units);
    f32b(&mut b, h.cal_max);
    f32b(&mut b, h.cal_min);
    f32b(&mut b, h.slice_duration);
    f32b(&mut b, h.toffset);
    b.extend_from_slice(&h.glmax.to_le_bytes());
    b.extend_from_slice(&h.glmin.to_le_bytes());
    let mut descrip = h.descrip.clone();
    descrip.resize(80, 0);
    b.extend_from_slice(&descrip);
    b.extend_from_slice(&h.aux_file);
    i16b(&mut b, h.qform_code);
    i16b(&mut b, h.sform_code);
    for v in [h.quatern_b, h.quatern_c, h.quatern_d, h.quatern_x, h.quatern_y, h.quatern_z] {
        f32b(&mut b, v);
    }
    for row in [h.srow_x, h.srow_y, h.srow_z] {
        row.iter().for_each(|&v| f32b(&mut b, v));
    }
    b.extend_from_slice(&h.intent_name);
    b.extend_from_slice(b"n+1\0");
    b.extend_from_slice(&[0u8; 4]);
    b
}

/// A float32 NIfTI written volume by volume (gzip level 1 for .gz, as nibabel).
struct VolWriter {
    w: Box<dyn Write>,
}

impl VolWriter {
    fn create(path: &Path, h: &NiftiHeader, dims: &[usize]) -> Result<VolWriter> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let f = BufWriter::new(File::create(path).with_context(|| format!("Failed to create {}", path.display()))?);
        let mut w: Box<dyn Write> = if path.to_string_lossy().ends_with(".gz") {
            Box::new(flate2::write::GzEncoder::new(f, flate2::Compression::new(1)))
        } else {
            Box::new(f)
        };
        w.write_all(&header_bytes(h, dims))?;
        Ok(VolWriter { w })
    }

    fn put(&mut self, v: &A3) -> Result<()> {
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for x in v.t().iter() {
            bytes.extend_from_slice(&(*x as f32).to_le_bytes());
        }
        self.w.write_all(&bytes)?;
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.w.flush()?;
        Ok(())
    }
}

fn write_3d(path: &Path, h: &NiftiHeader, v: &A3) -> Result<()> {
    let d = v.dim();
    let mut w = VolWriter::create(path, h, &[d.0, d.1, d.2])?;
    w.put(v)?;
    w.finish()
}

// ---------------------------------------------------------------- sidecars and phase encoding

fn stem(p: &Path) -> String {
    let s = p.to_string_lossy().to_string();
    s.strip_suffix(".nii.gz").or_else(|| s.strip_suffix(".nii")).unwrap_or(&s).to_string()
}

fn name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
}

fn sidecar(p: &Path) -> Result<serde_json::Value> {
    let j = PathBuf::from(stem(p) + ".json");
    if !j.exists() {
        return Ok(serde_json::Value::Null);
    }
    let txt = std::fs::read_to_string(&j)?;
    serde_json::from_str(&txt).with_context(|| format!("Failed to parse {}", j.display()))
}

fn json_str(m: &serde_json::Value, k: &str) -> Option<String> {
    m.get(k).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(String::from)
}

fn json_f64(m: &serde_json::Value, k: &str) -> Option<f64> {
    m.get(k).and_then(|v| v.as_f64()).filter(|&v| v != 0.0)
}

/// TotalReadoutTime (s) of a sidecar, else EffectiveEchoSpacing x (n_pe - 1) (BIDS).
fn readout_of(m: &serde_json::Value, n_pe: usize) -> Option<f64> {
    json_f64(m, "TotalReadoutTime").or_else(|| json_f64(m, "EffectiveEchoSpacing").map(|e| e * (n_pe as f64 - 1.0)))
}

/// The unit world (RAS+) vector of phase-encoding axis `ax` with polarity `pol`.
fn world(ax: usize, pol: f64, aff: &Mat4) -> [f64; 3] {
    let c = [aff[0][ax] * pol, aff[1][ax] * pol, aff[2][ax] * pol];
    let n = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    c.map(|v| v / n)
}

fn label(w: &[f64; 3]) -> &'static str {
    let i = (0..3).max_by(|&a, &b| w[a].abs().partial_cmp(&w[b].abs()).unwrap()).unwrap();
    [["RL", "LR"], ["AP", "PA"], ["SI", "IS"]][i][(w[i] > 0.0) as usize]
}

fn dotv(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// (axis, polarity) of the voxel axis along world direction w; an oblique grid must leave it unambiguous.
fn from_world(w: &[f64; 3], aff: &Mat4, f: &Path) -> Result<(usize, f64)> {
    let c: Vec<f64> = (0..3).map(|j| dotv(&world(j, 1.0, aff), w)).collect();
    let ax = (0..3).max_by(|&a, &b| c[a].abs().partial_cmp(&c[b].abs()).unwrap()).unwrap();
    if c[ax].abs() < MIN_COS {
        bail!("{}: direction {} is {:.0} degrees from the nearest voxel axis; give --pe as a voxel axis (i / j / k)", name(f), label(w), c[ax].abs().acos().to_degrees());
    }
    Ok((ax, if c[ax] > 0.0 { 1.0 } else { -1.0 }))
}

/// (axis, polarity) from a voxel code ('j-'; early-BIDS 'y-') or an anatomical direction ('AP').
fn parse_pe(code: &str, aff: &Mat4, f: &Path) -> Result<(usize, f64)> {
    let s = code.trim();
    if let Some((_, w)) = ANAT.iter().find(|(n, _)| n.eq_ignore_ascii_case(s)) {
        return from_world(w, aff, f);
    }
    let ok = ["i", "i-", "j", "j-", "k", "k-", "x", "x-", "y", "y-", "z", "z-"].contains(&s);
    if !ok {
        bail!("{}: phase-encoding direction {:?} (i, i-, j, j-, k, k- or AP, PA, LR, RL, IS, SI)", name(f), code);
    }
    let ax = "ijkxyz".find(s.chars().next().unwrap()).unwrap() % 3;
    Ok((ax, if s.ends_with('-') { -1.0 } else { 1.0 }))
}

/// (axis, polarity) of a scan in its voxel axes: JSON, else `given`, else `default` (a world direction), else the
/// header's dim_info.
fn phase(n: &Nii, meta: &serde_json::Value, given: Option<&str>, default: Option<[f64; 3]>) -> Result<(usize, f64)> {
    if let Some(code) = json_str(meta, "PhaseEncodingDirection").or(given.map(String::from)) {
        return parse_pe(&code, &n.aff, &n.path);
    }
    if let Some(w) = default {
        return from_world(&w, &n.aff, &n.path);
    }
    let ph = (n.header.dim_info >> 2) & 3;
    if ph == 0 {
        bail!("{}: no PhaseEncodingDirection (JSON), --pe or dim_info phase axis", name(&n.path));
    }
    eprintln!("  Warning: {}: phase axis {} from dim_info, polarity assumed positive (the field sign and the series correction depend on it)", name(&n.path), ph - 1);
    Ok((ph as usize - 1, 1.0))
}

/// The mean of the b < 50 volumes (from the .bval) or of all volumes.
fn b0(n: &Nii) -> Result<A3> {
    let nv = n.nvol();
    let mut use_v = vec![true; nv];
    let bv = PathBuf::from(stem(&n.path) + ".bval");
    if nv > 1 && bv.exists() {
        let b: Vec<f64> = std::fs::read_to_string(&bv)?.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        if b.len() == nv && b.iter().any(|&x| x < 50.0) {
            use_v = b.iter().map(|&x| x < 50.0).collect();
        }
    }
    let mut total = A3::zeros(n.dim3());
    for (v, keep) in n.volumes()?.zip(&use_v) {
        let v = v?;
        if *keep {
            total += &v;
        }
    }
    Ok(total / use_v.iter().filter(|&&k| k).count() as f64)
}

fn same_grid(a: &Nii, b: &Nii) -> bool {
    a.shape == b.shape && (0..16).all(|k| (a.aff[k / 4][k % 4] - b.aff[k / 4][k % 4]).abs() <= 1e-3)
}

/// Every volume of `img` unwarped by sign * disp (and M) onto `ref_`'s grid in one cubic resampling, written volume
/// by volume to `path`.
fn corrected(img: &Nii, ref_: &Nii, disp: &A3, ax: usize, m: Option<&Mat4>, sign: f64, path: &Path) -> Result<()> {
    let (idx, jac) = duope::unwarp_map(disp, &ref_.aff, ax, m, sign, Some(&img.aff));
    let hdr = if same_grid(img, ref_) { &img.header } else { &ref_.header };
    let mut dims = ref_.shape.to_vec();
    dims.extend(&img.extra);
    let mut w = VolWriter::create(path, hdr, &dims)?;
    for v in img.volumes()? {
        w.put(&duope::sample_map(&duope::Spline::new(&v?), &idx, &jac))?;
    }
    w.finish()
}

fn out_path(input: &Path, output: Option<&Path>, postfix: &str) -> PathBuf {
    let base = name(input);
    let st = base.strip_suffix(".nii.gz").or_else(|| base.strip_suffix(".nii")).unwrap_or(&base).to_string();
    let parent = output.map(Path::to_path_buf).unwrap_or_else(|| input.parent().unwrap_or(Path::new(".")).to_path_buf());
    parent.join(format!("{st}_{postfix}.nii.gz"))
}

// ---------------------------------------------------------------- one pair

type Row = Vec<(&'static str, String)>;

fn fmt_num(v: f64, digits: i32) -> String {
    let p = 10f64.powi(digits);
    let r = (v * p).round() / p;
    let s = format!("{r}");
    if s.contains('.') || s.contains('e') { s } else { format!("{s}.0") }
}

fn one(f: &Path, r: &Path, s: &Settings) -> Result<Row> {
    let t0 = std::time::Instant::now();
    let (fi, ri) = (Nii::open(f)?, Nii::open(r)?);
    let (fm, rm) = (sidecar(f)?, sidecar(r)?);
    let (ax_f, pol_f) = phase(&fi, &fm, s.pe.as_deref(), None)?;
    let w_f = world(ax_f, pol_f, &fi.aff);
    let (ax_r, pol_r) = phase(&ri, &rm, None, Some(w_f.map(|v| -v)))?;
    let w_r = world(ax_r, pol_r, &ri.aff);
    if dotv(&w_f, &w_r) > -0.9 {
        let code = |a: usize, p: f64| format!("{}{}", AXES[a], if p < 0.0 { "-" } else { "" });
        bail!("{} / {}: not a reverse phase-encoding pair (directions {} / {}: {} / {} in their voxel axes)", name(f), name(r), label(&w_f), label(&w_r), code(ax_f, pol_f), code(ax_r, pol_r));
    }
    let trt = readout_of(&fm, fi.shape[ax_f]).or(s.readout);
    let fwd = b0(&fi)?;
    let mut rev = b0(&ri)?;
    if !same_grid(&ri, &fi) {
        rev = linear_transform(&rev, &mat_mul(&mat_inv(&ri.aff), &fi.aff), fi.dim3());
    }
    // the core takes the positive polarity first; d is its displacement, M moves the negative-polarity scan
    let (pos, neg) = if pol_f > 0.0 { (&fwd, &rev) } else { (&rev, &fwd) };
    let (d, m, info) = duope::estimate(pos, neg, &fi.aff, ax_f, s.joint);
    let (m_f, m_r) = if pol_f > 0.0 { (None, Some(&m)) } else { (Some(&m), None) };
    let out = s.output.as_deref();
    let fpath = out_path(f, out, "duope");
    corrected(&fi, &fi, &d, ax_f, m_f, pol_f, &fpath)?;
    corrected(&ri, &fi, &d, ax_f, m_r, -pol_f, &out_path(r, out, "duope"))?;
    if s.save_field {
        let t = trt.ok_or_else(|| anyhow!("{}: --save-field needs the readout time (JSON TotalReadoutTime or --readout)", name(f)))?;
        write_3d(&out_path(f, out, "duope_field"), &fi.header, &(&d / t))?;
    }
    if s.save_disp {
        write_3d(&out_path(f, out, "duope_disp"), &fi.header, &(&d * pol_f))?;
    }

    let mut moved = Vec::new();
    if s.series_pe.len() > 1 && s.series_pe.len() != s.series.len() {
        bail!("--series-pe: one per series (or one for all)");
    }
    let fixed = if !s.series.is_empty() && s.series_align { Some(b0(&Nii::open(&fpath)?)?) } else { None };
    for (k, sf) in s.series.iter().enumerate() {
        let si = Nii::open(sf)?;
        let sm = sidecar(sf)?;
        let spe = s.series_pe.get(if s.series_pe.len() == 1 { 0 } else { k }).map(String::as_str);
        let (ax_s, pol_s) = phase(&si, &sm, spe, Some(w_f))?;
        let w_s = world(ax_s, pol_s, &si.aff);
        if dotv(&w_s, &w_f).abs() < 0.9 {
            bail!("{}: phase-encoding direction {}, the field is along {}", name(sf), label(&w_s), label(&w_f));
        }
        let trt_s = readout_of(&sm, si.shape[ax_s]);
        if !same_grid(&si, &fi) && !(trt_s.is_some() && trt.is_some()) {
            eprintln!("  Warning: {}: another grid than the forward scan and no readout time for both; the displacement in voxels is assumed equal", name(sf));
        }
        let scale = match (trt_s, trt) {
            (Some(a), Some(b)) => a / b,
            _ => 1.0,
        };
        let ds = &d * scale;
        let ms = match &fixed {
            Some(fx) => {
                let ms = duope::series_motion(&b0(&si)?, &si.aff, fx, &fi.aff, &ds, ax_s, pol_s);
                let c = [0, 1, 2].map(|a| (si.shape[a] as f64 - 1.0) / 2.0);
                let cw: [f64; 3] = std::array::from_fn(|i| (0..3).map(|j| si.aff[i][j] * c[j]).sum::<f64>() + si.aff[i][3]);
                let (mm, deg) = duope::motion_size(&ms, &cw);
                moved.push(format!("{mm:.2}/{deg:.2}"));
                Some(ms)
            }
            None => None,
        };
        let dsg = duope::field_on(&ds, &fi.aff, &si.aff, si.dim3(), ms.as_ref());
        corrected(&si, &si, &dsg, ax_s, None, pol_s, &out_path(sf, out, "duope"))?;
    }
    let secs = t0.elapsed().as_secs_f64();
    eprintln!("  {}: motion {:.2} mm / {:.2} deg, joint {}, {:.1} s", name(f), info.motion_mm, info.motion_deg, info.joint, secs);
    let py_bool = |b: bool| if b { "True" } else { "False" }.to_string();
    Ok(vec![
        ("forward", f.to_string_lossy().to_string()),
        ("reverse", r.to_string_lossy().to_string()),
        ("phase_axis", AXES[ax_f].to_string()),
        ("forward_polarity", format!("{}", pol_f as i32)),
        ("forward_direction", label(&w_f).to_string()),
        ("readout_s", trt.map(|t| format!("{t}")).unwrap_or_default()),
        ("motion_mm", fmt_num(info.motion_mm, 3)),
        ("motion_deg", fmt_num(info.motion_deg, 3)),
        ("joint", py_bool(info.joint)),
        ("seconds", fmt_num(secs, 1)),
        ("series", s.series.iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>().join(";")),
        ("series_motion_mm_deg", moved.join(";")),
    ])
}

// ---------------------------------------------------------------- summary CSV

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() }
}

fn csv_split(line: &str) -> Vec<String> {
    let (mut out, mut cur, mut q) = (Vec::new(), String::new(), false);
    let mut it = line.chars().peekable();
    while let Some(c) = it.next() {
        match (c, q) {
            ('"', true) if it.peek() == Some(&'"') => {
                cur.push('"');
                it.next();
            }
            ('"', _) => q = !q,
            (',', false) => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out
}

fn abs(p: &str) -> PathBuf {
    std::path::absolute(p).unwrap_or_else(|_| PathBuf::from(p))
}

/// Add the rows to the folder's summary; a pair run again replaces its earlier row.
fn write_summary(folder: &Path, rows: &[Row]) -> Result<()> {
    std::fs::create_dir_all(folder)?;
    let path = folder.join(SUMMARY_NAME);
    let cols: Vec<&str> = rows[0].iter().map(|(k, _)| *k).collect();
    let key = |f: &str, r: &str| (abs(f), abs(r));
    let done: Vec<(PathBuf, PathBuf)> = rows.iter().map(|r| key(&r[0].1, &r[1].1)).collect();
    let mut lines = vec![cols.join(",")];
    if path.exists() {
        let txt = std::fs::read_to_string(&path)?;
        let mut it = txt.lines();
        if let Some(head) = it.next() {
            let names = csv_split(head);
            for l in it.filter(|l| !l.is_empty()) {
                let v = csv_split(l);
                let get = |c: &str| names.iter().position(|n| n == c).and_then(|i| v.get(i)).cloned().unwrap_or_default();
                if done.contains(&key(&get("forward"), &get("reverse"))) {
                    continue;
                }
                lines.push(cols.iter().map(|c| csv_field(&get(c))).collect::<Vec<_>>().join(","));
            }
        }
    }
    for r in rows {
        lines.push(r.iter().map(|(_, v)| csv_field(v)).collect::<Vec<_>>().join(","));
    }
    std::fs::write(&path, lines.join("\r\n") + "\r\n")?;
    Ok(())
}

/// Every pair (forward, reverse) of `inputs`; a failing pair of several is reported and skipped.
pub fn run(inputs: &[String], s: &Settings) -> Result<()> {
    if inputs.len() < 2 || inputs.len() % 2 != 0 {
        bail!("--duope takes forward / reverse pairs: tigerbx --duope FORWARD REVERSE [FORWARD REVERSE ...]");
    }
    if !s.series.is_empty() && inputs.len() > 2 {
        bail!("--series goes with one pair");
    }
    let pairs: Vec<(PathBuf, PathBuf)> = inputs.chunks(2).map(|c| (PathBuf::from(&c[0]), PathBuf::from(&c[1]))).collect();
    let mut rows = Vec::new();
    let mut failed = Vec::new();
    for (f, r) in &pairs {
        eprintln!("duope: {} / {}", name(f), name(r));
        match one(f, r, s) {
            Ok(row) => rows.push(row),
            Err(e) if pairs.len() > 1 => {
                eprintln!("  FAILED {}: {e:#}", name(f));
                failed.push(name(f));
            }
            Err(e) => return Err(e),
        }
    }
    if !rows.is_empty() {
        let folder = match &s.output {
            Some(o) => o.clone(),
            None => abs(&pairs[0].0.to_string_lossy()).parent().unwrap().to_path_buf(),
        };
        write_summary(&folder, &rows)?;
    }
    if !failed.is_empty() {
        eprintln!("Warning: {} of {} pairs failed and were skipped: {}", failed.len(), pairs.len(), failed.join(", "));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::duope::tests::{phantom, distort, mae};

    fn header(aff: &Mat4, dims: &[usize]) -> NiftiHeader {
        let mut h = NiftiHeader::default();
        h.dim = [1; 8];
        h.dim[0] = dims.len() as u16;
        for (a, &d) in dims.iter().enumerate() {
            h.dim[a + 1] = d as u16;
        }
        for a in 0..3 {
            h.pixdim[a + 1] = (0..3).map(|i| aff[i][a] * aff[i][a]).sum::<f64>().sqrt() as f32;
        }
        h.sform_code = 1;
        h.srow_x = aff[0].map(|v| v as f32);
        h.srow_y = aff[1].map(|v| v as f32);
        h.srow_z = aff[2].map(|v| v as f32);
        h.dim_info = 1 << 2 | 1;
        h
    }

    fn aff() -> Mat4 {
        [[-2.5, 0.0, 0.0, 10.0], [0.0, 2.5, 0.0, -3.0], [0.0, 0.0, 2.5, 7.0], [0.0, 0.0, 0.0, 1.0]]
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("tigerbx_duope_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn save(path: &Path, vols: &[A3], pe: Option<&str>, trt: Option<f64>) {
        let d = vols[0].dim();
        let dims: Vec<usize> = if vols.len() > 1 { vec![d.0, d.1, d.2, vols.len()] } else { vec![d.0, d.1, d.2] };
        let mut w = VolWriter::create(path, &header(&aff(), &dims), &dims).unwrap();
        for v in vols {
            w.put(v).unwrap();
        }
        w.finish().unwrap();
        if let Some(pe) = pe {
            let mut j = serde_json::json!({ "PhaseEncodingDirection": pe });
            if let Some(t) = trt {
                j["TotalReadoutTime"] = serde_json::json!(t);
            }
            std::fs::write(stem(path) + ".json", j.to_string()).unwrap();
        }
    }

    fn read_all(p: &Path) -> Vec<A3> {
        Nii::open(p).unwrap().volumes().unwrap().map(|v| v.unwrap()).collect()
    }

    #[test]
    fn volume_writer_round_trip() {
        let d = tmp("io");
        let a = A3::from_shape_fn((5, 6, 4), |(i, j, k)| (i * 100 + j * 10 + k) as f64 * 0.5 - 3.0);
        for ext in ["nii", "nii.gz"] {
            let p = d.join(format!("a.{ext}"));
            save(&p, &[a.clone(), &a * 2.0], None, None);
            let n = Nii::open(&p).unwrap();
            assert_eq!((n.shape, n.extra.clone()), ([5, 6, 4], vec![2]));
            assert!((0..16).all(|k| (n.aff[k / 4][k % 4] - aff()[k / 4][k % 4]).abs() < 1e-6));
            let v = read_all(&p);
            assert!(v[0] == a && v[1] == &a * 2.0);
            assert_eq!(n.header.dim_info, 1 << 2 | 1);
        }
    }

    #[test]
    fn phase_encoding_codes_and_anatomical_directions() {
        let (a, f) = (aff(), Path::new("x.nii"));
        // voxel i runs to -x (towards the right), j to +y (anterior), k to +z
        assert_eq!(parse_pe("LR", &a, f).unwrap(), (0, -1.0));
        assert_eq!(parse_pe("rl", &a, f).unwrap(), (0, 1.0));
        assert_eq!(parse_pe("PA", &a, f).unwrap(), (1, 1.0));
        assert_eq!(parse_pe("AP", &a, f).unwrap(), (1, -1.0));
        assert_eq!(parse_pe("y-", &a, f).unwrap(), (1, -1.0));
        assert_eq!(parse_pe("SI", &a, f).unwrap(), (2, -1.0));
        assert!(parse_pe("up", &a, f).is_err());
        let (c, s) = (std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2);
        let oblique = [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]];
        assert!(parse_pe("AP", &oblique, f).unwrap_err().to_string().contains("degrees from the nearest voxel axis"));
    }

    #[test]
    fn pipeline_outputs_signs_series_and_summary() {
        let (u, d, inside) = phantom();
        let (fwd, rev) = (distort(&u, &d), distort(&u, &d.mapv(|v| -v)));
        let dir = tmp("pipe");
        let (fa, fp, sr) = (dir.join("a_epi.nii.gz"), dir.join("p_epi.nii.gz"), dir.join("s_dwi.nii.gz"));
        save(&fa, &[fwd.clone()], Some("j"), Some(0.05));
        save(&fp, &[rev.clone()], Some("j-"), None);
        save(&sr, &[rev.clone(), &rev * 0.5], Some("j-"), None);
        let out = dir.join("out");
        let s = Settings {
            output: Some(out.clone()),
            series: vec![sr.clone()],
            pe: None,
            readout: None,
            series_pe: vec![],
            series_align: true,
            joint: None,
            save_field: true,
            save_disp: true,
        };
        let args: Vec<String> = [&fa, &fp].iter().map(|p| p.to_string_lossy().to_string()).collect();
        run(&args, &s).unwrap();
        let disp = &read_all(&out.join("a_epi_duope_disp.nii.gz"))[0];
        assert!(mae(disp, &d, &inside) < 0.15);
        let field = &read_all(&out.join("a_epi_duope_field.nii.gz"))[0];
        assert!(field.iter().zip(disp.iter()).all(|(f, d)| (f * 0.05 - d).abs() < 1e-4));
        let corr = |a: &A3| {
            let (x, y): (Vec<f64>, Vec<f64>) = a.iter().zip(u.iter()).zip(inside.iter()).filter(|(_, &m)| m).map(|((a, b), _)| (*a, *b)).unzip();
            let (mx, my) = (x.iter().sum::<f64>() / x.len() as f64, y.iter().sum::<f64>() / y.len() as f64);
            let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
            sxy / (x.iter().map(|a| (a - mx).powi(2)).sum::<f64>() * y.iter().map(|b| (b - my).powi(2)).sum::<f64>()).sqrt()
        };
        let f_out = &read_all(&out.join("a_epi_duope.nii.gz"))[0];
        let p_out = &read_all(&out.join("p_epi_duope.nii.gz"))[0];
        // 1 - correlation with the undistorted image down at least 5x
        let better = |out: &A3, raw: &A3| 1.0 - corr(out) < 0.2 * (1.0 - corr(raw));
        assert!(better(f_out, &fwd) && better(p_out, &rev), "{} {} {} {}", corr(f_out), corr(&fwd), corr(p_out), corr(&rev));
        let s_out = read_all(&out.join("s_dwi_duope.nii.gz"));
        assert!(s_out.len() == 2 && better(&s_out[0], &rev));
        let csv = std::fs::read_to_string(out.join(SUMMARY_NAME)).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert!(lines[0].starts_with("forward,reverse,phase_axis,forward_polarity,forward_direction,readout_s"));
        assert_eq!(lines.len(), 2);
        let row = csv_split(lines[1]);
        assert_eq!((row[2].as_str(), row[3].as_str(), row[4].as_str(), row[5].as_str()), ("j", "1", "PA", "0.05"));
        // run again without the series: the pair's row is replaced; the reverse given first: displacement negated
        run(&args, &Settings { series: vec![], ..s }).unwrap();
        let csv = std::fs::read_to_string(out.join(SUMMARY_NAME)).unwrap();
        assert_eq!(csv.lines().count(), 2);
        assert_eq!(csv_split(csv.lines().nth(1).unwrap())[10], "");
        // not a reverse pair
        let same = dir.join("same.nii.gz");
        save(&same, &[rev.clone()], Some("j"), None);
        let bad: Vec<String> = [&fa, &same].iter().map(|p| p.to_string_lossy().to_string()).collect();
        let s2 = Settings { output: Some(dir.join("o2")), series: vec![], pe: None, readout: None, series_pe: vec![], series_align: true, joint: None, save_field: false, save_disp: false };
        assert!(run(&bad, &s2).unwrap_err().to_string().contains("not a reverse phase-encoding pair"));
    }

    #[test]
    fn csv_quotes_round_trip() {
        let v = ["a,b", "x\"y", "plain"].map(String::from);
        let line = v.iter().map(|s| csv_field(s)).collect::<Vec<_>>().join(",");
        assert_eq!(csv_split(&line), v.to_vec());
    }
}
