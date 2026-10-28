//! tigerbx.template (tigerbx/pipelines/template.py) as a native port: a population template on the TigerBx template
//! grid from a group of images. Per input: brain (tBX, or the non-zero voxels of a skull-stripped input), numreg
//! affine to the bundled template, intensity / brain median; then `iters` rounds of VFA to the current template,
//! voxel-wise median (or mean) and the inverse of the mean displacement; the last round samples every subject once
//! from its input grid. The numreg port is not bit-identical to Python (rust/README.md), so the template is close to,
//! not equal to, the Python one.
use anyhow::{bail, Context, Result};
use ndarray::{Array3, Axis, Zip};
use ort::session::Session;
use rayon::prelude::*;
use std::path::Path;

use crate::bx;
use crate::nifti_io;
use crate::numreg;
use crate::reg::{self, Template};
use crate::resample::{self, mat_inv, mat_mul, Kind, Mat4, Volume};

/// Options of the template pipeline (tigerbx.template arguments).
pub struct Options {
    pub iters: usize,
    pub median: bool,
    /// inputs are skull-stripped: the brain is their non-zero voxels (mask='none')
    pub stripped: bool,
    /// write template_tseg (majority vote of the subjects' TSEG, deep GM from DGM12)
    pub labels: bool,
    pub contrast: String,
}

/// One prepared input.
struct Subject {
    brain: Array3<f64>,
    mask: Array3<f64>,
    tseg: Option<Array3<f64>>,
    /// template voxel -> input voxel (inverse(input affine) . M . template affine)
    v: Mat4,
    scale: f64,
    vfa_in: Array3<f64>,
    affined: Array3<f64>,
}

/// numreg.resample coordinates: template voxel p (+ warp) -> input voxel.
fn at(v: &Mat4, p: [f64; 3]) -> [f64; 3] {
    let mut c = [0.0; 3];
    for (r, cr) in c.iter_mut().enumerate() {
        *cr = v[r][0] * p[0] + v[r][1] * p[1] + v[r][2] * p[2] + v[r][3];
    }
    c
}

fn min_max(x: &Array3<f64>) -> Array3<f64> {
    let lo = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    x.mapv(|v| (v - lo) / (hi - lo).max(1e-12))
}

/// Median of the brain voxels (> 0).
fn median_pos(x: &Array3<f64>) -> f64 {
    let mut v: Vec<f64> = x.iter().cloned().filter(|&a| a > 0.0).collect();
    median(&mut v)
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let n = v.len();
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

/// Warp component a at template voxel (i, j, k): w[a] = grid[a] - index (float32, as TigerBx).
fn disp(w: &[f32], n: usize, dim: (usize, usize, usize), a: usize, i: usize, j: usize, k: usize) -> f32 {
    w[a * n + (i * dim.1 + j) * dim.2 + k]
}

fn prepare(f: &Path, t: &Template, opts: &Options, tbx: Option<&mut Session>, tseg: Option<&mut Session>,
           dgm: Option<&mut Session>) -> Result<Subject> {
    let (header, data) = nifti_io::read_nifti(f)?;
    let input_affine = resample::best_affine(&header);
    let (brain, mask, labels) = if opts.stripped {
        let shape: Vec<usize> = data.shape().iter().cloned().filter(|&n| n != 1).collect();
        if shape.len() != 3 {
            bail!("Expected a 3D image, got shape {:?}", data.shape());
        }
        let d = data.to_shape((shape[0], shape[1], shape[2]))?.into_dimensionality::<ndarray::Ix3>()?.to_owned();
        let d = d.mapv(|v| v as f32 as f64);
        let mask = d.mapv(|v| if v != 0.0 { 1.0 } else { 0.0 });
        // tigerbx.template(mask='none', labels): tigerbx.run('t') (TSEG of its own tBX, no deep-GM merge)
        let labels = match (tbx, tseg) {
            (Some(b), Some(s)) => Some(bx::run_bx(&data, &header, b, Some(s), None, false)?.tseg.context("no TSEG")?),
            _ => None,
        };
        (d, mask, labels.map(|l| l.into_dimensionality::<ndarray::Ix3>().map(|a| a.mapv(|v| v as f64))).transpose()?)
    } else {
        let merge = tseg.is_some();
        let r = bx::run_bx(&data, &header, tbx.context("tBX session missing")?, tseg, dgm, merge)?;
        let mask = r.mask.into_dimensionality::<ndarray::Ix3>()?.mapv(|v| if v > 0 { 1.0 } else { 0.0 });
        let brain = r.brain.into_dimensionality::<ndarray::Ix3>()?.mapv(|v| v as f32 as f64);
        let labels = r.tseg.map(|l| l.into_dimensionality::<ndarray::Ix3>().map(|a| a.mapv(|v| v as f64))).transpose()?;
        (brain, mask, labels)
    };
    if !mask.iter().any(|&m| m > 0.0) {
        bail!("empty brain mask");
    }
    let scale = median_pos(&brain);
    // numreg affine on the reordered brain (templatereg.affine)
    let moving = resample::reorder(Volume { data: brain.clone(), affine: input_affine, kind: Kind::Float32 });
    let m = numreg::register(&moving.data, &moving.affine, &t.vol.data, &t.vol.affine);
    let v = mat_mul(&mat_mul(&mat_inv(&input_affine), &m), &t.vol.affine);
    let dim = t.vol.data.dim();
    let lin = resample::sample(&brain, dim, false, |i, j, k| at(&v, [i as f64, j as f64, k as f64]));
    let vfa_in = min_max(&lin.mapv(|x| x as f32 as f64));
    let affined = lin.mapv(|x| x as f32 as f64 / scale);
    Ok(Subject { brain, mask, tseg: labels, v, scale, vfa_in, affined })
}

/// Voxel-wise median / mean over subjects.
fn stat(stack: &[Array3<f64>], median_stat: bool) -> Array3<f64> {
    let dim = stack[0].dim();
    let mut out = Array3::<f64>::zeros(dim);
    out.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut plane)| {
        let mut v = vec![0.0; stack.len()];
        for ((j, k), o) in plane.indexed_iter_mut() {
            for (s, a) in stack.iter().enumerate() {
                v[s] = a[[i, j, k]];
            }
            *o = if median_stat { median(&mut v) } else { v.iter().sum::<f64>() / v.len() as f64 };
        }
    });
    out
}

/// map_coordinates(order=1, mode='nearest') at one point.
fn linear_nearest(a: &Array3<f64>, c: [f64; 3]) -> f64 {
    let d = a.dim();
    let dims = [d.0, d.1, d.2];
    let cl: [f64; 3] = std::array::from_fn(|x| c[x].clamp(0.0, (dims[x] - 1) as f64));
    let i0: [usize; 3] = std::array::from_fn(|x| cl[x].floor() as usize);
    let f: [f64; 3] = std::array::from_fn(|x| cl[x] - i0[x] as f64);
    let i1: [usize; 3] = std::array::from_fn(|x| (i0[x] + 1).min(dims[x] - 1));
    let mut s = 0.0;
    for (p, wp) in [(i0[0], 1.0 - f[0]), (i1[0], f[0])] {
        for (q, wq) in [(i0[1], 1.0 - f[1]), (i1[1], f[1])] {
            for (r, wr) in [(i0[2], 1.0 - f[2]), (i1[2], f[2])] {
                s += wp * wq * wr * a[[p, q, r]];
            }
        }
    }
    s
}

/// templatereg.invert_warp: v with (p + v) + warp(p + v) = p, by 20 fixed-point iterations v = -warp(p + v).
fn invert_warp(w: &[Array3<f64>; 3]) -> [Array3<f64>; 3] {
    let mut v: [Array3<f64>; 3] = std::array::from_fn(|a| w[a].mapv(|x| -(x as f32) as f64));
    let dim = w[0].dim();
    for _ in 0..20 {
        let next: Vec<Array3<f64>> = (0..3)
            .map(|a| {
                let mut o = Array3::<f64>::zeros(dim);
                Zip::indexed(&mut o).par_for_each(|(i, j, k), o| {
                    let c = [i as f64 + v[0][[i, j, k]], j as f64 + v[1][[i, j, k]], k as f64 + v[2][[i, j, k]]];
                    *o = -linear_nearest(&w[a], c);
                });
                o
            })
            .collect();
        v = [next[0].clone(), next[1].clone(), next[2].clone()];
    }
    v.map(|a| a.mapv(|x| x as f32 as f64))
}

/// templatereg.warp_on_grid: out(p) = data(p + warp(p)), map_coordinates mode 'constant' (order 0, 1 or 3).
fn warp_on_grid(data: &Array3<f64>, w: &[Array3<f64>; 3], order: usize) -> Array3<f64> {
    let dim = data.dim();
    let c = |i: usize, j: usize, k: usize| [i as f64 + w[0][[i, j, k]], j as f64 + w[1][[i, j, k]], k as f64 + w[2][[i, j, k]]];
    match order {
        0 => resample::sample_nearest(data, dim, c),
        1 => resample::sample(data, dim, false, c),
        _ => resample::sample(data, dim, true, c),
    }
}

/// Majority label per voxel among the non-zero labels inside `mask` (templatereg _vote).
fn vote(stack: &[Array3<f64>], mask: &Array3<bool>) -> Array3<f64> {
    let mut labels: Vec<u8> = stack.iter().flat_map(|a| a.iter().map(|&v| v as u8)).filter(|&v| v > 0).collect();
    labels.sort_unstable();
    labels.dedup();
    let dim = stack[0].dim();
    let mut out = Array3::<f64>::zeros(dim);
    Zip::indexed(&mut out).par_for_each(|(i, j, k), o| {
        if !mask[[i, j, k]] {
            return;
        }
        let mut best = (0usize, 0u8);
        for &l in &labels {
            let n = stack.iter().filter(|a| a[[i, j, k]] as u8 == l).count();
            if n > best.0 {
                best = (n, l);
            }
        }
        *o = best.1 as f64;
    });
    out
}

fn corr(a: &Array3<f64>, b: &Array3<f64>) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.sum() / n, b.sum() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b.iter()) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    sab / (saa * sbb).sqrt()
}

/// Build the template of `files` into `out_dir` (template, template_mask, template_tseg when labels, template_log.json).
#[allow(clippy::too_many_arguments)]
pub fn run(files: &[std::path::PathBuf], out_dir: &Path, t: &Template, opts: &Options, mut tbx: Option<&mut Session>,
           mut tseg: Option<&mut Session>, mut dgm: Option<&mut Session>, vfa: &mut Session) -> Result<()> {
    let t0 = std::time::Instant::now();
    let dim = t.vol.data.dim();
    let n = dim.0 * dim.1 * dim.2;
    let mut subs = Vec::new();
    let mut failed = Vec::new();
    for (i, f) in files.iter().enumerate() {
        println!("[prep {}/{}] {}", i + 1, files.len(), f.display());
        match prepare(f, t, opts, tbx.as_deref_mut(), tseg.as_deref_mut(), dgm.as_deref_mut()) {
            Ok(s) => subs.push(s),
            Err(e) => {
                eprintln!("  ERROR: {:#}", e);
                failed.push(f.file_name().unwrap_or_default().to_string_lossy().to_string());
            }
        }
    }
    if subs.len() < 2 {
        bail!("only {} input(s) could be prepared", subs.len());
    }
    let ns = subs.len() as f64;
    let mut cur = stat(&subs.iter().map(|s| s.affined.clone()).collect::<Vec<_>>(), opts.median);
    let mut rounds = Vec::new();
    let mut inv: [Array3<f64>; 3] = std::array::from_fn(|_| Array3::zeros(dim));
    let (mut masks, mut tsegs) = (Vec::new(), Vec::new());
    for k in 1..=opts.iters {
        let last = k == opts.iters;
        let fixed = min_max(&cur);
        let mut warped = Vec::with_capacity(subs.len());
        let mut mean_w: [Array3<f64>; 3] = std::array::from_fn(|_| Array3::zeros(dim));
        masks.clear();
        tsegs.clear();
        for (si, s) in subs.iter().enumerate() {
            print!("\r  round {k}/{}: {}/{}", opts.iters, si + 1, subs.len());
            std::io::Write::flush(&mut std::io::stdout()).ok();
            let grid = reg::run_vfa(vfa, &s.vfa_in, &fixed)?;
            // w = grid - identity (float32)
            let w: Vec<f32> = (0..3 * n)
                .map(|p| {
                    let (a, q) = (p / n, p % n);
                    let id = [q / (dim.1 * dim.2), (q / dim.2) % dim.1, q % dim.2][a] as f32;
                    grid[p] - id
                })
                .collect();
            let wa: [Array3<f64>; 3] = std::array::from_fn(|a| {
                Array3::from_shape_fn(dim, |(i, j, l)| disp(&w, n, dim, a, i, j, l) as f64)
            });
            for a in 0..3 {
                mean_w[a].zip_mut_with(&wa[a], |m, &x| *m += (x as f32 / ns as f32) as f64);
            }
            if last {
                // one sample from the input grid at the affine + warp; masks / labels from it too
                let c = |i: usize, j: usize, l: usize| {
                    let p = [i as f64 + wa[0][[i, j, l]], j as f64 + wa[1][[i, j, l]], l as f64 + wa[2][[i, j, l]]];
                    at(&s.v, p)
                };
                let lo = s.brain.iter().cloned().fold(0.0f64, f64::min);
                let hi = s.brain.iter().cloned().fold(0.0f64, f64::max);
                let b = resample::sample(&s.brain, dim, true, c).mapv(|x| x.clamp(lo, hi) / s.scale);
                warped.push(b);
                masks.push(resample::sample(&s.mask, dim, false, c).mapv(|x| if x > 0.5 { 1.0 } else { 0.0 }));
                if let Some(l) = s.tseg.as_ref() {
                    tsegs.push(resample::sample_nearest(l, dim, c));
                }
            } else {
                warped.push(warp_on_grid(&s.affined, &wa, 1));
            }
        }
        println!();
        let avg = stat(&warped, opts.median);
        inv = invert_warp(&mean_w);
        let prev = std::mem::replace(&mut cur, warp_on_grid(&avg, &inv, if last { 3 } else { 1 }));
        let md = Zip::from(&mean_w[0]).and(&mean_w[1]).and(&mean_w[2]).fold(0.0, |s, a, b, c| s + (a * a + b * b + c * c).sqrt()) / n as f64;
        let r = format!("{{\"round\": {k}, \"mean_disp_vox\": {md}, \"corr_prev\": {}, \"sec\": {:.1}}}", corr(&prev, &cur),
                        t0.elapsed().as_secs_f64());
        println!("  round {k}: mean displacement {md:.3} vox");
        rounds.push(r);
    }
    let mean_mask = stat(&masks, false);
    let brain_mask = warp_on_grid(&mean_mask, &inv, 1).mapv(|x| x > 0.5);
    let mut out = Zip::from(&cur).and(&brain_mask).map_collect(|&c, &m| if m { (c.max(0.0) as f32) as f64 } else { 0.0 });
    let refv: Vec<f64> = t.vol.data.iter().cloned().filter(|&v| v > 0.0).collect();
    let mut refv = refv;
    let ref_med = median(&mut refv);
    let mut inside: Vec<f64> = Zip::from(&out).and(&brain_mask).fold(Vec::new(), |mut acc, &v, &m| {
        if m {
            acc.push(v);
        }
        acc
    });
    let gain = ref_med / median(&mut inside).max(1e-6);
    out.mapv_inplace(|v| ((v * gain) as f32) as f64);
    let mut header = t.header.clone();
    reg::set_affine(&mut header, &t.vol.affine);
    std::fs::create_dir_all(out_dir)?;
    let p = out_dir.join("template.nii.gz");
    nifti_io::write_nifti_f32(&p, &header, &out.mapv(|v| v as f32).into_dyn())?;
    println!("  -> {}", p.display());
    let p = out_dir.join("template_mask.nii.gz");
    nifti_io::write_nifti_u8(&p, &header, &brain_mask.mapv(|m| m as u8).into_dyn())?;
    println!("  -> {}", p.display());
    if opts.labels && !tsegs.is_empty() {
        let lab = vote(&tsegs, &mean_mask.mapv(|x| x > 0.5));
        let lab = warp_on_grid(&lab, &inv, 0);
        let lab = Zip::from(&lab).and(&brain_mask).map_collect(|&l, &m| if m { l as u8 } else { 0 });
        let p = out_dir.join("template_tseg.nii.gz");
        nifti_io::write_nifti_u8(&p, &header, &lab.into_dyn())?;
        println!("  -> {}", p.display());
    }
    let failed_json = failed.iter().map(|f| format!("{:?}", f)).collect::<Vec<_>>().join(", ");
    let log = format!(
        "{{\n \"n_inputs\": {},\n \"contrast\": {:?},\n \"iters\": {},\n \"stat\": \"{}\",\n \"labels\": {},\n \"rounds\": [\n  {}\n ],\n \"n_used\": {},\n \"failed\": [{}],\n \"sec\": {:.1}\n}}",
        files.len(), opts.contrast, opts.iters, if opts.median { "median" } else { "mean" }, opts.labels,
        rounds.join(",\n  "), subs.len(), failed_json, t0.elapsed().as_secs_f64()
    );
    std::fs::write(out_dir.join("template_log.json"), log)?;
    Ok(())
}
