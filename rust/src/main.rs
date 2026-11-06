mod bx;
mod inference;
mod duope;
mod duope_pipe;
mod dwimc;
mod inspect;
mod models;
mod nifti_io;
mod npz;
mod numreg;
mod onnx_prune;
mod postprocess;
mod reg;
mod resample;
mod template;
mod tissuenorm;

use anyhow::{bail, Context, Result};
use clap::Parser;
use std::path::{Path, PathBuf};

/// DGM12 (-d) is not embedded: downloaded on first use into the TigerBx model cache (src/models.rs)
const DGM_MODEL: &str = "mprage_dgm12_v002_opt.onnx";
/// Embedded models (compiled into the binary): TigerBx's defaults, from the modelhub release
static EMBEDDED_TBX: &[u8] = include_bytes!("../models/mprage_tbxdual_v007_opt.onnx");
static EMBEDDED_TSEG: &[u8] = include_bytes!("../models/mprage_tseg_v1_opt.onnx");
static EMBEDDED_TREG: &[u8] = include_bytes!("../models/mprage_treg_v003_opt.onnx");
static EMBEDDED_INSPECT: &[u8] = include_bytes!("../models/inspect_v003.onnx");
/// TigerBx's bundled registration template (same file as the Python package)
static EMBEDDED_TEMPLATE: &[u8] = include_bytes!("../../tigerbx/template/mni_template.nii.gz");
/// Its tseg: tissue priors of the gnorm intensity normalisation of _norm
static EMBEDDED_TEMPLATE_TSEG: &[u8] = include_bytes!("../../tigerbx/template/mni_template_tseg.nii.gz");

/// Postfixes TigerBx writes; inputs ending in one of them are earlier outputs, not new images.
const OUTPUT_POSTFIXES: [&str; 11] = ["tbx", "tbxmask", "tbxoutermask", "tbxouter", "tseg", "dgm", "affine", "norm", "duope", "duope_field", "duope_disp"];

/// tigerbx — TigerBx brain extraction, TSEG and template registration as a native executable
///
/// Supports glob patterns, directories, and multiple files:
///   tigerbx T1w.nii.gz                    (no option: _tbx)
///   tigerbx data/*.nii.gz -o output_dir --seg
///   tigerbx data/T1w_dir -o results/ --norm -t
///   tigerbx data/ -o results/ --inspect     (inspect_summary.csv: what each image is)
///   tigerbx --duope ap.nii.gz pa.nii.gz -o out/ --series bold.nii.gz   (EPI distortion correction)
///   tigerbx --dwimc dwi.nii.gz -o out/ --reverse pa.nii.gz             (DWI motion + eddy-current correction)
#[derive(Parser, Debug)]
#[command(version, verbatim_doc_comment, about = "TigerBx tBX, TSEG and template registration using ONNX Runtime (no option: _tbx)")]
struct Args {
    /// Input NIfTI files, glob patterns, or directories
    inputs: Vec<String>,

    /// Output directory (default: same directory as input)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Pipeline tBX + TSEG: _tbx, _tseg (as tigerbx.seg); output options add to it
    #[arg(long)]
    seg: bool,

    /// Pipeline template normalisation (tBX, numreg affine, tReg): _tbx, _norm, _warp.npz (as tigerbx.norm); output
    /// options add to it
    #[arg(long)]
    norm: bool,

    /// Pipeline affine-only template normalisation (tBX, numreg affine; keeps the brain's size and shape, e.g.
    /// ventricle volume, which the tReg warp of --norm removes): _tbx, _affine, _warp.npz (as tigerbx.affine);
    /// output options add to it
    #[arg(long)]
    affine: bool,

    /// Pipeline inspect: contrast, skull-stripped, EPI, orientation, truncation, motion and header / data checks of
    /// every input in inspect_summary.csv (as tigerbx.inspect); runs on its own
    #[arg(long)]
    inspect: bool,

    /// With --inspect: every probability and check value in the summary (as tigerbx.inspect(detail=True))
    #[arg(long)]
    detail: bool,

    /// Pipeline template: a population template of the inputs (at least 2) in -o: template, template_mask,
    /// template_tseg (T1) and template_log.json (as tigerbx.template); runs on its own
    #[arg(long)]
    template: bool,

    /// With --template: rounds of tReg to the current template
    #[arg(long, default_value_t = 6)]
    iters: usize,

    /// With --template: the inputs are skull-stripped (their brain is the non-zero voxels; no tBX)
    #[arg(long)]
    skull_stripped: bool,

    /// With --template: input contrast; t1 also builds template_tseg
    #[arg(long, default_value = "t1")]
    contrast: String,

    /// Save the brain-extracted image (_tbx)
    #[arg(short = 'b', long)]
    tbx: bool,

    /// Save the brain mask (_tbxmask; the brain, following its surface)
    #[arg(short = 'm', long)]
    tbxmask: bool,

    /// Save the outer brain mask (_tbxoutermask; the brain with the surrounding CSF, up to the inner surface of the
    /// skull; the TSEG input); with -b also _tbxouter
    #[arg(short = 'x', long)]
    tbxoutermask: bool,

    /// Save the TSEG segmentation (_tseg; 32 structures, FreeSurfer numbers; as tigerbx.run('t'))
    #[arg(short = 't', long)]
    tseg: bool,

    /// Save the DGM12 deep gray-matter segmentation (_dgm; labels 1-12, as tigerbx.run('d')); its model is
    /// downloaded on first use into the TigerBx model cache
    #[arg(short = 'd', long)]
    dgm: bool,

    /// Intensities of _norm and of the --affine pipeline's _affine: raw (the input's units; any contrast), minmax
    /// ([0, 1]) or gnorm (T1 only: tigerbx tissue_norm, CSF 1/3, GM 2/3, WM 1; else img / p99 with a warning)
    #[arg(long, value_enum, default_value_t = Intensity::Raw)]
    intensity: Intensity,

    /// Pipeline duope: EPI susceptibility distortion correction from reverse phase-encoding pairs (inputs: FORWARD
    /// REVERSE [FORWARD REVERSE ...]): <input>_duope for both scans and each --series, duope_summary.csv (as
    /// tigerbx.duope); runs on its own, no model
    #[arg(long)]
    duope: bool,

    /// Pipeline dwimc: head motion and eddy-current correction of DWI series (inputs; .bval / .bvec sidecars):
    /// <input>_dwimc, .bval, .bvec (rotated), .csv (as tigerbx.dwimc); with --reverse / --field also the
    /// susceptibility distortion; runs on its own, no model
    #[arg(long)]
    dwimc: bool,

    /// With --dwimc: b0 volume(s) acquired with the reversed phase encoding (one for all inputs, or one per input)
    #[arg(long)]
    reverse: Vec<PathBuf>,

    /// With --dwimc: instead of --reverse, a field in Hz (as --duope --save-field) and the readout time
    #[arg(long)]
    field: Option<PathBuf>,

    /// With --duope: a DWI / BOLD series (any grid) to correct with the field; repeatable, one pair only
    #[arg(long)]
    series: Vec<PathBuf>,

    /// With --duope / --dwimc: the (forward) scan's phase-encoding direction without a JSON sidecar: i, i-, j, j-,
    /// k, k- or AP, PA, LR, RL, IS, SI (from -> to, through the affine); the reverse scan is the opposite
    #[arg(long)]
    pe: Option<String>,

    /// With --duope / --dwimc: the (forward) scan's total readout time (s) without a JSON sidecar
    #[arg(long)]
    readout: Option<f64>,

    /// With --duope: also write <forward>_duope_field, the field in Hz (needs the readout time)
    #[arg(long)]
    save_field: bool,

    /// Show the version, ONNX Runtime, threads, the GPU (with -g) and the embedded models, then exit (as
    /// tigerbx.check())
    #[arg(long)]
    check: bool,

    /// Run the models on the GPU: CUDA on Linux (GPU build of ONNX Runtime + CUDA 12 / cuDNN 9 libraries), DirectML
    /// on Windows (any DirectX 12 GPU); falls back to the CPU with a warning
    #[arg(short = 'g', long)]
    gpu: bool,

    /// Threads (default: 70% of available CPUs): ONNX Runtime intra-op threads; with --dwimc the volumes processed in
    /// parallel
    #[arg(short = 'j', long)]
    threads: Option<usize>,
}

/// Template-space intensities (tigerbx.affine / tigerbx.norm `intensity`).
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq)]
enum Intensity {
    Raw,
    Minmax,
    Gnorm,
}

/// Which outputs to write.
#[derive(Clone, Copy)]
struct Outputs {
    tbx: bool,
    tbxmask: bool,
    tbxoutermask: bool,
    tseg: bool,
    dgm: bool,
    /// `--affine`: _affine scaled by --intensity and _warp.npz with the affine only (tigerbx.affine)
    affine_pipe: bool,
    norm: bool,
    intensity: Intensity,
}

/// The embedded model `embedded` (its size printed).
fn model_bytes(embedded: &'static [u8], name: &str) -> &'static [u8] {
    println!("{name} model: embedded ({:.1} MB)", embedded.len() as f64 / 1_048_576.0);
    embedded
}

/// Release model `name` from the TigerBx model cache (downloaded on first use).
fn cached_model_bytes(name: &str, what: &str) -> Result<Vec<u8>> {
    let p = models::get(name)?;
    println!("{what} model: {}", p.display());
    Ok(std::fs::read(&p)?)
}

/// ONNX Runtime is loaded at run time: ORT_DYLIB_PATH if set, else a library next to the executable, else the
/// system search path. Loaded here, so that a missing library is an error message, not a panic at the first model.
#[cfg(feature = "dynamic-ort")]
fn init_onnxruntime() -> Result<Option<PathBuf>> {
    let lib = if let Some(p) = std::env::var_os("ORT_DYLIB_PATH") {
        Some(PathBuf::from(p))
    } else {
        std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)).and_then(|dir| {
            let mut names: Vec<PathBuf> = ["onnxruntime.dll", "libonnxruntime.dylib", "libonnxruntime.so"]
                .iter()
                .map(|n| dir.join(n))
                .collect();
            if let Ok(entries) = std::fs::read_dir(&dir) {
                // versioned Linux library, e.g. libonnxruntime.so.1.24.4
                names.extend(entries.flatten().map(|e| e.path()).filter(|p| {
                    p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("libonnxruntime.so."))
                }));
            }
            names.into_iter().find(|p| p.is_file())
        })
    };
    let system = if cfg!(windows) { "onnxruntime.dll" } else if cfg!(target_os = "macos") { "libonnxruntime.dylib" } else { "libonnxruntime.so" };
    let path = lib.clone().unwrap_or_else(|| PathBuf::from(system));
    ort::init_from(&path)
        .with_context(|| format!("ONNX Runtime library {} could not be loaded: keep the library of the tigerbx \
                                  package next to the executable, or set ORT_DYLIB_PATH", path.display()))?
        .commit();
    Ok(lib)
}

/// ONNX Runtime is linked into the executable (static-ort).
#[cfg(not(feature = "dynamic-ort"))]
fn init_onnxruntime() -> Result<Option<PathBuf>> {
    Ok(None)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let other_pipelines = args.seg || args.norm || args.affine || args.tbx || args.tbxmask || args.tbxoutermask || args.tseg
        || args.dgm || args.template || args.inspect || args.check;
    if args.dwimc {
        if other_pipelines || args.duope {
            bail!("--dwimc runs on its own; run the other pipelines / outputs separately");
        }
        if !args.series.is_empty() || args.save_field {
            bail!("--series / --save-field go with --duope");
        }
        let settings = dwimc::Settings {
            output: args.output.clone(),
            reverse: args.reverse.clone(),
            field: args.field.clone(),
            pe: args.pe.clone(),
            readout: args.readout,
            order: 2,
            iters: dwimc::OUTER,
        };
        let threads = args.threads.unwrap_or_else(inference::default_threads);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
        return pool.install(|| dwimc::run(&args.inputs, &settings));
    }
    if !args.reverse.is_empty() || args.field.is_some() {
        bail!("--reverse / --field go with --dwimc");
    }
    if args.duope {
        let others = args.seg || args.norm || args.affine || args.tbx || args.tbxmask || args.tbxoutermask || args.tseg
            || args.dgm || args.template || args.inspect || args.check;
        if others {
            bail!("--duope runs on its own; run the other pipelines / outputs separately");
        }
        let settings = duope_pipe::Settings {
            output: args.output.clone(),
            series: args.series.clone(),
            pe: args.pe.clone(),
            readout: args.readout,
            series_pe: Vec::new(),
            series_align: true,
            joint: None,
            save_field: args.save_field,
            save_disp: false,
            alpha: duope::ALPHA,
            max_mm: duope::MAX_MM,
        };
        return duope_pipe::run(&args.inputs, &settings);
    }
    if !args.series.is_empty() || args.pe.is_some() || args.readout.is_some() || args.save_field {
        bail!("--series / --pe / --readout / --save-field go with --duope");
    }
    let ort_lib = init_onnxruntime()?;
    if args.check {
        return run_check(&args, ort_lib.as_deref());
    }

    if args.inputs.is_empty() {
        bail!("No input files specified. Usage: tigerbx <input.nii.gz> [-o output_dir]");
    }

    let others = args.seg || args.norm || args.affine || args.tbx || args.tbxmask || args.tbxoutermask || args.tseg
        || args.dgm;
    if args.template {
        if others || args.inspect {
            bail!("--template runs on its own; run the other pipelines / outputs separately");
        }
        return run_template(&args);
    }
    if args.inspect {
        if others {
            bail!("--inspect runs on its own; run the other pipelines / outputs separately");
        }
        return run_inspect(&args);
    }
    if args.detail {
        bail!("--detail goes with --inspect");
    }

    // Output options, plus the pipelines'; nothing given: _tbx
    let (seg, norm) = (args.seg, args.norm);
    let mut outputs = Outputs {
        tbx: args.tbx || seg || norm || args.affine,
        tbxmask: args.tbxmask,
        tbxoutermask: args.tbxoutermask,
        tseg: args.tseg || seg,
        dgm: args.dgm,
        affine_pipe: args.affine,
        norm,
        intensity: args.intensity,
    };
    if !(outputs.tbx || outputs.tbxmask || outputs.tbxoutermask || outputs.tseg || outputs.dgm || outputs.affine_pipe || outputs.norm) {
        outputs.tbx = true;
    }

    // Expand inputs (globs, directories, literal files)
    let files = expand_inputs(&args.inputs)?;
    if files.is_empty() {
        bail!("No NIfTI files found matching the given inputs.");
    }

    // Create output directory if specified
    if let Some(ref out_dir) = args.output {
        std::fs::create_dir_all(out_dir)?;
    }

    // Build the ONNX sessions once, reuse them for all files
    let threads = args.threads.unwrap_or_else(inference::default_threads);
    let mut tbx_session = inference::create_session(model_bytes(EMBEDDED_TBX, "tBX"), threads, args.gpu)?;
    let mut tseg_session = if outputs.tseg {
        Some(inference::create_session(model_bytes(EMBEDDED_TSEG, "TSEG"), threads, args.gpu)?)
    } else {
        None
    };
    let mut dgm_session = if outputs.dgm {
        Some(inference::create_session(&cached_model_bytes(DGM_MODEL, "DGM12")?, threads, args.gpu)?)
    } else {
        None
    };
    let mut treg_session = if outputs.norm {
        Some(inference::create_session(model_bytes(EMBEDDED_TREG, "tReg"), threads, args.gpu)?)
    } else {
        None
    };
    let template = if outputs.affine_pipe || outputs.norm { Some(reg::load_template(EMBEDDED_TEMPLATE)?) } else { None };
    let prior = if (outputs.norm || outputs.affine_pipe) && args.intensity == Intensity::Gnorm {
        Some(tissuenorm::template_prior(&reg::load_labels(EMBEDDED_TEMPLATE_TSEG)?))
    } else {
        None
    };
    println!("Models loaded ({}).\n", if inference::on_gpu() { "GPU".to_string() } else { format!("CPU, {threads} threads") });

    println!("Found {} file(s) to process\n", files.len());

    // Process each file; a failing input is reported and skipped
    let mut failed = 0usize;
    for (i, input) in files.iter().enumerate() {
        println!("[{}/{}] {}", i + 1, files.len(), input.display());
        let sessions = Sessions {
            tbx: &mut tbx_session,
            tseg: tseg_session.as_mut(),
            dgm: dgm_session.as_mut(),
            treg: treg_session.as_mut(),
        };
        if let Err(e) = process_file(input, args.output.as_deref(), outputs, sessions, template.as_ref(), prior.as_ref()) {
            eprintln!("  ERROR: {:#}", e);
            failed += 1;
        }
        println!();
    }

    if failed > 0 {
        println!("Done: {} of {} file(s) failed.", failed, files.len());
    } else {
        println!("Done!");
    }
    Ok(())
}

/// `--check`: what the executable runs with.
fn run_check(args: &Args, ort_lib: Option<&Path>) -> Result<()> {
    println!("tigerbx {}", env!("CARGO_PKG_VERSION"));
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let threads = args.threads.unwrap_or_else(inference::default_threads);
    println!("CPUs {cpus}, ONNX Runtime threads {threads} (-j)");
    match ort_lib {
        Some(lib) => println!("ONNX Runtime library: {}", lib.display()),
        None if cfg!(feature = "dynamic-ort") => println!("ONNX Runtime library: from the system library path"),
        None => println!("ONNX Runtime: linked into the executable"),
    }
    // a session on the smallest embedded model: the runtime loads and runs (and, with -g, whether on the GPU)
    let _session = inference::create_session(EMBEDDED_INSPECT, threads, args.gpu)?;
    println!("ONNX Runtime: loads and runs");
    if args.gpu {
        println!("GPU: {}", if inference::on_gpu() { "yes" } else { "no, the CPU is used" });
    }
    println!("embedded models:");
    for (name, bytes) in [("mprage_tbxdual_v007_opt", EMBEDDED_TBX), ("mprage_tseg_v1_opt", EMBEDDED_TSEG),
                          ("mprage_treg_v003_opt", EMBEDDED_TREG), ("inspect_v003", EMBEDDED_INSPECT)] {
        println!("  {name:28} {:6.1} MB", bytes.len() as f64 / 1_048_576.0);
    }
    let dir = models::model_dir()?;
    let dgm = if dir.join(DGM_MODEL).exists() { "cached" } else { "downloaded on first use" };
    println!("model cache: {} ({DGM_MODEL}: {dgm})", dir.display());
    Ok(())
}

/// `--template`: tigerbx.template of all inputs into -o.
fn run_template(args: &Args) -> Result<()> {
    let Some(out_dir) = args.output.as_deref() else {
        bail!("--template needs an output directory (-o)");
    };
    if args.iters < 1 {
        bail!("--iters must be >= 1");
    }
    let files = expand_inputs(&args.inputs)?;
    if files.len() < 2 {
        bail!("a template needs at least 2 inputs, got {}", files.len());
    }
    let labels = args.contrast.eq_ignore_ascii_case("t1");
    let threads = args.threads.unwrap_or_else(inference::default_threads);
    // tBX: the brain of non-stripped inputs, and the TSEG input of stripped ones (labels)
    let mut tbx = if !args.skull_stripped || labels {
        Some(inference::create_session(model_bytes(EMBEDDED_TBX, "tBX"), threads, args.gpu)?)
    } else {
        None
    };
    let mut tseg = if labels {
        Some(inference::create_session(model_bytes(EMBEDDED_TSEG, "TSEG"), threads, args.gpu)?)
    } else {
        None
    };
    let mut treg = inference::create_session(model_bytes(EMBEDDED_TREG, "tReg"), threads, args.gpu)?;
    let t = reg::load_template(EMBEDDED_TEMPLATE)?;
    let opts = template::Options {
        iters: args.iters,
        median: true,
        stripped: args.skull_stripped,
        labels,
        contrast: args.contrast.clone(),
    };
    println!("Building a template of {} input(s) into {}\n", files.len(), out_dir.display());
    template::run(&files, out_dir, &t, &opts, tbx.as_mut(), tseg.as_mut(), &mut treg)?;
    println!("Done!");
    Ok(())
}

/// `--inspect`: one row per input (in the order of the inputs) in <output or the first input's folder>/
/// inspect_summary.csv; a failing input gets status failed and its error, the others go on.
fn run_inspect(args: &Args) -> Result<()> {
    let files = expand_inputs(&args.inputs)?;
    if files.is_empty() {
        bail!("No NIfTI files found matching the given inputs.");
    }
    let threads = args.threads.unwrap_or_else(inference::default_threads);
    let mut session = inference::create_session(model_bytes(EMBEDDED_INSPECT, "inspect"), threads, args.gpu)?;
    let summary_dir = match &args.output {
        Some(d) => d.clone(),
        None => files[0].parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from(".")),
    };
    std::fs::create_dir_all(&summary_dir)?;
    let mut rows = Vec::new();
    let mut failed = 0usize;
    for (i, f) in files.iter().enumerate() {
        println!("[{}/{}] {}", i + 1, files.len(), f.display());
        let mut row: inspect::Row = vec![("file", f.display().to_string())];
        match nifti_io::read_nifti_raw(f).and_then(|(h, d)| inspect::inspect_one(&h, &d, &mut session)) {
            Ok(r) => {
                row.push(("status", "ok".to_string()));
                row.extend(r);
            }
            Err(e) => {
                eprintln!("  ERROR: {:#}", e);
                failed += 1;
                row.push(("status", "failed".to_string()));
                row.push(("error", format!("{:#}", e)));
            }
        }
        rows.push(row);
    }
    let columns: &[&str] = if args.detail { &inspect::DETAIL_COLUMNS } else { &inspect::SUMMARY_COLUMNS };
    let out = summary_dir.join(inspect::SUMMARY_NAME);
    inspect::write_summary(&out, &rows, columns)?;
    println!("\n  -> {}", out.display());
    if failed > 0 {
        println!("Done: {} of {} file(s) failed.", failed, files.len());
    } else {
        println!("Done!");
    }
    Ok(())
}

/// Process a single NIfTI file and write the requested outputs.
struct Sessions<'a> {
    tbx: &'a mut ort::session::Session,
    tseg: Option<&'a mut ort::session::Session>,
    dgm: Option<&'a mut ort::session::Session>,
    treg: Option<&'a mut ort::session::Session>,
}

fn process_file(
    input: &Path,
    output_dir: Option<&Path>,
    outputs: Outputs,
    sessions: Sessions,
    template: Option<&reg::Template>,
    prior: Option<&ndarray::Array4<f64>>,
) -> Result<()> {
    let start = std::time::Instant::now();

    let (header, data) = nifti_io::read_nifti(input)?;
    let res = bx::run_bx(&data, &header, sessions.tbx, sessions.tseg, sessions.dgm)?;

    if outputs.tbx {
        let p = output_path(input, output_dir, "tbx");
        nifti_io::write_tbx(&p, &header, &res.brain)?;
        println!("  -> {}", p.display());
    }
    if outputs.tbxmask {
        let p = output_path(input, output_dir, "tbxmask");
        nifti_io::write_nifti_u8(&p, &header, &res.mask)?;
        println!("  -> {}", p.display());
    }
    if outputs.tbxoutermask {
        let outer = res.outer.as_ref().context("--tbxoutermask needs a dual mask tBX model (mode tbxdual)")?;
        let p = output_path(input, output_dir, "tbxoutermask");
        nifti_io::write_nifti_u8(&p, &header, outer)?;
        println!("  -> {}", p.display());
        if outputs.tbx {
            let d = data.to_shape(outer.raw_dim()).context("Expected 3D input")?;
            let brain = ndarray::Zip::from(&d).and(outer).map_collect(|&v, &m| if m > 0 { v } else { 0.0 });
            let p = output_path(input, output_dir, "tbxouter");
            nifti_io::write_tbx(&p, &header, &brain)?;
            println!("  -> {}", p.display());
        }
    }
    if let Some(tseg) = res.tseg.as_ref() {
        let p = output_path(input, output_dir, "tseg");
        nifti_io::write_nifti_u8(&p, &header, tseg)?;
        println!("  -> {}", p.display());
    }
    if let (true, Some(dgm)) = (outputs.dgm, res.dgm.as_ref()) {
        let p = output_path(input, output_dir, "dgm");
        nifti_io::write_nifti_u8(&p, &header, dgm)?;
        println!("  -> {}", p.display());
    }

    if let Some(t) = template {
        let brain = res.brain.view().into_dimensionality::<ndarray::Ix3>()?.to_owned();
        let mask = res.mask.view().into_dimensionality::<ndarray::Ix3>()?.to_owned();
        let r = reg::run_reg(&brain, prior.map(|_| &mask), &header, t, sessions.treg)?;
        if outputs.affine_pipe {
            let affine = scale(r.affine.clone(), r.affine_mask.as_ref(), outputs.intensity, prior, "affine");
            let p = output_path(input, output_dir, "affine");
            nifti_io::write_nifti_f32(&p, &t.header, &affine.into_dyn())?;
            println!("  -> {}", p.display());
        }
        if let Some(dir) = std::env::var_os("TIGERBX_DUMP_REG") {
            // validation aid: the registration (template -> input world matrix, tReg grid) for applying it to labels
            let stem = output_path(input, Some(Path::new(&dir)), "reg");
            let stem = stem.to_string_lossy().replace("_reg.nii.gz", "");
            let m: Vec<String> = r.matrix.iter().flatten().map(|v| format!("{v:.17e}")).collect();
            std::fs::write(format!("{stem}_M.txt"), m.join(" "))?;
            if let Some(m) = r.norm_mask.as_ref() {
                std::fs::write(format!("{stem}_normmask.u8"), m.iter().map(|&b| b as u8).collect::<Vec<u8>>())?;
            }
            if let Some(g) = r.grid.as_ref() {
                let bytes: Vec<u8> = g.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(format!("{stem}_grid.f32"), bytes)?;
            }
        }
        if let Some(norm) = r.norm {
            let norm = scale(norm, r.norm_mask.as_ref(), outputs.intensity, prior, "norm");
            let p = output_path(input, output_dir, "norm");
            nifti_io::write_nifti_f32(&p, &t.header, &norm.into_dyn())?;
            println!("  -> {}", p.display());
        }
        if outputs.norm || outputs.affine_pipe {
            let p = output_path(input, output_dir, "warp").with_extension("").with_extension("npz");
            write_warp(&p, &header, data.shape(), t, &r.matrix, r.grid.as_deref())?;
            println!("  -> {}", p.display());
        }
    }

    println!("  Done in {:.1}s", start.elapsed().as_secs_f64());
    Ok(())
}

/// Template-space intensities by `intensity` (tigerbx.pipelines._common.scale); `mask` = the brain (gnorm).
fn scale(img: ndarray::Array3<f32>, mask: Option<&ndarray::Array3<bool>>, intensity: Intensity, prior: Option<&ndarray::Array4<f64>>, what: &str) -> ndarray::Array3<f32> {
    match (intensity, prior) {
        (Intensity::Minmax, _) => reg::min_max_f32(&img),
        (Intensity::Gnorm, Some(prior)) => {
            let (y, pk) = tissuenorm::tissue_norm(&img, mask, prior);
            println!("  gnorm peaks ({what}) CSF {:.4} GM {:.4} WM {:.4}", pk[0], pk[1], pk[2]);
            y
        }
        _ => img,
    }
}

/// `<stem>_warp.npz`: the registration as TigerBx `core.warpfile` (plain arrays; tigerbx.transform applies it):
/// template grid, input grid, affine_M (template world -> input world) and, when tReg ran, treg_warp (3, X, Y, Z:
/// displacement in template voxels, = tReg sampling grid - identity).
fn write_warp(path: &Path, header: &nifti::NiftiHeader, shape: &[usize], t: &reg::Template, matrix: &[[f64; 4]; 4], grid: Option<&[f32]>) -> Result<()> {
    use npz::Npy;
    let (x, y, z) = t.vol.data.dim();
    let n = x * y * z;
    let disp: Option<Vec<f32>> = grid.map(|g| {
        let mut d = g.to_vec();
        for (idx, v) in d.iter_mut().enumerate() {
            let (c, p) = (idx / n, idx % n);
            *v -= [p / (y * z), (p / z) % y, p % z][c] as f32;
        }
        d
    });
    let flat = |m: &[[f64; 4]; 4]| m.iter().flatten().cloned().collect::<Vec<f64>>();
    let (taff, iaff, m) = (flat(&t.vol.affine), flat(&resample::best_affine(header)), flat(matrix));
    let tshape = [x as i64, y as i64, z as i64];
    let ishape: Vec<i64> = shape.iter().take(3).map(|&v| v as i64).collect();
    let mut arrays = vec![
        ("format", Npy::Str("tigerbx-warp/2")),
        ("template_affine", Npy::F64(vec![4, 4], &taff)),
        ("template_shape", Npy::I64(vec![3], &tshape)),
        ("input_affine", Npy::F64(vec![4, 4], &iaff)),
        ("input_shape", Npy::I64(vec![3], &ishape)),
        ("affine_M", Npy::F64(vec![4, 4], &m)),
    ];
    if let Some(d) = disp.as_ref() {
        arrays.push(("treg_warp", Npy::F32(vec![3, x, y, z], d)));
    }
    npz::write_npz(path, &arrays)
}

/// Expand input arguments into a list of NIfTI file paths.
/// - Contains `*` or `?` → glob pattern
/// - Is a directory → scan for .nii / .nii.gz files
/// - Otherwise → literal file path
fn expand_inputs(inputs: &[String]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    for input in inputs {
        if input.contains('*') || input.contains('?') {
            // Glob pattern
            let paths = glob::glob(input)
                .map_err(|e| anyhow::anyhow!("Invalid glob pattern '{}': {}", input, e))?;
            for entry in paths {
                let path = entry?;
                if is_nifti(&path) {
                    files.push(path);
                }
            }
        } else {
            let path = PathBuf::from(input);
            if path.is_dir() {
                // Directory: scan for NIfTI files
                scan_directory(&path, &mut files)?;
            } else if path.exists() {
                files.push(path);
            } else {
                eprintln!("Warning: '{}' not found, skipping.", input);
            }
        }
    }

    files.sort();
    files.dedup();
    Ok(files)
}

/// Scan a directory (non-recursive) for .nii and .nii.gz files.
fn scan_directory(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if is_nifti(&path) {
            files.push(path);
        }
    }
    Ok(())
}

/// Check if a path looks like an input NIfTI file (not an earlier TigerBx output).
fn is_nifti(path: &Path) -> bool {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let Some(stem) = name.strip_suffix(".nii.gz").or_else(|| name.strip_suffix(".nii")) else {
        return false;
    };
    !OUTPUT_POSTFIXES.iter().any(|p| stem.ends_with(&format!("_{p}")))
}

/// Output path `<stem>_<postfix>.nii.gz` in `output_dir`, or next to the input.
fn output_path(input: &Path, output_dir: Option<&Path>, postfix: &str) -> PathBuf {
    let stem = input
        .file_name()
        .unwrap()
        .to_string_lossy()
        .replace(".nii.gz", "")
        .replace(".nii", "");
    let parent = match output_dir {
        Some(dir) => dir,
        None => input.parent().unwrap_or(Path::new(".")),
    };
    parent.join(format!("{stem}_{postfix}.nii.gz"))
}
