mod bx;
mod inference;
mod inspect;
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

/// Embedded models (compiled into the binary): TigerBx's defaults, from the modelhub release
static EMBEDDED_TBX: &[u8] = include_bytes!("../models/mprage_tbx_v005_opt.onnx");
static EMBEDDED_TSEG: &[u8] = include_bytes!("../models/mprage_tseg_v0011_opt.onnx");
static EMBEDDED_DGM: &[u8] = include_bytes!("../models/mprage_dgm12_v002_opt.onnx");
static EMBEDDED_VFA: &[u8] = include_bytes!("../models/mprage_reg_v005_vfa.onnx");
static EMBEDDED_INSPECT: &[u8] = include_bytes!("../models/inspect_v003.onnx");
/// TigerBx's bundled registration template (same file as the Python package)
static EMBEDDED_TEMPLATE: &[u8] = include_bytes!("../../tigerbx/template/mni_template.nii.gz");
/// Its tseg: tissue priors of the gnorm intensity normalisation of _norm
static EMBEDDED_TEMPLATE_TSEG: &[u8] = include_bytes!("../../tigerbx/template/mni_template_tseg.nii.gz");

/// Postfixes TigerBx writes; inputs ending in one of them are earlier outputs, not new images.
const OUTPUT_POSTFIXES: [&str; 6] = ["tbx", "tbxmask", "tseg", "dgm", "affine", "norm"];

/// tigerbx — TigerBx brain extraction, TSEG and template registration as a native executable
///
/// Supports glob patterns, directories, and multiple files:
///   tigerbx T1w.nii.gz                    (no option: _tbx)
///   tigerbx data/*.nii.gz -o output_dir --seg
///   tigerbx data/T1w_dir -o results/ -t -A
///   tigerbx data/ -o results/ --inspect     (inspect_summary.csv: what each image is)
#[derive(Parser, Debug)]
#[command(version, verbatim_doc_comment, about = "TigerBx tBX, TSEG and template registration using ONNX Runtime (no option: _tbx)")]
struct Args {
    /// Input NIfTI files, glob patterns, or directories
    inputs: Vec<String>,

    /// Output directory (default: same directory as input)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Pipeline tBX + TSEG: _tbx, _tseg with its 12 deep-GM structures from DGM12 (more accurate; as
    /// tigerbx.seg); output options add to it
    #[arg(long)]
    seg: bool,

    /// Pipeline template normalisation (tBX, numreg affine, VFA): _tbx, _norm, _warp.npz (as tigerbx.norm); output
    /// options add to it
    #[arg(long)]
    norm: bool,

    /// Pipeline affine-only template normalisation (tBX, numreg affine; keeps the brain's size and shape, e.g.
    /// ventricle volume, which the VFA warp of --norm removes): _tbx, _affine, _warp.npz (as tigerbx.affine);
    /// output options add to it
    #[arg(long)]
    affine: bool,

    /// Pipelines seg + norm: _tbx, _tseg, _norm, _warp.npz
    #[arg(long)]
    all: bool,

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

    /// With --template: rounds of VFA to the current template
    #[arg(long, default_value_t = 6)]
    iters: usize,

    /// With --template: voxel-wise statistic of the registered images
    #[arg(long, value_enum, default_value_t = Stat::Median)]
    stat: Stat,

    /// With --template: the inputs are skull-stripped (their brain is the non-zero voxels; no tBX)
    #[arg(long)]
    skull_stripped: bool,

    /// With --template: input contrast; t1 also builds template_tseg (labels auto)
    #[arg(long, default_value = "t1")]
    contrast: String,

    /// With --template: template_tseg yes / no (auto: for --contrast t1)
    #[arg(long, value_enum, default_value_t = Labels::Auto)]
    labels: Labels,

    /// Save the brain-extracted image (_tbx)
    #[arg(short = 'b', long)]
    tbx: bool,

    /// Save the brain mask (_tbxmask)
    #[arg(short = 'm', long)]
    tbxmask: bool,

    /// Save the TSEG segmentation (_tseg; 43 structures + 99 Unlabeled-Intracranial; the model output, as
    /// tigerbx.run('t'))
    #[arg(short = 't', long)]
    tseg: bool,

    /// Save the DGM12 deep gray-matter segmentation (_dgm; labels 1-12, as tigerbx.run('d'))
    #[arg(short = 'd', long)]
    dgm: bool,

    /// Save the tBX brain affinely registered to the template (_affine; numreg, min-max scaled, as tigerbx.reg('A'))
    #[arg(short = 'A')]
    affine_out: bool,

    /// Save the tBX brain normalised to the template (_norm; numreg affine + VFA, intensities by --intensity) and
    /// its registration (_warp.npz: affine + VFA warp as plain arrays, for tigerbx.transform)
    #[arg(short = 'n')]
    norm_out: bool,

    /// Intensities of _norm and of the --affine pipeline's _affine: raw (the input's units; any contrast), minmax
    /// ([0, 1]) or gnorm (T1 only: tigerbx tissue_norm, CSF 1/3, GM 2/3, WM 1; else img / p99 with a warning)
    #[arg(long, value_enum, default_value_t = Intensity::Raw)]
    intensity: Intensity,

    /// tBX ONNX model file (default: embedded mprage_tbx_v005_opt)
    #[arg(long)]
    tbx_model: Option<PathBuf>,

    /// TSEG ONNX model file (default: embedded mprage_tseg_v0011_opt)
    #[arg(long)]
    tseg_model: Option<PathBuf>,

    /// DGM12 ONNX model file (default: embedded mprage_dgm12_v002_opt)
    #[arg(long)]
    dgm_model: Option<PathBuf>,

    /// VFA registration ONNX model file (default: embedded mprage_reg_v005_vfa)
    #[arg(long)]
    vfa_model: Option<PathBuf>,

    /// Inspect ONNX model file (default: embedded inspect_v003)
    #[arg(long)]
    inspect_model: Option<PathBuf>,

    /// Show the version, ONNX Runtime, threads, the GPU (with -g) and the embedded models, then exit (as
    /// tigerbx.check())
    #[arg(long)]
    check: bool,

    /// Run the models on the GPU: CUDA on Linux (GPU build of ONNX Runtime + CUDA 12 / cuDNN 9 libraries), DirectML
    /// on Windows (any DirectX 12 GPU); falls back to the CPU with a warning
    #[arg(short = 'g', long)]
    gpu: bool,

    /// ONNX Runtime intra-op threads (default: 70% of available CPUs)
    #[arg(short = 'j', long)]
    threads: Option<usize>,
}

/// Voxel-wise statistic of the template pipeline.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq)]
enum Stat {
    Median,
    Mean,
}

/// template_tseg of the template pipeline.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq)]
enum Labels {
    Auto,
    Yes,
    No,
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
    tseg: bool,
    dgm: bool,
    affine: bool,
    /// `--affine`: _affine scaled by --intensity and _warp.npz with the affine only (tigerbx.affine)
    affine_pipe: bool,
    norm: bool,
    /// `--seg` / `--all`: _tseg with its 12 deep-GM structures from DGM12 (tigerbx.seg)
    merge_dgm: bool,
    intensity: Intensity,
}

/// Model bytes from `path`, or the embedded model.
fn model_bytes(path: &Option<PathBuf>, embedded: &'static [u8], name: &str) -> Result<std::borrow::Cow<'static, [u8]>> {
    Ok(match path {
        Some(p) => {
            if !p.exists() {
                bail!("Model file not found: {}", p.display());
            }
            println!("{name} model: {} (external)", p.display());
            std::borrow::Cow::Owned(std::fs::read(p)?)
        }
        None => {
            println!("{name} model: embedded ({:.1} MB)", embedded.len() as f64 / 1_048_576.0);
            std::borrow::Cow::Borrowed(embedded)
        }
    })
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
    let ort_lib = init_onnxruntime()?;
    if args.check {
        return run_check(&args, ort_lib.as_deref());
    }

    if args.inputs.is_empty() {
        bail!("No input files specified. Usage: tigerbx <input.nii.gz> [-o output_dir]");
    }

    let others = args.seg || args.norm || args.affine || args.all || args.tbx || args.tbxmask || args.tseg
        || args.dgm || args.affine_out || args.norm_out;
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
    let (seg, norm) = (args.seg || args.all, args.norm || args.all);
    let mut outputs = Outputs {
        tbx: args.tbx || seg || norm || args.affine,
        tbxmask: args.tbxmask,
        tseg: args.tseg || seg,
        dgm: args.dgm,
        affine: args.affine_out,
        affine_pipe: args.affine,
        norm: args.norm_out || norm,
        merge_dgm: seg,
        intensity: args.intensity,
    };
    if !(outputs.tbx || outputs.tbxmask || outputs.tseg || outputs.dgm || outputs.affine || outputs.affine_pipe || outputs.norm) {
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
    let mut tbx_session = inference::create_session(&model_bytes(&args.tbx_model, EMBEDDED_TBX, "tBX")?, threads, args.gpu)?;
    let mut tseg_session = if outputs.tseg {
        Some(inference::create_session(&model_bytes(&args.tseg_model, EMBEDDED_TSEG, "TSEG")?, threads, args.gpu)?)
    } else {
        None
    };
    // DGM12: for -d, and for the seg pipeline (its deep-GM structures replace those of TSEG)
    let mut dgm_session = if outputs.dgm || outputs.merge_dgm {
        Some(inference::create_session(&model_bytes(&args.dgm_model, EMBEDDED_DGM, "DGM12")?, threads, args.gpu)?)
    } else {
        None
    };
    let mut vfa_session = if outputs.norm {
        Some(inference::create_session(&model_bytes(&args.vfa_model, EMBEDDED_VFA, "VFA")?, threads, args.gpu)?)
    } else {
        None
    };
    let template = if outputs.affine || outputs.affine_pipe || outputs.norm { Some(reg::load_template(EMBEDDED_TEMPLATE)?) } else { None };
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
            vfa: vfa_session.as_mut(),
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
    for (name, bytes) in [("mprage_tbx_v005_opt", EMBEDDED_TBX), ("mprage_tseg_v0011_opt", EMBEDDED_TSEG),
                          ("mprage_dgm12_v002_opt", EMBEDDED_DGM), ("mprage_reg_v005_vfa", EMBEDDED_VFA),
                          ("inspect_v003", EMBEDDED_INSPECT)] {
        println!("  {name:24} {:6.1} MB", bytes.len() as f64 / 1_048_576.0);
    }
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
    let labels = match args.labels {
        Labels::Auto => args.contrast.eq_ignore_ascii_case("t1"),
        Labels::Yes => true,
        Labels::No => false,
    };
    let threads = args.threads.unwrap_or_else(inference::default_threads);
    // tBX: the brain of non-stripped inputs, and the TSEG input of stripped ones (labels)
    let mut tbx = if !args.skull_stripped || labels {
        Some(inference::create_session(&model_bytes(&args.tbx_model, EMBEDDED_TBX, "tBX")?, threads, args.gpu)?)
    } else {
        None
    };
    let mut tseg = if labels {
        Some(inference::create_session(&model_bytes(&args.tseg_model, EMBEDDED_TSEG, "TSEG")?, threads, args.gpu)?)
    } else {
        None
    };
    let mut dgm = if labels && !args.skull_stripped {
        Some(inference::create_session(&model_bytes(&args.dgm_model, EMBEDDED_DGM, "DGM12")?, threads, args.gpu)?)
    } else {
        None
    };
    let mut vfa = inference::create_session(&model_bytes(&args.vfa_model, EMBEDDED_VFA, "VFA")?, threads, args.gpu)?;
    let t = reg::load_template(EMBEDDED_TEMPLATE)?;
    let opts = template::Options {
        iters: args.iters,
        median: args.stat == Stat::Median,
        stripped: args.skull_stripped,
        labels,
        contrast: args.contrast.clone(),
    };
    println!("Building a template of {} input(s) into {}\n", files.len(), out_dir.display());
    template::run(&files, out_dir, &t, &opts, tbx.as_mut(), tseg.as_mut(), dgm.as_mut(), &mut vfa)?;
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
    let mut session = inference::create_session(&model_bytes(&args.inspect_model, EMBEDDED_INSPECT, "inspect")?, threads, args.gpu)?;
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
    vfa: Option<&'a mut ort::session::Session>,
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
    let res = bx::run_bx(&data, &header, sessions.tbx, sessions.tseg, sessions.dgm, outputs.merge_dgm)?;

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
        let r = reg::run_reg(&brain, prior.map(|_| &mask), &header, t, sessions.vfa)?;
        if outputs.affine || outputs.affine_pipe {
            // -A alone: min-max as tigerbx.reg('A'); the --affine pipeline: --intensity
            let affine = if outputs.affine_pipe {
                scale(r.affine.clone(), r.affine_mask.as_ref(), outputs.intensity, prior, "affine")
            } else {
                reg::min_max_f32(&r.affine)
            };
            let p = output_path(input, output_dir, "affine");
            nifti_io::write_nifti_f32(&p, &t.header, &affine.into_dyn())?;
            println!("  -> {}", p.display());
        }
        if let Some(dir) = std::env::var_os("TIGERBX_DUMP_REG") {
            // validation aid: the registration (template -> input world matrix, VFA grid) for applying it to labels
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
/// template grid, input grid, affine_M (template world -> input world) and, when VFA ran, vfa_warp (3, X, Y, Z:
/// displacement in template voxels, = VFA sampling grid - identity).
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
        arrays.push(("vfa_warp", Npy::F32(vec![3, x, y, z], d)));
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
