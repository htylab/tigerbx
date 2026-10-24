mod bx;
mod inference;
mod nifti_io;
mod numreg;
mod onnx_prune;
mod postprocess;
mod reg;
mod resample;
mod tissuenorm;

use anyhow::{bail, Result};
use clap::Parser;
use std::path::{Path, PathBuf};

/// Embedded models (compiled into the binary): TigerBx's defaults, from the modelhub release
static EMBEDDED_BET: &[u8] = include_bytes!("../models/mprage_bet_v005_opt.onnx");
static EMBEDDED_ASEG: &[u8] = include_bytes!("../models/mprage_aseg43_v0011_opt.onnx");
static EMBEDDED_DGM: &[u8] = include_bytes!("../models/mprage_dgm12_v002_opt.onnx");
static EMBEDDED_VFA: &[u8] = include_bytes!("../models/mprage_reg_v004_vfa.onnx");
/// TigerBx's bundled registration template (same file as the Python package)
static EMBEDDED_TEMPLATE: &[u8] = include_bytes!("../../tigerbx/template/mni_template.nii.gz");
/// Its aseg: tissue priors of the gnorm intensity normalisation of _norm
static EMBEDDED_TEMPLATE_ASEG: &[u8] = include_bytes!("../../tigerbx/template/mni_template_aseg.nii.gz");

/// Postfixes TigerBx writes; inputs ending in one of them are earlier outputs, not new images.
const OUTPUT_POSTFIXES: [&str; 6] = ["tbet", "tbetmask", "aseg", "dgm", "affine", "norm"];

/// tigerbx — TigerBx brain extraction, ASEG and template registration as a native executable
///
/// Supports glob patterns, directories, and multiple files:
///   tigerbx T1w.nii.gz                    (no option: _tbet)
///   tigerbx data/*.nii.gz -o output_dir --seg
///   tigerbx data/T1w_dir -o results/ -a -A
#[derive(Parser, Debug)]
#[command(version, verbatim_doc_comment, about = "TigerBx tBET, ASEG and template registration using ONNX Runtime (no option: _tbet)")]
struct Args {
    /// Input NIfTI files, glob patterns, or directories
    inputs: Vec<String>,

    /// Output directory (default: same directory as input)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Pipeline tBET + ASEG: _tbet, _aseg (as tigerbx.run('ba')); output options add to it
    #[arg(long)]
    seg: bool,

    /// Pipeline tBET + template normalisation: _tbet, _norm (gnorm); output options add to it
    #[arg(long)]
    norm: bool,

    /// Pipelines seg + norm: _tbet, _aseg, _norm
    #[arg(long)]
    all: bool,

    /// Save the brain-extracted image (_tbet)
    #[arg(short = 'b', long)]
    bet: bool,

    /// Save the brain mask (_tbetmask)
    #[arg(short = 'm', long)]
    betmask: bool,

    /// Save the ASEG segmentation (_aseg; 43 structures + 99 Unlabeled-Intracranial), its 12 deep-GM
    /// structures taken from DGM12
    #[arg(short = 'a', long)]
    aseg: bool,

    /// Save the DGM12 deep gray-matter segmentation (_dgm; labels 1-12, as tigerbx.run('d'))
    #[arg(short = 'd', long)]
    dgm: bool,

    /// Save the tBET brain affinely registered to the template (_affine; numreg, as tigerbx.reg('A'))
    #[arg(short = 'A', long)]
    affine: bool,

    /// Save the tBET brain normalised to the template (_norm; numreg affine + VFA, as tigerbx.reg('AD') / VBM),
    /// intensity-normalised with gnorm (tigerbx tissue_norm: CSF 1/3, GM 2/3, WM 1; T1 only, else img / p99)
    #[arg(short = 'n')]
    norm_out: bool,

    /// Keep _norm min-max scaled as tigerbx.reg('AD') instead of gnorm
    #[arg(long)]
    norm_minmax: bool,

    /// tBET ONNX model file (default: embedded mprage_bet_v005_opt)
    #[arg(long)]
    bet_model: Option<PathBuf>,

    /// ASEG ONNX model file (default: embedded mprage_aseg43_v0011_opt)
    #[arg(long)]
    aseg_model: Option<PathBuf>,

    /// DGM12 ONNX model file (default: embedded mprage_dgm12_v002_opt)
    #[arg(long)]
    dgm_model: Option<PathBuf>,

    /// VFA registration ONNX model file (default: embedded mprage_reg_v004_vfa)
    #[arg(long)]
    vfa_model: Option<PathBuf>,

    /// Run the models on the GPU: CUDA on Linux (GPU build of ONNX Runtime + CUDA 12 / cuDNN 9 libraries), DirectML
    /// on Windows (any DirectX 12 GPU); falls back to the CPU with a warning
    #[arg(short = 'g', long)]
    gpu: bool,

    /// ONNX Runtime intra-op threads (default: 70% of available CPUs)
    #[arg(short, long)]
    threads: Option<usize>,
}

/// Which outputs to write.
#[derive(Clone, Copy)]
struct Outputs {
    bet: bool,
    betmask: bool,
    aseg: bool,
    dgm: bool,
    affine: bool,
    norm: bool,
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

/// ONNX Runtime's own log: fatal messages only (failures still come back as errors and are reported per input);
/// TIGERBX_ORT_LOG=1 also shows its errors. Its other errors are handled internally, e.g. cuDNN >= 9.2x logs "No
/// execution plans support the graph" for strict-FP32 convolutions (TF32 off) and falls back to another algorithm.
fn ort_logger() -> ort::logging::LoggerFunction {
    let verbose = std::env::var_os("TIGERBX_ORT_LOG").is_some();
    std::sync::Arc::new(move |level: ort::logging::LogLevel, _category: &str, _id: &str, location: &str, message: &str| {
        use ort::logging::LogLevel;
        if matches!(level, LogLevel::Fatal) || (verbose && matches!(level, LogLevel::Error)) {
            eprintln!("onnxruntime: {message} ({location})");
        }
    })
}

/// ONNX Runtime is loaded at run time: ORT_DYLIB_PATH if set, else a library next to the executable, else the
/// system search path.
#[cfg(feature = "dynamic-ort")]
fn init_onnxruntime() -> Result<()> {
    let lib = if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        None
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
    match lib {
        Some(lib) => ort::init_from(&lib)?.with_logger(ort_logger()).commit(),
        None => ort::init().with_logger(ort_logger()).commit(),
    };
    Ok(())
}

/// ONNX Runtime is linked into the executable (static-ort).
#[cfg(not(feature = "dynamic-ort"))]
fn init_onnxruntime() -> Result<()> {
    ort::init().with_logger(ort_logger()).commit();
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    init_onnxruntime()?;

    if args.inputs.is_empty() {
        bail!("No input files specified. Usage: tigerbx <input.nii.gz> [-o output_dir]");
    }

    // Output options, plus the pipelines'; nothing given: _tbet
    let (seg, norm) = (args.seg || args.all, args.norm || args.all);
    let mut outputs = Outputs {
        bet: args.bet || seg || norm,
        betmask: args.betmask,
        aseg: args.aseg || seg,
        dgm: args.dgm,
        affine: args.affine,
        norm: args.norm_out || norm,
    };
    if !(outputs.bet || outputs.betmask || outputs.aseg || outputs.dgm || outputs.affine || outputs.norm) {
        outputs.bet = true;
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
    let mut bet_session = inference::create_session(&model_bytes(&args.bet_model, EMBEDDED_BET, "tBET")?, threads, args.gpu)?;
    let mut aseg_session = if outputs.aseg {
        Some(inference::create_session(&model_bytes(&args.aseg_model, EMBEDDED_ASEG, "ASEG")?, threads, args.gpu)?)
    } else {
        None
    };
    // DGM12 also runs for -a: the ASEG deep-GM structures come from it
    let mut dgm_session = if outputs.dgm || outputs.aseg {
        Some(inference::create_session(&model_bytes(&args.dgm_model, EMBEDDED_DGM, "DGM12")?, threads, args.gpu)?)
    } else {
        None
    };
    let mut vfa_session = if outputs.norm {
        Some(inference::create_session(&model_bytes(&args.vfa_model, EMBEDDED_VFA, "VFA")?, threads, args.gpu)?)
    } else {
        None
    };
    let template = if outputs.affine || outputs.norm { Some(reg::load_template(EMBEDDED_TEMPLATE)?) } else { None };
    let prior = if outputs.norm && !args.norm_minmax {
        Some(tissuenorm::template_prior(&reg::load_labels(EMBEDDED_TEMPLATE_ASEG)?))
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
            bet: &mut bet_session,
            aseg: aseg_session.as_mut(),
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

/// Process a single NIfTI file and write the requested outputs.
struct Sessions<'a> {
    bet: &'a mut ort::session::Session,
    aseg: Option<&'a mut ort::session::Session>,
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
    let res = bx::run_bx(&data, &header, sessions.bet, sessions.aseg, sessions.dgm)?;

    if outputs.bet {
        let p = output_path(input, output_dir, "tbet");
        nifti_io::write_tbet(&p, &header, &res.brain)?;
        println!("  -> {}", p.display());
    }
    if outputs.betmask {
        let p = output_path(input, output_dir, "tbetmask");
        nifti_io::write_nifti_u8(&p, &header, &res.mask)?;
        println!("  -> {}", p.display());
    }
    if let Some(aseg) = res.aseg.as_ref() {
        let p = output_path(input, output_dir, "aseg");
        nifti_io::write_nifti_u8(&p, &header, aseg)?;
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
        if outputs.affine {
            let p = output_path(input, output_dir, "affine");
            nifti_io::write_nifti_f32(&p, &t.header, &r.affine.into_dyn())?;
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
        if let Some(mut norm) = r.norm {
            if let Some(prior) = prior {
                let (y, pk) = tissuenorm::tissue_norm(&norm, r.norm_mask.as_ref(), prior);
                println!("  gnorm peaks CSF {:.4} GM {:.4} WM {:.4}", pk[0], pk[1], pk[2]);
                norm = y;
            }
            let p = output_path(input, output_dir, "norm");
            nifti_io::write_nifti_f32(&p, &t.header, &norm.into_dyn())?;
            println!("  -> {}", p.display());
        }
    }

    println!("  Done in {:.1}s", start.elapsed().as_secs_f64());
    Ok(())
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
