mod bx;
mod inference;
mod nifti_io;
mod postprocess;
mod resample;

use anyhow::{bail, Result};
use clap::Parser;
use std::path::{Path, PathBuf};

/// Embedded models (compiled into the binary): TigerBx's defaults, from the modelhub release
static EMBEDDED_BET: &[u8] = include_bytes!("../models/mprage_bet_v005_opt.onnx");
static EMBEDDED_ASEG: &[u8] = include_bytes!("../models/mprage_aseg43_v0011_opt.onnx");

/// Postfixes rustbx / TigerBx write; inputs ending in one of them are earlier outputs, not new images.
const OUTPUT_POSTFIXES: [&str; 3] = ["tbet", "tbetmask", "aseg"];

/// rustbx — TigerBx brain extraction and ASEG as a native executable
///
/// Supports glob patterns, directories, and multiple files:
///   rustbx T1w.nii.gz                    (all outputs: _tbet, _tbetmask, _aseg)
///   rustbx data/*.nii.gz -o output_dir -m
///   rustbx data/T1w_dir -o results/ -b -a
#[derive(Parser, Debug)]
#[command(version, about = "TigerBx tBET and ASEG using ONNX Runtime (no option: -b -m -a)")]
struct Args {
    /// Input NIfTI files, glob patterns, or directories
    inputs: Vec<String>,

    /// Output directory (default: same directory as input)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Save the brain-extracted image (_tbet)
    #[arg(short = 'b', long)]
    bet: bool,

    /// Save the brain mask (_tbetmask)
    #[arg(short = 'm', long)]
    betmask: bool,

    /// Save the ASEG segmentation (_aseg; 43 structures + 99 Unlabeled-Intracranial)
    #[arg(short = 'a', long)]
    aseg: bool,

    /// tBET ONNX model file (default: embedded mprage_bet_v005_opt)
    #[arg(long)]
    bet_model: Option<PathBuf>,

    /// ASEG ONNX model file (default: embedded mprage_aseg43_v0011_opt)
    #[arg(long)]
    aseg_model: Option<PathBuf>,

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
/// system search path.
fn init_onnxruntime() -> Result<()> {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return Ok(());
    }
    let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) else {
        return Ok(());
    };
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
    if let Some(lib) = names.into_iter().find(|p| p.is_file()) {
        ort::init_from(&lib)?.commit();
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    init_onnxruntime()?;

    if args.inputs.is_empty() {
        bail!("No input files specified. Usage: rustbx <input.nii.gz> [-o output_dir]");
    }

    // No output option: all of them
    let outputs = if args.bet || args.betmask || args.aseg {
        Outputs { bet: args.bet, betmask: args.betmask, aseg: args.aseg }
    } else {
        Outputs { bet: true, betmask: true, aseg: true }
    };

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
    let mut bet_session = inference::create_session(&model_bytes(&args.bet_model, EMBEDDED_BET, "tBET")?, threads)?;
    let mut aseg_session = if outputs.aseg {
        Some(inference::create_session(&model_bytes(&args.aseg_model, EMBEDDED_ASEG, "ASEG")?, threads)?)
    } else {
        None
    };
    println!("Models loaded ({} threads).\n", threads);

    println!("Found {} file(s) to process\n", files.len());

    // Process each file; a failing input is reported and skipped
    let mut failed = 0usize;
    for (i, input) in files.iter().enumerate() {
        println!("[{}/{}] {}", i + 1, files.len(), input.display());
        if let Err(e) = process_file(input, args.output.as_deref(), outputs, &mut bet_session, aseg_session.as_mut()) {
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
fn process_file(
    input: &Path,
    output_dir: Option<&Path>,
    outputs: Outputs,
    bet_session: &mut ort::session::Session,
    aseg_session: Option<&mut ort::session::Session>,
) -> Result<()> {
    let start = std::time::Instant::now();

    let (header, data) = nifti_io::read_nifti(input)?;
    let res = bx::run_bx(&data, &header, bet_session, aseg_session)?;

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

/// Check if a path looks like an input NIfTI file (not an earlier rustbx / TigerBx output).
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
