mod bx;
mod inference;
mod nifti_io;
mod postprocess;
mod resample;

use anyhow::{bail, Result};
use clap::Parser;
use std::path::{Path, PathBuf};

/// Embedded tBET model (compiled into the binary): TigerBx's default, from the modelhub release
static EMBEDDED_MODEL: &[u8] = include_bytes!("../models/mprage_bet_v005_opt.onnx");

/// rustbx — Brain extraction tool
///
/// Supports glob patterns, directories, and multiple files:
///   rustbx T1w.nii.gz
///   rustbx data/*.nii.gz -o output_dir
///   rustbx data/T1w_dir -o results/
#[derive(Parser, Debug)]
#[command(version, about = "Brain extraction tool using ONNX Runtime")]
struct Args {
    /// Input NIfTI files, glob patterns, or directories
    inputs: Vec<String>,

    /// Output directory (default: same directory as input)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Path to ONNX model file (default: embedded model)
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// ONNX Runtime intra-op threads (default: 70% of available CPUs)
    #[arg(short, long)]
    threads: Option<usize>,
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

    // Resolve model bytes
    let external_model;
    let model_bytes: &[u8] = if let Some(ref model_path) = args.model {
        if !model_path.exists() {
            bail!("Model file not found: {}", model_path.display());
        }
        println!("Model: {} (external)", model_path.display());
        external_model = std::fs::read(model_path)?;
        &external_model
    } else {
        println!("Model: embedded ({:.1} MB)", EMBEDDED_MODEL.len() as f64 / 1_048_576.0);
        EMBEDDED_MODEL
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

    // Build ONNX session once, reuse for all files
    println!("Loading model...");
    let threads = args.threads.unwrap_or_else(inference::default_threads);
    let mut session = inference::create_session(model_bytes, threads)?;
    println!("Model loaded ({} threads).\n", threads);

    println!("Found {} file(s) to process\n", files.len());

    // Process each file
    for (i, input) in files.iter().enumerate() {
        println!("[{}/{}] {}", i + 1, files.len(), input.display());
        if let Err(e) = process_file(input, args.output.as_deref(), &mut session) {
            eprintln!("  ERROR: {:#}", e);
        }
        println!();
    }

    println!("Done!");
    Ok(())
}

/// Process a single NIfTI file through the brain-extraction pipeline.
fn process_file(input: &Path, output_dir: Option<&Path>, session: &mut ort::session::Session) -> Result<()> {
    let start = std::time::Instant::now();
    let (tbx_path, tbxmask_path) = output_paths(input, output_dir);

    // 1. Read NIfTI
    let (header, data) = nifti_io::read_nifti(input)?;

    // 2. Run brain extraction
    let (mask, brain) = bx::run_bx(&data, &header, session)?;

    // 3. Write outputs
    nifti_io::write_nifti(&tbx_path, &header, &brain)?;
    println!("  -> {}", tbx_path.display());

    nifti_io::write_nifti_u8(&tbxmask_path, &header, &mask)?;
    println!("  -> {}", tbxmask_path.display());

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

/// Check if a path looks like a NIfTI file.
fn is_nifti(path: &Path) -> bool {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    name.ends_with(".nii.gz") || name.ends_with(".nii")
}

/// Generate output file paths.
/// If `output_dir` is Some, put outputs there; otherwise next to input.
fn output_paths(input: &Path, output_dir: Option<&Path>) -> (PathBuf, PathBuf) {
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

    let tbx = parent.join(format!("{}_tbet.nii.gz", stem));
    let tbxmask = parent.join(format!("{}_tbetmask.nii.gz", stem));
    (tbx, tbxmask)
}
