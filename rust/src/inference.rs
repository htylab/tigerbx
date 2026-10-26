use anyhow::{Context, Result};
use ndarray::{Array3, ArrayView3};
use ort::session::Session;
use ort::session::run_options::OutputSelector;
use ort::value::TensorRef;
use std::sync::atomic::{AtomicBool, Ordering};

/// Set once a session runs on the GPU (CUDA: arena shrinkage then also covers gpu:0).
static ON_GPU: AtomicBool = AtomicBool::new(false);
static ON_CUDA: AtomicBool = AtomicBool::new(false);
/// The CPU-fallback warning is printed once.
static WARNED: AtomicBool = AtomicBool::new(false);

/// Optional uint8 mask output of an optimised TigerBx model.
const LABELS_OUTPUT: &str = "labels";

/// Number of intra-op threads: 70% of the CPUs this process may use (same default as TigerBx).
/// Filling every core makes ONNX Runtime's worker threads compete with each other and with other jobs.
pub fn default_threads() -> usize {
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    ((n as f64 * 0.7) as usize).max(1)
}

fn builder(threads: usize) -> Result<ort::session::builder::SessionBuilder> {
    Ok(Session::builder()
        .context("Failed to create session builder")?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .context("Failed to set optimization level")?
        .with_intra_threads(threads)
        .context("Failed to set intra-op threads")?
        .with_inter_threads(1)
        .context("Failed to set inter-op threads")?
        // Every input has a different shape, so a memory pattern planned for one input is never reused.
        .with_memory_pattern(false)
        .context("Failed to disable memory pattern")?)
}

/// CUDA execution provider for inputs whose shape changes with every subject: allocate exactly what each run needs,
/// pick cuDNN convolution algorithms heuristically (an exhaustive search reruns for every new shape) and do not let
/// cuDNN take its maximum workspace (allocated outside the arena).
fn cuda() -> ort::ep::ExecutionProviderDispatch {
    ort::ep::CUDA::default()
        .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
        .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
        .with_conv_max_workspace(false)
        // TF32 (default on): convolutions 1.5-3x faster, ~100 TSEG voxels per subject differ from the CPU;
        // TIGERBX_TF32=0 turns it off
        .with_tf32(std::env::var("TIGERBX_TF32").map_or(true, |v| v != "0"))
        .build()
        .error_on_failure()
}

/// GPU execution provider: DirectML on Windows (any DirectX 12 GPU, in the static build), CUDA elsewhere.
fn gpu_ep() -> (ort::ep::ExecutionProviderDispatch, bool) {
    if cfg!(windows) {
        (ort::ep::DirectML::default().build().error_on_failure(), false)
    } else {
        (cuda(), true)
    }
}

/// Create an ONNX inference session from model bytes, on the GPU (CUDA, or DirectML on Windows) when `gpu` is set and
/// the ONNX Runtime in use can, else on the CPU with a warning (printed once).
///
/// This should be called once and the session reused for all files.
pub fn create_session(model_bytes: &[u8], threads: usize, gpu: bool) -> Result<Session> {
    if gpu {
        let (ep, is_cuda) = gpu_ep();
        match builder(threads)?.with_execution_providers([ep]) {
            Ok(b) => {
                // GPU: keep only the logits (CUDA ArgMax of the labels output needs ~10x the logits as workspace);
                // the argmax runs on the CPU
                let pruned = crate::onnx_prune::output_names(model_bytes)
                    .filter(|names| names.len() > 1)
                    .and_then(|names| crate::onnx_prune::keep_outputs(model_bytes, &names[..1]));
                let bytes = pruned.as_deref().unwrap_or(model_bytes);
                let session = b.commit_from_memory(bytes).context("Failed to load ONNX model from memory")?;
                ON_GPU.store(true, Ordering::Relaxed);
                ON_CUDA.store(is_cuda, Ordering::Relaxed);
                return Ok(session);
            }
            Err(e) => {
                if !WARNED.swap(true, Ordering::Relaxed) {
                    if is_cuda {
                        eprintln!("Warning: -g given but the CUDA execution provider is not available ({e}); running on CPU.");
                        eprintln!("         Needs the GPU build of ONNX Runtime with CUDA 12 + cuDNN 9 libraries on the library path.");
                    } else {
                        eprintln!("Warning: -g given but DirectML is not available ({e}); running on CPU.");
                    }
                }
            }
        }
    }
    builder(threads)?.commit_from_memory(model_bytes).context("Failed to load ONNX model from memory")
}

/// True when at least one session runs on the GPU.
pub fn on_gpu() -> bool {
    ON_GPU.load(Ordering::Relaxed)
}

/// Run options that release the memory arena after each run (inputs differ in size; otherwise the arena keeps the
/// buffers of every size seen and memory grows over a batch, > 100 GB on mixed-resolution data).
pub fn run_options() -> Result<ort::session::RunOptions> {
    let mut opts = ort::session::RunOptions::new().context("Failed to create run options")?;
    let devices = if ON_CUDA.load(Ordering::Relaxed) { "cpu:0;gpu:0" } else { "cpu:0" };
    opts.add_config_entry("memory.enable_memory_arena_shrinkage", devices).context("Failed to enable arena shrinkage")?;
    Ok(opts)
}

/// Run inference on a pre-built session and return a label per voxel.
///
/// Input: 3D f32 view (H, W, D), passed to ONNX Runtime as (1, 1, H, W, D) without copying when it is contiguous.
/// Output (H, W, D) u8: a model with a ``labels`` output (tigerbx.dev.onnx_optimize --labels, e.g. the embedded
/// mprage_tbx_v005_opt / mprage_tseg_v0011_opt) returns it and only that output is requested; otherwise the
/// logits [1, C, H, W, D] give logit >= 0 (C = 1, i.e. sigmoid >= 0.5) or the argmax over channels.
pub fn run_onnx(session: &mut Session, input: ArrayView3<f32>) -> Result<Array3<u8>> {
    let input = input.as_standard_layout();
    let (h, w, d) = input.dim();
    let flat = input.as_slice().context("Input is not contiguous")?;
    let input_tensor = TensorRef::from_array_view(([1usize, 1, h, w, d], flat))
        .context("Failed to create input tensor")?;

    let input_name = session.inputs()[0].name().to_string();

    if session.outputs().iter().any(|o| o.name() == LABELS_OUTPUT) {
        let opts = run_options()?.with_outputs(OutputSelector::no_default().with(LABELS_OUTPUT));
        let outputs = session
            .run_with_options(ort::inputs![input_name.as_str() => input_tensor], &opts)
            .context("ONNX inference failed")?;
        // (1, H, W, D) uint8
        let (_, labels) = outputs[LABELS_OUTPUT]
            .try_extract_tensor::<u8>()
            .context("Failed to extract labels output")?;
        return Array3::from_shape_vec((h, w, d), labels.to_vec()).context("Unexpected output size");
    }

    let logits_name = session.outputs()[0].name().to_string();
    let opts = run_options()?.with_outputs(OutputSelector::no_default().with(logits_name.as_str()));
    let outputs = session
        .run_with_options(ort::inputs![input_name.as_str() => input_tensor], &opts)
        .context("ONNX inference failed")?;
    let (shape, logits) = outputs[logits_name.as_str()]
        .try_extract_tensor::<f32>()
        .context("Failed to extract output tensor")?;
    let channels = shape[1] as usize;
    let n = h * w * d;
    if logits.len() != channels * n {
        anyhow::bail!("Unexpected output size");
    }
    let labels: Vec<u8> = if channels == 1 {
        // sigmoid(x) >= 0.5 <=> x >= 0 (same rule as TigerBx and the labels output)
        logits.iter().map(|&x| (x >= 0.0) as u8).collect()
    } else {
        if channels > 256 {
            anyhow::bail!("{channels} channels do not fit in u8 labels");
        }
        (0..n)
            .map(|v| {
                let mut best = 0usize;
                for c in 1..channels {
                    if logits[c * n + v] > logits[best * n + v] {
                        best = c;
                    }
                }
                best as u8
            })
            .collect()
    };
    Array3::from_shape_vec((h, w, d), labels).context("Unexpected output size")
}
