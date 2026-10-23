use anyhow::{Context, Result};
use ndarray::{Array3, ArrayView3};
use ort::session::Session;
use ort::session::run_options::OutputSelector;
use ort::value::TensorRef;

/// Optional uint8 mask output of an optimised TigerBx model.
const LABELS_OUTPUT: &str = "labels";

/// Number of intra-op threads: 70% of the CPUs this process may use (same default as TigerBx).
/// Filling every core makes ONNX Runtime's worker threads compete with each other and with other jobs.
pub fn default_threads() -> usize {
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    ((n as f64 * 0.7) as usize).max(1)
}

/// Create an ONNX inference session from model bytes.
///
/// This should be called once and the session reused for all files.
pub fn create_session(model_bytes: &[u8], threads: usize) -> Result<Session> {
    let session = Session::builder()
        .context("Failed to create session builder")?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .context("Failed to set optimization level")?
        .with_intra_threads(threads)
        .context("Failed to set intra-op threads")?
        .with_inter_threads(1)
        .context("Failed to set inter-op threads")?
        // Every input has a different shape, so a memory pattern planned for one input is never reused.
        .with_memory_pattern(false)
        .context("Failed to disable memory pattern")?
        .commit_from_memory(model_bytes)
        .context("Failed to load ONNX model from memory")?;

    Ok(session)
}

/// Run inference on a pre-built session and return a label per voxel.
///
/// Input: 3D f32 view (H, W, D), passed to ONNX Runtime as (1, 1, H, W, D) without copying when it is contiguous.
/// Output (H, W, D) u8: a model with a ``labels`` output (tigerbx.dev.onnx_optimize --labels, e.g. the embedded
/// mprage_bet_v005_opt / mprage_aseg43_v0011_opt) returns it and only that output is requested; otherwise the
/// logits [1, C, H, W, D] give logit >= 0 (C = 1, i.e. sigmoid >= 0.5) or the argmax over channels.
pub fn run_onnx(session: &mut Session, input: ArrayView3<f32>) -> Result<Array3<u8>> {
    let input = input.as_standard_layout();
    let (h, w, d) = input.dim();
    let flat = input.as_slice().context("Input is not contiguous")?;
    let input_tensor = TensorRef::from_array_view(([1usize, 1, h, w, d], flat))
        .context("Failed to create input tensor")?;

    let input_name = session.inputs()[0].name().to_string();

    // Inputs differ in size, so release the arena after each run; otherwise it keeps the buffers
    // of every size seen and memory grows over a batch (> 100 GB on mixed-resolution data).
    const SHRINK: (&str, &str) = ("memory.enable_memory_arena_shrinkage", "cpu:0");

    if session.outputs().iter().any(|o| o.name() == LABELS_OUTPUT) {
        let mut opts = ort::session::RunOptions::new()
            .context("Failed to create run options")?
            .with_outputs(OutputSelector::no_default().with(LABELS_OUTPUT));
        opts.add_config_entry(SHRINK.0, SHRINK.1).context("Failed to enable arena shrinkage")?;
        let outputs = session
            .run_with_options(ort::inputs![input_name.as_str() => input_tensor], &opts)
            .context("ONNX inference failed")?;
        // (1, H, W, D) uint8
        let (_, labels) = outputs[LABELS_OUTPUT]
            .try_extract_tensor::<u8>()
            .context("Failed to extract labels output")?;
        return Array3::from_shape_vec((h, w, d), labels.to_vec()).context("Unexpected output size");
    }

    let mut opts = ort::session::RunOptions::new().context("Failed to create run options")?;
    opts.add_config_entry(SHRINK.0, SHRINK.1).context("Failed to enable arena shrinkage")?;
    let outputs = session
        .run_with_options(ort::inputs![input_name.as_str() => input_tensor], &opts)
        .context("ONNX inference failed")?;
    let (shape, logits) = outputs[0]
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
