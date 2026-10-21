---
name: tigerbx
description: "Use TigerBx 0.2.3+ for research-use neuroimaging workflows: T1w brain extraction and segmentation, HLC parcellation, registration, VBM, EPI/DTI distortion correction, NERVE embeddings, and image metrics. Trigger when choosing or calling tigerbx Python APIs/CLI, interpreting outputs or labels, or building TigerBx batch pipelines."
---

# TigerBx

Use this skill to select the smallest correct TigerBx API and avoid mixing pipeline contracts.

## Route the task

| Need | API | Read |
|---|---|---|
| BET, mask, ASEG, DeepGM, WMH, SynthSeg, CT, tissue PVE | `tigerbx.run()` | [bx.md](bx.md) |
| HLC parcellation or HLC tissue outputs | `tigerbx.hlc()` | [hlc.md](hlc.md) |
| Rigid/affine/nonlinear registration; apply saved warp | `tigerbx.reg()`, `tigerbx.transform()` | [reg.md](reg.md) |
| VBM | `tigerbx.vbm()` or `tigerbx.pipeline('vbm', ...)` | [reg.md](reg.md) |
| EPI/DTI distortion correction | `tigerbx.gdm()` | [gdm.md](gdm.md) |
| Hippocampus/amygdala VAE latent/reconstruction | `tigerbx.nerve()` | [nerve.md](nerve.md) |
| Dice, IoU, distances, IQA, MI, classification metrics | `tigerbx.eval()` | [eval.md](eval.md) |
| Numeric segmentation labels | — | [labels.md](labels.md) |
| Validate / compare models on benchmark datasets | `tigerbx.val()` | [val.md](val.md) |
| Use or ship a custom / retrained ONNX model | `model=` override | [models.md](models.md) |

Read only the routed file. Read `labels.md` only when label meaning is needed.

## Non-negotiable contracts

- Treat all outputs as research-use only; do not make clinical or diagnostic claims.
- `reg` plans are ordered uppercase steps: `R/A/V/N/C/F`. VBM is a separate API, never a reg step.
- Single NIfTI input normally returns a dict of nibabel objects; batch input returns a list of dicts containing saved paths. NERVE returns named path dicts because its primary artifact is NPZ.
- Access a nibabel image array with `.get_fdata()`.
- Models download automatically on first execution. Do not install packages or download models unless the user asks to run the workflow and the environment lacks them.
- For GPU: use `g` inside `tigerbx.run()` flags; use the documented `GPU=True` or `gpu=True` spelling for other APIs.

## Pitfalls

- **Files are written by default.** `run()` saves outputs (`save_outputs=True`); with `output=None` they go next to
  the input. Always pass an output folder, or `save_outputs=False` when you only need the returned images.
- Directory / glob inputs are sorted and skip `<name>_<tigerbx suffix>` files that sit next to `<name>`, but any other
  extra NIfTI in the folder is processed as a subject.
- GPU memory: ORT allocates only what each input needs; on a shared GPU cap it with `TIGERBX_GPU_MEM_LIMIT_GB=<int>`.
  CPU threads: `TIGERBX_ORT_INTRA_THREADS` / `TIGERBX_ORT_INTER_THREADS`.
- `GPU=True` silently runs on CPU if onnxruntime-gpu does not match the system CUDA (1.27+ is built for CUDA 13;
  `tigerbx[cu12]` pins `<1.27`). tigerbx warns once (`RuntimeWarning: GPU=True but running on CPU`); check
  `onnxruntime.get_available_providers()` and that the process appears in `nvidia-smi`.
- Use the public API (`tigerbx.run/hlc/reg/val/eval/...`). Internals moved in 0.2.x (`lib_tool.create_session` →
  `tigerbx.core.onnx`), so do not import or monkey-patch private modules.
- Model filenames carry the preprocessing contract — see [models.md](models.md).

## Execution check

Only when executing code, verify once:

```python
import tigerbx
print(tigerbx.__version__)  # expected >= 0.2.3
```

Prefer the Python API for composed workflows and the `tiger` CLI for one-off shell runs.
