# TigerBx Release Notes — v0.2.4

## Highlights

- **New default ASEG model** — `mprage_aseg43_v0011_allseg.onnx` replaces
  `mprage_aseg43_v007_16ksynth.onnx` (same 43 labels, same I/O)
- **GPU memory** — no more out-of-memory errors when many differently sized
  inputs run in one session
- **`tigerbx.val()`** — a failing file no longer skips the whole dataset
- **Input folders** — previous tigerbx outputs are no longer picked up as new
  subjects
- **Model contract docs** — `doc/models.md` and agent-facing skills pages

---

## 1. New default ASEG model (v0011)

`tiger bx -a` / `tigerbx.run('a', ...)` now uses
`mprage_aseg43_v0011_allseg.onnx`, a larger U-Net (10.0M parameters, base 32).

### Accuracy

| model | IBSR18, 30 structures (manual) | seg123, 12 deep-GM structures |
|---|---|---|
| v007 (0.2.3 default) | 0.8076 | 0.8313 |
| **v0011 (0.2.4 default)** | **0.8129** | **0.8344** |

v0011 is better in 17/18 IBSR18 subjects (Wilcoxon p = 1.5e-5) and 92/123
seg123 subjects (p = 5e-11). Largest gains: ventral DC (+0.03/+0.04), cerebral
cortex (+0.013/+0.018), brain stem (+0.015); accumbens is slightly lower
(-0.01/-0.02).

### Compatibility

- Output labels are identical (`label_all['aseg43']`, 43 structures); every
  label, including vessel, choroid plexus, WM-hypointensities and optic
  chiasm, is still produced.
- **Volumes change.** Do not mix ASEG results from 0.2.3 and 0.2.4 in one
  study; keep the old model with
  `model={'aseg': 'mprage_aseg43_v007_16ksynth.onnx'}`.
- Cost: roughly 1.3x the CPU time of v007 (8 threads, 141×180×138 crop, busy
  shared server: v0011 2.5-3.4 s vs v007 1.7-2.5 s per run) and peak RAM
  4.6 GB vs 3.9 GB; GPU time is unchanged (~0.2 s).

---

## 2. Runtime and validation fixes

- **GPU memory** (`core/onnx.py`): the CUDA provider allocates as requested
  (`arena_extend_strategy=kSameAsRequested`) instead of growing a
  power-of-two arena for every new crop shape. Optional cap:
  `TIGERBX_GPU_MEM_LIMIT_GB`.
- **`tigerbx.val()`** (`validate.py`): a failing file is printed as
  `FAILED <file>: <error>`, excluded from the mean and counted in the summary;
  batch runs fall back to file-by-file.
- **Input resolution** (`core/io.py`): directory / glob inputs are sorted and
  `<stem>_<tigerbx postfix>` files are skipped (with a warning) when `<stem>`
  is present; explicit file lists are untouched.
- **Model filenames** (`lib_bx.get_mode`): a clear `ValueError` for names
  that do not follow `<prefix>_<mode>_<version>_<tag>.onnx`.

---

## 3. Docs and tests

- New `doc/models.md`: filename rules, input/output contract and export
  checklist for replacement models.
- New `skills/tigerbx/val.md` and `skills/tigerbx/models.md`; `SKILL.md`
  pitfalls section.
- Tests: input resolver, `get_mode`, `_run_loop` failure handling.
- `tests/run_and_compare.py bx` reference outputs need to be regenerated:
  ASEG outputs differ by design.
- package version is bumped from **`0.2.3`** to **`0.2.4`**
