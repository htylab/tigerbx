# Validation

```python
summary = tigerbx.val(val_dir, output_dir=None, model=None, GPU=False, full=False,
                      template=None, task=None, bet_model=None, seg_model=None)
```

Datasets are auto-discovered under `val_dir` (and its immediate sub-folders) by file pattern:

| Folder pattern | Dataset | Tasks |
|---|---|---|
| `*/image.nii.gz` | SynthStrip | `bet_synstrip` |
| `*/*T1w.nii.gz` | NFBS | `bet_NFBS` |
| `raw123/*.nii.gz` + `label123/` (same filenames) | seg123 = CANDI + MindBoggle | `aseg_123`, `dgm_123`, `syn_123`, `hlc_123` |
| `raw60/*.nii.gz` | reg60 | `reg_60` |

- `task` is a substring filter (`'aseg'`, `'dgm'`, `'syn'`, `'hlc'`, `'bet'`, `'reg'`).
- seg123 scores **only the 12 deep-GM structures** (thalamus, caudate, putamen, pallidum, hippocampus, amygdala, L/R)
  even for `aseg`; it says nothing about cortex, WM, ventricles or cerebellum.
- `full=False` (default) = lite mode: ≤ 20 files per dataset, the file list is cached in `val_dir/lite_list.json`.
  Use `full=True` for numbers you report.
- `output_dir` gets one CSV per task (`val_aseg_123.csv`, ...); `None` writes no CSV.
- Return value `{name: metric}`; `metric` is `None` when the whole dataset failed (printed as `SKIPPED`).
  A file that fails (missing label, bad header, OOM) is printed as `FAILED <file>: <error>`, excluded from the mean
  and counted in the summary line.
- tBET is recomputed for every task/model; comparing N models costs N × BET time.

Testing a new segmentation model (filename rules: [models.md](models.md)):

```python
r = tigerbx.val('/data/validate', output_dir='val_out/v010', GPU=True, full=True,
                task='aseg', seg_model='mprage_aseg43_v010_test.onnx')
```

Keep `raw123/` clean: anything matching `<name>_tbet.nii.gz`, `<name>_aseg.nii.gz`, ... next to `<name>.nii.gz`
is skipped with a warning, but unrelated extra NIfTI files are treated as subjects and fail for lack of a label.
