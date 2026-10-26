# Validation

```python
summary = tigerbx.val(val_dir, output_dir=None, model=None, GPU=False, full=False,
                      template=None, task=None, tbx_model=None, seg_model=None)
```

Datasets are auto-discovered under `val_dir` (and its immediate sub-folders) by file pattern:

| Folder pattern | Dataset | Tasks |
|---|---|---|
| `*/image.nii.gz` | strip2 (skull stripping) | `tbx_strip2` |
| `*/*T1w.nii.gz` | strip1 (skull stripping) | `tbx_strip1` |
| `raw123/*.nii.gz` + `label123/` (same filenames) | seg123 | `tseg_123`, `dgm_123`, `syn_123`, `hlc_123` |
| `raw60/*.nii.gz` | reg60 | `reg_60` |

- `task` is a substring filter (`'tseg'`, `'dgm'`, `'hlc'`, `'tbx'`, `'reg'`; `'tseg_18'` = seg18, 30 TSEG structures, use `full=True`).
- seg123 scores **only the 12 deep-GM structures** (thalamus, caudate, putamen, pallidum, hippocampus, amygdala, L/R)
  even for `tseg`; it says nothing about cortex, WM, ventricles or cerebellum.
- `full=False` (default) = lite mode: ≤ 20 files per dataset, the file list is cached in `val_dir/lite_list.json`.
  Use `full=True` for numbers you report.
- `output_dir` gets one CSV per task (`val_tseg_123.csv`, ...); `None` writes no CSV.
- Return value `{name: metric}`; `metric` is `None` when the whole dataset failed (printed as `SKIPPED`).
  A file that fails (missing label, bad header, OOM) is printed as `FAILED <file>: <error>`, excluded from the mean
  and counted in the summary line.
- tBX is recomputed for every task/model; comparing N models costs N × tBX time.

Testing a new segmentation model (filename rules: [models.md](models.md)):

```python
r = tigerbx.val('/data/validate', output_dir='val_out/v010', GPU=True, full=True,
                task='tseg', seg_model='mprage_tseg_v010_test.onnx')
```

Keep `raw123/` clean: anything matching `<name>_tbx.nii.gz`, `<name>_tseg.nii.gz`, ... next to `<name>.nii.gz`
is skipped with a warning, but unrelated extra NIfTI files are treated as subjects and fail for lack of a label.
