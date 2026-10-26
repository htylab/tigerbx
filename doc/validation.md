# TigerBx Validation Guide

---

## 1. Basic Testing

Verify overall performance of the installed version.
Run from the directory containing your validation datasets:

```python
import tigerbx

# Lite mode (default): ≤20 randomly sampled files per dataset,
# cached in lite_list.json for reproducibility
tigerbx.val(GPU=True)

# Specify a directory explicitly
tigerbx.val('/data/val_home', GPU=True)

# Full dataset + save CSV results
tigerbx.val('/data/val_home', GPU=True, full=True, output_dir='val_out')
```

`val()` scans the given directory (defaults to cwd) and all immediate
sub-directories for known dataset patterns. Directory names are not fixed.

**Auto-detection rules:**

| Dataset       | File pattern              | Tasks run                        |
|---------------|---------------------------|----------------------------------|
| strip2        | `*/image.nii.gz`          | `tbx_strip2`                   |
| strip1        | `*/*T1w.nii.gz`           | `tbx_strip1`                       |
| seg123        | `raw123/*.nii.gz`         | `tseg`, `dgm`, `hlc`             |
| seg18        | `raw18/*.nii.gz` + `label18/` (same filenames) | `tseg_18`         |
| reg60         | `raw60/*.nii.gz`          | `reg`                            |

---

## 2. Development: Testing Individual Models

When training a new model, test only the relevant task while keeping all
other models at their defaults.

### Test a new tBX model

```python
# Runs both skull-stripping sets
tigerbx.val(GPU=True, task='tbx', tbx_model='mprage_tbx_v006_test.onnx', output_dir='val_out')
```

### Test a new TSEG model (tBX stays default)

```python
tigerbx.val(GPU=True, task='tseg', seg_model='mprage_tseg_v009_test.onnx', output_dir='val_out')
```

### Test other segmentation models

```python
tigerbx.val(GPU=True, task='dgm', seg_model='mprage_dgm12_v003_test.onnx',  output_dir='val_out')
tigerbx.val(GPU=True, task='hlc', seg_model='new_hlc.onnx',  output_dir='val_out')
tigerbx.val(GPU=True, task='reg', seg_model='new_reg.onnx',  output_dir='val_out')
```

### Override both tBX and segmentation model

```python
tigerbx.val(GPU=True, task='tseg',
            tbx_model='mprage_tbx_v006_new.onnx',
            seg_model='mprage_tseg_v010_new.onnx',
            output_dir='val_out')
```

tbx / tseg / dgm / syn model files must be named `<prefix>_<mode>_<version>_<tag>.onnx` (mode = `tbx`, `tseg`,
`dgm12`, ...): the mode in the filename selects the preprocessing and label table — see [models.md](models.md).

**Valid `task` values:** `'tbx'`, `'tseg'`, `'dgm'`, `'hlc'`, `'reg'` (substring match: `'tseg'` runs `tseg_123` and
`tseg_18`; `'tseg_18'` only seg18).

### seg18 (all TSEG structures)

`tseg_18` is the main TSEG accuracy check: the full pipeline (tBX + TSEG) on the raw seg18 T1w,
the prediction reoriented to RAS (the label grid), Dice against the manual labels with
the harmonisation in `validate.SEG18_STRUCTURES` (WM bilateral incl. 77 / 251-255, choroid plexus counted as
lateral ventricle, optic chiasm as background), summarised over 30 structures (CSF and vessels reported only).

```python
tigerbx.val(val_dir='/data/validate', task='tseg_18', full=True, output_dir='val_out')
# -> val_out/val_tseg_18.csv; default model mprage_tseg_v0011_opt: 0.8131
tigerbx.val(val_dir='/data/validate', task='tseg_18', full=True,
            seg_model='mprage_tseg_v0014_test.onnx')       # another TSEG model
```

Use `full=True`: lite mode scores a random subset.

---

## 3. Validation Datasets

| Set     | Task              |
|---------|-------------------|
| strip1  | Skull stripping   |
| strip2  | Skull stripping   |
| seg123  | Deep gray matter  |
| seg18   | TSEG              |
| reg60   | Registration      |

---

## 4. Validation Results — Version 0.1.20

### Skull Stripping (Dice)

| Dataset    | Dice  |
|------------|-------|
| strip1     | 0.973 |
| strip2     | 0.971 |

### Deep Gray Matter (Dice)

| Structure   | L/R | tseg  | dgm   | syn   | hlc   | L/R | tseg  | dgm   | syn   | hlc   |
|-------------|-----|-------|-------|-------|-------|-----|-------|-------|-------|-------|
| Thalamus    | L   | 0.879 | 0.898 | 0.890 | 0.898 | R   | 0.889 | 0.902 | 0.884 | 0.902 |
| Caudate     | L   | 0.875 | 0.874 | 0.850 | 0.874 | R   | 0.875 | 0.872 | 0.845 | 0.870 |
| Putamen     | L   | 0.873 | 0.885 | 0.847 | 0.884 | R   | 0.862 | 0.880 | 0.829 | 0.879 |
| Pallidum    | L   | 0.827 | 0.827 | 0.828 | 0.824 | R   | 0.814 | 0.815 | 0.794 | 0.814 |
| Hippocampus | L   | 0.808 | 0.828 | 0.789 | 0.825 | R   | 0.810 | 0.831 | 0.779 | 0.829 |
| Amygdala    | L   | 0.737 | 0.764 | 0.716 | 0.756 | R   | 0.727 | 0.750 | 0.711 | 0.743 |
| **Mean**    | L   | 0.833 | 0.846 | 0.820 | 0.844 | R   | 0.829 | 0.841 | 0.807 | 0.839 |

### Registration (Dice)

`tigerbx.val(task='reg')`, reg60, Dice against the template tseg after warping each subject's labels.

| Method                                 | Dice  |
|----------------------------------------|-------|
| `AD`: numreg affine + VFA (default)     | 0.852 |
| `AV`: C2FViT + VMnet (default up to 0.2.3) | 0.797 |
