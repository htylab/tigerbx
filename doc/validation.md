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
| SynthStrip    | `*/image.nii.gz`          | `bet_synstrip`                   |
| NFBS          | `*/*T1w.nii.gz`           | `bet_NFBS`                       |
| seg123        | `raw123/*.nii.gz`         | `aseg`, `dgm`, `syn`, `hlc`      |
| reg60         | `raw60/*.nii.gz`          | `reg`                            |

---

## 2. Development: Testing Individual Models

When training a new model, test only the relevant task while keeping all
other models at their defaults.

### Test a new BET model

```python
# Runs both SynthStrip and NFBS skull-stripping datasets
tigerbx.val(GPU=True, task='bet', bet_model='mprage_bet_v006.onnx', output_dir='val_out')
```

### Test a new ASEG model (BET stays default)

```python
tigerbx.val(GPU=True, task='aseg', seg_model='mprage_aseg_v009.onnx', output_dir='val_out')
```

### Test other segmentation models

```python
tigerbx.val(GPU=True, task='dgm', seg_model='new_dgm.onnx',  output_dir='val_out')
tigerbx.val(GPU=True, task='syn', seg_model='new_syn.onnx',  output_dir='val_out')
tigerbx.val(GPU=True, task='hlc', seg_model='new_hlc.onnx',  output_dir='val_out')
tigerbx.val(GPU=True, task='reg', seg_model='new_reg.onnx',  output_dir='val_out')
```

### Override both BET and segmentation model

```python
tigerbx.val(GPU=True, task='aseg',
            bet_model='new_bet.onnx',
            seg_model='new_aseg.onnx',
            output_dir='val_out')
```

**Valid `task` values:** `'bet'`, `'aseg'`, `'dgm'`, `'syn'`, `'hlc'`, `'reg'`

---

## 3. QC Calibration (BET)

BET validation records a `QC_raw` confidence score for each image alongside
its Dice coefficient. Use `qc_stat()` to analyse the QC–Dice relationship
and find an appropriate warning threshold.

### Step 1 — Run BET validation and save CSVs

```python
tigerbx.val(GPU=True, task='bet', full=True, output_dir='val_out')
# Produces: val_out/val_bet_synstrip.csv  and  val_out/val_bet_NFBS.csv
```

### Step 2 — Analyse the QC–Dice relationship

```python
# Single CSV
tigerbx.qc_stat('val_out/val_bet_synstrip.csv')

# Combine both datasets (recommended — more samples, better statistics)
tigerbx.qc_stat(['val_out/val_bet_synstrip.csv',
                 'val_out/val_bet_NFBS.csv'])
```

**Output:**
- Distribution statistics for `QC_raw` and `DICE`
- Pearson r between `QC_raw` and `DICE`
- Suggested `qc_score` warning threshold (captures ≥ 90 % of failed cases)

### Step 3 — Apply the suggested threshold

```
Example output:
  Suggested threshold : QC_raw = 0.6123  →  qc_score = 81
  Recommendation: change the warning threshold in bx.py run_args()
    from:  if qc_score < 66:
    to:    if qc_score < 81:
```

### QC formula

```
qc_raw      = 1 - mean(entropy[brain_voxels]) / ln2
_QC_RAW_GOOD = 0.7581           # calibrated: qc_raw at Dice ≈ 0.95
qc_score    = int(clip(qc_raw / _QC_RAW_GOOD × 100, 0, 100))

entropy  = -(p · log p + (1−p) · log(1−p))   # binary entropy, p = P(brain)
```

- Score = 100: qc_raw ≥ 0.7581 (predicted Dice ≥ 0.95 — good extraction)
- Score < 100: qc_raw below calibrated good threshold; extraction may be imperfect
- Score → 0: model is very uncertain; extraction likely poor
- Only predicted-brain voxels are used — captures false positives, not false negatives

---

## 4. Validation Datasets

| #  | Task              | Dataset                                                                              |
|----|-------------------|--------------------------------------------------------------------------------------|
| 1  | Skull Stripping   | [NFBS Dataset](http://preprocessed-connectomes-project.org/NFB_skullstripped)       |
| 2  | Skull Stripping   | [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip)                    |
| 3  | Deep Gray Matter  | [MindBoggle-101](https://mindboggle.info/)                                          |
| 4  | Deep Gray Matter  | [CANDI](https://www.nitrc.org/projects/candi_share/)                                |
| 5  | Registration      | [CC359](https://sites.google.com/view/calgary-campinas-dataset/home)                |

**Internal set — seg123:** MindBoggle-101 + CANDI combined (N = 123), deep gray matter labels

---

## 5. Validation Results — Version 0.1.20

### Skull Stripping (Dice)

| Dataset    | Dice  |
|------------|-------|
| NFBS       | 0.973 |
| SynthStrip | 0.971 |

### Deep Gray Matter (Dice)

| Structure   | L/R | aseg  | dgm   | syn   | hlc   | L/R | aseg  | dgm   | syn   | hlc   |
|-------------|-----|-------|-------|-------|-------|-----|-------|-------|-------|-------|
| Thalamus    | L   | 0.879 | 0.898 | 0.890 | 0.898 | R   | 0.889 | 0.902 | 0.884 | 0.902 |
| Caudate     | L   | 0.875 | 0.874 | 0.850 | 0.874 | R   | 0.875 | 0.872 | 0.845 | 0.870 |
| Putamen     | L   | 0.873 | 0.885 | 0.847 | 0.884 | R   | 0.862 | 0.880 | 0.829 | 0.879 |
| Pallidum    | L   | 0.827 | 0.827 | 0.828 | 0.824 | R   | 0.814 | 0.815 | 0.794 | 0.814 |
| Hippocampus | L   | 0.808 | 0.828 | 0.789 | 0.825 | R   | 0.810 | 0.831 | 0.779 | 0.829 |
| Amygdala    | L   | 0.737 | 0.764 | 0.716 | 0.756 | R   | 0.727 | 0.750 | 0.711 | 0.743 |
| **Mean**    | L   | 0.833 | 0.846 | 0.820 | 0.844 | R   | 0.829 | 0.841 | 0.807 | 0.839 |

### Registration (Dice)

| Method    | Dice  |
|-----------|-------|
| Default   | 0.797 |
| FuseMorph | 0.804 |
