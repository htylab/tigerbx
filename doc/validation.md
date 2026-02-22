### TigerBx: Validation Pipeline

---

## Quick start

```python
import tigerbx

# Lite mode (default): ≤20 randomly sampled files per dataset, cached in lite_list.json
tigerbx.val('/data/val_home')

# Full dataset
tigerbx.val('/data/val_home', full=True, output_dir='val_out')

# GPU
tigerbx.val('/data/val_home', GPU=True, output_dir='val_out')
```

`val()` auto-discovers dataset types under `val_home` and all immediate sub-directories
by probing for known file patterns — directory names are not fixed.

---

## Running a single task

Call the task-specific functions directly for more control:

```python
import tigerbx
from tigerbx.validate import (
    val_bet_synstrip, val_bet_NFBS,
    val_hlc_123, val_reg_60,
)
from tigerbx.validate import _val_seg_123  # aseg / dgm / syn

df, metric    = val_bet_synstrip('synthstrip_data_v1.4', output_dir='val_out', GPU=True)
df, metric    = val_bet_NFBS('NFBS_Dataset', output_dir='val_out', GPU=True)
data, metrics = _val_seg_123('aseg', 'a', 'aseg_data', output_dir='val_out', GPU=True)
data, metrics = _val_seg_123('dgm',  'd', 'aseg_data', output_dir='val_out', GPU=True)
data, metrics = _val_seg_123('syn',  'S', 'aseg_data', output_dir='val_out', GPU=True)
data, metrics = val_hlc_123('aseg_data', output_dir='val_out', GPU=True)
data, metrics = val_reg_60('reg_data',   output_dir='val_out', GPU=True,
                           template='Template_T1_tbet.nii.gz')
```

---

## Overriding models during validation

Every validation function accepts `bet_model` and `seg_model` keyword arguments.
`val()` accepts a `model` dict whose keys mirror those used in `run()`.

### Override BET only (applies to all tasks)

```python
# via val()
tigerbx.val('/data/val_home', model={'bet': 'new_bet_v2.onnx'}, output_dir='val_out')

# via direct call
val_bet_synstrip('synthstrip_data_v1.4', bet_model='new_bet_v2.onnx', output_dir='val_out')
```

### Override a segmentation model, keep default BET

```python
# Test new ASEG model — BET stays at default
tigerbx.val('/data/val_home', model={'aseg': 'mprage_aseg_v009.onnx'}, output_dir='val_out')

# Or call directly
_val_seg_123('aseg', 'a', 'aseg_data', seg_model='mprage_aseg_v009.onnx', output_dir='val_out')
```

### Override both BET and a segmentation model

```python
tigerbx.val('/data/val_home',
            model={'bet': 'new_bet.onnx', 'aseg': 'new_aseg.onnx'},
            output_dir='val_out')
```

### Model dict keys

| Key     | Applies to                        |
|---------|-----------------------------------|
| `'bet'` | all tasks (BET step)              |
| `'aseg'`| `_val_seg_123('aseg', ...)` only  |
| `'dgm'` | `_val_seg_123('dgm', ...)`  only  |
| `'syn'` | `_val_seg_123('syn', ...)`  only  |
| `'hlc'` | `val_hlc_123`               only  |
| `'reg'` | `val_reg_60`                only  |

---

## QC calibration

BET validation functions write CSVs containing `DICE` and `QC_raw` columns.
Use `qc_stat()` to analyse how well QC_raw predicts segmentation quality:

```python
import tigerbx

# Run BET validation (writes val_bet_synstrip.csv, val_bet_NFBS.csv)
tigerbx.val('/data/val_home', output_dir='val_out', full=True)

# Analyse QC–Dice relationship across one or both CSVs
tigerbx.qc_stat('val_out/val_bet_synstrip.csv')
tigerbx.qc_stat(['val_out/val_bet_synstrip.csv',
                 'val_out/val_bet_NFBS.csv'])
```

Output includes:
- Distribution stats for `QC_raw` and `DICE`
- Pearson r between `QC_raw` and `DICE`
- A suggested `qc_score` warning threshold (flags ≥ 90 % of failed cases)

### QC formula

```
qc_raw   = 1 - mean(entropy[brain_voxels]) / ln2
qc_score = int(clip(qc_raw * 100, 0, 100))

entropy  = -(p · log p + (1-p) · log(1-p))   # binary entropy, p = P(brain)
```

- Scores near 100 → confident, sharp brain boundary
- Scores near 0 → model uncertain across most voxels (likely poor extraction)
- Only predicted-brain voxels are used, so QC captures false positives but not false negatives

---

## Validation Datasets

1. **Skull Stripping** — [NFBS Dataset](http://preprocessed-connectomes-project.org/NFB_skullstripped)
2. **Skull Stripping** — [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip)
3. **Deep Gray Matter** — [MindBoggle-101](https://mindboggle.info/)
4. **Deep Gray Matter** — [CANDI](https://www.nitrc.org/projects/candi_share/)
5. **Registration** — [CC359](https://sites.google.com/view/calgary-campinas-dataset/home)

### Internal Validation Sets

- **seg123**: MindBoggle-101 + CANDI combined (N=123) with deep gray matter labels

---

## Validation Results — Version 0.1.20

### Deep Gray Matter: Dice Coefficients

| Structure   | L/R | aseg  | dgm   | syn   | hlc   | L/R | aseg  | dgm   | syn   | hlc   |
|-------------|-----|-------|-------|-------|-------|-----|-------|-------|-------|-------|
| Thalamus    | L   | 0.879 | 0.898 | 0.890 | 0.898 | R   | 0.889 | 0.902 | 0.884 | 0.902 |
| Caudate     | L   | 0.875 | 0.874 | 0.850 | 0.874 | R   | 0.875 | 0.872 | 0.845 | 0.870 |
| Putamen     | L   | 0.873 | 0.885 | 0.847 | 0.884 | R   | 0.862 | 0.880 | 0.829 | 0.879 |
| Pallidum    | L   | 0.827 | 0.827 | 0.828 | 0.824 | R   | 0.814 | 0.815 | 0.794 | 0.814 |
| Hippocampus | L   | 0.808 | 0.828 | 0.789 | 0.825 | R   | 0.810 | 0.831 | 0.779 | 0.829 |
| Amygdala    | L   | 0.737 | 0.764 | 0.716 | 0.756 | R   | 0.727 | 0.750 | 0.711 | 0.743 |
| **Mean**    | L   | 0.833 | 0.846 | 0.820 | 0.844 | R   | 0.829 | 0.841 | 0.807 | 0.839 |

### Registration

    mean Dice: 0.797
    mean Dice: 0.804 (FuseMorph)

### Skull Stripping

    bet_NFBS:     0.973
    bet_synstrip: 0.971
