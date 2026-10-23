# `tigerbx.core.tissuenorm` — Tissue-anchored T1 intensity normalisation

Maps a skull-stripped T1 image on the MNI template grid (e.g. the `_affine` output of `tigerbx.reg('A', ...)`) to a
fixed tissue scale: background 0, CSF 1/3, GM 2/3, WM 1, piecewise linear in between (continued above WM).
Pure numpy/scipy, about 1.5 s per image on one thread.

Not used by any default pipeline: a model only benefits if it was trained on `tissue_norm` input (and its fixed
image normalised the same way).

---

### `tissue_norm(img, mask=None, seg=None, prior_aseg=None, bias=True, sigma=12.0, iters=3)`

| Parameter    | Type    | Description |
|--------------|---------|-------------|
| `img`        | ndarray | Skull-stripped T1 on the template grid |
| `mask`       | ndarray | Optional brain mask (same grid); default `img > 0` |
| `seg`        | ndarray | Optional aseg of the image itself; when given, tissue probabilities come from it |
| `prior_aseg` | ndarray | Template aseg used as the tissue prior (default: bundled `mni_template_aseg.nii.gz`) |
| `bias`       | bool    | Estimate and remove a multiplicative bias field (2 mm grid) |
| `sigma`      | float   | Bias-field smoothing in voxels |
| `iters`      | int     | Bias / tissue-fit iterations |

**Returns:** `(np.ndarray, np.ndarray)` — normalised image (float32) and the CSF / GM / WM anchor intensities.

Tissue probabilities: a 3-class GMM whose mixing weights are the smoothed template aseg (σ 2 mm), so no subject
segmentation is needed. Anchors are medians of high-probability (> 0.75), low-gradient voxels of each class; the CSF
anchor uses the ventricles only (aseg 4, 43, 14, 15, 24).

**Failure handling.** It never raises, so batch runs do not stop. If the anchors are not ordered CSF < GM < WM
(T2 / PD input, failed skull stripping, image not on the template grid), the mask does not match the image, or the
fit fails, a warning is logged
(`tigerbx` logger), `y` falls back to the image divided by the brain's 99th percentile (background 0, roughly the
same 0-1 range; non-finite voxels become 0, and `img > 0` replaces an unusable mask) and `peaks` is
`[nan, nan, nan]`. Check `np.isnan(peaks).any()` to flag the case in a QC report.

```python
from tigerbx.core.tissuenorm import tissue_norm

y, peaks = tissue_norm(t1_mni)
```
