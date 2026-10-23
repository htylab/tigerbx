import os

import nibabel as nib
import numpy as np

import tigerbx
from tigerbx.core.tissuenorm import LAB, tissue_norm

TPL = os.path.join(os.path.dirname(tigerbx.__file__), 'template')


def _template():
    img = np.asarray(nib.load(os.path.join(TPL, 'mni_template.nii.gz')).dataobj, np.float32)
    aseg = np.asarray(nib.load(os.path.join(TPL, 'mni_template_aseg.nii.gz')).dataobj).astype(int)
    return img, img > 0, aseg


def test_tissue_norm_anchors_template_tissues():
    img, mask, aseg = _template()
    # a smooth multiplicative bias and a global scale should not move the anchors
    g = np.stack(np.meshgrid(*[np.linspace(-1, 1, n) for n in img.shape], indexing='ij'))
    y, peaks = tissue_norm(img * 3.0 * (1 + 0.15 * g[0]))
    assert 0 < peaks[0] < peaks[1] < peaks[2]
    assert y[~mask].max() == 0
    for k, target in (('csf', 1 / 3), ('gm', 2 / 3), ('wm', 1.0)):
        assert abs(np.median(y[np.isin(aseg, LAB[k]) & mask]) - target) < 0.08, k


def test_tissue_norm_inverted_contrast_falls_back_without_raising():
    img, mask, _ = _template()
    y, peaks = tissue_norm(np.where(mask, img.max() - img, 0))
    assert np.isnan(peaks).all()
    assert y[~mask].max() == 0 and 0.9 < np.percentile(y[mask & (y > 0)], 99) < 1.1


def test_tissue_norm_bad_mask_or_nan_does_not_raise():
    img, mask, _ = _template()
    y, peaks = tissue_norm(img, mask=mask[:-1])                # mask of the wrong shape
    assert np.isnan(peaks).all() and y.shape == img.shape and np.isfinite(y).all()
    x = img.copy()
    x[~mask] = np.nan                                          # NaN background
    y, peaks = tissue_norm(x)
    assert np.isfinite(y).all() and y[~mask].max() == 0


def test_template_prior_cache_key_uses_content():
    from tigerbx.core.tissuenorm import template_prior
    _, _, aseg = _template()
    p = template_prior(aseg).copy()
    q = template_prior(aseg[::-1].copy())                      # same shape and label sum, different anatomy
    np.testing.assert_allclose(q, p[:, ::-1], atol=1e-6)
