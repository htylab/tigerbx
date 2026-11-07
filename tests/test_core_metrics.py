import numpy as np

import pytest

from tigerbx.core.metrics import asd, dice, hd95, iou, ksg_mi, ssim


def test_segmentation_metrics_include_prediction_only_labels():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 2])

    for metric in (dice, iou, hd95, asd):
        scores = metric(y_true, y_pred)
        assert set(scores) == {'1', '2', 'mean'}


def test_explicit_labels_remain_authoritative():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 2])

    scores = iou(y_true, y_pred, labels=[1])

    assert set(scores) == {'1', 'mean'}


def test_ssim_handles_constant_images():
    constant = np.ones((5, 5), dtype=float)

    assert ssim(constant, constant) == 1.0
    assert ssim(constant, np.zeros_like(constant)) == 0.0


def test_ksg_mi_validates_inputs_and_returns_finite_value():
    x = np.arange(10, dtype=float)
    y = x * 2

    assert np.isfinite(ksg_mi(x, y, k=2))
    with pytest.raises(ValueError, match='same number'):
        ksg_mi(x, y[:-1], k=2)
    with pytest.raises(ValueError, match='k must'):
        ksg_mi(x, y, k=len(x))
