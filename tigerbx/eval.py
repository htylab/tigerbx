"""
tigerbx.eval -- image quality, segmentation, and classification metrics.

Facade module that preserves the public `tigerbx.eval(...)` API while metric
implementations live in `tigerbx.core.metrics`.
"""

from tigerbx.core.metrics import (
    accuracy,
    asd,
    dice,
    f1,
    hd95,
    iou,
    ksg_mi,
    mae,
    mi,
    mse,
    ncc,
    precision,
    psnr,
    recall,
    ssim,
)

_REGISTRY = {
    "dice": dice,
    "iou": iou,
    "hd95": hd95,
    "asd": asd,
    "mae": mae,
    "mse": mse,
    "psnr": psnr,
    "ssim": ssim,
    "ncc": ncc,
    "mi": mi,
    "ksg_mi": ksg_mi,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
}


def eval(y_true, y_pred, metrics, **kwargs):
    """Evaluate one or more metrics between y_true and y_pred."""
    if isinstance(metrics, str):
        metrics = [metrics]

    unknown = [m for m in metrics if m not in _REGISTRY]
    if unknown:
        raise ValueError(f"Unknown metric(s): {unknown}. Available: {sorted(_REGISTRY)}")

    results = {}
    for name in metrics:
        fn = _REGISTRY[name]
        import inspect

        valid_params = inspect.signature(fn).parameters
        filtered = {k: v for k, v in kwargs.items() if k in valid_params}
        results[name] = fn(y_true, y_pred, **filtered)

    return results


__all__ = [
    "eval",
    "dice",
    "iou",
    "hd95",
    "asd",
    "mae",
    "mse",
    "psnr",
    "ssim",
    "ncc",
    "mi",
    "ksg_mi",
    "accuracy",
    "precision",
    "recall",
    "f1",
]
