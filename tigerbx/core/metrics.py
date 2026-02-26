"""Core metric implementations used by :mod:`tigerbx.eval` and internal modules."""

import numpy as np


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------

def _to_array(x):
    """Convert path / nibabel image / numpy array to a numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, str):
        import nibabel as nib
        return nib.load(x).get_fdata()
    # nibabel image (Nifti1Image, Nifti2Image, etc.)
    if hasattr(x, 'get_fdata'):
        return x.get_fdata()
    raise TypeError(f"Unsupported input type: {type(x)}")


def _label_list(y_true, labels, ignore_background):
    """Resolve the list of labels to evaluate."""
    if labels is not None:
        return list(labels)
    unique = np.unique(y_true.astype(int))
    if ignore_background:
        unique = unique[unique != 0]
    return list(unique)


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

def dice(y_true, y_pred, labels=None, ignore_background=True):
    """Dice similarity coefficient per label.

    Returns
    -------
    dict with integer-string keys ('10', '11', ?? and 'mean'.
    """
    y_true = _to_array(y_true).astype(int)
    y_pred = _to_array(y_pred).astype(int)
    if labels is None:
        label_list = np.unique(
            np.concatenate([y_true.astype(int).ravel(), y_pred.astype(int).ravel()])
        )
        if ignore_background:
            label_list = label_list[label_list != 0]
        label_list = list(label_list)
    else:
        label_list = _label_list(y_true, labels, ignore_background)

    scores = {}
    for lbl in label_list:
        gt = (y_true == lbl)
        pr = (y_pred == lbl)
        intersection = np.logical_and(gt, pr).sum()
        denom = gt.sum() + pr.sum()
        scores[str(lbl)] = float(2 * intersection / denom) if denom > 0 else 1.0

    scores['mean'] = float(np.mean(list(scores.values()))) if scores else float('nan')
    return scores


def iou(y_true, y_pred, labels=None, ignore_background=True):
    """Intersection-over-union (Jaccard index) per label.

    Returns
    -------
    dict with integer-string keys and 'mean'.
    """
    y_true = _to_array(y_true).astype(int)
    y_pred = _to_array(y_pred).astype(int)
    label_list = _label_list(y_true, labels, ignore_background)

    scores = {}
    for lbl in label_list:
        gt = (y_true == lbl)
        pr = (y_pred == lbl)
        inter = np.logical_and(gt, pr).sum()
        union = np.logical_or(gt, pr).sum()
        scores[str(lbl)] = float(inter / union) if union > 0 else 1.0

    scores['mean'] = float(np.mean(list(scores.values()))) if scores else float('nan')
    return scores


def _binary_surface_distances(gt, pr, voxel_spacing=None):
    """Return all surface-to-surface distances between two binary masks."""
    from scipy.ndimage import binary_erosion, generate_binary_structure

    def _surface(mask):
        struct = generate_binary_structure(mask.ndim, 1)
        eroded = binary_erosion(mask, structure=struct, border_value=1)
        return mask & ~eroded

    surf_gt = _surface(gt)
    surf_pr = _surface(pr)

    from scipy.spatial import KDTree

    coords_gt = np.argwhere(surf_gt)
    coords_pr = np.argwhere(surf_pr)

    if len(coords_gt) == 0 or len(coords_pr) == 0:
        return None  # undefined

    if voxel_spacing is not None:
        coords_gt = coords_gt * np.array(voxel_spacing)
        coords_pr = coords_pr * np.array(voxel_spacing)

    tree_pr = KDTree(coords_pr)
    dist_gt_to_pr, _ = tree_pr.query(coords_gt)

    tree_gt = KDTree(coords_gt)
    dist_pr_to_gt, _ = tree_gt.query(coords_pr)

    return np.concatenate([dist_gt_to_pr, dist_pr_to_gt])


def hd95(y_true, y_pred, labels=None, ignore_background=True, voxel_spacing=None):
    """95th-percentile Hausdorff distance per label (in voxels by default).

    Parameters
    ----------
    voxel_spacing : sequence of float, optional
        Physical voxel size (e.g. [1.0, 1.0, 1.0] mm). If None, distances
        are in voxels.

    Returns
    -------
    dict with integer-string keys and 'mean'.
    """
    y_true = _to_array(y_true).astype(int)
    y_pred = _to_array(y_pred).astype(int)
    label_list = _label_list(y_true, labels, ignore_background)

    scores = {}
    for lbl in label_list:
        dists = _binary_surface_distances(y_true == lbl, y_pred == lbl, voxel_spacing)
        if dists is None:
            scores[str(lbl)] = float('nan')
        else:
            scores[str(lbl)] = float(np.percentile(dists, 95))

    scores['mean'] = float(np.nanmean(list(scores.values()))) if scores else float('nan')
    return scores


def asd(y_true, y_pred, labels=None, ignore_background=True, voxel_spacing=None):
    """Average symmetric surface distance per label.

    Returns
    -------
    dict with integer-string keys and 'mean'.
    """
    y_true = _to_array(y_true).astype(int)
    y_pred = _to_array(y_pred).astype(int)
    label_list = _label_list(y_true, labels, ignore_background)

    scores = {}
    for lbl in label_list:
        dists = _binary_surface_distances(y_true == lbl, y_pred == lbl, voxel_spacing)
        if dists is None:
            scores[str(lbl)] = float('nan')
        else:
            scores[str(lbl)] = float(np.mean(dists))

    scores['mean'] = float(np.nanmean(list(scores.values()))) if scores else float('nan')
    return scores


# ---------------------------------------------------------------------------
# Reconstruction metrics
# ---------------------------------------------------------------------------

def mae(y_true, y_pred):
    """Mean absolute error."""
    y_true = _to_array(y_true).astype(float)
    y_pred = _to_array(y_pred).astype(float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred):
    """Mean squared error."""
    y_true = _to_array(y_true).astype(float)
    y_pred = _to_array(y_pred).astype(float)
    return float(np.mean((y_true - y_pred) ** 2))


def psnr(y_true, y_pred, data_range=None):
    """Peak signal-to-noise ratio (dB).

    Parameters
    ----------
    data_range : float, optional
        Maximum possible value. Defaults to max(y_true) - min(y_true).
    """
    y_true = _to_array(y_true).astype(float)
    y_pred = _to_array(y_pred).astype(float)
    if data_range is None:
        data_range = y_true.max() - y_true.min()
    if data_range == 0:
        return float('inf')
    mse_val = np.mean((y_true - y_pred) ** 2)
    if mse_val == 0:
        return float('inf')
    return float(10 * np.log10(data_range ** 2 / mse_val))


def ssim(y_true, y_pred, data_range=None, sigma=1.5):
    """Structural similarity index (global, using Gaussian-weighted statistics).

    Parameters
    ----------
    data_range : float, optional
        Defaults to max(y_true) - min(y_true).
    sigma : float
        Standard deviation for Gaussian kernel (default 1.5).
    """
    from scipy.ndimage import gaussian_filter

    y_true = _to_array(y_true).astype(float)
    y_pred = _to_array(y_pred).astype(float)

    if data_range is None:
        data_range = y_true.max() - y_true.min()

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = gaussian_filter(y_true, sigma)
    mu2 = gaussian_filter(y_pred, sigma)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(y_true ** 2, sigma) - mu1_sq
    sigma2_sq = gaussian_filter(y_pred ** 2, sigma) - mu2_sq
    sigma12 = gaussian_filter(y_true * y_pred, sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))


def ncc(y_true, y_pred):
    """Normalized cross-correlation (global, scalar)."""
    y_true = _to_array(y_true).astype(float).ravel()
    y_pred = _to_array(y_pred).astype(float).ravel()

    y_true_c = y_true - y_true.mean()
    y_pred_c = y_pred - y_pred.mean()

    denom = np.sqrt((y_true_c ** 2).sum() * (y_pred_c ** 2).sum())
    if denom == 0:
        return float('nan')
    return float(np.dot(y_true_c, y_pred_c) / denom)


def mi(y_true, y_pred, bins=64):
    """Mutual information (histogram-based).

    Parameters
    ----------
    bins : int
        Number of histogram bins (default 64).
    """
    y_true = _to_array(y_true).astype(float).ravel()
    y_pred = _to_array(y_pred).astype(float).ravel()

    hist_2d, _, _ = np.histogram2d(y_true, y_pred, bins=bins)
    p_xy = hist_2d / hist_2d.sum()

    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    mi_val = 0.0
    for i in range(p_xy.shape[0]):
        for j in range(p_xy.shape[1]):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi_val += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))

    return float(mi_val)


def ksg_mi(y_true, y_pred, k=5):
    """Mutual information via the KSG k-NN estimator.

    Parameters
    ----------
    k : int
        Number of nearest neighbours (default 5).
    """
    from scipy.special import digamma
    from scipy.spatial import KDTree

    y_true = _to_array(y_true).astype(float).ravel()
    y_pred = _to_array(y_pred).astype(float).ravel()

    n = len(y_true)
    data = np.column_stack([y_true, y_pred])

    tree_xy = KDTree(data)
    tree_x = KDTree(y_true.reshape(-1, 1))
    tree_y = KDTree(y_pred.reshape(-1, 1))

    # Chebyshev distance to k-th neighbour in joint space (excluding self)
    dists, _ = tree_xy.query(data, k=k + 1, workers=-1)
    eps = dists[:, -1]  # k-th neighbour distance (Chebyshev)

    # Count points within eps in marginal spaces
    nx = np.array([len(tree_x.query_ball_point([[x]], r, workers=1)[0]) - 1
                   for x, r in zip(y_true, eps)])
    ny = np.array([len(tree_y.query_ball_point([[y]], r, workers=1)[0]) - 1
                   for y, r in zip(y_pred, eps)])

    mi_val = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return float(mi_val)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def _clf_prf(y_true, y_pred, average):
    """Per-label and aggregated (precision, recall, F1) ??pure numpy."""
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n = len(labels)
    tps = np.empty(n, dtype=float)
    fps = np.empty(n, dtype=float)
    fns = np.empty(n, dtype=float)
    for i, lbl in enumerate(labels):
        pp = y_pred == lbl
        tp = y_true == lbl
        tps[i] = (pp & tp).sum()
        fps[i] = (pp & ~tp).sum()
        fns[i] = (~pp & tp).sum()
    sup = tps + fns
    ps = np.where(tps + fps > 0, tps / (tps + fps), 0.0)
    rs = np.where(tps + fns > 0, tps / (tps + fns), 0.0)
    fs = np.where(ps + rs > 0, 2 * ps * rs / (ps + rs), 0.0)

    if average == 'micro':
        tp_s, fp_s, fn_s = tps.sum(), fps.sum(), fns.sum()
        p = tp_s / (tp_s + fp_s) if tp_s + fp_s > 0 else 0.0
        r = tp_s / (tp_s + fn_s) if tp_s + fn_s > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    elif average == 'weighted':
        s = sup.sum()
        w = sup / s if s > 0 else np.ones(n) / n
        p, r, f = float(np.dot(ps, w)), float(np.dot(rs, w)), float(np.dot(fs, w))
    elif average == 'binary':
        p, r, f = float(ps[-1]), float(rs[-1]), float(fs[-1])
    else:  # macro
        p, r, f = float(ps.mean()), float(rs.mean()), float(fs.mean())

    return p, r, f


def accuracy(y_true, y_pred):
    """Classification accuracy."""
    y_true = _to_array(y_true).astype(int).ravel()
    y_pred = _to_array(y_pred).astype(int).ravel()
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred, average='macro'):
    """Classification precision.

    Parameters
    ----------
    average : str
        Averaging strategy: 'macro', 'micro', 'weighted', 'binary'.
    """
    y_true = _to_array(y_true).astype(int).ravel()
    y_pred = _to_array(y_pred).astype(int).ravel()
    p, _, _ = _clf_prf(y_true, y_pred, average)
    return p


def recall(y_true, y_pred, average='macro'):
    """Classification recall."""
    y_true = _to_array(y_true).astype(int).ravel()
    y_pred = _to_array(y_pred).astype(int).ravel()
    _, r, _ = _clf_prf(y_true, y_pred, average)
    return r


def f1(y_true, y_pred, average='macro'):
    """F1 score."""
    y_true = _to_array(y_true).astype(int).ravel()
    y_pred = _to_array(y_pred).astype(int).ravel()
    _, _, f = _clf_prf(y_true, y_pred, average)
    return f
