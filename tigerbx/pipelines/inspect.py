"""Image inspection: what kind of brain MRI an input is, from one small 3D CNN (about 1 s per image on one CPU thread).

Per input (any contrast, any orientation; 4D: the first volume):
  contrast         T1w (contrast-enhanced T1 included), T2w, FLAIR, b0, DWI (b >= ~800) or other (PD, GRE, ASL,
                   angiography, quantitative maps, ...), with the probability of each class
  skull_stripped   the image has already been skull-stripped
  epi              an EPI acquisition (diffusion, ASL)
  orientation      the anatomical direction of the image's +x, +y, +z after its header: 'RAS' = as the header says;
                   e.g. 'RPS' = anterior-posterior flipped, 'SAR' = axes swapped. Left-right cannot be told apart
                   (a brain is near symmetric), so the first letter of an R-L axis is always R
  truncated        part of the head lies outside the field of view
  motion           motion artifact score 0-1 (T1w, T2w and FLAIR only; empty for other contrasts) and motion_flag
                   (score > 0.5)
Rule-based checks (header and data, no model): number of volumes, voxel size and thick slices, obliqueness, field of
view, non-finite voxels, label map / mask instead of an image, empty image, missing spatial header; `warnings` lists
the ones that apply.

Input of the network: a 256 mm cube at 2.67 mm (96^3) on RAS axes (from the header) centred on the intensity centre
of mass, with four channels: intensity / p99, the fraction of empty (0 / non-finite) voxels, the inside-field-of-view
mask and |image - 3x3x3 box mean| / p99 computed on the native grid (motion ringing and blur, which a coarse grid
alone loses), plus header features (voxel size, matrix, oblique, sform / qform codes, float data). The contrast
comes from the image only, not from the header.

Results are written to ``inspect_summary.csv`` in the output folder (one row per input). Research use only.
"""
import csv
import json
import os
import logging
from types import SimpleNamespace

import numpy as np
import nibabel as nib
from scipy import ndimage
from tqdm import tqdm

from tigerbx import lib_tool
from tigerbx.core import models
from tigerbx.core.io import resolve_inputs
from tigerbx.core.onnx import create_session

_logger = logging.getLogger('tigerbx')
_logger.addHandler(logging.NullHandler())

DEFAULT_MODEL = models.INSPECT
N, STEP, BLOCK = 96, 256 / 96, 2.0       # output cube, spacing (mm), block average target before the anti-alias
CONTRASTS = ('T1w', 'T2w', 'FLAIR', 'b0', 'DWI', 'other')
MOTION_CONTRASTS = ('T1w', 'T2w', 'FLAIR')
PERMS = [(a, b, c) for a in range(3) for b in range(3) for c in range(3) if len({a, b, c}) == 3]
SUMMARY_NAME = 'inspect_summary.csv'
CHECK_COLUMNS = ('n_volumes', 'ndim', 'matrix', 'voxel_mm', 'thick_slices', 'oblique_deg', 'fov_mm', 'nonfinite_frac',
                 'label_map')
# detail=True: every value; default: the decisions, with the flagged findings spelled out in `warnings`
DETAIL_COLUMNS = (('file', 'status', 'contrast', 'p_contrast') + tuple(f'p_{c}' for c in CONTRASTS) +
                  ('skull_stripped', 'p_skull_stripped', 'epi', 'p_epi', 'orientation', 'p_orientation',
                   'truncated', 'p_truncated', 'motion', 'motion_flag') + CHECK_COLUMNS + ('warnings', 'error'))
SUMMARY_COLUMNS = ('file', 'status', 'contrast', 'p_contrast', 'skull_stripped', 'epi', 'orientation', 'truncated',
                   'motion', 'ndim', 'matrix', 'voxel_mm', 'warnings', 'error')


def _volume(img):
    d = np.asarray(img.dataobj, np.float32)
    if d.ndim > 3:
        d = d.reshape(d.shape[:3] + (-1,))[..., 0]
    return d


def header_features(img, shape):
    """log voxel size (3), matrix / 256 (3), obliqueness, sform 4, qform 4, (unused, 0), float data."""
    A = img.affine
    z = np.sqrt((A[:3, :3] ** 2).sum(0))
    oblique = float(np.abs(np.abs(A[:3, :3] / z) - np.round(np.abs(A[:3, :3] / z))).max())
    h = img.header
    sform = int(h['sform_code']) if 'sform_code' in h else 0
    qform = int(h['qform_code']) if 'qform_code' in h else 0
    return np.r_[np.log(np.maximum(z, 0.1)), np.array(shape) / 256, oblique, sform == 4, qform == 4, 0,
                 img.get_data_dtype().kind == 'f'].astype(np.float32)


def make_input(img, data=None):
    """(4, 96, 96, 96) network input. The high-frequency map is taken on the native grid; then image, empty fraction
    and high-frequency map are block-averaged by an integer factor per axis to <= 2 mm (the box takes part of the
    anti-alias, a Gaussian the rest) and resampled trilinearly; p99 and the centre of mass come from every k-th voxel
    of the full-resolution data."""
    d = _volume(img) if data is None else np.asarray(data, np.float32)
    finite = np.isfinite(d)
    d = np.where(finite, d, 0)
    empty = ((~finite) | (d == 0)).astype(np.float32)
    A0 = img.affine
    z0 = np.sqrt((A0[:3, :3] ** 2).sum(0))
    k = np.maximum(np.floor(BLOCK / z0), 1).astype(int)
    shape0 = np.array(d.shape)
    sub, esub = d[::k[0], ::k[1], ::k[2]], empty[::k[0], ::k[1], ::k[2]] > 0
    nz = sub[~esub]
    p99 = float(np.percentile(nz, 99)) if nz.size else 1.0
    fg = sub > 0.05 * p99
    idx = (np.argwhere(fg) if fg.any() else np.argwhere(np.ones(sub.shape, bool))) * k
    w = sub[fg] if fg.any() else np.ones(len(idx))
    c = A0[:3, :3] @ (idx * w[:, None]).sum(0) / w.sum() + A0[:3, 3]           # world centre of mass
    hf = np.abs(d - ndimage.uniform_filter(d, 3))
    vols = [d, empty, hf]
    A = A0
    if (k > 1).any():
        n = shape0 // k
        sl = tuple(slice(0, n[i] * k[i]) for i in range(3))
        vols = [a[sl].reshape(n[0], k[0], n[1], k[1], n[2], k[2]).mean((1, 3, 5)) for a in vols]
        A = A0.copy()
        A[:3, :3] = A0[:3, :3] * k
        A[:3, 3] = A0[:3, :3] @ ((k - 1) / 2) + A0[:3, 3]
    # Gaussian of sigma0 = sqrt((STEP / z)^2 - 1) / 2 voxels at the full resolution, minus the box's variance
    var0 = (np.sqrt(np.maximum((STEP / z0) ** 2 - 1, 0)) / 2) ** 2
    sig = np.sqrt(np.maximum(var0 - (k ** 2 - 1) / 12, 0)) / k
    if sig.any():
        vols = [ndimage.gaussian_filter(a, sig) for a in vols]
    g = (np.indices((N, N, N)).reshape(3, -1).T - (N - 1) / 2) * STEP + c       # world, RAS axes
    v = (np.linalg.inv(A) @ np.c_[g, np.ones(len(g))].T)[:3]
    v0 = (np.linalg.inv(A0) @ np.c_[g, np.ones(len(g))].T)[:3]
    inside = np.all((v0 >= -0.5) & (v0 <= shape0[:, None] - 0.5), 0)
    ds, es, hs = (ndimage.map_coordinates(a, v, order=1, mode='nearest') for a in vols)
    x = np.stack([np.clip(ds * inside / max(p99, 1e-6), 0, 1.5), np.where(inside, es, 1), inside.astype(np.float32),
                  np.clip(hs * inside / max(p99, 1e-6), 0, 1.5)]).reshape(4, N, N, N)
    return x.astype(np.float32)


def _clean(d):
    """(data with NaN / Inf set to 0, fraction of such voxels): only the report sees the non-finite voxels."""
    finite = np.isfinite(d)
    return np.where(finite, d, 0).astype(np.float32), float(1 - finite.mean())


def rule_checks(img, d=None, nonfinite=None):
    """Header / data checks without a model -> dict of CHECK_COLUMNS + `warnings` (text, '; ' separated).
    d: the first volume with NaN / Inf set to 0 and nonfinite their fraction (_clean); None: from img."""
    shape = img.shape
    n_vol = int(np.prod(shape[3:])) if len(shape) > 3 else 1
    if d is None:
        d, nonfinite = _clean(_volume(img))
    nonfinite = nonfinite or 0.0
    A = img.affine
    z = np.sqrt((A[:3, :3] ** 2).sum(0))
    # angle of each voxel axis to its nearest world axis
    oblique = float(np.degrees(np.arccos(np.clip(np.abs(A[:3, :3] / z).max(0), 0, 1))).max())
    fov = z * np.array(shape[:3])
    q = np.maximum(np.array(d.shape) // 64, 1)                     # a strided sample for the value statistics
    sub = d[::q[0], ::q[1], ::q[2]].ravel()
    integer = sub.size > 0 and bool(np.all(sub == np.round(sub)))
    uniq = np.unique(sub) if integer else np.array([])
    span = float(uniq[-1] - uniq[0] + 1) if uniq.size else 0.0
    label_map = integer and uniq.size > 1 and (uniq.size <= 20 or (uniq.size <= 300 and uniq.size / span < 0.5))
    thick = bool(z.max() >= 3 or z.max() / max(z.min(), 1e-6) >= 3)
    h = img.header
    no_space = ('sform_code' in h and int(h['sform_code']) == 0 and int(h['qform_code']) == 0)
    w = []
    if n_vol > 1:
        w.append(f'{n_vol} volumes, only the first is used')
    if thick:
        w.append('thick slices / low resolution')
    if oblique > 20:                    # tilted acquisitions of a few degrees are common and handled
        w.append(f'oblique header ({oblique:.0f} deg)')
    if fov.min() < 80:
        w.append(f'slab: field of view {fov.min():.0f} mm along one axis')
    if nonfinite > 0:
        w.append(f'non-finite voxels ({100 * nonfinite:.2g}%)')
    if label_map:
        w.append('a label map / mask, not an image')
    if sub.size == 0 or float(sub.min()) == float(sub.max()):
        w.append('empty or constant image')
    if no_space:
        w.append('no spatial header (sform and qform codes 0): the orientation is a guess')
    return dict(n_volumes=n_vol, ndim=len(shape), matrix='x'.join(str(int(v)) for v in shape),   # full shape
                voxel_mm='x'.join(f'{v:.2f}' for v in z), thick_slices=thick,
                oblique_deg=round(oblique, 1), fov_mm='x'.join(f'{v:.0f}' for v in fov),
                nonfinite_frac=round(nonfinite, 6), label_map=bool(label_map), warnings='; '.join(w))


def orientation_code(k):
    """Orientation class -> anatomical direction of the header's +x, +y, +z ('RAS' = class 0)."""
    p, f_ap, f_si = PERMS[k // 4], (k // 2) % 2, k % 2
    return ''.join(('R', 'PA'[1 - f_ap], 'IS'[1 - f_si])[a] for a in p)


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def _sigmoid(z):
    return float(1 / (1 + np.exp(-z)))


def inspect_one(img, session):
    """Predictions for one nibabel image (dict, as a row of inspect_summary.csv without file / status)."""
    d, nonfinite = _clean(_volume(img))          # NaN / Inf only reported (nonfinite_frac); every check sees 0
    out = session.run(None, {'x': make_input(img, d)[None], 'h': header_features(img, d.shape)[None]})
    contrast, stripped, epi, orient, trunc, motion = (o[0] for o in out)
    pc, po = _softmax(contrast), _softmax(orient)
    k = int(po.argmax())
    r = dict(contrast=CONTRASTS[int(pc.argmax())], p_contrast=round(float(pc.max()), 4),
             **{f'p_{c}': round(float(p), 4) for c, p in zip(CONTRASTS, pc)})
    for name, o in (('skull_stripped', stripped), ('epi', epi), ('truncated', trunc)):
        p = _sigmoid(float(o))
        r[name], r[f'p_{name}'] = p > 0.5, round(p, 4)
    r['orientation'], r['p_orientation'] = orientation_code(k), round(float(po[k]), 4)
    if r['contrast'] in MOTION_CONTRASTS:
        p = _sigmoid(float(motion))
        r['motion'], r['motion_flag'] = round(p, 4), p > 0.5
    else:
        r['motion'] = r['motion_flag'] = None
    checks = rule_checks(img, d, nonfinite)
    found = []
    if r['orientation'] != 'RAS':
        found.append(f"orientation {r['orientation']}: the data do not match the header")
    if r['truncated']:
        found.append('head cut by the field of view')
    if r['motion_flag']:
        found.append('motion artifacts')
    checks['warnings'] = '; '.join(found + ([checks['warnings']] if checks['warnings'] else []))
    r.update(checks)
    return r


def _save_summary(summary_dir, rows, columns):
    path = os.path.join(summary_dir, SUMMARY_NAME)
    try:
        os.makedirs(summary_dir, exist_ok=True)
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=columns)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, '') for k in columns})
    except OSError as exc:
        _logger.warning('could not write %s: %s', path, exc)


def inspect(input=None, output=None, model=None, gpu=False, detail=False, verbose=0):
    """Contrast, skull-stripped, EPI, orientation, truncation and motion of each input (module docstring).

    input: a NIfTI file, a directory, a glob pattern or a list. output: folder of ``inspect_summary.csv`` (None: the
    first input's folder). detail: every probability and check value (DETAIL_COLUMNS) instead of the decisions
    (SUMMARY_COLUMNS). A failing input is logged and skipped (a single input raises). Returns the result dict for a
    single input, otherwise the list of result dicts (one per input, with ``file`` and ``status``).
    """
    args = SimpleNamespace(input=input if isinstance(input, list) else [input], output=output,
                           model=model or DEFAULT_MODEL, gpu=gpu, detail=detail, verbose=verbose)
    return run_args(args)


def run_args(args):
    verbose = getattr(args, 'verbose', 0)
    input_file_list, _ = resolve_inputs(args.input)
    if not input_file_list:
        return []
    session = create_session(lib_tool.get_model(getattr(args, 'model', None) or DEFAULT_MODEL), args.gpu)
    is_single = len(input_file_list) == 1
    summary_dir = args.output or os.path.dirname(os.path.abspath(input_file_list[0]))
    columns = DETAIL_COLUMNS if getattr(args, 'detail', False) else SUMMARY_COLUMNS
    rows = []
    for f in tqdm(input_file_list, desc='tigerbx-inspect', unit='file', disable=(verbose > 0)):
        try:
            r = dict(file=f, status='ok', **inspect_one(nib.load(f), session))
            r = {k: r[k] for k in columns if k in r}
        except Exception as exc:                 # noqa: BLE001 - one bad input must not stop a batch
            msg = f'{type(exc).__name__}: {exc}'
            rows.append(dict(file=f, status='failed', error=msg))
            if is_single:
                _save_summary(summary_dir, rows, columns)
                raise
            _logger.warning('FAILED %s: %s', os.path.basename(f), msg)
            continue
        rows.append(r)
        if verbose >= 1:
            _logger.info('%s: %s', os.path.basename(f), json.dumps({k: r.get(k) for k in (
                'contrast', 'skull_stripped', 'epi', 'orientation', 'truncated', 'motion', 'warnings')}))
    _save_summary(summary_dir, rows, columns)
    return rows[0] if is_single else rows
