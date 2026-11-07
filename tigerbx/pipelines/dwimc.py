"""dwimc: head motion and eddy-current correction of a diffusion series (see tigerbx.core.dwimc).

    tigerbx.dwimc('sub-01_dwi.nii.gz', 'out/', reverse='sub-01_dir-PA_epi.nii.gz')

The series is acquired in one phase-encoding direction; a few b0 volumes with the reversed direction (`reverse`) give
the susceptibility field (core.duope), which then moves with the head. Without `reverse` or `field` only motion and
eddy currents are corrected.
"""
import csv
import logging
import os
import tempfile
import time
import warnings

import nibabel as nib
import numpy as np
from scipy import ndimage

from tigerbx.core import dwimc as core
from tigerbx.core import duope
from tigerbx.core.io import get_template, load_nifti, save_nib_volumes
from tigerbx.pipelines.duope import _b0, _json, _on_grid, _phase, _readout, _sidecar

_logger = logging.getLogger('tigerbx')
_logger.addHandler(logging.NullHandler())



def _brain_mask(kind, data, b, disp, ax, img, gpu=False):
    """The estimation mask on the input grid (the frame of the first b0), or None for core's threshold mask.
    kind: 'tbx' (tBX on the mean b0, unwarped by the susceptibility displacement when there is one; the threshold
    mask, with a warning, if tBX is unavailable or its mask is implausible), 'threshold', or a mask file."""
    if kind is None or kind == 'threshold':
        return None
    if kind != 'tbx':
        mi = nib.load(kind)
        return _on_grid(np.asarray(mi.dataobj, np.float64), mi, img) > 0.5
    b0 = data[..., b < core.B0_MAX].mean(-1).astype(np.float64)
    if disp is not None:                                   # observed at x + disp along ax: back to the head frame
        x = np.indices(b0.shape, np.float64)
        x[ax] += disp
        b0 = ndimage.map_coordinates(b0, x, order=3, mode='nearest') * (1 + np.gradient(disp, axis=ax))
    try:
        from tigerbx.core import models
        from tigerbx.modules.bx import produce_tbxdual
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, 'b0.nii.gz')
            nib.save(nib.Nifti1Image(np.maximum(b0, 0).astype(np.float32), img.affine), f)
            m = np.asarray(produce_tbxdual(models.TBX, f, GPU=gpu)[0].dataobj) > 0      # the outer mask
    except Exception as exc:                     # noqa: BLE001 - fall back, the mask only guides the estimation
        warnings.warn(f'dwimc: tBX mask unavailable ({type(exc).__name__}: {exc}); using the threshold mask',
                      RuntimeWarning, stacklevel=3)
        return None
    ref = core.head_mask(b0)
    if not 0.5 * ref.sum() < m.sum() < 2.0 * ref.sum():
        warnings.warn(f'dwimc: the tBX mask ({int(m.sum())} voxels) is implausible against the threshold mask '
                      f'({int(ref.sum())}); using the threshold mask', RuntimeWarning, stacklevel=3)
        return None
    return m


def _ec_names(order_t, K):
    """EC column names in the input's voxel axes (i, j, k): const, then the linear and quadratic terms."""
    a, b, c = ('ijk'[order_t[0]], 'ijk'[order_t[1]], 'ijk'[order_t[2]])
    return ('const', a, b, c, a + a, b + b, c + c, a + b, a + c, b + c)[:K]


def dwimc(input, output=None, *, reverse=None, field=None, bval=None, bvec=None, pe=None, readout=None, order=2,
          iters=core.OUTER, workers=None, mask='threshold', gpu=False, resume=False, gz=True, verbose=0):
    """Correct head motion and eddy currents (and susceptibility distortion, given `reverse` or `field`) of a DWI
    series in one cubic resampling.

    input: a 4D DWI series (or a list). bval / bvec: FSL-format files (default: the input's .bval / .bvec).
    reverse: b0 volume(s) acquired with the reversed phase encoding (a list: one per input); the field is estimated
    from them and the series' b0s. field: instead, a field in Hz (as tigerbx.duope save_field) with the readout time
    (JSON or `readout`). pe: the series' phase-encoding direction when there is no JSON ('j', 'j-', 'AP', ...).
    order: the eddy-current polynomial (1: linear, 2: quadratic). iters: outer iterations. workers: processes
    (default 70% of the CPUs). mask: the voxels the estimation uses: 'threshold' (the b0 intensity head mask),
    'tbx' (tBX on the mean b0, unwarped by the field when there is one; gpu: tBX on the GPU) or a mask file on the
    input grid. resume: skip inputs whose output is already there.

    Outputs per input: <input>_dwimc (corrected series in the frame of its first b0), <input>_dwimc.bval / .bvec
    (bvecs rotated by each volume's head rotation) and <input>_dwimc.csv (per volume: b, translation mm, rotation
    degrees in voxel axes, EC coefficients in voxels along the phase axis (terms in voxel axes i / j / k scaled
    to [-1, 1] over the grid), intensity scale, RMS displacement by motion and eddy currents
    in the brain in mm). Returns the files dict for one input, else a list of them (a failing input is logged and
    skipped, a warning lists them). Research use only.
    """
    inputs = list(input) if isinstance(input, (list, tuple)) else [input]
    n = len(inputs)

    def each(v, name):
        if isinstance(v, (list, tuple)) and n > 1:
            if len(v) != n:
                raise ValueError(f'{name}: one entry per input')
            return list(v)
        return [v] * n

    revs, fields, bvals, bvecs = each(reverse, 'reverse'), each(field, 'field'), each(bval, 'bval'), each(bvec, 'bvec')
    workers = workers or core.default_workers()
    out, failed = [], []
    for k, f in enumerate(inputs):
        try:
            ftemplate, _ = get_template(f, output, gz)
            if resume and os.path.isfile(ftemplate.replace('@@@@', 'dwimc')):
                _logger.info('%s: output exists, skipped', os.path.basename(f))
                continue
            out.append(_one(f, ftemplate, revs[k], fields[k], bvals[k], bvecs[k], pe, readout, order, iters, workers,
                            verbose, mask, gpu))
        except Exception as exc:                 # noqa: BLE001 - one bad input must not stop a batch
            if n == 1:
                raise
            _logger.warning('FAILED %s: %s: %s', os.path.basename(f), type(exc).__name__, exc)
            failed.append(f'{f}: {type(exc).__name__}: {exc}')
    if failed:
        warnings.warn(f'tigerbx.dwimc: {len(failed)} of {n} inputs failed and were skipped:\n  ' + '\n  '.join(failed),
                      RuntimeWarning, stacklevel=2)
    return out[0] if n == 1 and out else out


def _flip(aff):
    """FSL bvecs are in voxel axes with x negated when the affine's determinant is positive."""
    return np.diag([-1.0 if np.linalg.det(aff[:3, :3]) > 0 else 1.0, 1.0, 1.0])


def _one(f, ftemplate, reverse, field, bval, bvec, pe, readout, order, iters, workers, verbose, mask='threshold',
         gpu=False):
    t0 = time.time()
    img = load_nifti(f)
    if img.ndim != 4:
        raise ValueError(f'{os.path.basename(f)}: a 4D DWI series is needed')
    meta = _json(f)
    b = np.loadtxt(bval or _sidecar(f, '.bval')).ravel()
    g = np.loadtxt(bvec or _sidecar(f, '.bvec'))
    g = g.T if g.shape[0] == 3 and g.shape[1] != 3 else g
    n = img.shape[3]
    if len(b) != n or g.shape != (n, 3):
        raise ValueError(f'{os.path.basename(f)}: {n} volumes but {len(b)} b-values / {g.shape} b-vectors')
    aff = img.affine
    zooms = np.array(img.header.get_zooms()[:3], float)
    data = np.nan_to_num(np.asarray(img.dataobj, np.float32))
    ax, pol = _phase(img, meta, pe, f)
    F = _flip(aff)
    g_vox = g @ F                                         # voxel axes (mm)

    disp = None                                           # the series' susceptibility displacement, voxels along ax
    if reverse is not None:
        ri = load_nifti(reverse)
        rev = _on_grid(_b0(ri, reverse), ri, img)
        fwd = data[..., b < core.B0_MAX].mean(-1).astype(np.float64)
        pos, neg = (fwd, rev) if pol > 0 else (rev, fwd)  # duope.estimate takes the positive polarity first
        d, _, _ = duope.estimate(pos, neg, aff, pe_axis=ax)
        disp = pol * d
    elif field is not None:
        trt = _readout(meta, img.shape[ax]) or readout
        if not trt:
            raise ValueError(f'{os.path.basename(f)}: field= needs the readout time (JSON TotalReadoutTime or readout=)')
        fi = nib.load(field)
        disp = pol * trt * _on_grid(np.asarray(fi.dataobj, np.float64), fi, img)

    order_t = core.SWAP[ax]
    tr = lambda a: np.transpose(a, order_t + (3,)) if a.ndim == 4 else np.transpose(a, order_t)
    P = np.eye(4)[list(order_t) + [3]]                   # voxel axes permutation (its own inverse)
    m = _brain_mask(mask, data, b, disp, ax, img, gpu)
    est = core.estimate(tr(data), b, g_vox[:, list(order_t)], zooms[list(order_t)],
                        field=None if disp is None else tr(disp), mask=None if m is None else tr(m), order=order,
                        outer=iters, workers=workers, verbose=verbose)
    rms = []

    def volumes():
        for v, r in core.correct_all(tr(data), est, None if disp is None else tr(disp), zooms[list(order_t)],
                                     workers):
            rms.append(r)
            yield np.transpose(v, order_t)

    path = save_nib_volumes(img.header, aff, img.shape, volumes(), ftemplate, 'dwimc')
    stem = path[:-7] if path.endswith('.nii.gz') else path[:-4]

    # rotated b-vectors (FSL convention), per-volume table
    rows, g_out = [], np.zeros_like(g)
    for i in range(n):
        M = P @ est['M'][i] @ P                           # original voxel axes
        R = core.rotation(M, zooms)
        g_out[i] = (R.T @ g_vox[i]) @ F if b[i] > core.B0_MAX else g[i]
        p = core.params_of(M, zooms, est['c_mm'][list(order_t)])
        row = dict(volume=i, b=b[i], tx_mm=p[0], ty_mm=p[1], tz_mm=p[2], rx_deg=np.degrees(p[3]),
                   ry_deg=np.degrees(p[4]), rz_deg=np.degrees(p[5]))
        row.update({f'ec_{nm}_vox': v for nm, v in zip(_ec_names(order_t, len(est['c'][i])), est['c'][i])})
        row.update(scale=est['s'][i], rms_mm=rms[i])
        rows.append({k: (round(float(v), 5) if not isinstance(v, int) else v) for k, v in row.items()})
    np.savetxt(stem + '.bval', b[None], fmt='%g')
    np.savetxt(stem + '.bvec', g_out.T, fmt='%.6f')
    with open(stem + '.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if verbose >= 1:
        _logger.info('%s: %d volumes, %.1f s', os.path.basename(f), n, time.time() - t0)
    return {'dwimc': path, 'bval': stem + '.bval', 'bvec': stem + '.bvec', 'csv': stem + '.csv'}
