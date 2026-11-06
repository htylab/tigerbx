"""epinorm: a DWI / BOLD series onto its subject's T1 and the template, in one resampling.

    tigerbx.epinorm('sub-01_task-rest_bold.nii.gz', 'sub-01_T1w.nii.gz',
                    'sub-01_dir-AP_epi.nii.gz', 'sub-01_dir-PA_epi.nii.gz', 'out/')

Steps: (1) the susceptibility field from the reverse phase-encoding pair (forward, reverse; tigerbx.duope), carried
onto the series' grid, readout time and head position (without a pair none is corrected, with a warning); (2) the
corrected forward scan's brain (tBX; without a pair the series' b0 / mean) registered rigidly to the T1's brain
(numreg, Mattes MI, 6 dof); (3) the T1 to the template (tBX, numreg affine, tReg, as tigerbx.norm); (4) every output
voxel sampled once from the raw series through field, head position, rigid, affine and warp composed (cubic
B-spline, times the field's Jacobian). The EPI -> T1 step is deliberately rigid: after the distortion correction a
deformable EPI -> T1 registration matched the T1's appearance rather than the residual distortion.
"""
import csv
import importlib
import logging
import os
import tempfile
import time
import warnings

import nibabel as nib
import numpy as np
from scipy import ndimage

from tigerbx.core import duope as core
from tigerbx.core import numreg, templatereg
from tigerbx.core.io import get_template, load_nifti, save_nib, save_nib_volumes
from tigerbx.pipelines import _common

_logger = logging.getLogger('tigerbx')
_logger.addHandler(logging.NullHandler())
_D = importlib.import_module('tigerbx.pipelines.duope')       # the duope pipeline's sidecar / phase helpers

SUMMARY_NAME = 'epinorm_summary.csv'
SPACES = ('t1', 'mni')


def epinorm(series, t1, forward=None, reverse=None, output=None, *, pe=None, readout=None, series_pe=None,
            series_align=True, joint='auto', space=SPACES, res=2, template=None, gpu=False, gz=True,
            resume=False, verbose=0):
    """A DWI / BOLD series onto the subject's T1 and the template.

    series: the series (NIfTI, 3D or 4D, on any grid); t1: the subject's T1w (raw; tBX runs on it); forward,
    reverse: the reverse phase-encoding pair for the distortion correction (tigerbx.duope; e.g. AP, PA; the
    polarities come from the sidecars / pe; None, None: no correction). pe / readout: the forward scan's direction
    and readout time without JSON; series_pe: the series' direction without JSON (default: the forward scan's);
    series_align: move the field to the series' head position; joint: as tigerbx.duope. space: the outputs, 't1'
    (the T1 grid) and / or 'mni' (the template grid). res: the output voxel size in mm (default 2; 'epi': the
    series' smallest voxel size; None: the T1's / template's own grid), over the same field of view. template:
    the registration template (default: bundled). A list of series (one per subject) with lists or single values
    of the others runs a batch. resume: skip series whose first output (_epi2<space>) is already there
    (continuing a batch; they are left out of the list).

    Outputs: <series>_epi2t1 / _epi2mni (every volume), <t1>_tbx, <series>_epinorm.npz (the transform chain: field,
    head position, rigid, affine, warp) and one row per series in epinorm_summary.csv. Returns the images dict for
    one series, else a list of dicts of the written files. Research use only.
    """
    batch = isinstance(series, (list, tuple))
    sers = list(series) if batch else [series]
    n = len(sers)
    each = lambda v, name: _D._each(v, n, batch, name)
    t1s, fwds, revs = each(t1, 't1'), each(forward, 'forward'), each(reverse, 'reverse')
    pes, readouts, spes = each(pe, 'pe'), each(readout, 'readout'), each(series_pe, 'series_pe')
    space = (space,) if isinstance(space, str) else tuple(space)
    bad = set(space) - set(SPACES)
    if bad or not space:
        raise ValueError(f'space: {sorted(bad) or "empty"} (use {SPACES})')
    if not (res is None or res == 'epi' or (np.isscalar(res) and float(res) > 0)):
        raise ValueError(f"res: {res!r} (a voxel size in mm, 'epi' or None)")
    tpl = templatereg.template(template)
    results, failed, rows = [], [], []
    for args in zip(sers, t1s, fwds, revs, pes, readouts, spes):
        if resume and os.path.isfile(get_template(args[0], output, gz)[0].replace('@@@@', f'epi2{space[0]}')):
            continue
        try:
            images, saved, row = _one(*args, output, series_align, joint, space, res, tpl, gpu, gz, verbose)
            rows.append(row)
        except Exception as exc:                 # noqa: BLE001 - one bad subject must not stop a batch
            if n == 1:
                raise
            _logger.warning('FAILED %s: %s: %s', os.path.basename(args[0]), type(exc).__name__, exc)
            failed.append(f'{args[0]}: {type(exc).__name__}: {exc}')
            continue
        results.append(images if n == 1 else saved)
    if rows:
        _write_summary(output or os.path.dirname(os.path.abspath(sers[0])), rows)
    if failed:
        warnings.warn(f'tigerbx.epinorm: {len(failed)} of {n} series failed and were skipped:\n  '
                      + '\n  '.join(failed), RuntimeWarning, stacklevel=2)
    return results[0] if n == 1 and results else results


def _write_summary(folder, rows):
    """Add the rows to the folder's summary; a series run again replaces its earlier row."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, SUMMARY_NAME)
    key = lambda row: os.path.abspath(row['series'])
    done = {key(row) for row in rows}
    old = []
    if os.path.exists(path):
        with open(path, newline='') as fh:
            old = [row for row in csv.DictReader(fh) if key(row) not in done]
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), extrasaction='ignore')
        w.writeheader()
        w.writerows(old + rows)


# ---------------------------------------------------------------- the chain

class Source:
    """How a raw EPI image (on src_aff) gives its corrected image on corr_aff: corrected(x) = raw(post(x) moved by
    sign * disp(x) along pe_axis) * (1 + sign * d disp / d pe), disp and x on the corr_aff grid; pre maps the
    corrected forward scan's world to this image's corrected world (a series' head position)."""

    def __init__(self, src_aff, corr_aff, disp, pe_axis, sign, post=None, pre=None):
        self.src_aff, self.corr_aff = np.asarray(src_aff, float), np.asarray(corr_aff, float)
        self.disp, self.ax, self.sign = np.asarray(disp, np.float32), int(pe_axis), float(sign)
        self.grad = np.gradient(self.disp, axis=self.ax).astype(np.float32)
        self.post = np.eye(4) if post is None else np.asarray(post, float)
        self.pre = np.eye(4) if pre is None else np.asarray(pre, float)

    def coords(self, world):
        """Raw-image voxel coordinates (3, N) and Jacobian (N) of corrected-forward world points (3, N)."""
        y = self.pre[:3, :3] @ world + self.pre[:3, 3:]
        ci = np.linalg.inv(self.corr_aff)
        u = ci[:3, :3] @ y + ci[:3, 3:]                                       # corrected-grid voxels
        d = ndimage.map_coordinates(self.disp, u, order=1, mode='nearest')
        g = ndimage.map_coordinates(self.grad, u, order=1, mode='nearest')
        V = ci @ self.post @ self.corr_aff                                    # motion, in corrected-grid voxels
        v = V[:3, :3] @ u + V[:3, 3:]
        v[self.ax] += self.sign * d
        S = np.linalg.inv(self.src_aff) @ self.corr_aff                       # onto the raw image's own grid
        return S[:3, :3] @ v + S[:3, 3:], 1.0 + self.sign * g


def _resized(aff, shape, res):
    """A grid of res mm voxels over the field of view of (aff, shape), its voxels in the old grid's voxel
    coordinates (3, N), its affine and shape."""
    aff = np.asarray(aff, float)
    if res is None:
        return templatereg.identity_grid(shape).reshape(3, -1), aff, tuple(shape)
    scale = float(res) / np.linalg.norm(aff[:3, :3], axis=0)
    new = tuple(int(n) for n in np.maximum(1, np.round(np.asarray(shape) / scale)))
    q = (templatereg.identity_grid(new).reshape(3, -1) * scale[:, None].astype(np.float32)
         + ((scale - 1) / 2)[:, None].astype(np.float32))                    # the boxes' edges coincide
    out = aff.copy()
    out[:3, :3] = aff[:3, :3] * scale
    out[:3, 3] = aff[:3, :3] @ ((scale - 1) / 2) + aff[:3, 3]
    return q, out, new


def _target_world(space, t1_img, tpl, A, warp, R, res):
    """Corrected-forward world points (3, N) of the target grid's voxels, with the grid's (affine, shape)."""
    if space == 't1':
        q, aff, shape = _resized(t1_img.affine, t1_img.shape[:3], res)
        old = t1_img.affine
        x = old[:3, :3] @ q + old[:3, 3:]
    else:
        q, aff, shape = _resized(tpl.affine, tpl.shape[:3], res)
        old = tpl.affine
        w = warp.reshape(3, -1) if res is None else np.stack(
            [ndimage.map_coordinates(c, q, order=1, mode='nearest') for c in warp])
        p = q + w
        x = A[:3, :3] @ (old[:3, :3] @ p + old[:3, 3:]) + A[:3, 3:]           # template world -> T1 world
    return R[:3, :3] @ x + R[:3, 3:], aff, shape                              # T1 world -> EPI world


def _write(img, src, world, aff, shape, ftemplate, postfix):
    """Every volume of img through src at the world points, cubic, written volume by volume."""
    c, jac = src.coords(world)

    def volumes():
        for _, block in _D._chunks(img):
            for k in range(block.shape[3]):
                v = ndimage.map_coordinates(block[..., k].astype(np.float64), c, order=3, mode='constant')
                yield (v * jac).reshape(shape)
    hdr = nib.Nifti1Header()
    hdr.set_xyzt_units('mm', 'sec')
    if len(img.shape) > 3:
        hdr['pixdim'][4] = img.header['pixdim'][4]
    path = save_nib_volumes(hdr, aff, tuple(shape) + img.shape[3:], volumes(), ftemplate, postfix)
    return path, nib.load(path)


def _rigid(epi_brain, t1_brain):
    """Rigid M (T1 world -> EPI world) of an EPI brain to the T1 brain: numreg, Mattes MI, 6 dof, one start
    (the setting that held up against boundary-based and feature-based costs)."""
    eb, tb = np.asarray(epi_brain.dataobj, np.float32), np.asarray(t1_brain.dataobj, np.float32)
    return numreg.register(nib.Nifti1Image(eb, epi_brain.affine), nib.Nifti1Image(tb, t1_brain.affine),
                           moving_mask=eb > 0, fixed_mask=tb > 0, dof=6, metric='mi', n_starts=1)


def _epi_brain(epi, f, gpu):
    """The tBX brain of an EPI reference image (3D array on the grid aff) through a temporary file."""
    with tempfile.TemporaryDirectory() as tmp:
        rf = os.path.join(tmp, 'epi_ref.nii.gz')
        nib.save(nib.Nifti1Image(np.clip(np.asarray(epi, np.float32), 0, None), f.affine), rf)
        _, brain, _, _ = _common.brain(rf, None, gpu)
    return brain


def to_epi(t1, epi, output=None, *, labels=None, gpu=False, gz=True):
    """The T1 (and label maps on its grid) onto an EPI's grid, rigidly: tBX on both, the EPI's b0 / mean brain
    registered to the T1 brain (Mattes MI, 6 dof, the EPI -> T1 step of epinorm), the T1 resampled cubic and each
    label map nearest. The EPI is taken as it is (its distortion is not corrected: run duope first, or epinorm).

    Writes <t1>_t12epi and <label>_t12epi (next to the T1, or in output). Returns {'t1_t12epi', '<label>_t12epi',
    ..., 'rigid_M' (T1 world -> EPI world)} as images. Research use only.
    """
    ei = load_nifti(epi)
    ref = nib.Nifti1Image(_D._b0(ei, epi).astype(np.float32), ei.affine)
    t1_img, t1_brain, _, _ = _common.brain(t1, None, gpu)
    R = _rigid(_epi_brain(ref.dataobj, ref, gpu), t1_brain)
    Ri = np.linalg.inv(R)                                                    # EPI world -> T1 world
    ftemplate, _ = get_template(t1, output, gz)
    images = {'t1_t12epi': nib.Nifti1Image(numreg.resample(t1_img, ref, Ri, order=3), ref.affine)}
    save_nib(images['t1_t12epi'], ftemplate, 't12epi')
    for lf in ([labels] if isinstance(labels, (str, os.PathLike)) else labels or []):
        li = load_nifti(lf)
        data = np.asarray(li.dataobj)
        key = os.path.basename(lf).split('.')[0] + '_t12epi'
        images[key] = nib.Nifti1Image(numreg.resample(li, ref, Ri, order=0).astype(data.dtype), ref.affine)
        save_nib(images[key], get_template(lf, output, gz)[0], 't12epi')
    images['rigid_M'] = R
    return images


# ---------------------------------------------------------------- one series

def _field(f, r, pe, readout, joint):
    """The forward scan's field as tigerbx.duope: (affine, displacement, phase axis, polarity, world direction,
    readout time, corrected reference, info, whether the direction was guessed)."""
    fi, ri = load_nifti(f), load_nifti(r)
    fm, rm = _D._json(f), _D._json(r)
    aff = fi.affine
    fwd = _D._b0(fi, f)
    rev = _D._on_grid(_D._b0(ri, r), ri, fi)
    ax_f, pol_f, fit, guessed = _D._pair_phase(fi, ri, fm, rm, pe, f, r, fwd, rev, dict(joint=joint))
    w_f = _D._world(ax_f, pol_f, aff)
    trt = _D._readout(fm, fi.shape[ax_f]) or readout
    ax_r, pol_r = _D._phase(ri, rm, None, r, default=-w_f)
    if w_f @ _D._world(ax_r, pol_r, ri.affine) > -0.9:
        raise ValueError(f'{os.path.basename(f)} / {os.path.basename(r)}: not a reverse phase-encoding pair')
    pos, neg = (fwd, rev) if pol_f > 0 else (rev, fwd)
    d, M, info = fit or core.estimate(pos, neg, aff, pe_axis=ax_f, joint=joint)
    ref = core.unwarp(fwd, d, aff, ax_f, M=None if pol_f > 0 else M, sign=pol_f)
    return aff, d, ax_f, pol_f, w_f, trt, ref, info, guessed


def _one(sf, t1f, f, r, pe, readout, series_pe, output, series_align, joint, space, res, tpl, gpu, gz,
         verbose):
    t0 = time.time()
    if (f is None) != (r is None):
        raise ValueError(f'{os.path.basename(sf)}: forward and reverse go together (or neither)')
    si = nib.load(sf, keep_file_open=True)
    sm = _D._json(sf)
    sref = _D._b0(si, sf)
    info = dict(motion_mm=0.0, motion_deg=0.0, joint=False)
    ax_s = pol_s = pol_f = trt = scale = M_s = None
    sinfo = {}
    if f is not None:
        aff, d, ax_f, pol_f, w_f, trt, ref, info, guessed = _field(f, r, pe, readout, joint)
        trt_s = _D._readout(sm, si.shape[_D._from_world(w_f, si.affine, sf)[0]])
        scale = trt_s / trt if (trt_s and trt) else 1.0
        ax_s, pol_s, M_s, sinfo = _D._series_phase(si, sm, series_pe, sf, sref, ref, aff, scale * d, w_f, guessed)
        if not series_align:
            M_s = None
        d_s = core.field_on(scale * d, aff, si.affine, si.shape[:3], M_s)
        pre = None if M_s is None else np.linalg.inv(M_s)                    # forward world -> series world
    else:
        warnings.warn(f'{os.path.basename(sf)}: no reverse phase-encoding pair, the susceptibility distortion is not '
                      'corrected', RuntimeWarning, stacklevel=4)
        ref, aff, d_s, pre = sref, si.affine, np.zeros(si.shape[:3], np.float32), None
    src = Source(si.affine, si.affine, d_s, 1 if ax_s is None else ax_s, pol_s or 1.0, pre=pre)

    if res == 'epi':
        res = float(np.min(si.header.get_zooms()[:3]))
    stemplate, _ = get_template(sf, output, gz)
    t1template, _ = get_template(t1f, output, gz)
    images, saved = {}, {}
    # T1: brain (tBX) and the template registration
    t1_img, t1_brain, _, tbx = _common.brain(t1f, None, gpu)
    images['t1_tbx'], saved['t1_tbx'] = tbx, save_nib(tbx, t1template, 'tbx')
    A = warp = None
    if 'mni' in space:
        A = templatereg.affine(t1_brain, tpl)
        warp = templatereg.treg_warp(t1_brain, tpl, A, None, gpu)
    # EPI: brain of the corrected reference (tBX), rigid to the T1 brain (numreg MI)
    with tempfile.TemporaryDirectory() as tmp:
        rf = os.path.join(tmp, 'epi_ref.nii.gz')
        nib.save(nib.Nifti1Image(np.clip(ref, 0, None).astype(np.float32), aff), rf)
        _, epi_brain, _, _ = _common.brain(rf, None, gpu)
    R = _rigid(epi_brain, t1_brain)                                          # T1 world -> EPI world

    for sp in space:
        world, gaff, gshape = _target_world(sp, t1_img, tpl, A, warp, R, res)
        k = f'epi2{sp}'
        saved[k], images[k] = _write(si, src, world, gaff, gshape, stemplate, k)

    chain = dict(series_affine=si.affine, field_disp=d_s.astype(np.float32), pe_axis=-1 if ax_s is None else ax_s,
                 polarity=pol_s or 0, head_M=np.eye(4) if pre is None else pre, rigid_M=R,
                 reference_affine=aff, t1_affine=t1_img.affine, t1_shape=np.array(t1_img.shape[:3]))
    if A is not None:
        chain.update(affine_M=A, treg_warp=warp.astype(np.float32), template_affine=tpl.affine,
                     template_shape=np.array(tpl.shape[:3]))
    saved['chain'] = stemplate.split('@@@@')[0] + 'epinorm.npz'
    np.savez(saved['chain'], **chain)
    rmm, rdeg = core.motion_size(R, (t1_img.affine @ np.r_[(np.array(t1_img.shape[:3]) - 1) / 2, 1])[:3])
    smm, sdeg = ((0.0, 0.0) if M_s is None
                  else core.motion_size(M_s, (aff @ np.r_[(np.array(ref.shape) - 1) / 2, 1])[:3]))
    row = dict(series=sf, t1=t1f, forward=f or '', reverse=r or '', corrected=f is not None,
               phase_axis='' if ax_s is None else _D._AXES[ax_s], series_polarity=pol_s or '',
               series_direction=sinfo.get('source', ''),
               series_ncc=('%.3f/%.3f' % (sinfo['ncc_forward'], sinfo['ncc_reverse'])) if sinfo else '',
               forward_polarity=pol_f or '', readout_s=trt or '', readout_scale=round(scale, 4) if scale else '',
               pair_motion_mm=round(info['motion_mm'], 3), pair_motion_deg=round(info['motion_deg'], 3),
               joint=info['joint'], series_mm=round(smm, 2), series_deg=round(sdeg, 2), rigid_mm=round(rmm, 2),
               rigid_deg=round(rdeg, 2), space=';'.join(space), res=res or '', seconds=round(time.time() - t0, 1))
    if verbose >= 1:
        _logger.info('%s: rigid %.1f mm / %.1f deg, %.1f s', os.path.basename(sf), rmm, rdeg, time.time() - t0)
    return images, saved, row
