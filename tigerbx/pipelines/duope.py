"""DuoPE: susceptibility distortion correction from a reverse phase-encoding pair (see tigerbx.core.duope).

    tigerbx.duope('sub-01_dir-AP_epi.nii.gz', 'sub-01_dir-PA_epi.nii.gz', 'out/')
    tigerbx.duope(ap, pa, 'out/', series='sub-01_task-rest_bold.nii.gz')   # also correct a BOLD / DWI series

The phase-encoding axis and polarity of each scan and the total readout time come from the BIDS JSON sidecars
(PhaseEncodingDirection, TotalReadoutTime), else from `pe` / `readout` (`pe` also as an anatomical direction, 'AP',
'PA', 'LR', ..., converted with the scan's affine), else the axis from the header's dim_info (polarity then assumed
positive for `forward`, with a warning). The reverse scan's direction is compared in world space, so it may be
stored in another orientation. A 4D input contributes the mean of its b < 50 volumes
(from its .bval) or of all volumes. All displacements are in NIfTI voxel axes (unlike FSL, which flips voxel x of
images with a positive-determinant affine).
"""
import csv
import json
import logging
import os
import time
import warnings

import nibabel as nib
import numpy as np
from scipy import ndimage

from tigerbx.core import duope as core
from tigerbx.core.io import get_template, load_nifti, save_nib, save_nib_volumes

_logger = logging.getLogger('tigerbx')
_logger.addHandler(logging.NullHandler())

_AXES = 'ijk'
_CODES = ('i', 'i-', 'j', 'j-', 'k', 'k-', 'x', 'x-', 'y', 'y-', 'z', 'z-')    # x / y / z: early BIDS
# anatomical phase-encoding directions (from -> to) as RAS+ world vectors
_ANAT = {'LR': (1, 0, 0), 'RL': (-1, 0, 0), 'PA': (0, 1, 0), 'AP': (0, -1, 0), 'IS': (0, 0, 1), 'SI': (0, 0, -1)}
_MIN_COS = 0.75                                  # an anatomical direction must lie within ~40 degrees of a voxel axis


SUMMARY_NAME = 'duope_summary.csv'


def duope(forward, reverse, output=None, *, series=None, pe=None, readout=None, series_pe=None, series_align=True,
          joint='auto', save_field=False, save_disp=False, gz=True, verbose=0):
    """Estimate the field of a forward / reverse phase-encoding pair and unwarp it (and optional series).

    forward, reverse: the two scans (NIfTI paths; or equal-length lists of pairs). series: a DWI / BOLD series (or a
    list) to unwarp with the field, on its own grid (the field resampled onto it), each by its own phase-encoding
    polarity (JSON, else the forward scan's); series_align: move the field to the series' head position (its b0 /
    mean image registered rigidly to the corrected forward scan; False: the same position assumed); for a list of pairs, a list with one entry (path, list or None) per pair. pe: the forward
    scan's phase-encoding direction when there is no JSON, a voxel axis code ('i', 'i-', 'j', 'j-', 'k', 'k-') or
    an anatomical direction from -> to ('AP', 'PA', 'LR', 'RL', 'IS', 'SI'; through the affine); the reverse scan
    without JSON is taken as the opposite. readout: the forward scan's total readout time in s. series_pe: the
    direction of a series without JSON (default: the forward scan's), one per series for a list. For a list of
    pairs, pe / readout / series_pe may be lists with one entry per pair. joint: 'auto' (field and motion jointly when the motion between the scans exceeds 1 mm /
    1 degree), True or False.

    Outputs: every input corrected, all its volumes, as <input>_duope (forward, reverse and each series; one cubic
    resampling with the motion of its scan); save_field: <forward>_duope_field in Hz (a positive-polarity scan moves
    field x readout time voxels along +axis; needs the readout time); save_disp: <forward>_duope_disp, the forward
    scan's displacement in voxels along its phase axis (corrected(x) = forward(x + disp)); one row per pair in
    duope_summary.csv (phase axis, polarity, direction, readout time, motion, joint step, time). Returns the images dict for one
    pair, else a list of dicts of the written files (a failing pair is logged and skipped, a warning lists them).
    Research use only.
    """
    pairs = list(zip(forward, reverse)) if isinstance(forward, (list, tuple)) else [(forward, reverse)]
    if isinstance(forward, (list, tuple)) and len(forward) != len(reverse):
        raise ValueError(f'{len(forward)} forward and {len(reverse)} reverse scans')
    if isinstance(forward, (list, tuple)):
        series = list(series) if series is not None else [None] * len(pairs)
        if len(series) != len(pairs):
            raise ValueError('series: one entry per pair')
    else:
        series = [series]
    batch = isinstance(forward, (list, tuple))
    pes, readouts, series_pes = (_each(v, len(pairs), batch, name) for v, name in
                                 ((pe, 'pe'), (readout, 'readout'), (series_pe, 'series_pe')))
    results, failed, rows = [], [], []
    for (f, r), s, p, t, sp in zip(pairs, series, pes, readouts, series_pes):
        try:
            images, saved, row = _one(f, r, s, output, p, t, sp, series_align, joint, save_field, save_disp, gz,
                                      verbose)
            rows.append(row)
        except Exception as exc:                 # noqa: BLE001 - one bad pair must not stop a batch
            if len(pairs) == 1:
                raise
            _logger.warning('FAILED %s: %s: %s', os.path.basename(f), type(exc).__name__, exc)
            failed.append(f'{f}: {type(exc).__name__}: {exc}')
            continue
        results.append(images if len(pairs) == 1 else saved)
    if rows:
        _write_summary(output or os.path.dirname(os.path.abspath(pairs[0][0])), rows)
    if failed:
        warnings.warn(f'tigerbx.duope: {len(failed)} of {len(pairs)} pairs failed and were skipped:\n  '
                      + '\n  '.join(failed), RuntimeWarning, stacklevel=2)
    return results[0] if len(pairs) == 1 and results else results


def _each(value, n, batch, name):
    """A per-pair option: a list (batch only) gives one value per pair, anything else applies to all pairs."""
    if batch and isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f'{name}: one entry per pair')
        return list(value)
    return [value] * n


def _sidecar(f, ext):
    stem = f[:-7] if f.endswith('.nii.gz') else os.path.splitext(f)[0]
    return stem + ext


def _json(f):
    j = _sidecar(f, '.json')
    return json.load(open(j)) if os.path.exists(j) else {}


def _readout(meta, n_pe):
    """TotalReadoutTime (s) of a sidecar, else EffectiveEchoSpacing x (n_pe - 1) (BIDS); None if neither."""
    if meta.get('TotalReadoutTime'):
        return float(meta['TotalReadoutTime'])
    if meta.get('EffectiveEchoSpacing'):
        return float(meta['EffectiveEchoSpacing']) * (n_pe - 1)
    return None


CHUNK_BYTES = 2 ** 28                            # a 4D input is read in blocks of about this size


def _chunks(img):
    """(first volume, float32 block x, y, z, n) over the volumes of img, non-finite voxels set to 0."""
    shape, n = img.shape[:3], int(np.prod(img.shape[3:]))
    if img.ndim != 4:
        yield 0, np.nan_to_num(np.asarray(img.dataobj, np.float32).reshape(shape + (n,)), nan=0, posinf=0, neginf=0)
        return
    step = max(1, CHUNK_BYTES // (4 * int(np.prod(shape))))
    for a in range(0, n, step):
        yield a, np.nan_to_num(np.asarray(img.dataobj[..., a:a + step], np.float32), nan=0, posinf=0, neginf=0)


def _b0(img, f):
    """The mean of the b < 50 volumes (from f's .bval) or of all volumes, read block by block."""
    n = int(np.prod(img.shape[3:]))
    use = np.ones(n, bool)
    bv = _sidecar(f, '.bval')
    if n > 1 and os.path.exists(bv):
        b = np.array([float(x) for x in open(bv).read().split()])
        if len(b) == n and (b < 50).any():
            use = b < 50
    total = np.zeros(img.shape[:3])
    for a, block in _chunks(img):
        total += block[..., use[a:a + block.shape[3]]].sum(-1, dtype=np.float64)
    return total / use.sum()


def _world(ax, pol, aff):
    """The unit world (RAS+) vector of phase-encoding axis `ax` with polarity `pol`."""
    c = pol * np.asarray(aff, float)[:3, ax]
    return c / np.linalg.norm(c)


def _label(w):
    """The anatomical name ('AP', 'LR', ...) of the world axis closest to direction w."""
    i = int(np.argmax(np.abs(w)))
    return (('RL', 'LR'), ('AP', 'PA'), ('SI', 'IS'))[i][int(w[i] > 0)]


def _from_world(w, aff, f):
    """(axis, polarity) of the voxel axis along world direction w; an oblique grid must leave it unambiguous."""
    cols = np.asarray(aff, float)[:3, :3]
    c = (cols / np.linalg.norm(cols, axis=0)).T @ np.asarray(w, float)
    ax = int(np.argmax(np.abs(c)))
    if abs(c[ax]) < _MIN_COS:
        raise ValueError(f'{os.path.basename(f)}: direction {_label(w)} is {np.degrees(np.arccos(abs(c[ax]))):.0f} '
                         'degrees from the nearest voxel axis; give pe as a voxel axis (i / j / k)')
    return ax, (1 if c[ax] > 0 else -1)


def _parse(code, aff, f):
    """(axis, polarity) from a voxel code ('j-'; early-BIDS 'y-') or an anatomical direction ('AP')."""
    s = code.strip() if isinstance(code, str) else None
    if s and s.upper() in _ANAT:
        return _from_world(_ANAT[s.upper()], aff, f)
    if s not in _CODES:
        raise ValueError(f'{os.path.basename(f)}: phase-encoding direction {code!r} (i, i-, j, j-, k, k- or AP, PA, '
                         'LR, RL, IS, SI)')
    return 'ijkxyz'.index(s[0]) % 3, (-1 if s.endswith('-') else 1)


def _phase(img, meta, given, f, default=None):
    """(axis, polarity +1 / -1) of a scan in its voxel axes: JSON, else `given` (code or anatomical), else `default`
    (a world direction), else the header's dim_info."""
    code = meta.get('PhaseEncodingDirection') or given
    if code:
        return _parse(code, img.affine, f)
    if default is not None:
        return _from_world(default, img.affine, f)
    phase = img.header.get_dim_info()[1] if hasattr(img.header, 'get_dim_info') else None
    if phase is None:
        raise ValueError(f'{os.path.basename(f)}: no PhaseEncodingDirection (JSON), pe= or dim_info phase axis')
    warnings.warn(f'{os.path.basename(f)}: phase axis {phase} from dim_info, polarity assumed positive (the field '
                  'sign and the series correction depend on it)', RuntimeWarning, stacklevel=4)
    return int(phase), 1


def _same_grid(img, ref):
    return img.shape[:3] == ref.shape[:3] and np.allclose(img.affine, ref.affine, atol=1e-3)


def _on_grid(data, img, ref):
    """data (img's grid) on ref's grid, linear through world coordinates (no-op when the grids match)."""
    if data.shape == ref.shape[:3] and _same_grid(img, ref):
        return data
    vox = np.linalg.inv(img.affine) @ ref.affine
    return ndimage.affine_transform(data, vox[:3, :3], vox[:3, 3], output_shape=ref.shape[:3], order=1)


def _write_summary(folder, rows):
    """Add the rows to the folder's summary; a pair run again replaces its earlier row."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, SUMMARY_NAME)
    key = lambda row: (os.path.abspath(row['forward']), os.path.abspath(row['reverse']))
    done = {key(row) for row in rows}
    old = []
    if os.path.exists(path):
        with open(path, newline='') as fh:
            old = [row for row in csv.DictReader(fh) if key(row) not in done]
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), extrasaction='ignore')
        w.writeheader()
        w.writerows(old + rows)


def _corrected(img, ref, disp, ax, ftemplate, M=None, sign=1.0):
    """Every volume of img unwarped by sign * disp (and M) onto ref's grid in one cubic resampling, read in blocks and
    written volume by volume as <input>_duope; returns (path, the image backed by the file)."""
    shape = ref.shape[:3]
    idx, jac = core.unwarp_map(disp, ref.affine, ax, M=M, sign=sign, src_aff=img.affine)

    def volumes():
        for _, block in _chunks(img):
            for k in range(block.shape[3]):
                v = ndimage.map_coordinates(block[..., k].astype(np.float64), idx, order=3, mode='nearest')
                yield v.reshape(shape) * jac

    hdr = img.header if _same_grid(img, ref) else ref.header
    path = save_nib_volumes(hdr, ref.affine, shape + img.shape[3:], volumes(), ftemplate, 'duope')
    return path, nib.load(path)


def _one(f, r, series, output, pe, readout, series_pe, series_align, joint, save_field, save_disp, gz, verbose):
    t0 = time.time()
    fi, ri = load_nifti(f), load_nifti(r)
    fm, rm = _json(f), _json(r)
    ax_f, pol_f = _phase(fi, fm, pe, f)
    w_f = _world(ax_f, pol_f, fi.affine)
    ax_r, pol_r = _phase(ri, rm, None, r, default=-w_f)
    w_r = _world(ax_r, pol_r, ri.affine)
    if w_f @ w_r > -0.9:                                          # in world space: the grids may differ
        raise ValueError(f'{os.path.basename(f)} / {os.path.basename(r)}: not a reverse phase-encoding pair '
                         f'(directions {_label(w_f)} / {_label(w_r)}: {_AXES[ax_f]}{"-" if pol_f < 0 else ""} / '
                         f'{_AXES[ax_r]}{"-" if pol_r < 0 else ""} in their voxel axes)')
    trt = _readout(fm, fi.shape[ax_f]) or readout
    fwd = _b0(fi, f)
    rev = _on_grid(_b0(ri, r), ri, fi)
    aff = fi.affine
    pos, neg = (fwd, rev) if pol_f > 0 else (rev, fwd)          # the core takes the positive polarity first
    d, M, info = core.estimate(pos, neg, aff, pe_axis=ax_f, joint=joint)
    # d is the positive-polarity scan's displacement; M moves the negative-polarity scan
    m_f, m_r = (None, M) if pol_f > 0 else (M, None)
    ftemplate, _ = get_template(f, output, gz)
    rtemplate, _ = get_template(r, output, gz)
    images, saved = {}, {}
    saved['forward'], images['forward'] = _corrected(fi, fi, d, ax_f, ftemplate, M=m_f, sign=pol_f)
    saved['reverse'], images['reverse'] = _corrected(ri, fi, d, ax_f, rtemplate, M=m_r, sign=-pol_f)
    hdr = fi.header.copy()
    hdr.set_data_dtype(np.float32)
    hdr.set_slope_inter(1, 0)
    mk = lambda a: nib.Nifti1Image(np.asarray(a, np.float32), aff, _header3(hdr, a))
    if save_field:
        if not trt:
            raise ValueError(f'{os.path.basename(f)}: save_field needs the readout time (JSON TotalReadoutTime or '
                             'readout=)')
        images['field'] = mk(d / trt)
        saved['field'] = save_nib(images['field'], ftemplate, 'duope_field')
    if save_disp:
        images['disp'] = mk(pol_f * d)
        saved['disp'] = save_nib(images['disp'], ftemplate, 'duope_disp')

    series = [] if series is None else list(series) if isinstance(series, (list, tuple)) else [series]
    spes = list(series_pe) if isinstance(series_pe, (list, tuple)) else [series_pe] * len(series)
    if len(spes) != len(series):
        raise ValueError('series_pe: one entry per series')
    fixed = _b0(images['forward'], f) if series and series_align else None      # in the field's head position
    moved = []
    for sf, spe in zip(series, spes):
        si = nib.load(sf, keep_file_open=True)               # read in blocks; non-finite voxels set to 0 there
        sm = _json(sf)
        ax_s, pol_s = _phase(si, sm, spe, sf, default=w_f)
        if abs(_world(ax_s, pol_s, si.affine) @ w_f) < 0.9:
            raise ValueError(f'{os.path.basename(sf)}: phase-encoding direction '
                             f'{_label(_world(ax_s, pol_s, si.affine))}, the field is along {_label(w_f)}')
        trt_s = _readout(sm, si.shape[ax_s])
        if not _same_grid(si, fi) and not (trt_s and trt):
            warnings.warn(f'{os.path.basename(sf)}: another grid than the forward scan and no readout time for both; '
                          'the displacement in voxels is assumed equal', RuntimeWarning, stacklevel=4)
        scale = trt_s / trt if (trt_s and trt) else 1.0
        # the displacement in voxels (field x readout time) along the phase direction, moved to the series' head
        M_s = None
        if series_align:
            M_s = core.series_motion(_b0(si, sf), si.affine, fixed, aff, scale * d, ax_s, pol_s)
            c = (si.affine @ np.r_[(np.array(si.shape[:3]) - 1) / 2, 1])[:3]
            moved.append('%.2f/%.2f' % core.motion_size(M_s, c))
        stemplate, _ = get_template(sf, output, gz)
        key = os.path.basename(sf).split('.')[0]
        saved[key], images[key] = _corrected(si, si, core.field_on(scale * d, aff, si.affine, si.shape[:3], M_s),
                                             ax_s, stemplate, sign=pol_s)

    row = dict(forward=f, reverse=r, phase_axis=_AXES[ax_f], forward_polarity=pol_f, forward_direction=_label(w_f),
               readout_s=trt or '',
               motion_mm=round(info['motion_mm'], 3), motion_deg=round(info['motion_deg'], 3),
               joint=info['joint'], seconds=round(time.time() - t0, 1),
               series=';'.join(series), series_motion_mm_deg=';'.join(moved))
    if verbose >= 1:
        _logger.info('%s: motion %.2f mm / %.2f deg, joint %s, %.1f s', os.path.basename(f), info['motion_mm'],
                     info['motion_deg'], info['joint'], time.time() - t0)
    return images, saved, row


def _header3(hdr, a):
    h = hdr.copy()
    h.set_data_shape(np.shape(a))
    return h
