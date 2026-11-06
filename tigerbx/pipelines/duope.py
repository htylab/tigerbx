"""DuoPE: susceptibility distortion correction from a reverse phase-encoding pair (see tigerbx.core.duope).

    tigerbx.duope('sub-01_dir-AP_epi.nii.gz', 'sub-01_dir-PA_epi.nii.gz', 'out/')
    tigerbx.duope(ap, pa, 'out/', series='sub-01_task-rest_bold.nii.gz')   # also correct a BOLD / DWI series

The phase-encoding axis and polarity of each scan and the total readout time come from the BIDS JSON sidecars
(PhaseEncodingDirection, TotalReadoutTime), else from `pe` / `readout` (`pe` also as an anatomical direction, 'AP',
'PA', 'LR', ..., converted with the scan's affine), else from the data (with a warning): the phase axis is the one
along which the corrected pair agrees best (the polarity is then taken as positive: the corrected images do not depend
on it, the field's sign does), and a series runs along the forward or the reverse scan's direction by which of the two
unwarps it onto the corrected forward scan. A series' JSON direction is checked against the data. The reverse scan's
direction is compared in world space, so it may be
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
          joint='auto', alpha=core.ALPHA, max_mm=core.MAX_MM, save_field=False, save_disp=False,
          resume=False, gz=True, verbose=0):
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
    1 degree), True or False. alpha / max_mm: the field's bending weight and its finest estimation grid in mm (0 =
    the native grid), as core.duope.estimate; the defaults are tuned for correcting images, a field map for its own
    sake (QSM, ...) may want max_mm=0 and a smaller alpha. resume: skip pairs whose <forward>_duope is already
    there (continuing a batch; they are left out of the list).

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
        if resume and os.path.isfile(get_template(f, output, gz)[0].replace('@@@@', 'duope')):
            continue
        try:
            images, saved, row = _one(f, r, s, output, p, t, sp, series_align,
                                      dict(joint=joint, alpha=alpha, max_mm=max_mm), save_field, save_disp, gz,
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
    (a world direction); without any, the caller finds it from the data (_pair_phase / _series_phase)."""
    code = meta.get('PhaseEncodingDirection') or given
    if code:
        return _parse(code, img.affine, f)
    if default is None:
        raise ValueError(f'{os.path.basename(f)}: no PhaseEncodingDirection (JSON) or pe=')
    return _from_world(default, img.affine, f)


def _same_grid(img, ref):
    return img.shape[:3] == ref.shape[:3] and np.allclose(img.affine, ref.affine, atol=1e-3)


def _on_grid(data, img, ref):
    """data (img's grid) on ref's grid, linear through world coordinates (no-op when the grids match)."""
    if data.shape == ref.shape[:3] and _same_grid(img, ref):
        return data
    vox = np.linalg.inv(img.affine) @ ref.affine
    return ndimage.affine_transform(data, vox[:3, :3], vox[:3, 3], output_shape=ref.shape[:3], order=1)


# ---------------------------------------------------------------- the phase encoding from the data (no JSON / pe)

def _known(meta, given):
    return bool(meta.get('PhaseEncodingDirection') or given)


def _ncc(a, b, m):
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-12))


def _pair_axis(fwd, rev, img, f, est):
    """The phase axis of a pair from the data: the field estimated along each voxel axis, the axis whose corrected
    pair agrees best (NCC in the head mask). Returns (axis, (d, M, info) along it with fwd as the positive scan)."""
    aff, fits, score = img.affine, {}, {}
    for ax in range(3):
        d, M, info = core.estimate(fwd, rev, aff, pe_axis=ax, **est)
        cf, cr = core.unwarp(fwd, d, aff, ax, sign=1.0), core.unwarp(rev, d, aff, ax, M=M, sign=-1.0)
        fits[ax], score[ax] = (d, M, info), _ncc(cf, cr, core.head_mask(cf, cr))
    ax = max(score, key=score.get)
    dim = img.header.get_dim_info()[1] if hasattr(img.header, 'get_dim_info') else None
    if dim is not None and int(dim) != ax:
        warnings.warn(f'{os.path.basename(f)}: the data give phase axis {_AXES[ax]}, the header (dim_info) '
                      f'{_AXES[int(dim)]}; the data are used', RuntimeWarning, stacklevel=5)
    warnings.warn(f'{os.path.basename(f)}: no PhaseEncodingDirection (JSON) or pe=: phase axis {_AXES[ax]} from the '
                  f'data (corrected-pair NCC {", ".join(f"{_AXES[a]} {s:.3f}" for a, s in score.items())}), polarity '
                  'taken as positive (the corrected images do not depend on it, the field sign does)',
                  RuntimeWarning, stacklevel=5)
    return ax, fits[ax]


def _pair_phase(fi, ri, fm, rm, pe, f, r, fwd, rev, est):
    """(axis, polarity, (d, M, info) or None, guessed) of the forward scan: its JSON / pe, else the opposite of the
    reverse scan's JSON, else the axis from the data (polarity +1, guessed)."""
    if _known(fm, pe):
        return (*_phase(fi, fm, pe, f), None, False)
    if _known(rm, None):
        ax_r, pol_r = _phase(ri, rm, None, r)
        return (*_from_world(-_world(ax_r, pol_r, ri.affine), fi.affine, f), None, False)
    ax, fit = _pair_axis(fwd, rev, fi, f, est)
    return ax, 1, fit, True


def _series_direction(ref_s, s_aff, fixed, aff, d, w_f, sf):
    """Whether a series runs along the forward scan's direction w_f (+1) or the opposite (-1), from the data: its
    b0 / mean unwarped either way (the field d, forward grid, moved to its head position), NCC with the corrected
    forward image `fixed`. Returns (sign, its rigid M series world -> forward world)."""
    score, Ms = {}, {}
    for s in (1, -1):
        ax, pol = _from_world(s * w_f, s_aff, sf)
        Ms[s] = core.series_motion(ref_s, s_aff, fixed, aff, d, ax, pol)
        u = core.unwarp(ref_s, core.field_on(d, aff, s_aff, ref_s.shape, Ms[s]), s_aff, ax, sign=pol)
        v = np.linalg.inv(aff) @ Ms[s] @ s_aff
        g = ndimage.affine_transform(np.asarray(fixed, np.float64), v[:3, :3], v[:3, 3], output_shape=ref_s.shape,
                                     order=1)
        score[s] = _ncc(u, g, core.head_mask(u, u) & (g > 0))
    return max(score, key=score.get), Ms, score


SERIES_MARGIN = 0.03   # the data overrule a series' JSON direction when one direction's NCC is higher by this much


def _series_phase(si, sm, spe, sf, ref_s, fixed, aff, d, w_f, guessed_f):
    """(axis, polarity, M series world -> forward world, info) of a series. Its direction relative to the forward
    scan's comes from the data (its b0 / mean unwarped either way onto the corrected forward scan); series_pe= is
    always followed; a JSON direction is followed unless the data clearly (NCC margin >= SERIES_MARGIN) say the
    other one. info: source ('series_pe', 'json', 'data', 'data_over_json') and the two NCCs."""
    given = None
    if _known(sm, spe):
        ax, pol = _phase(si, sm, spe, sf)
        if abs(_world(ax, pol, si.affine) @ w_f) < 0.9:
            raise ValueError(f'{os.path.basename(sf)}: phase-encoding direction '
                             f'{_label(_world(ax, pol, si.affine))}, the field is along {_label(w_f)}')
        if not guessed_f:                         # its direction relative to the forward scan's is known
            given = 1 if _world(ax, pol, si.affine) @ w_f > 0 else -1
    s, Ms, score = _series_direction(ref_s, si.affine, fixed, aff, d, w_f, sf)
    margin = abs(score[1] - score[-1])
    how = f'NCC {score[1]:.3f} along the forward scan\'s direction, {score[-1]:.3f} along the reverse'
    side = lambda v: 'the forward' if v > 0 else 'the reverse'
    name = os.path.basename(sf)
    info = dict(ncc_forward=score[1], ncc_reverse=score[-1])
    if given is not None and (spe or given == s or margin < SERIES_MARGIN):
        if given != s:
            why = 'series_pe' if spe else f'JSON (the data are not clear: margin {margin:.3f} < {SERIES_MARGIN})'
            warnings.warn(f'{name}: the {why} gives {side(given)} scan\'s direction, the data {side(s)} ({how}); '
                          f'{"series_pe" if spe else "the JSON"} is used', RuntimeWarning, stacklevel=5)
        return ax, pol, Ms[given], dict(info, source='series_pe' if spe else 'json')
    if given is not None:
        warnings.warn(f'{name}: its JSON PhaseEncodingDirection gives {side(given)} scan\'s direction, the data '
                      f'clearly {side(s)} ({how}); the data are used (give series_pe= to force the JSON one)',
                      RuntimeWarning, stacklevel=5)
        source = 'data_over_json'
    else:
        if not _known(sm, spe):
            warnings.warn(f'{name}: no PhaseEncodingDirection (JSON) or series_pe=: {side(s)} scan\'s direction '
                          f'from the data ({how})', RuntimeWarning, stacklevel=5)
        source = 'data'
    ax, pol = _from_world(s * w_f, si.affine, sf)
    return ax, pol, Ms[s], dict(info, source=source)


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


def _one(f, r, series, output, pe, readout, series_pe, series_align, est, save_field, save_disp, gz, verbose):
    t0 = time.time()
    fi, ri = load_nifti(f), load_nifti(r)
    fm, rm = _json(f), _json(r)
    fwd = _b0(fi, f)
    rev = _on_grid(_b0(ri, r), ri, fi)
    ax_f, pol_f, fit, guessed_f = _pair_phase(fi, ri, fm, rm, pe, f, r, fwd, rev, est)
    w_f = _world(ax_f, pol_f, fi.affine)
    ax_r, pol_r = _phase(ri, rm, None, r, default=-w_f)
    w_r = _world(ax_r, pol_r, ri.affine)
    if w_f @ w_r > -0.9:                                          # in world space: the grids may differ
        raise ValueError(f'{os.path.basename(f)} / {os.path.basename(r)}: not a reverse phase-encoding pair '
                         f'(directions {_label(w_f)} / {_label(w_r)}: {_AXES[ax_f]}{"-" if pol_f < 0 else ""} / '
                         f'{_AXES[ax_r]}{"-" if pol_r < 0 else ""} in their voxel axes)')
    trt = _readout(fm, fi.shape[ax_f]) or readout
    aff = fi.affine
    pos, neg = (fwd, rev) if pol_f > 0 else (rev, fwd)          # the core takes the positive polarity first
    d, M, info = fit or core.estimate(pos, neg, aff, pe_axis=ax_f, **est)
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
    fixed = _b0(images['forward'], f) if series else None      # in the field's head position
    moved, dirs = [], []
    for sf, spe in zip(series, spes):
        si = nib.load(sf, keep_file_open=True)               # read in blocks; non-finite voxels set to 0 there
        sm = _json(sf)
        trt_s = _readout(sm, si.shape[_from_world(w_f, si.affine, sf)[0]])
        if not _same_grid(si, fi) and not (trt_s and trt):
            warnings.warn(f'{os.path.basename(sf)}: another grid than the forward scan and no readout time for both; '
                          'the displacement in voxels is assumed equal', RuntimeWarning, stacklevel=4)
        scale = trt_s / trt if (trt_s and trt) else 1.0
        # the displacement in voxels (field x readout time) along the phase direction, moved to the series' head
        ax_s, pol_s, M_s, sinfo = _series_phase(si, sm, spe, sf, _b0(si, sf), fixed, aff, scale * d, w_f,
                                                guessed_f)
        dirs.append('%s %.3f/%.3f' % (sinfo['source'], sinfo['ncc_forward'], sinfo['ncc_reverse']))
        if series_align:
            c = (si.affine @ np.r_[(np.array(si.shape[:3]) - 1) / 2, 1])[:3]
            moved.append('%.2f/%.2f' % core.motion_size(M_s, c))
        else:
            M_s = None
        stemplate, _ = get_template(sf, output, gz)
        key = os.path.basename(sf).split('.')[0]
        saved[key], images[key] = _corrected(si, si, core.field_on(scale * d, aff, si.affine, si.shape[:3], M_s),
                                             ax_s, stemplate, sign=pol_s)

    row = dict(forward=f, reverse=r, phase_axis=_AXES[ax_f], forward_polarity=pol_f, forward_direction=_label(w_f),
               readout_s=trt or '',
               motion_mm=round(info['motion_mm'], 3), motion_deg=round(info['motion_deg'], 3),
               joint=info['joint'], seconds=round(time.time() - t0, 1),
               series=';'.join(series), series_motion_mm_deg=';'.join(moved),
               series_direction=';'.join(dirs))
    if verbose >= 1:
        _logger.info('%s: motion %.2f mm / %.2f deg, joint %s, %.1f s', os.path.basename(f), info['motion_mm'],
                     info['motion_deg'], info['joint'], time.time() - t0)
    return images, saved, row


def _header3(hdr, a):
    h = hdr.copy()
    h.set_data_shape(np.shape(a))
    return h
