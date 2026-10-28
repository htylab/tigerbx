import csv
import importlib
import json
import os
import warnings

import nibabel as nib
import numpy as np
import pytest
from scipy import ndimage

import tigerbx
from tigerbx.core import duope as core

SHAPE = (30, 38, 24)
AFF = np.diag([-2.5, 2.5, 2.5, 1.0])
TRT = 0.05


def _phantom():
    rng = np.random.default_rng(0)
    g = np.indices(SHAPE).astype(float)
    c = [(n - 1) / 2 for n in SHAPE]
    r2 = sum(((g[i] - c[i]) / (0.38 * SHAPE[i])) ** 2 for i in range(3))
    body = ndimage.gaussian_filter((r2 < 1).astype(float), 1.0)
    tex = ndimage.gaussian_filter(rng.standard_normal(SHAPE), 1.5)
    u = body * (1 + 0.6 * tex / tex.std())
    d = 2.0 * np.exp(-(((g[0] - c[0]) / 6) ** 2 + ((g[1] - c[1] - 4) / 7) ** 2 + ((g[2] - c[2]) / 5) ** 2))
    return np.clip(u, 0, None) * 100, d, body > 0.5


def _distort(u, d):
    """Signal-conserving distortion along axis 1: the scan whose unwarping is u(x) = obs(x + d)(1 + dd/dy)."""
    n = u.shape[1]
    x = np.arange(n, dtype=float)
    edges = np.arange(-0.5, n + 0.5)
    out = np.zeros_like(u)
    for i in range(u.shape[0]):
        for k in range(u.shape[2]):
            y = np.maximum.accumulate(x + d[i, :, k] + 1e-6 * x)
            c = np.concatenate([[0.0], np.cumsum(u[i, :, k])])
            xe = np.interp(edges, np.r_[y[0] - 0.5, y, y[-1] + 0.5], np.r_[-0.5, x, n - 0.5])
            out[i, :, k] = np.diff(np.interp(xe, np.arange(-0.5, n), c))
    return out


@pytest.fixture(scope='module')
def pair():
    u, d, inside = _phantom()
    return _distort(u, d), _distort(u, -d), d, inside


def _mae(a, b, m):
    e = a - b
    return float(np.abs(e - np.median(e[m]))[m].mean())


def test_estimate_recovers_the_displacement(pair):
    fwd, rev, d_true, inside = pair
    d, M, info = core.estimate(fwd, rev, AFF, pe_axis=1)
    assert _mae(d, d_true, inside) < 0.15 < _mae(np.zeros_like(d), d_true, inside)
    assert info['motion_mm'] < 0.5 and not info['joint']


def test_phase_axis_0_is_the_transposed_problem(pair):
    fwd, rev, d_true, inside = pair
    T = lambda a: np.transpose(a, (1, 0, 2))
    aff = AFF[:, [1, 0, 2, 3]]                                     # same world, voxel axes swapped
    d, _, _ = core.estimate(T(fwd), T(rev), aff, pe_axis=0)
    assert _mae(d, T(d_true), T(inside)) < 0.15


def _save(tmp, name, data, pe, trt=TRT, bval=None):
    f = str(tmp / f'{name}.nii.gz')
    nib.save(nib.Nifti1Image(np.asarray(data, np.float32), AFF), f)
    meta = {'PhaseEncodingDirection': pe}
    if trt:
        meta['TotalReadoutTime'] = trt
    json.dump(meta, open(str(tmp / f'{name}.json'), 'w'))
    if bval is not None:
        open(str(tmp / f'{name}.bval'), 'w').write(' '.join(str(b) for b in bval))
    return f


def test_pipeline_outputs_signs_and_series(tmp_path, pair):
    fwd, rev, d_true, inside = pair
    fa = _save(tmp_path, 'sub_dir-AP_epi', fwd, 'j')
    fp = _save(tmp_path, 'sub_dir-PA_epi', rev, 'j-')
    series = _save(tmp_path, 'sub_dir-PA_dwi', np.stack([rev, 0.5 * rev], -1), 'j-')
    out = tmp_path / 'out'
    r = tigerbx.duope(fa, fp, str(out), series=series, save_field=True, save_disp=True)
    disp = r['disp'].get_fdata()
    assert _mae(disp, d_true, inside) < 0.15
    assert np.allclose(r['field'].get_fdata(), disp / TRT, atol=1e-4)               # positive polarity: field x TRT
    for name in ('sub_dir-AP_epi_duope', 'sub_dir-PA_epi_duope', 'sub_dir-PA_dwi_duope', 'sub_dir-AP_epi_duope_disp',
                 'sub_dir-AP_epi_duope_field'):
        assert (out / f'{name}.nii.gz').exists()
    rows = list(csv.DictReader(open(out / 'duope_summary.csv')))
    assert len(rows) == 1 and rows[0]['phase_axis'] == 'j' and rows[0]['forward_polarity'] == '1'
    u_true = _phantom()[0]
    cc = lambda a: np.corrcoef(a[inside], u_true[inside])[0, 1]
    for key, raw in (('forward', fwd), ('reverse', rev)):
        assert cc(r[key].get_fdata()) > cc(raw) + 0.05
    series_out = r['sub_dir-PA_dwi'].get_fdata()
    assert series_out.shape == SHAPE + (2,) and cc(series_out[..., 0]) > cc(rev) + 0.05
    # by default only the corrected inputs; the reverse scan given first: its displacement is the negative one
    r2 = tigerbx.duope(fp, fa, str(tmp_path / 'out2'), save_disp=True)
    assert sorted(r2) == ['disp', 'forward', 'reverse']
    assert _mae(r2['disp'].get_fdata(), -d_true, inside) < 0.15
    assert sorted(os.listdir(tmp_path / 'out2')) == ['duope_summary.csv', 'sub_dir-AP_epi_duope.nii.gz',
                                                     'sub_dir-PA_epi_duope.nii.gz', 'sub_dir-PA_epi_duope_disp.nii.gz']


def test_4d_input_uses_its_b0_volumes(tmp_path, pair):
    fwd, rev, d_true, inside = pair
    fa = _save(tmp_path, 'a_dwi', np.stack([0.3 * fwd, fwd, 0.3 * fwd], -1), 'j', bval=[1000, 0, 1000])
    fp = _save(tmp_path, 'p_epi', rev, 'j-')
    r = tigerbx.duope(fa, fp, str(tmp_path / 'out'), save_disp=True)
    assert _mae(r['disp'].get_fdata(), d_true, inside) < 0.15
    assert r['forward'].shape == SHAPE + (3,)                                      # every volume corrected


def test_bad_pairs(tmp_path, pair):
    fwd, rev, _, _ = pair
    fa = _save(tmp_path, 'a', fwd, 'j')
    same = _save(tmp_path, 'b', rev, 'j')
    other_axis = _save(tmp_path, 'c', rev, 'i-')
    for bad in (same, other_axis):
        with pytest.raises(ValueError, match='not a reverse phase-encoding pair'):
            tigerbx.duope(fa, bad, str(tmp_path / 'out'))
    good = _save(tmp_path, 'g', rev, 'j-')
    with pytest.warns(RuntimeWarning, match='1 of 2 pairs failed'):
        res = tigerbx.duope([fa, fa], [same, good], str(tmp_path / 'out3'))
    assert len(res) == 1 and os.path.exists(res[0]['forward'])


def test_no_json_needs_pe_or_dim_info(tmp_path, pair):
    fwd, rev, d_true, inside = pair
    fa, fp = str(tmp_path / 'a.nii.gz'), str(tmp_path / 'b.nii.gz')
    nib.save(nib.Nifti1Image(fwd.astype(np.float32), AFF), fa)
    nib.save(nib.Nifti1Image(rev.astype(np.float32), AFF), fp)
    with pytest.raises(ValueError, match='PhaseEncodingDirection'):
        tigerbx.duope(fa, fp, str(tmp_path / 'o1'))
    r = tigerbx.duope(fa, fp, str(tmp_path / 'o2'), pe='j', readout=TRT, save_field=True)
    assert _mae(r['field'].get_fdata() * TRT, d_true, inside) < 0.15
    img = nib.load(fa)
    img.header.set_dim_info(phase=1)
    nib.save(img, fa)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        tigerbx.duope(fa, fp, str(tmp_path / 'o3'))
    assert any('dim_info' in str(x.message) for x in w)
    with pytest.raises(ValueError, match='readout time'):
        tigerbx.duope(fa, fp, str(tmp_path / 'o4'), save_field=True)


def test_anatomical_pe_per_pair_and_reoriented_reverse(tmp_path, pair):
    P = importlib.import_module('tigerbx.pipelines.duope')
    # AFF: voxel i runs to -x (towards the right), j to +y (anterior), k to +z
    assert P._parse('LR', AFF, 'a') == (0, -1) and P._parse('rl', AFF, 'a') == (0, 1)
    assert P._parse('PA', AFF, 'a') == (1, 1) and P._parse('AP', AFF, 'a') == (1, -1)
    assert P._parse('y-', AFF, 'a') == (1, -1) and P._parse('SI', AFF, 'a') == (2, -1)
    c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
    oblique = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])
    with pytest.raises(ValueError, match='degrees from the nearest voxel axis'):
        P._parse('AP', oblique, 'a')
    with pytest.raises(ValueError, match='phase-encoding direction'):
        P._parse('up', AFF, 'a')

    fwd, rev, d_true, inside = pair
    fa, fp = str(tmp_path / 'a.nii.gz'), str(tmp_path / 'p.nii.gz')
    nib.save(nib.Nifti1Image(fwd.astype(np.float32), AFF), fa)
    nib.save(nib.Nifti1Image(rev.astype(np.float32), AFF), fp)
    # the reverse scan stored with j flipped: the same world direction is 'j' in its own voxels
    flip = AFF.copy()
    flip[:3, 1] *= -1
    flip[:3, 3] = AFF[:3, 3] + AFF[:3, 1] * (SHAPE[1] - 1)
    ff = _save(tmp_path, 'p_flipped', rev[:, ::-1], 'j')
    nib.save(nib.Nifti1Image(rev[:, ::-1].astype(np.float32), flip), ff)
    sr = str(tmp_path / 's.nii.gz')                                         # a series without JSON, PE as the reverse
    nib.save(nib.Nifti1Image(rev.astype(np.float32), AFF), sr)
    out = tmp_path / 'out'
    res = tigerbx.duope([fa, fa], [fp, ff], str(out), pe=['PA', 'j'], readout=TRT, series=[sr, None],
                        series_pe=['AP', None], save_disp=True)
    assert len(res) == 2
    for r in res:
        assert _mae(nib.load(r['disp']).get_fdata(), d_true, inside) < 0.15
    rows = list(csv.DictReader(open(out / 'duope_summary.csv')))
    assert [(x['phase_axis'], x['forward_polarity'], x['forward_direction']) for x in rows] == [('j', '1', 'PA')] * 2
    u_true = _phantom()[0]
    cc = lambda a: np.corrcoef(a[inside], u_true[inside])[0, 1]
    assert cc(nib.load(res[0]['s']).get_fdata()) > cc(rev) + 0.05
    # 'AP' for the forward scan reverses both polarities: same correction, displacement negated
    r = tigerbx.duope(fa, fp, str(tmp_path / 'o2'), pe='AP', save_disp=True)
    assert _mae(r['disp'].get_fdata(), d_true, inside) < 0.15                  # forward's own displacement
    with pytest.raises(ValueError, match='one entry per pair'):
        tigerbx.duope([fa, fa], [fp, ff], str(out), pe=['PA'])


def test_series_on_another_grid_and_rerun_summary(tmp_path, pair):
    fwd, rev, d_true, inside = pair
    fa = _save(tmp_path, 'a', fwd, 'j')
    fp = _save(tmp_path, 'p', rev, 'j-')
    # a series acquired like the reverse scan, stored with j flipped and 4 slices fewer (another grid)
    aff = AFF.copy()
    aff[:3, 1] *= -1
    aff[:3, 3] = AFF[:3, 3] + AFF[:3, 1] * (SHAPE[1] - 1) + AFF[:3, 2] * 2
    sub = lambda a: a[:, ::-1, 2:-2]
    sf = _save(tmp_path, 's_bold', sub(rev), 'j')                     # reverse polarity in its own (flipped) voxels
    nib.save(nib.Nifti1Image(np.stack([sub(rev)] * 2, -1).astype(np.float32), aff), sf)
    out = tmp_path / 'out'
    r = tigerbx.duope(fa, fp, str(out), series=sf)
    s = r['s_bold']
    assert s.shape == sub(rev).shape + (2,) and np.allclose(s.affine, aff)
    u_true = sub(_phantom()[0])
    m = sub(inside)
    cc = lambda a: np.corrcoef(a[m], u_true[m])[0, 1]
    assert cc(s.get_fdata()[..., 1]) > cc(sub(rev)) + 0.05
    # the same series on the forward grid gives the same correction
    sg = _save(tmp_path, 's_grid', np.stack([rev] * 2, -1), 'j-')
    rg = tigerbx.duope(fa, fp, str(tmp_path / 'out_g'), series=sg)
    assert np.abs(sub(rg['s_grid'].get_fdata()[..., 0]) - s.get_fdata()[..., 0])[m].mean() < 0.01 * u_true[m].mean()
    # a pair run again replaces its row in the summary
    tigerbx.duope(fa, fp, str(out))
    rows = list(csv.DictReader(open(out / 'duope_summary.csv')))
    assert len(rows) == 1 and rows[0]['series'] == ''
    bare = str(tmp_path / 'bare.nii.gz')                              # no JSON: series_pe applies
    nib.save(nib.Nifti1Image(rev.astype(np.float32), AFF), bare)
    with pytest.raises(ValueError, match='the field is along'):
        tigerbx.duope(fa, fp, str(out), series=bare, series_pe='LR')


def test_series_align_moves_the_field_with_the_head(tmp_path, pair):
    fwd, rev, d_true, inside = pair
    u_true = _phantom()[0]
    fa = _save(tmp_path, 'a', fwd, 'j')
    fp = _save(tmp_path, 'p', rev, 'j-')
    sh = lambda a: ndimage.shift(a, (0, 0, 3), order=1, mode='nearest')     # head moved 3 voxels (7.5 mm) along k
    sf = _save(tmp_path, 's', np.stack([_distort(sh(u_true), -sh(d_true))] * 2, -1), 'j-')
    out = tmp_path / 'out'
    on = tigerbx.duope(fa, fp, str(out), series=sf)['s'].get_fdata()[..., 0]
    off = tigerbx.duope(fa, fp, str(tmp_path / 'off'), series=sf, series_align=False)['s'].get_fdata()[..., 0]
    m = sh(inside.astype(float)) > 0.5
    err = lambda a: np.abs(a - sh(u_true))[m].mean()
    assert err(on) < 0.6 * err(off)
    mm, deg = map(float, list(csv.DictReader(open(out / 'duope_summary.csv')))[0]['series_motion_mm_deg'].split('/'))
    assert abs(mm - 7.5) < 1.0 and deg < 1.0


@pytest.mark.parametrize('ext', ['.nii', '.nii.gz'])
def test_volume_by_volume_writer_matches_nib_save(tmp_path, ext):
    from tigerbx.core.io import save_nib_volumes
    rng = np.random.default_rng(1)
    data = rng.standard_normal((5, 6, 4, 3)).astype(np.float32)
    aff = np.array([[0, -2, 0, 10], [2.5, 0, 0, -3], [0, 0, 3, 7], [0, 0, 0, 1.0]])
    src = nib.Nifti1Image(data.astype('>i2'), aff)                 # big-endian int source header
    src.header.set_dim_info(freq=0, phase=1, slice=2)
    src.header.set_xyzt_units('mm', 'sec')
    src.header['pixdim'][4] = 1.4
    path = save_nib_volumes(src.header, aff, data.shape, (data[..., k] for k in range(3)),
                            str(tmp_path / f'a_@@@@{ext}'), 'duope')
    ref = nib.Nifti1Image(data, aff, src.header.copy())
    ref.header.set_data_dtype(np.float32)
    nib.save(ref, str(tmp_path / f'b{ext}'))
    a, b = nib.load(path), nib.load(str(tmp_path / f'b{ext}'))
    assert path.endswith(f'a_duope{ext}') and np.array_equal(a.get_fdata(), b.get_fdata())
    assert np.allclose(a.affine, b.affine) and a.header.get_dim_info() == (0, 1, 2)
    assert a.header.get_xyzt_units() == ('mm', 'sec') and a.header['pixdim'][4] == pytest.approx(1.4)
    assert a.get_data_dtype() == np.dtype('<f4')
