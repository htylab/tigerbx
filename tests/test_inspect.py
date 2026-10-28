import csv
import importlib

import nibabel as nib
import numpy as np

inspect_mod = importlib.import_module('tigerbx.pipelines.inspect')    # tigerbx.inspect is the inspect() function


def test_orientation_codes():
    assert inspect_mod.orientation_code(0) == 'RAS'
    assert inspect_mod.orientation_code(2) == 'RPS'           # A-P flipped
    assert inspect_mod.orientation_code(1) == 'RAI'           # S-I flipped
    codes = {inspect_mod.orientation_code(k) for k in range(24)}
    assert len(codes) == 24 and all(sorted(c.replace('P', 'A').replace('I', 'S')) == ['A', 'R', 'S'] for c in codes)


def test_make_input_centres_the_object():
    d = np.zeros((96, 96, 96), np.float32)
    d[20:40, 50:70, 30:60] = 100.0                           # a box off the array centre
    img = nib.Nifti1Image(d, np.diag([2.0, 2.0, 2.0, 1.0]))
    x = inspect_mod.make_input(img)
    assert x.shape == (4, 96, 96, 96) and x.dtype == np.float32
    com = np.array(np.nonzero(x[0] > 0.5)).mean(1)
    np.testing.assert_allclose(com, [47.5, 47.5, 47.5], atol=1.0)
    assert x[2].min() == 0 and x[2].max() == 1                # the 256 mm cube is larger than the 192 mm image
    assert np.all(x[1][x[2] == 0] == 1)                       # outside the field of view counts as empty
    assert x[3].max() > 0                                     # edges of the box in the high-frequency channel
    assert x[3][44:51, 44:51, 44:51].max() < 1e-3             # flat interior: no high frequencies


def test_header_features():
    img = nib.Nifti1Image(np.zeros((91, 109, 91), np.float32), np.diag([-2.0, 2.0, 2.0, 1.0]))
    h = inspect_mod.header_features(img, (91, 109, 91))
    assert h.shape == (11,)
    np.testing.assert_allclose(h[:3], np.log(2.0), rtol=1e-6)
    assert h[6] == 0 and h[9] == 0 and h[10] == 1


def test_rule_checks():
    d = np.random.default_rng(0).normal(100, 20, (80, 90, 70)).astype(np.float32)
    r = inspect_mod.rule_checks(nib.Nifti1Image(d, np.diag([1.5, 1.5, 1.5, 1.0])))      # 105-135 mm field of view
    assert r['warnings'] == '' and not r['label_map'] and r['voxel_mm'] == '1.50x1.50x1.50'
    assert r['matrix'] == '80x90x70' and r['ndim'] == 3
    assert 'slab' in inspect_mod.rule_checks(nib.Nifti1Image(d, np.eye(4)))['warnings']   # 70 mm along z
    lab = np.zeros((80, 90, 70), np.int16)
    lab[10:30, 10:40, 10:50], lab[40:60, 10:40, 10:50] = 17, 53
    assert inspect_mod.rule_checks(nib.Nifti1Image(lab, np.eye(4)))['label_map']
    a = np.deg2rad(30)
    R = np.eye(4)
    R[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    r = inspect_mod.rule_checks(nib.Nifti1Image(np.stack([d, d], -1), R @ np.diag([1.0, 1.0, 5.0, 1.0])))
    assert r['n_volumes'] == 2 and r['thick_slices'] and abs(r['oblique_deg'] - 30) < 0.1
    assert r['matrix'] == '80x90x70x2' and r['ndim'] == 4
    assert 'volumes' in r['warnings'] and 'thick' in r['warnings'] and 'oblique' in r['warnings']
    d2 = d.copy()
    d2[0, 0, 0] = np.nan
    assert 'non-finite' in inspect_mod.rule_checks(nib.Nifti1Image(d2, np.eye(4)))['warnings']


def test_inspect_batch_continues_after_failure(tmp_path, monkeypatch):
    files = []
    for name in ('bad', 'a'):                                 # rows keep the order of the given list
        f = tmp_path / f'{name}.nii.gz'
        nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.float32), np.eye(4)), str(f))
        files.append(str(f))

    def fake(img, session):
        if img.get_filename().endswith('bad.nii.gz'):
            raise ValueError('broken')
        return dict(contrast='T1w', orientation='RAS')

    monkeypatch.setattr(inspect_mod, 'inspect_one', fake)
    monkeypatch.setattr(inspect_mod, 'create_session', lambda *a, **k: None)
    monkeypatch.setattr(inspect_mod.lib_tool, 'get_model', lambda m: m)
    rows = inspect_mod.inspect(files, str(tmp_path / 'out'))
    assert [r['status'] for r in rows] == ['failed', 'ok']
    with open(tmp_path / 'out' / inspect_mod.SUMMARY_NAME) as fh:
        rd = csv.DictReader(fh)
        got = list(rd)
    assert tuple(rd.fieldnames) == inspect_mod.SUMMARY_COLUMNS
    assert [r['status'] for r in got] == ['failed', 'ok'] and got[1]['contrast'] == 'T1w'
    inspect_mod.inspect(files, str(tmp_path / 'out2'), detail=True)
    with open(tmp_path / 'out2' / inspect_mod.SUMMARY_NAME) as fh:
        assert tuple(csv.DictReader(fh).fieldnames) == inspect_mod.DETAIL_COLUMNS
