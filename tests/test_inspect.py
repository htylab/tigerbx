import csv
import importlib

import nibabel as nib
import numpy as np

inspect_mod = importlib.import_module('tigerbx.inspect')     # tigerbx.inspect is also the inspect() function


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


def test_inspect_batch_continues_after_failure(tmp_path, monkeypatch):
    files = []
    for name in ('a', 'bad'):
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
    assert [r['status'] for r in rows] == ['ok', 'failed']
    with open(tmp_path / 'out' / inspect_mod.SUMMARY_NAME) as fh:
        got = list(csv.DictReader(fh))
    assert [r['status'] for r in got] == ['ok', 'failed'] and got[0]['contrast'] == 'T1w'
