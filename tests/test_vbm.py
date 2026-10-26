import csv
import importlib
import inspect

import nibabel as nib
import numpy as np
import pytest

vbm_mod = importlib.import_module('tigerbx.pipelines.vbm')


def test_vbm_new_options_are_keyword_only():
    params = inspect.signature(vbm_mod.vbm).parameters
    names = list(params)
    assert names[:10] == ['input', 'output', 'model', 'template', 'reg_plan', 'gpu', 'gz',
                          'save_displacement', 'affine_type', 'verbose']        # 0.2.x positional order
    assert params['fwhm'].kind is inspect.Parameter.KEYWORD_ONLY
    assert params['save_modulated'].kind is inspect.Parameter.KEYWORD_ONLY


def test_check_fwhm():
    assert vbm_mod._check_fwhm(None) == vbm_mod.DEFAULT_FWHM
    assert vbm_mod._check_fwhm('8') == 8.0
    for bad in (0, -1, float('nan'), float('inf'), 'x'):
        with pytest.raises(ValueError):
            vbm_mod._check_fwhm(bad)


def test_final_image_follows_plan_order():
    rd = dict(affine=1, VFA=2, VXM=3)
    assert vbm_mod._final_image_key('ADV', rd) == 'VXM'
    assert vbm_mod._final_image_key('AVD', rd) == 'VFA'
    assert vbm_mod._final_image_key('A', dict(affine=1)) == 'affine'


def test_vbm_single_failure_writes_summary(tmp_path, monkeypatch):
    f = tmp_path / 'a.nii.gz'
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.float32), np.eye(4)), str(f))

    def boom(*a, **k):
        raise RuntimeError('boom')
    monkeypatch.setattr(vbm_mod, '_run_one', boom)
    out = tmp_path / 'out'
    with pytest.raises(RuntimeError):
        vbm_mod.vbm(str(f), str(out))
    rows = list(csv.DictReader(open(out / vbm_mod.SUMMARY_NAME)))
    assert [r['status'] for r in rows] == ['failed'] and 'boom' in rows[0]['error']
    assert (out / 'a_vbmfail.log').exists()


def test_fail_log_helpers_never_raise(tmp_path):
    from tigerbx.core.io import remove_quietly, write_fail_log
    blocker = tmp_path / 'file'
    blocker.write_text('')
    with pytest.warns(UserWarning):
        write_fail_log(str(blocker / 'sub' / 'x_vbmfail.log'), 'x', 'err')   # parent is a file
    remove_quietly(str(tmp_path / 'missing.log'))


def test_qc_flags_combine_quality_outliers_and_homogeneity():
    rng = np.random.default_rng(1)
    base = rng.random((6, 6, 6)) + 1
    rows = []
    for i in range(12):
        m = base + rng.normal(0, 0.01, base.shape)
        if i == 2:
            m = m[::-1]                                  # flipped map: homogeneity outlier
        r = dict(file=f'{i}.nii.gz', status='ok', noise=0.05 + 0.001 * i, bias=0.2, contrast=0.4 + 0.001 * i,
                 edge=0.3 + 0.001 * i, res=1.0, _homog_map=m)
        rows.append(r)
    rows[5]['noise'] = 0.5                                # noisy image
    rows.append(dict(file='bad.nii.gz', status='failed', error='boom'))
    counts = vbm_mod._set_qc_flags(rows)
    assert rows[2]['qc_flags'] == 'homog'
    assert rows[5]['qc_flags'] == 'noise'
    assert rows[-1]['qc_flags'] == 'fail'
    assert sum(1 for r in rows if r['qc_flags']) == 3
    assert counts == {'fail': 1, 'homog': 1, 'noise': 1}
    assert not any('_homog_map' in r or '_flags' in r for r in rows)


def test_qc_flags_need_a_sample():
    rows = [dict(file=f'{i}', status='ok', noise=1.0 if i else 9.0, _homog_map=np.ones((2, 2, 2))) for i in range(5)]
    vbm_mod._set_qc_flags(rows)
    assert all(r['qc_flags'] == '' for r in rows)
