import os
import importlib
import shutil
import tempfile
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from tigerbx import gdmi

from tigerbx.modules import bx



def _write_nifti(filename, data):
    nib.save(nib.Nifti1Image(data, np.eye(4)), filename)


@pytest.fixture
def work_dir():
    root = os.path.join(os.getcwd(), 'temp')
    os.makedirs(root, exist_ok=True)
    dirname = tempfile.mkdtemp(prefix='pytest-', dir=root)
    yield dirname
    shutil.rmtree(dirname)


def test_gdm_single_and_multiple_return_contracts(work_dir, monkeypatch):
    output_dir = os.path.join(work_dir, 'out')
    first = os.path.join(work_dir, 'first.nii.gz')
    second = os.path.join(work_dir, 'second.nii.gz')
    data = np.ones((3, 3, 3), dtype=np.float32)
    _write_nifti(first, data)
    _write_nifti(second, data)

    monkeypatch.setattr(gdmi.lib_tool, 'get_model', lambda name: name)
    monkeypatch.setattr(gdmi, 'run', lambda *args, **kwargs: (data, data))

    base_args = dict(output=output_dir, b0_index='0', dmap=True,
                     no_resample=False, gpu=False, verbose=0)
    single = gdmi.run_args(SimpleNamespace(input=[first], **base_args))
    multiple = gdmi.run_args(SimpleNamespace(input=[first, second], **base_args))

    assert isinstance(single['gdmi'], nib.spatialimages.SpatialImage)
    assert isinstance(single['gdm'], nib.spatialimages.SpatialImage)
    assert len(multiple) == 2
    assert all(isinstance(item['gdmi'], str) for item in multiple)


def test_gdm_accepts_empty_inputs():
    gdm_args = SimpleNamespace(input=[], output=None, b0_index='0', dmap=False,
                               no_resample=False, gpu=False, verbose=0)

    assert gdmi.run_args(gdm_args) == []


def test_bx_rejects_empty_tbx_mask(monkeypatch):
    mask = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4))
    monkeypatch.setattr(bx.lib_tool, 'get_model', lambda model: model)
    monkeypatch.setattr(bx, '_infer_mask', lambda *args, **kwargs: (mask, mask))

    try:
        bx.produce_tbxmask('model.onnx', 'empty.nii.gz')
    except RuntimeError as exc:
        assert 'empty brain mask' in str(exc)
    else:
        raise AssertionError('Expected an empty tBX mask to fail')


def test_cgw_model_and_hlc_tissue_outputs_are_removed():
    import tigerbx
    with pytest.raises(ValueError, match='CGW model'):
        bx.run('mC', 'x.nii.gz')
    with pytest.raises(ValueError, match='seg\\(save_tissue=True\\)'):
        tigerbx.hlc('x.nii.gz', save='hg')
    with pytest.raises(ValueError, match="run\\('c'\\)"):
        tigerbx.hlc('x.nii.gz', save='ht')


def test_tissue_pve_mixes_boundary_voxels_by_intensity():
    from tigerbx.core.cgw import tissue_pve
    img = np.zeros((5, 5, 9), np.float32)
    lab = np.zeros((5, 5, 9), int)
    img[..., 0:3], lab[..., 0:3] = 20, 4          # CSF (lateral ventricle)
    img[..., 3:6], lab[..., 3:6] = 60, 3          # GM (cortex)
    img[..., 6:9], lab[..., 6:9] = 100, 2         # WM
    img[..., 3], img[..., 5] = 40, 80             # boundary voxels halfway between two tissues
    p = tissue_pve(img, lab, np.ones(img.shape, bool), (1, 1, 1))
    np.testing.assert_allclose(p[:, 2, 2, 3], [0.5, 0.5, 0], atol=0.01)
    np.testing.assert_allclose(p[:, 2, 2, 5], [0, 0.5, 0.5], atol=0.01)
    np.testing.assert_allclose(p[:, 2, 2, 4], [0, 1, 0])
    np.testing.assert_allclose(p.sum(0), 1, atol=1e-6)

def test_vbm_pipeline_resolves_reg_functions():
    import tigerbx
    vbm_mod = importlib.import_module('tigerbx.pipelines.vbm')   # tigerbx.pipelines.vbm is the vbm() function
    for name in ('run_case', 'apply_warp', 'modulate', 'smooth'):
        assert callable(getattr(vbm_mod, name)), name
    assert callable(tigerbx.reg)


def test_reg_modulate_with_zero_warp_is_identity():
    reg_mod = importlib.import_module('tigerbx.modules.reg')
    vol = np.random.default_rng(0).random((6, 7, 8)).astype(np.float32)
    img = nib.Nifti1Image(vol, np.eye(4))
    ctx = reg_mod.RegistrationContext(ftemplate='', input_nib=img, template_nib=img, affine_nib=None,
                                      affine_matrix=None, init_flow=None, warp=np.zeros((3,) + vol.shape),
                                      pad_width=None, displacement_dict={})
    assert np.allclose(reg_mod.modulate(img, ctx).get_fdata(), vol)


def test_registration_state_keeps_scaled_int16_intensities():
    # int16 + large scl_slope (some Philips exports): scaled values exceed int16 and must not wrap around
    reg_mod = importlib.import_module('tigerbx.modules.reg')
    data = np.full((20, 20, 20), 700, np.int16)
    img = nib.Nifti1Image(data, np.eye(4))
    img.header.set_data_dtype(np.int16)
    img.header.set_slope_inter(3000.0, 0.0)
    img = nib.Nifti1Image.from_bytes(img.to_bytes())          # round trip so dataobj carries the scaling
    mask = nib.Nifti1Image(np.ones((20, 20, 20), np.uint8), np.eye(4))
    state = reg_mod._prepare_registration_state(img, mask, {'template': None})
    assert np.isclose(state['tbx_data'].max(), 700 * 3000.0)
    assert state['tbx_data'].min() >= 0


def test_reg_plan_vfa_step_and_defaults():
    import inspect
    reg_mod = importlib.import_module('tigerbx.modules.reg')
    vbm_mod = importlib.import_module('tigerbx.pipelines.vbm')
    assert reg_mod._parse_reg_plan('AD') == ['A', 'D']
    assert reg_mod._parse_reg_plan('RAVD') == ['R', 'A', 'V', 'D']
    with pytest.raises(ValueError):
        reg_mod._parse_reg_plan('D')                     # D needs a prior affine step
    run_d = reg_mod._build_run_dict(SimpleNamespace(plan='AD'))
    assert run_d['vfa'] and not run_d['registration'] and run_d['affine_type'] == 'numreg'
    assert reg_mod._build_omodel({'vfa': 'x.onnx'})['vfa'] == 'x.onnx'
    assert inspect.signature(reg_mod.reg).parameters['affine_type'].default == 'numreg'
    params = inspect.signature(vbm_mod.vbm).parameters
    assert params['reg_plan'].default == 'AD' and params['affine_type'].default == 'numreg'


def test_reg_plan_fusemorph_removed():
    reg_mod = importlib.import_module('tigerbx.modules.reg')
    with pytest.raises(ValueError, match='FuseMorph'):
        reg_mod._parse_reg_plan('AF')
    assert 'fusemorph' not in reg_mod._build_run_dict(SimpleNamespace(plan='AD'))


def test_reg_plan_syn_steps_removed():
    reg_mod = importlib.import_module('tigerbx.modules.reg')
    for plan in ('AN', 'AC'):
        with pytest.raises(ValueError, match='SyN'):
            reg_mod._parse_reg_plan(plan)


def test_vbm_batch_continues_after_a_failure_and_writes_summary(tmp_path, monkeypatch):
    import csv
    vbm_mod = importlib.import_module('tigerbx.pipelines.vbm')
    files = []
    for name in ('a', 'bad', 'c'):
        f = tmp_path / f'{name}.nii.gz'
        nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.float32), np.eye(4)), str(f))
        files.append(str(f))

    def fake_run_one(args, f, *a):
        if 'bad' in f:
            raise RuntimeError('boom')
        return {}, {'Smoothed_GM': f + '_SmoothedGM'}, dict(file=f, status='ok', TIV_ml=1.0)
    monkeypatch.setattr(vbm_mod, '_run_one', fake_run_one)

    out = tmp_path / 'out'
    res = vbm_mod.vbm(files, str(out))
    assert [r['Smoothed_GM'] for r in res] == [files[0] + '_SmoothedGM', files[2] + '_SmoothedGM']
    rows = list(csv.DictReader(open(out / vbm_mod.SUMMARY_NAME)))
    assert [r['status'] for r in rows] == ['ok', 'failed', 'ok']
    assert [r['qc_flags'] for r in rows] == ['', 'fail', '']
    assert 'boom' in rows[1]['error']
    assert sorted(p.name for p in out.glob('*_vbmfail.log')) == ['bad_vbmfail.log']
    assert 'boom' in (out / 'bad_vbmfail.log').read_text()

    with pytest.raises(RuntimeError, match='boom'):          # a single input still raises (and logs)
        vbm_mod.vbm(files[1], str(out))
    assert (out / 'bad_vbmfail.log').exists()

    monkeypatch.setattr(vbm_mod, '_run_one', lambda args, f, *a: ({}, {}, dict(file=f, status='ok')))
    vbm_mod.vbm(files, str(out))                             # a later success removes the stale fail log
    assert not list(out.glob('*_vbmfail.log'))


def test_reg_smooth_fwhm_is_in_mm():
    reg_mod = importlib.import_module('tigerbx.modules.reg')
    vol = np.zeros((41, 41, 41), np.float32)
    vol[20, 20, 20] = 1
    for zoom in (1.0, 2.0):
        img = nib.Nifti1Image(vol, np.diag([zoom, zoom, zoom, 1]))
        out = reg_mod.smooth(img, fwhm=8.0).get_fdata()[:, 20, 20]
        half = out.max() / 2
        width_vox = np.interp(half, out[20:][::-1], np.arange(20, 41)[::-1]) - 20
        assert abs(2 * width_vox * zoom - 8.0) < 0.5, zoom


def test_tseg_unlabeled_intracranial_fills_mask_only():
    from tigerbx import lib_bx
    tseg = np.zeros((6, 6, 6), np.uint8)
    tseg[2:4, 2:4, 2:4] = 17
    mask = np.zeros((6, 6, 6), np.uint8)
    mask[1:5, 1:5, 1:5] = 1
    out = np.asarray(bx._fill_unlabeled_intracranial(nib.Nifti1Image(tseg, np.eye(4)),
                                                     nib.Nifti1Image(mask, np.eye(4))).dataobj)
    assert out.dtype == np.uint8
    assert (out == 17).sum() == 8                                  # structures unchanged
    assert (out == lib_bx.TSEG_UNLABELED_INTRACRANIAL).sum() == 64 - 8
    assert (out > 0).sum() == mask.sum() and out[mask == 0].max() == 0


@pytest.mark.parametrize('flags', ['q', 'S', 'W', 'bmaq'])
def test_bx_removed_flags_raise(flags):
    with pytest.raises(ValueError, match='have been removed'):
        bx.run(flags, 'x.nii.gz')


def test_pipelines_are_top_level_functions():
    import tigerbx
    for name in ('bx', 'seg', 'affine', 'norm', 'vbm', 'template'):
        assert callable(getattr(tigerbx, name))
    assert not hasattr(tigerbx, 'pipeline')
    assert callable(tigerbx.run) and callable(tigerbx.reg) and callable(tigerbx.hlc)     # modules: tigerbx.modules


def test_template_pipelines_check_intensity():
    from tigerbx.pipelines import _common
    with pytest.raises(ValueError):
        _common.check_intensity('zscore', None)
    with pytest.raises(ValueError):
        _common.check_intensity('gnorm', 'my_template.nii.gz')
    _common.check_intensity('raw', 'my_template.nii.gz')
    img = np.array([[[0, 2], [4, 10]]], np.float32)
    assert np.array_equal(_common.scale(img, None, 'raw'), img)
    assert _common.scale(img, None, 'minmax').max() == 1


def test_brain_mask_options(tmp_path):
    from tigerbx.pipelines import _common
    data = np.zeros((6, 6, 6), np.float32)
    data[1:5, 1:5, 1:5] = 3
    data[2, 2, 2] = np.nan
    f = str(tmp_path / 'brain.nii.gz')
    _write_nifti(f, data)
    _, b, m, tbx = _common.brain(f, 'none')                  # already skull-stripped; NaN -> 0
    assert tbx is None and np.asarray(m.dataobj).sum() == 63 and np.isfinite(b.get_fdata()).all()
    mask = np.zeros((6, 6, 6), np.uint8)
    mask[2:4, 2:4, 2:4] = 1
    fm = str(tmp_path / 'mask.nii.gz')
    _write_nifti(fm, mask)
    _, b, m, _ = _common.brain(f, fm)
    assert np.asarray(m.dataobj).sum() == 8 and b.get_fdata().sum() == 3 * 7


def test_template_vote_stat_and_warp_inverse():
    from scipy import ndimage
    from tigerbx.core.templatereg import invert_warp, identity_grid
    from tigerbx.pipelines.template import _stat, _vote
    stack = np.zeros((5, 4, 4, 4), np.uint8)
    stack[:3, 0] = 17
    stack[3:, 0] = 53
    stack[:2, 1] = 0                         # two subjects miss the region: the other labels still win
    stack[2:, 1] = 3
    out = _vote(stack, np.ones((4, 4, 4), bool))
    assert (out[0] == 17).all() and (out[1] == 3).all() and (out[2:] == 0).all()
    x = np.arange(5, dtype=np.float32)[:, None, None, None] * np.ones((5, 4, 4, 4), np.float32)
    assert (_stat(x, 'median') == 2).all() and np.allclose(_stat(x, 'mean'), 2)
    rng = np.random.default_rng(0)
    w = np.stack([ndimage.gaussian_filter(rng.normal(0, 1, (24, 26, 22)), 4) * 15 for _ in range(3)]).astype(np.float32)
    v = invert_warp(w)
    c = identity_grid(w.shape[1:]) + v
    back = np.stack([ndimage.map_coordinates(w[a], c, order=1, mode='nearest') for a in range(3)])
    assert np.abs(v + back)[:, 3:-3, 3:-3, 3:-3].max() < 1e-3


def test_template_rejects_bad_arguments(tmp_path):
    import tigerbx
    with pytest.raises(ValueError):
        tigerbx.template(['a.nii.gz', 'b.nii.gz'], str(tmp_path), stat='max')
    with pytest.raises(ValueError):
        tigerbx.template(['a.nii.gz', 'b.nii.gz'], str(tmp_path), mask='mask.nii.gz')
