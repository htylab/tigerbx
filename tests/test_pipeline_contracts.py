import os
import importlib
import shutil
import tempfile
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from tigerbx import bx, gdmi



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


def test_bx_rejects_empty_bet_mask(monkeypatch):
    mask = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4))
    monkeypatch.setattr(bx.lib_tool, 'get_model', lambda model: model)
    monkeypatch.setattr(bx, '_infer_mask', lambda *args, **kwargs: (mask, mask))

    try:
        bx.produce_betmask('model.onnx', 'empty.nii.gz')
    except RuntimeError as exc:
        assert 'empty brain mask' in str(exc)
    else:
        raise AssertionError('Expected an empty BET mask to fail')


def test_cgw_rejects_zero_intensity_input(work_dir):
    input_file = os.path.join(work_dir, 'zero.nii.gz')
    zero_nib = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
    nib.save(zero_nib, input_file)

    try:
        bx._run_cgw('cgw', None, 'model.onnx', input_file,
                    {'tbet111_crop': zero_nib}, False, False, save_outputs=False)
    except ValueError as exc:
        assert 'finite positive image maximum' in str(exc)
    else:
        raise AssertionError('Expected a zero-intensity CGW input to fail')


def test_vbm_pipeline_resolves_reg_functions():
    # tigerbx.reg is the reg() function after `import tigerbx`; vbm must not use it as the module
    import tigerbx
    vbm_mod = importlib.import_module('tigerbx.pipelines.vbm')   # tigerbx.pipelines.vbm is the vbm() function too
    for name in ('run_case', 'apply_warp', 'modulate', 'smooth'):
        assert callable(getattr(vbm_mod, name)), name
    assert callable(tigerbx.reg)


def test_reg_modulate_with_zero_warp_is_identity():
    reg_mod = importlib.import_module('tigerbx.reg')
    vol = np.random.default_rng(0).random((6, 7, 8)).astype(np.float32)
    img = nib.Nifti1Image(vol, np.eye(4))
    ctx = reg_mod.RegistrationContext(ftemplate='', input_nib=img, template_nib=img, affine_nib=None,
                                      affine_matrix=None, init_flow=None, warp=np.zeros((3,) + vol.shape),
                                      pad_width=None, displacement_dict={})
    assert np.allclose(reg_mod.modulate(img, ctx).get_fdata(), vol)


def test_registration_state_keeps_scaled_int16_intensities():
    # int16 + large scl_slope (some Philips exports): scaled values exceed int16 and must not wrap around
    reg_mod = importlib.import_module('tigerbx.reg')
    data = np.full((20, 20, 20), 700, np.int16)
    img = nib.Nifti1Image(data, np.eye(4))
    img.header.set_data_dtype(np.int16)
    img.header.set_slope_inter(3000.0, 0.0)
    img = nib.Nifti1Image.from_bytes(img.to_bytes())          # round trip so dataobj carries the scaling
    mask = nib.Nifti1Image(np.ones((20, 20, 20), np.uint8), np.eye(4))
    state = reg_mod._prepare_registration_state(img, mask, {'template': None})
    assert np.isclose(state['bet_data'].max(), 700 * 3000.0)
    assert state['bet_data'].min() >= 0


def test_reg_plan_vfa_step_and_defaults():
    import inspect
    reg_mod = importlib.import_module('tigerbx.reg')
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
    reg_mod = importlib.import_module('tigerbx.reg')
    with pytest.raises(ValueError, match='FuseMorph'):
        reg_mod._parse_reg_plan('AF')
    assert 'fusemorph' not in reg_mod._build_run_dict(SimpleNamespace(plan='AD'))


def test_reg_plan_syn_steps_removed():
    reg_mod = importlib.import_module('tigerbx.reg')
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
    reg_mod = importlib.import_module('tigerbx.reg')
    vol = np.zeros((41, 41, 41), np.float32)
    vol[20, 20, 20] = 1
    for zoom in (1.0, 2.0):
        img = nib.Nifti1Image(vol, np.diag([zoom, zoom, zoom, 1]))
        out = reg_mod.smooth(img, fwhm=8.0).get_fdata()[:, 20, 20]
        half = out.max() / 2
        width_vox = np.interp(half, out[20:][::-1], np.arange(20, 41)[::-1]) - 20
        assert abs(2 * width_vox * zoom - 8.0) < 0.5, zoom


def test_aseg_unlabeled_intracranial_fills_mask_only():
    from tigerbx import lib_bx
    aseg = np.zeros((6, 6, 6), np.uint8)
    aseg[2:4, 2:4, 2:4] = 17
    mask = np.zeros((6, 6, 6), np.uint8)
    mask[1:5, 1:5, 1:5] = 1
    out = np.asarray(bx._fill_unlabeled_intracranial(nib.Nifti1Image(aseg, np.eye(4)),
                                                     nib.Nifti1Image(mask, np.eye(4))).dataobj)
    assert out.dtype == np.uint8
    assert (out == 17).sum() == 8                                  # structures unchanged
    assert (out == lib_bx.ASEG_UNLABELED_INTRACRANIAL).sum() == 64 - 8
    assert (out > 0).sum() == mask.sum() and out[mask == 0].max() == 0


@pytest.mark.parametrize('flags', ['q', 'S', 'W', 'bmaq'])
def test_bx_removed_flags_raise(flags):
    with pytest.raises(ValueError, match='have been removed'):
        bx.run(flags, 'x.nii.gz')
