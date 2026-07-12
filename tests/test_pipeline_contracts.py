import os
import importlib
import shutil
import tempfile
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from tigerbx import bx, gdmi

nerve = importlib.import_module('tigerbx.nerve')


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


def test_gdm_and_nerve_accept_empty_inputs():
    gdm_args = SimpleNamespace(input=[], output=None, b0_index='0', dmap=False,
                               no_resample=False, gpu=False, verbose=0)
    nerve_args = SimpleNamespace(input=[], output=None, model=None, method='NERVE',
                                 save_patch=False, encode=True, decode=False,
                                 evaluate=False, gpu=False, sigma=False, verbose=0)

    assert gdmi.run_args(gdm_args) == []
    assert nerve.run_args(nerve_args) == []


def test_nerve_returns_structured_results(work_dir, monkeypatch):
    first = os.path.join(work_dir, 'first.nii.gz')
    second = os.path.join(work_dir, 'second.nii.gz')
    output_dir = os.path.join(work_dir, 'out')
    monkeypatch.setattr(nerve.lib_tool, 'get_model', lambda name: name)
    monkeypatch.setattr(
        nerve, 'encode_nii',
        lambda raw_path, **kwargs: (raw_path + '_nerve.npz', raw_path + '_patch.nii.gz'))

    base_args = dict(output=output_dir, model=None, method='NERVE', save_patch=True,
                     encode=True, decode=False, evaluate=False, gpu=False,
                     sigma=False, verbose=0)
    single = nerve.run_args(SimpleNamespace(input=[first], **base_args))
    multiple = nerve.run_args(SimpleNamespace(input=[first, second], **base_args))

    assert set(single) == {'latent', 'patch'}
    assert len(multiple) == 2
    assert all(set(item) == {'latent', 'patch'} for item in multiple)


def test_bx_rejects_empty_bet_mask(monkeypatch):
    mask = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4))
    prob = np.ones((2, 2, 2, 2), dtype=np.float32) * 0.5
    monkeypatch.setattr(bx.lib_tool, 'get_model', lambda model: model)
    monkeypatch.setattr(bx, '_infer_mask', lambda *args, **kwargs: (mask, mask, prob))

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
