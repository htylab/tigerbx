import os

import nibabel as nib
import numpy as np
import pytest

from tigerbx import gdmi


def _img(shape=(8, 9, 6), dtype=np.int16, affine=None):
    data = (np.arange(np.prod(shape)).reshape(shape) % 50 + 10).astype(dtype)
    return nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0]) if affine is None else affine)


@pytest.fixture
def no_network(monkeypatch):
    """gdm without the model: a constant VDM of 0.5 voxels."""
    monkeypatch.setattr(gdmi.lib_tool, 'get_model', lambda name: name)
    monkeypatch.setattr(gdmi, 'create_session', lambda model, gpu: None)
    monkeypatch.setattr(gdmi, 'predict_single_output', lambda session, x: np.full(x.shape, 0.5, np.float32))


def test_b0_volume_index_and_bval(tmp_path):
    img4 = nib.Nifti1Image(np.zeros((2, 2, 2, 3), np.float32), np.eye(4))
    assert gdmi.b0_volume(_img(), 2) == 0                 # 3D: the image itself
    assert gdmi.b0_volume(img4, 1) == 1
    assert gdmi.b0_volume(img4, '2') == 2
    assert gdmi.b0_volume(img4, -1) == 2
    bval = tmp_path / 'dwi.bval'
    bval.write_text('1000\t5  1000\n')                    # tabs, double spaces, b = 5 counts as the b0
    assert gdmi.b0_volume(img4, str(bval)) == 1
    assert gdmi.b0_volume(img4, str(tmp_path / 'dwi')) == 1
    with pytest.raises(ValueError, match='out of range'):
        gdmi.b0_volume(img4, 3)
    bval.write_text('0 1000\n')
    with pytest.raises(ValueError, match='2 b-values for 3 volumes'):
        gdmi.b0_volume(img4, str(bval))


def test_zero_vdm_leaves_the_image_unchanged(monkeypatch):
    monkeypatch.setattr(gdmi, 'predict_single_output', lambda session, x: np.zeros(x.shape, np.float32))
    img = _img(dtype=np.float32)
    out, vdm = gdmi.correct(None, img, resample=False)
    assert np.allclose(out, img.get_fdata()) and not vdm.any()


def test_outputs_types_and_4d(tmp_path, no_network):
    f3 = str(tmp_path / 'b0.nii.gz')
    nib.save(_img(), f3)
    r = gdmi.gdm(f3, str(tmp_path / 'out'), dmap=True, no_resample=True)
    assert r['gdmi'].get_data_dtype() == np.int16           # the input's type, rounded
    assert r['gdm'].get_data_dtype() == np.float32          # the VDM keeps its sub-voxel values
    assert np.allclose(r['gdm'].get_fdata()[2:-2, 2:-2, :], 0.5 * 2.0 / 2.0, atol=1e-6)
    assert os.path.exists(tmp_path / 'out' / 'b0_gdmi.nii.gz')

    d4 = np.stack([_img(dtype=np.float32).get_fdata()] * 3, -1).astype(np.float32)
    f4 = str(tmp_path / 'dwi.nii.gz')
    nib.save(nib.Nifti1Image(d4, np.diag([2.0, 2.0, 2.0, 1.0])), f4)
    r = gdmi.gdm(f4, str(tmp_path / 'out'), b0_index=1, no_resample=True)
    assert r['gdmi'].shape == d4.shape and 'gdm' not in r


def test_batch_skips_a_failing_input(tmp_path, no_network):
    good = str(tmp_path / 'a.nii.gz')
    nib.save(_img(), good)
    bad = str(tmp_path / 'b.nii.gz')
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 2), np.float32), np.eye(4)), bad)
    with pytest.warns(RuntimeWarning, match=r'tigerbx\.gdm: 1 of 2 inputs failed'):
        res = gdmi.gdm([good, bad], str(tmp_path / 'out'), b0_index=5, no_resample=True)
    assert [os.path.basename(r['gdmi']) for r in res] == ['a_gdmi.nii.gz']


def test_gpu_keyword_and_deprecated_GPU(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(gdmi.lib_tool, 'get_model', lambda name: name)
    monkeypatch.setattr(gdmi, 'create_session', lambda model, gpu: seen.append(gpu))
    monkeypatch.setattr(gdmi, 'correct', lambda *a, **k: (np.zeros((8, 9, 6)), np.zeros((8, 9, 6))))
    f = str(tmp_path / 'b0.nii.gz')
    nib.save(_img(), f)
    gdmi.gdm(f, str(tmp_path), gpu=True)
    with pytest.warns(DeprecationWarning, match='gpu='):
        gdmi.gdm(f, str(tmp_path), GPU=True)
    assert seen == [True, True]


def test_phase_axis_warnings(no_network):
    img = _img(dtype=np.float32)
    img.header.set_dim_info(phase=0)
    with pytest.warns(RuntimeWarning, match='phase encoding along axis 0'):
        gdmi.correct(None, img, resample=False)
    coronal = np.array([[2.0, 0, 0, 0], [0, 0, 2.0, 0], [0, 2.0, 0, 0], [0, 0, 0, 1]])   # second axis runs S
    with pytest.warns(RuntimeWarning, match='second image axis runs S'):
        gdmi.correct(None, _img(dtype=np.float32, affine=coronal), resample=False)
