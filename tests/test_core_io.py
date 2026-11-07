import os
import warnings

import numpy as np
import pytest

from tigerbx.core.io import resolve_nifti_inputs


def _touch(d, *names):
    for n in names:
        open(os.path.join(d, n), 'w').close()


def test_resolve_dir_input_is_sorted_and_skips_previous_outputs(tmp_path):
    _touch(tmp_path, 'b.nii.gz', 'a.nii.gz', 'a_tbx.nii.gz', 'a_tseg.nii.gz', 'a_tseg_tseg.nii.gz', 'b_cgw_pve0.nii')
    with pytest.warns(UserWarning, match='tigerbx outputs'):
        out = resolve_nifti_inputs([str(tmp_path)])
    assert [os.path.basename(f) for f in out] == ['a.nii.gz', 'b.nii.gz']


def test_resolve_keeps_suffix_lookalikes_without_a_matching_input(tmp_path):
    _touch(tmp_path, 'patient_ct.nii.gz', 'other.nii.gz')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = resolve_nifti_inputs([str(tmp_path)])
    assert [os.path.basename(f) for f in out] == ['other.nii.gz', 'patient_ct.nii.gz']


def test_resolve_explicit_file_list_is_untouched():
    files = ['z.nii.gz', 'z_tbx.nii.gz']
    assert resolve_nifti_inputs(files) == files


def test_load_nifti_sets_non_finite_voxels_to_zero(tmp_path):
    import nibabel as nib
    from tigerbx.core.io import load_nifti
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    data[0, 0, :2] = np.nan
    data[1, 2, 3] = np.inf
    f = tmp_path / 'nan.nii.gz'
    nib.save(nib.Nifti1Image(data, np.eye(4)), f)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        img = load_nifti(f)
    assert any('3 non-finite voxels' in str(x.message) for x in w)
    out = img.get_fdata()
    assert np.isfinite(out).all() and out[0, 0, 0] == 0 and out[1, 2, 3] == 0 and out[1, 0, 0] == 12
    nib.save(nib.Nifti1Image(np.arange(24, dtype=np.int16).reshape(2, 3, 4), np.eye(4)), tmp_path / 'i.nii.gz')
    assert load_nifti(tmp_path / 'i.nii.gz').get_data_dtype() == np.int16


def test_qform_only_orientation_survives_import(tmp_path):
    import importlib
    import nibabel as nib
    import tigerbx
    for m in ('lib_tool', 'lib_reg', 'lib_bx', 'gdmi'):
        importlib.import_module(f'tigerbx.{m}')
    aff = np.eye(4)
    aff[:3, :3] = [[0, 0, -1], [1, 0, 0], [0, 1, 0]]
    img = nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), None)
    img.header.set_qform(aff, 1)
    img.header.set_sform(None, 0)
    nib.save(img, tmp_path / 'q.nii.gz')
    assert np.allclose(nib.load(tmp_path / 'q.nii.gz').affine, aff, atol=1e-6)
