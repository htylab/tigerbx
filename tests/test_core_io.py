import os
import warnings

import numpy as np
import pytest

from tigerbx.core.io import _pick_output_dtype, resolve_nifti_inputs


def test_pick_output_dtype_rejects_values_below_unsigned_minimum():
    data = np.array([-1, 1], dtype=np.int16)

    assert _pick_output_dtype(data, np.dtype('uint8')) is None


def test_pick_output_dtype_rejects_values_outside_integer_bounds():
    assert _pick_output_dtype(np.array([-129, 0]), np.dtype('int8')) is None
    assert _pick_output_dtype(np.array([0, 256]), np.dtype('uint8')) is None


def test_pick_output_dtype_keeps_compatible_reference_dtype():
    dtype = np.dtype('uint8')

    assert _pick_output_dtype(np.array([0, 255]), dtype) == dtype


def test_pick_output_dtype_rejects_non_finite_values():
    assert _pick_output_dtype(np.array([0.0, np.nan]), np.dtype('float32')) is None


def _touch(d, *names):
    for n in names:
        open(os.path.join(d, n), 'w').close()


def test_resolve_dir_input_is_sorted_and_skips_previous_outputs(tmp_path):
    _touch(tmp_path, 'b.nii.gz', 'a.nii.gz', 'a_tbet.nii.gz', 'a_aseg.nii.gz', 'a_aseg_aseg.nii.gz', 'b_cgw_pve0.nii')
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
    files = ['z.nii.gz', 'z_tbet.nii.gz']
    assert resolve_nifti_inputs(files) == files
