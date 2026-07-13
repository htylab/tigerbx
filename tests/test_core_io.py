import numpy as np

from tigerbx.core.io import _pick_output_dtype


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
