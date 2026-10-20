import pytest

from tigerbx import lib_bx
from tigerbx import validate


def test_get_mode_parses_release_names():
    assert lib_bx.get_mode('/m/mprage_aseg43_v007_16ksynth.onnx') == ('aseg43', 'v007', '16ksynth.onnx')
    assert lib_bx.get_mode('mprage_synthseg_v003_r111.onnx')[0] == 'synthseg'


@pytest.mark.parametrize('name', ['mprage_aseg_v009.onnx', 'mprage_aseg_v009_test.onnx', 'new_aseg.onnx'])
def test_get_mode_rejects_bad_names_with_a_clear_error(name):
    with pytest.raises(ValueError, match='<prefix>_<mode>_<version>_<tag>'):
        lib_bx.get_mode(name)


def test_run_loop_keeps_going_after_a_failed_file():
    def compute(f, _tmp):
        if f == 'bad.nii.gz':
            raise FileNotFoundError('no label')
        return 1.0

    validate._FAILED.clear()
    f_list, res = validate._run_loop(['a.nii.gz', 'bad.nii.gz', 'c.nii.gz'], compute)
    assert f_list == ['a.nii.gz', 'c.nii.gz'] and res == [1.0, 1.0]
    assert [f for f, _ in validate._FAILED] == ['bad.nii.gz']
