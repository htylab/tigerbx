import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from tigerbx import lib_bx
from tigerbx.core import onnx as core_onnx
from tigerbx import validate


def test_get_mode_parses_release_names():
    assert lib_bx.get_mode('/m/mprage_tseg_v007_16ksynth.onnx') == ('tseg', 'v007', '16ksynth.onnx')
    assert lib_bx.get_mode('mprage_dgm12_v002_mix6.onnx')[0] == 'dgm12'


@pytest.mark.parametrize('name', ['mprage_tseg_v009.onnx', 'mprage_seg_v009_test.onnx', 'new_tseg.onnx'])
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


def test_create_session_warns_once_on_gpu_fallback(monkeypatch):
    # onnxruntime-gpu built for another CUDA major only logs the failed provider load and runs on CPU.
    class FakeSession:
        def __init__(self, model_ff, providers, sess_options):
            self.model_ff = model_ff

        def get_providers(self):
            return ['CPUExecutionProvider']

    class FakeOptions:
        pass

    fake_ort = SimpleNamespace(__version__='9.9.9', SessionOptions=FakeOptions, InferenceSession=FakeSession,
                               get_device=lambda: 'GPU', set_default_logger_severity=lambda level: None)
    monkeypatch.setitem(sys.modules, 'onnxruntime', fake_ort)
    monkeypatch.setattr(core_onnx, '_GPU_FALLBACK_WARNED', False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        core_onnx.create_session('m.onnx', GPU=True)
        core_onnx.create_session('m.onnx', GPU=True)
        core_onnx.create_session('m.onnx', GPU=False)
    msgs = [str(x.message) for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(msgs) == 1 and 'running on CPU' in msgs[0]


def test_seg18_protocol_has_30_scored_structures():
    scored = [k for k in validate.SEG18_STRUCTURES if '*' not in k]
    assert len(scored) == 30
    assert any(ds['id'] == 'seg18' and ds['funcs'][0][0] == 'tseg_18' for ds in validate.DATASET_REGISTRY)


def test_seg_pipeline_merge_dgm_takes_dgm12_and_fills_holes():
    from tigerbx.pipelines.seg import merge_dgm
    tseg = np.zeros((5, 5, 1), int)
    tseg[:, :2] = 2                                    # cerebral WM
    tseg[:, 2:] = 10                                   # TSEG left thalamus
    dgm = np.zeros((5, 5, 1), int)
    dgm[:, 4] = 3                                      # DGM12 label 3 = left caudate (11)
    out = merge_dgm(tseg, dgm)
    assert (out[:, 4] == 11).all()                     # DGM12 wins
    assert (out[:, 2:4] == 2).all()                    # TSEG thalamus that DGM12 drops: nearest non-deep-GM label
    assert (out[:, :2] == 2).all()
