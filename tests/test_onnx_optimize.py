"""tigerbx.dev.onnx_optimize: rewritten GroupNorm / InstanceNorm and the labels output give the same result."""
import numpy as np
import pytest

onnx = pytest.importorskip('onnx')
ort = pytest.importorskip('onnxruntime')
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from tigerbx.dev import onnx_optimize as oo  # noqa: E402

RNG = np.random.default_rng(0)


def _init(name, arr):
    return numpy_helper.from_array(np.asarray(arr, np.float32), name)


def _model(out_ch, norm='group', opset=12):
    """Conv(1->8) -> norm -> Relu -> Conv(8->out_ch); norm is a PyTorch-style exported GroupNorm(2, 8) or a
    plain InstanceNorm with affine."""
    nodes = [helper.make_node('Conv', ['x', 'w1', 'b1'], ['c1'], kernel_shape=[1, 1, 1])]
    inits = [_init('w1', RNG.normal(size=(8, 1, 1, 1, 1))), _init('b1', RNG.normal(size=8))]
    if norm == 'group':
        inits += [numpy_helper.from_array(np.array([0, 2, -1], np.int64), 'gshape'),
                  _init('s_g', np.ones(2)), _init('b_g', np.zeros(2)),
                  _init('gw', RNG.normal(size=(8, 1, 1, 1))), _init('gb', RNG.normal(size=(8, 1, 1, 1)))]
        nodes += [helper.make_node('Reshape', ['c1', 'gshape'], ['r0']),
                  helper.make_node('InstanceNormalization', ['r0', 's_g', 'b_g'], ['in'], epsilon=1e-5),
                  helper.make_node('Shape', ['c1'], ['shp']),
                  helper.make_node('Reshape', ['in', 'shp'], ['r1']),
                  helper.make_node('Mul', ['r1', 'gw'], ['mul']),
                  helper.make_node('Add', ['mul', 'gb'], ['n'])]
    else:
        inits += [_init('s', RNG.normal(size=8)), _init('b', RNG.normal(size=8))]
        nodes += [helper.make_node('InstanceNormalization', ['c1', 's', 'b'], ['n'], epsilon=1e-5)]
    nodes += [helper.make_node('Relu', ['n'], ['a']),
              helper.make_node('Conv', ['a', 'w2', 'b2'], ['y'], kernel_shape=[1, 1, 1])]
    inits += [_init('w2', RNG.normal(size=(out_ch, 8, 1, 1, 1))), _init('b2', RNG.normal(size=out_ch))]
    g = helper.make_graph(nodes, 'm', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 'X', 'Y', 'Z'])],
                          [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, out_ch, 'X', 'Y', 'Z'])], inits)
    # pin the IR version: a newer onnx package defaults to an IR that an older onnxruntime cannot load
    return helper.make_model(g, opset_imports=[helper.make_opsetid('', opset)], ir_version=8)


def _run(model, x, outputs=None):
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    return sess.run(outputs, {'x': x})


@pytest.mark.parametrize('norm,opset,kind', [('group', 12, 'groupnorm'), ('instance', 12, 'instancenorm'),
                                             ('instance', 18, 'instancenorm')])
def test_rewrite_matches_original(norm, opset, kind):
    model = _model(3, norm, opset)
    opt, stats = oo.optimize_model(model)
    assert stats[kind] == 1 and stats['skipped'] == 0
    assert not any(n.op_type == 'InstanceNormalization' for n in opt.graph.node)
    x = RNG.random((1, 1, 6, 5, 4), dtype=np.float32)
    np.testing.assert_allclose(_run(opt, x)[0], _run(model, x)[0], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('out_ch', [1, 3])
def test_labels_output(out_ch):
    model = oo.add_label_output(oo.optimize_model(_model(out_ch))[0])
    x = RNG.random((1, 1, 6, 5, 4), dtype=np.float32)
    logits, labels = _run(model, x)
    assert labels.dtype == np.uint8 and labels.shape == (1, 6, 5, 4)
    ref = (logits[:, 0] >= 0) if out_ch == 1 else np.argmax(logits, axis=1)
    np.testing.assert_array_equal(labels, ref)
    with pytest.raises(ValueError):
        oo.add_label_output(model)


def test_lib_bx_run_same_with_labels_output(tmp_path):
    import nibabel as nib
    from tigerbx import lib_bx

    img = nib.Nifti1Image(RNG.random((6, 5, 4)).astype(np.float32), np.eye(4))
    plain = tmp_path / 'mprage_dgm12_v999_test.onnx'
    labelled = tmp_path / 'mprage_dgm12_v999lab_test.onnx'
    model = _model(3)
    onnx.save(model, plain)
    onnx.save(oo.add_label_output(model), labelled)
    a = np.asarray(lib_bx.run(str(plain), img, GPU=False).dataobj)
    b = np.asarray(lib_bx.run(str(labelled), img, GPU=False).dataobj)
    np.testing.assert_array_equal(a, b)
