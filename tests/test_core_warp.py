import os

import numpy as np
import pytest

from tigerbx.core.warp import affine_transform, dense_transform, init_shift


def _init_flow(shape, shift_xyz):
    axes = [np.linspace(-1, 1, n, dtype=np.float32) for n in shape]
    z, y, x = np.meshgrid(*axes, indexing='ij')
    return (np.stack([x, y, z], -1) + np.asarray(shift_xyz, np.float32))[None]


@pytest.mark.parametrize('interp', ['nearest', 'linear'])
def test_dense_identity_and_integer_shift(interp):
    vol = np.random.default_rng(0).random((8, 9, 10), np.float32)
    assert np.allclose(dense_transform(vol, np.zeros((3,) + vol.shape), interp), vol)
    warp = np.zeros((3,) + vol.shape, np.float32)
    warp[2] = 1                                            # sample one voxel further along the last axis
    out = dense_transform(vol, warp, interp)
    assert np.allclose(out[..., :-1], vol[..., 1:])
    # outside the volume: nearest clamps to the border, linear is zero
    assert np.allclose(out[..., -1], vol[..., -1] if interp == 'nearest' else 0)


def test_dense_linear_midpoint():
    vol = np.zeros((4, 4, 4), np.float32)
    vol[:, :, 1] = 2
    warp = np.zeros((3, 4, 4, 4), np.float32)
    warp[2] = 0.5
    assert np.allclose(dense_transform(vol, warp, 'linear')[:, :, 0], 1)


def test_init_shift_roundtrip():
    shift = np.array([-18, 12, -8], np.float32) / 127.5
    assert np.allclose(init_shift(_init_flow((16, 16, 16), shift)), shift, atol=1e-6)
    assert np.array_equal(init_shift(shift), shift)


@pytest.mark.parametrize('interp', ['nearest', 'linear'])
def test_affine_identity_and_shift_only(interp):
    n = 16
    vol = np.random.default_rng(1).random((n,) * 3, np.float32)
    theta = np.eye(3, 4, dtype=np.float32)
    assert np.allclose(affine_transform(vol, _init_flow(vol.shape, 0), theta, interp), vol, atol=1e-5)
    shift = np.array([2, 0, 0], np.float32) * 2 / (n - 1)  # +2 voxels along x (last axis)
    out = affine_transform(vol, shift, theta, interp)
    assert np.allclose(out[..., :-2], vol[..., 2:], atol=1e-5)
    assert np.allclose(out, affine_transform(vol, _init_flow(vol.shape, shift), theta, interp), atol=1e-5)


ONNX = os.environ.get('TIGERBX_COMPARE_ONNX') == '1'


@pytest.mark.skipif(not ONNX, reason='set TIGERBX_COMPARE_ONNX=1 to download and compare the old ONNX models')
@pytest.mark.parametrize('interp,suffix', [('nearest', 'near'), ('linear', 'bili')])
def test_matches_former_onnx_models(interp, suffix):
    import onnxruntime as ort
    from tigerbx import lib_tool

    rng = np.random.default_rng(2)
    run = lambda name, feeds: ort.InferenceSession(lib_tool.get_model(name), providers=['CPUExecutionProvider']).run(None, feeds)[0][0, 0]
    pick = lambda shape: rng.integers(0, 44, shape).astype(np.float32) if interp == 'nearest' else rng.random(shape, np.float32)

    def check(ref, out):
        if interp == 'nearest':                            # only ties at .5 may round differently
            assert np.mean(ref != out) < 1e-4
        else:
            assert np.abs(ref - out).max() < 1e-3

    shape = (160, 224, 192)
    seg, warp = pick(shape), (rng.standard_normal((3,) + shape) * 4).astype(np.float32)
    ref = run(f'mprage_transform_v002_{suffix}.onnx', {'seg': seg[None, None], 'warp': warp[None]})
    check(ref, dense_transform(seg, warp, interp))

    vol, flow = pick((256,) * 3), _init_flow((256,) * 3, np.array([-18, 12, -8], np.float32) / 127.5)
    theta = (np.eye(3, 4) + rng.standard_normal((3, 4)) * 0.05).astype(np.float32)[None]
    ref = run(f'mprage_affinetransform_v002_{suffix}.onnx', {'moving_seg': vol[None, None], 'init_flow': flow, 'affine_matrix': theta})
    check(ref, affine_transform(vol, flow, theta, interp))


def test_resample_same_grid_shortcut_matches_interpolation():
    import nibabel as nib
    from scipy.ndimage import affine_transform
    from tigerbx.core.resample import resample_img, resample_to_img

    rng = np.random.default_rng(0)
    aff = np.array([[0, 0, -1.2, 90], [1.1, 0, 0, -100], [0, 0.9, 0, -80], [0, 0, 0, 1]], float)
    for data in (rng.random((9, 8, 7)).astype(np.float32), (rng.random((9, 8, 7)) * 200).astype(np.uint8),
                 (rng.random((9, 8, 7)) * 3000).astype(np.int16)):
        img = nib.Nifti1Image(data, aff)
        for interp, order in (('nearest', 0), ('linear', 1), ('continuous', 3)):
            out = resample_to_img(img, img, interpolation=interp)
            ref_dtype = np.float64 if (order == 3 and data.dtype.kind == 'i') else data.dtype
            ref = affine_transform(data, np.ones(3), output_shape=data.shape, order=order,
                                   output=ref_dtype, mode='constant')
            np.testing.assert_allclose(np.asarray(out.dataobj), ref, rtol=0, atol=1e-6)
            assert np.asarray(out.dataobj).dtype == ref_dtype
    # a different shape still interpolates
    img = nib.Nifti1Image(rng.random((9, 8, 7)).astype(np.float32), aff)
    assert resample_img(img, target_affine=aff, target_shape=(5, 5, 5)).shape == (5, 5, 5)
