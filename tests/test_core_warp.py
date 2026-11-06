import os

import numpy as np
import pytest

from tigerbx.core.warp import dense_transform


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


def test_epi_template_on_the_template_grid():
    from tigerbx.core import templatereg
    t = templatereg.template(None)
    for kind in ('se', 'bold'):
        e = templatereg.epi_template(kind)
        assert e.shape == t.shape and np.allclose(e.affine, t.affine, atol=1e-3)
        x = e.get_fdata()
        assert x.min() >= 0 and (x > 0).mean() > 0.1
    with pytest.raises(ValueError):
        templatereg.epi_template('flair')


def test_model_box_and_warp_from_box():
    import nibabel as nib
    from tigerbx.core import templatereg
    # the bundled template is on the model grid: no box (outputs as before)
    assert templatereg.model_box(templatereg.template(None)) is None
    # a 1.5 mm template of another size: a 1 mm box of the model size on its content
    data = np.zeros((90, 110, 100), np.float32)
    data[20:70, 25:85, 20:80] = 1.0
    aff = np.diag([1.5, 1.5, 1.5, 1.0])
    aff[:3, 3] = (-60, -80, -70)
    tpl = nib.Nifti1Image(data, aff)
    box = templatereg.model_box(tpl)
    assert box.shape == templatereg.MODEL_SHAPE and np.allclose(box.affine[:3, :3], np.eye(3))
    centre = box.affine @ np.r_[(np.array(box.shape) - 1) / 2, 1]
    assert np.allclose(centre[:3], (aff @ np.r_[np.argwhere(data > 0).mean(0), 1])[:3])
    # 3 mm along the first axis on the box = 2 template voxels; 0 outside the box
    warp = np.zeros((3,) + box.shape, np.float32)
    warp[0] = 3.0
    w = templatereg.warp_from_box(warp, box, tpl)
    assert w.shape == (3,) + data.shape
    assert np.allclose(w[0][data > 0], 2.0, atol=1e-5) and np.allclose(w[1:], 0)
    big = nib.Nifti1Image(np.ones((200, 120, 120), np.float32), np.eye(4))      # wider than the box along x, no background
    wb = templatereg.warp_from_box(np.ones((3,) + templatereg.MODEL_SHAPE, np.float32),
                                   templatereg.model_box(big), big)
    assert wb[0, 0].max() == 0 and wb[0, 100].min() == 1.0
    # an image moved by the carried-back displacement: out(p) = data(p + w(p)), the same move as on the box
    moved = templatereg.warp_on_grid(data, w)
    assert np.allclose(moved[22:66, 30:80, 25:75], data[24:68, 30:80, 25:75])
