import importlib
from types import SimpleNamespace

import nibabel as nib
import numpy as np

from tigerbx.core import numreg


def _phantom(shape=(64, 72, 60)):
    """Smooth ellipsoid 'brain' with an off-centre bright blob and a dark 'ventricle' (asymmetric on purpose)."""
    g = np.stack(np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing='ij'))
    brain = ((g[0] / 0.8) ** 2 + (g[1] / 0.9) ** 2 + (g[2] / 0.7) ** 2) < 1
    img = brain * 100.0
    img += 80 * np.exp(-((g[0] - 0.3) ** 2 + (g[1] + 0.2) ** 2 + (g[2] - 0.1) ** 2) / 0.02)
    img -= 60 * (((g[0] + 0.1) ** 2 / 0.05 + (g[1] - 0.2) ** 2 / 0.1 + g[2] ** 2 / 0.03) < 1) * brain
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = -np.array(shape) + 1.0
    return nib.Nifti1Image(img.astype(np.float32), aff)


def _rotation(deg, axis=2):
    t = np.deg2rad(deg)
    R = np.eye(4)
    i, j = [k for k in range(3) if k != axis]
    R[i, i], R[i, j], R[j, i], R[j, j] = np.cos(t), -np.sin(t), np.sin(t), np.cos(t)
    return R


def test_numreg_recovers_known_affine():
    fixed = _phantom()
    M_true = _rotation(8) @ np.diag([1.05, 0.95, 1.0, 1.0])
    M_true[:3, 3] = [6.0, -4.0, 3.0]
    # moving(y) = fixed(M_true^-1 y): resampling moving through M_true gives fixed back
    moving = nib.Nifti1Image(numreg.apply(fixed, fixed, np.linalg.inv(M_true)), fixed.affine)
    M = numreg.register(moving, fixed)
    pts = np.c_[np.random.default_rng(0).uniform(-40, 40, (200, 3)), np.ones(200)].T
    err = np.linalg.norm((M @ pts - M_true @ pts)[:3], axis=0)
    assert err.mean() < 1.5, err.mean()                      # mm, 2 mm voxels


def test_reg_apply_affine_numreg_branch_matches_numreg_apply():
    reg = importlib.import_module('tigerbx.reg')
    img = _phantom()
    M = _rotation(5)
    M[:3, 3] = [2.0, 1.0, -3.0]
    disp = {'affine_matrix': {'M': M}, 'reference_info': {'affine': img.affine, 'shape': img.shape}}
    out = reg._apply_affine_to_volume(img.get_fdata().astype(np.float32), img, {'displacement_dict': disp},
                                      {'affine_type': 'numreg'}, None, interpolation='nearestNeighbor')
    ref = numreg.apply(img, SimpleNamespace(affine=img.affine, shape=img.shape), M, order=0)
    np.testing.assert_array_equal(out, ref)


def test_itk_mat_roundtrip(tmp_path):
    M = _rotation(12, axis=0)
    M[:3, 3] = [1.5, -2.0, 4.0]
    numreg.write_itk_mat(M, str(tmp_path / 'a.mat'), center=[3.0, 2.0, 1.0])
    np.testing.assert_allclose(numreg.read_itk_mat(str(tmp_path / 'a.mat')), M, atol=1e-5)


def test_legacy_ants_affine_warp_npz_converts_to_numreg():
    lib_reg = importlib.import_module('tigerbx.lib_reg')
    M = _rotation(7, axis=1)
    M[:3, 3] = [3.0, -1.0, 2.0]
    c = np.array([1.0, -4.0, 6.0])                   # ITK centre of rotation (fixed_parameters)
    A = M[:3, :3]
    t = M[:3, 3] - c + A @ c
    ref_aff = np.diag([1.0, 1.0, 1.0, 1.0])
    ref_aff[:3, 3] = [-80.0, -110.0, -70.0]
    M2, ref = lib_reg.legacy_ants_affine(
        {'parameters': np.r_[A.ravel(), t], 'fixed_parameters': c},
        {'origin': ref_aff[:3, 3], 'spacing': (1.0, 1.0, 1.0), 'direction': np.eye(3), 'shape': (160, 224, 192)})
    np.testing.assert_allclose(M2, M, atol=1e-12)
    np.testing.assert_allclose(ref.affine, ref_aff)
    assert ref.shape == (160, 224, 192)


def test_affine_type_ants_is_an_alias_of_numreg():
    reg = importlib.import_module('tigerbx.reg')
    import pytest
    with pytest.warns(FutureWarning):
        run_d = reg._build_run_dict(SimpleNamespace(plan='AD', affine_type='ANTs'))
    assert run_d['affine_type'] == 'numreg'


def test_numreg_rigid_mode_recovers_rotation_and_stays_rigid():
    fixed = _phantom()
    M_true = _rotation(10, axis=0) @ _rotation(-6, axis=2)
    M_true[:3, 3] = [-5.0, 3.0, 4.0]
    moving = nib.Nifti1Image(numreg.apply(fixed, fixed, np.linalg.inv(M_true)), fixed.affine)
    M = numreg.register(moving, fixed, dof=6)
    np.testing.assert_allclose(M[:3, :3] @ M[:3, :3].T, np.eye(3), atol=1e-8)   # pure rotation
    pts = np.c_[np.random.default_rng(1).uniform(-40, 40, (200, 3)), np.ones(200)].T
    assert np.linalg.norm((M @ pts - M_true @ pts)[:3], axis=0).mean() < 1.5
