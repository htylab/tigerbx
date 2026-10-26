import numpy as np
import pytest

from tigerbx.core.warpfile import FORMAT, load_warp, save_warp


def test_warpfile_roundtrip_plain_arrays(tmp_path):
    M = np.eye(4)
    M[:3, 3] = [1.0, -2.0, 3.0]
    warp = np.random.default_rng(0).normal(size=(3, 4, 5, 6)).astype(np.float32)
    p = save_warp(str(tmp_path / 'x_warp.npz'), np.eye(4), (4, 5, 6), 2 * np.eye(4), (8, 9, 10, 1),
                  affine_M=M, vfa_warp=warp, rigid_M=None)
    with np.load(p) as z:                           # readable without allow_pickle
        assert 'rigid_M' not in z.files and all(z[k].dtype != object for k in z.files)
    w = load_warp(p)
    assert str(w['format']) == FORMAT
    np.testing.assert_array_equal(w['affine_M'], M)
    np.testing.assert_array_equal(w['vfa_warp'], warp)
    assert w['vfa_warp'].dtype == np.float32 and tuple(w['input_shape']) == (8, 9, 10)


def test_warpfile_rejects_pickled_and_foreign_npz(tmp_path):
    old = tmp_path / 'old.npz'
    np.savez(old, affine_matrix={'M': np.eye(4)}, rigid_matrix=None)
    with pytest.raises(ValueError, match='re-run the registration'):
        load_warp(str(old))
    other = tmp_path / 'other.npz'
    np.savez(other, a=np.zeros(3))
    with pytest.raises(ValueError, match=FORMAT):
        load_warp(str(other))
    with pytest.raises(ValueError, match='unknown'):
        save_warp(str(tmp_path / 'y.npz'), np.eye(4), (1, 1, 1), np.eye(4), (1, 1, 1), M=np.eye(4))
