import nibabel as nib
import numpy as np

from tigerbx.core.qa import block_mean, homogeneity, image_quality, quality_outliers, robust_z


def _phantom(noise=0.0, blur=0.0, seed=0):
    """Nested boxes: ventricle (4) < GM (3) < WM (2) intensities, T1 ordering."""
    from scipy import ndimage
    a = np.zeros((96, 96, 96), np.int16)
    a[8:88, 8:88, 8:88] = 3
    a[16:80, 16:80, 16:80] = 2
    a[42:54, 42:54, 42:54] = 4
    lut = {0: 0.0, 3: 60.0, 2: 100.0, 4: 20.0}
    x = np.vectorize(lut.get)(a).astype(np.float32)
    if blur:
        x = ndimage.gaussian_filter(x, blur)
    x += np.random.default_rng(seed).normal(0, noise, x.shape).astype(np.float32)
    return nib.Nifti1Image(x, np.eye(4)), nib.Nifti1Image(a, np.eye(4))


def test_image_quality_tracks_noise_and_blur():
    q0 = image_quality(*_phantom(noise=1.0))
    q1 = image_quality(*_phantom(noise=5.0))
    q2 = image_quality(*_phantom(noise=1.0, blur=1.5))
    assert q1['noise'] > 3 * q0['noise']
    assert q2['edge'] < 0.7 * q0['edge']
    assert abs(q0['contrast'] - 0.5) < 0.05 and q0['res'] == 1.0


def test_homogeneity_flags_a_flipped_map():
    rng = np.random.default_rng(0)
    base = rng.random((16, 16, 16)).astype(np.float32)
    maps = [base + 0.05 * rng.random(base.shape).astype(np.float32) for _ in range(30)]
    maps.append(maps[0][::-1].copy())
    _, z = homogeneity(np.stack(maps))
    assert z[-1] < -3 and (z[:-1] > -3).all()
    assert block_mean(np.ones((8, 8, 8)), 4).shape == (2, 2, 2)


def test_image_quality_thick_slices_do_not_fail():
    img, tseg = _phantom(noise=1.0)
    thick = [nib.Nifti1Image(np.asarray(x.dataobj)[:, :, ::4], np.diag([1, 1, 4, 1])) for x in (img, tseg)]
    q = image_quality(*thick)
    assert all(np.isfinite(v) for v in q.values()) and q['res'] > 2


def test_homogeneity_matches_full_matrix_and_skips_unusable_maps():
    rng = np.random.default_rng(1)
    base = rng.random((12, 12, 12)).astype(np.float32)
    maps = np.stack([base + 0.1 * rng.random(base.shape).astype(np.float32) for _ in range(20)])
    s, z = homogeneity(maps)
    X = maps.reshape(20, -1) - maps.reshape(20, -1).mean(1, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    C = X @ X.T
    np.fill_diagonal(C, np.nan)
    np.testing.assert_allclose(s, np.nanmean(C, 1), atol=1e-5)
    bad = maps.copy()
    bad[3] = np.nan                               # unusable maps: NaN, and flat
    bad[5] = 1.0
    bad[-1] = bad[0][::-1]
    s2, z2 = homogeneity(bad)
    assert np.isnan(z2[[3, 5]]).all() and z2[-1] < -3
    assert np.isfinite(z2[[0, 1, 2, 4]]).all()


def test_image_quality_without_tissues_keeps_res_and_accepts_4d():
    img, tseg = _phantom(noise=1.0)
    a = np.asarray(tseg.dataobj).copy()
    a[a == 2] = 0                                 # no WM: contrast-relative measures are NaN, res is kept
    q = image_quality(img, nib.Nifti1Image(a, np.eye(4)))
    assert q['res'] == 1.0 and np.isnan([q[k] for k in ('noise', 'bias', 'contrast', 'edge')]).all()
    q4 = image_quality(*[nib.Nifti1Image(np.asarray(x.dataobj)[..., None], np.eye(4)) for x in (img, tseg)])
    assert q4 == image_quality(img, tseg)


def test_quality_outliers_flag_only_the_worse_direction():
    rng = np.random.default_rng(0)
    n = 20
    v = {k: rng.normal(1.0, 0.01, n) for k in ('noise', 'bias', 'contrast', 'edge', 'res')}
    v['noise'][3] = 2.0       # noisier: flagged
    v['noise'][4] = 0.0       # less noisy: not flagged
    v['edge'][5] = 0.0        # blurred: flagged
    v['contrast'][6] = np.nan # missing: not flagged
    v['res'][7] = 3.0         # different voxel size: never flagged
    flags = quality_outliers(v)
    assert flags[3] == ['noise'] and flags[5] == ['edge']
    assert all(not flags[i] for i in range(n) if i not in (3, 5))


def test_robust_z_constant_sample_is_nan():
    assert np.isnan(robust_z(np.ones(10))).all()
    assert np.isnan(robust_z([1.0, 2.0])).all()          # fewer than 3 values
