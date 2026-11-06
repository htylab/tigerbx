import csv
import os

import nibabel as nib
import numpy as np
import pytest
from scipy import ndimage

import tigerbx
from tigerbx.core import dwimc as core

SHAPE = (32, 40, 28)
ZOOMS = (2.5, 2.5, 2.5)
AFF = np.diag([-2.5, 2.5, 2.5, 1.0])


def _directions(n, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(n, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    for _ in range(500):                                  # antipodal repulsion
        d1, d2 = g[:, None] - g[None], g[:, None] + g[None]
        f = (d1 / (np.linalg.norm(d1, axis=2, keepdims=True) ** 3 + 1e-9)).sum(1) \
            + (d2 / (np.linalg.norm(d2, axis=2, keepdims=True) ** 3 + 1e-9)).sum(1)
        g = g + 0.01 * f / n
        g /= np.linalg.norm(g, axis=1, keepdims=True)
    return g


def _phantom():
    """Head-space S0 and a two-compartment anisotropic attenuation (fibres along x on one side, along z on the
    other), smooth enough for 2.5 mm voxels."""
    rng = np.random.default_rng(1)
    x = np.indices(SHAPE).astype(float)
    c = [(n - 1) / 2 for n in SHAPE]
    r2 = sum(((x[i] - c[i]) / (0.4 * SHAPE[i])) ** 2 for i in range(3))
    body = ndimage.gaussian_filter((r2 < 1).astype(float), 1.0)
    tex = ndimage.gaussian_filter(rng.standard_normal(SHAPE), 1.2)
    S0 = np.clip(body * (1 + 0.5 * tex / tex.std()), 0, None) * 1000
    left = ndimage.gaussian_filter((x[0] < c[0]).astype(float), 3.0)
    return S0, left, body > 0.5


def _series(n_dir=16, seed=2, ec_scale=1.0, aniso=1.0):
    """A series (b0 first, one more b0 in the middle) with known motion and linear eddy currents, no field."""
    S0, left, mask = _phantom()
    g = _directions(n_dir)
    b = np.r_[0, np.full(n_dir // 2, 1000), 0, np.full(n_dir - n_dir // 2, 1000)]
    gv = np.r_[[np.zeros(3)], g[:n_dir // 2], [np.zeros(3)], g[n_dir // 2:]]
    rng = np.random.default_rng(seed)
    zooms = np.array(ZOOMS)
    c_mm = zooms * np.argwhere(mask).mean(0)
    B = ec_scale * np.array([[0.4, -0.3, 0.2], [0.3, 0.2, -0.3], [-0.4, 0.3, 0.2], [0.2, 0.3, -0.3]])   # (4, 3)
    ec = core.ECBasis(SHAPE, 1)
    u = core.grid(SHAPE)
    vols, Ms, cs = [], [], []
    for i, (bi, gi) in enumerate(zip(b, gv)):
        p = np.r_[rng.normal(0, 0.5, 3), np.radians(rng.normal(0, 0.5, 3))] if i else np.zeros(6)
        M = core.rigid(p, zooms, c_mm)
        cc = B @ gi if bi > 0 else np.zeros(4)
        if bi > 0:
            fa = aniso * 0.8e-3 * (left * gi[0] ** 2 + (1 - left) * gi[2] ** 2) + 0.4e-3
            T = S0 * np.exp(-bi * fa)
        else:
            T = S0
        F = (cc @ ec.phi(u)).reshape(SHAPE)
        xs = u.copy()
        xs[1] += core.line_inverse(F).ravel()
        Fy = core.sample(np.gradient(F, axis=1), xs)
        v = core.sample(T, core.apply(np.linalg.inv(M), xs), order=3) / (1 + Fy)
        vols.append(v.reshape(SHAPE))
        Ms.append(M)
        cs.append(cc)
    return np.stack(vols, -1).astype(np.float32), b, gv, mask, np.array(Ms), np.array(cs)


def _b0_with_dark_core():
    """A bright head with a dark deep region that a one-voxel channel joins to the outside (a 3D fill keeps it out)."""
    x, y, z = np.indices((40, 40, 30))
    head = ((x - 20) / 16.0) ** 2 + ((y - 20) / 16.0) ** 2 + ((z - 15) / 12.0) ** 2 < 1
    core_ = ((x - 20) / 5.0) ** 2 + ((y - 20) / 5.0) ** 2 + ((z - 15) / 4.0) ** 2 < 1
    b0 = np.where(head, 1000.0, 0.0)
    b0[core_] = 50.0
    b0[20, 20, 15:] = 0.0                                    # the channel to the top of the head
    return b0, core_


def test_head_mask_has_no_holes():
    b0, core_ = _b0_with_dark_core()
    assert not ndimage.binary_fill_holes(b0 > 150)[core_].all()      # the case: a 3D fill leaves the hole
    m = core.head_mask(b0)
    assert m[core_].all()


def test_line_inverse_inverts_a_monotone_map():
    rng = np.random.default_rng(0)
    F = ndimage.gaussian_filter(rng.normal(size=(4, 30, 3)), (0, 4, 0))
    F *= 0.6 / np.abs(np.gradient(F, axis=1)).max()            # |dF/dy| <= 0.6: monotone
    g = core.line_inverse(F)
    y = np.arange(30.0)
    for i in range(4):
        for k in range(3):
            ys = y + g[i, :, k]
            u = y + F[i, :, k]
            inside = (y > u.min() + 1) & (y < u.max() - 1)      # observed voxels the map reaches
            back = ys + np.interp(ys, y, F[i, :, k])          # u = x_s + F(x_s)
            assert np.abs(back[inside] - y[inside]).max() < 0.05


def test_sh_predictor_is_leave_one_out():
    g = _directions(20)
    rng = np.random.default_rng(3)
    S = rng.normal(size=(20, 500)) + 5
    H = core.sh_predictor(g, S, 4)
    assert np.allclose(np.diag(H), 0)                       # a volume never predicts itself
    assert np.allclose(H.sum(1), 1, atol=0.05)              # a constant shell is predicted (nearly) unchanged


def test_estimate_recovers_motion_and_eddy_currents():
    data, b, g, mask, M_true, c_true = _series()
    est = core.estimate(data, b, g, ZOOMS, mask=mask, order=1, workers=2)
    assert est['ref'] == 0
    xh = core.grid(SHAPE)[:, mask.ravel()]
    zero = np.zeros(SHAPE)
    ec = core.ECBasis(SHAPE, 1)
    err0, err = [], []
    for i in range(1, len(b)):
        t = core.observed_position(xh, M_true[i], zero, ec, c_true[i])
        e = core.observed_position(xh, est['M'][i], zero, ec, est['c'][i])
        err.append(np.median(np.linalg.norm(t - e, axis=0)))
        err0.append(np.median(np.linalg.norm(t - xh, axis=0)))
    # anisotropic contrast between the two halves biases a few volumes (a strong, sharp pattern for 16 directions)
    assert np.median(err) < 0.5 * np.median(err0), (np.median(err), np.median(err0))
    assert np.median(err) < 0.2                             # voxels


def test_estimate_is_exact_on_isotropic_contrast():
    data, b, g, mask, M_true, c_true = _series(aniso=0.0)
    est = core.estimate(data, b, g, ZOOMS, mask=mask, order=1, workers=2)
    xh = core.grid(SHAPE)[:, mask.ravel()]
    zero = np.zeros(SHAPE)
    ec = core.ECBasis(SHAPE, 1)
    err = [np.median(np.linalg.norm(core.observed_position(xh, M_true[i], zero, ec, c_true[i])
                                    - core.observed_position(xh, est['M'][i], zero, ec, est['c'][i]), axis=0))
           for i in range(len(b))]
    assert max(err) < 0.05                                  # voxels


def test_dwimc_pipeline_writes_outputs(tmp_path):
    data, b, g, mask, _, _ = _series(seed=4)
    f = str(tmp_path / 'sub_dwi.nii.gz')
    nib.save(nib.Nifti1Image(data, AFF), f)
    np.savetxt(str(tmp_path / 'sub_dwi.bval'), b[None], fmt='%g')
    np.savetxt(str(tmp_path / 'sub_dwi.bvec'), g.T, fmt='%.6f')
    out = tigerbx.dwimc(f, str(tmp_path / 'out'), pe='j', order=1, iters=2, workers=2)
    img = nib.load(out['dwimc'])
    assert img.shape == data.shape and np.allclose(img.affine, AFF)
    assert np.asarray(img.dataobj).min() >= 0
    bv = np.loadtxt(out['bvec'])
    assert bv.shape == (3, len(b))
    n = np.linalg.norm(bv, axis=0)
    assert np.allclose(n[b > 0], 1, atol=1e-4) and np.allclose(n[b == 0], 0)
    rows = list(csv.DictReader(open(out['csv'])))
    assert len(rows) == len(b) and {'tx_mm', 'rz_deg', 'ec_j_vox', 'rms_mm'} <= set(rows[0])
    assert float(rows[0]['rms_mm']) == pytest.approx(0, abs=1e-6)      # the reference b0 does not move
    again = tigerbx.dwimc(f, str(tmp_path / 'out'), pe='j', resume=True)
    assert again == []


def test_dwimc_mask_option(tmp_path, monkeypatch):
    import importlib
    pipe = importlib.import_module('tigerbx.pipelines.dwimc')       # the module (the package exports the function)
    data, b, _, mask, _, _ = _series(seed=4)
    img = nib.Nifti1Image(data, AFF)
    assert pipe._brain_mask('threshold', data, b, None, 1, img) is None           # core's own head mask
    mf = str(tmp_path / 'mask.nii.gz')
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), AFF), mf)
    assert np.array_equal(pipe._brain_mask(mf, data, b, None, 1, img), mask)       # a mask file on the grid

    def no_model(*a, **k):
        raise RuntimeError('no model')
    monkeypatch.setattr('tigerbx.modules.bx.produce_tbxdual', no_model)
    with pytest.warns(RuntimeWarning, match='threshold mask'):
        assert pipe._brain_mask('tbx', data, b, None, 1, img) is None              # falls back


def test_dwimc_rejects_a_series_without_b0(tmp_path):
    data, b, g, _, _, _ = _series()
    f = str(tmp_path / 'x_dwi.nii.gz')
    nib.save(nib.Nifti1Image(data, AFF), f)
    np.savetxt(str(tmp_path / 'x_dwi.bval'), np.full((1, len(b)), 1000), fmt='%g')
    np.savetxt(str(tmp_path / 'x_dwi.bvec'), np.r_[g[1:2], g[1:]].T, fmt='%.6f')
    with pytest.raises(ValueError, match='b0'):
        tigerbx.dwimc(f, str(tmp_path / 'out'), pe='j')
