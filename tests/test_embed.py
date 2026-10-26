import csv
import importlib

import nibabel as nib
import numpy as np
import pytest

embed_mod = importlib.import_module('tigerbx.embed')       # tigerbx.embed is also the embed() function


def test_roi_means_block_weights_and_missing_labels():
    seg = np.zeros((8, 8, 8), np.uint8)
    seg[:4] = 17                                  # left half -> label 17
    seg[4:, :, :2] = 53                           # a thin slab, partial in 4x4x4 blocks
    f = np.zeros((2, 2, 2, 2), np.float32)        # 2 channels on a 2x2x2 grid (block 4)
    f[0, 0] = 1.0                                 # channel 0 = 1 in the left blocks
    f[1] = np.arange(8).reshape(2, 2, 2)
    out = embed_mod.roi_means([f], seg, labels=(17, 53, 4))
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[0], [1.0, np.mean([0, 1, 2, 3])])        # label 17 fills blocks 0..3
    np.testing.assert_allclose(out[1], [0.0, np.mean([4, 6])])              # slab touches blocks (1,*,0) equally
    assert np.isnan(out[2]).all()                                           # absent label -> NaN row


def test_roi_means_two_levels_concatenate():
    seg = np.full((8, 8, 8), 2, np.uint8)
    a, b = np.ones((3, 4, 4, 4), np.float32), np.full((5, 2, 2, 2), 2.0, np.float32)
    out = embed_mod.roi_means([a, b], seg, labels=(2,))
    np.testing.assert_allclose(out, [[1, 1, 1, 2, 2, 2, 2, 2]])


def test_embed_batch_continues_after_failure(tmp_path, monkeypatch):
    files = []
    for name in ('a', 'bad'):
        f = tmp_path / f'{name}.nii.gz'
        nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.float32), np.eye(4)), str(f))
        files.append(str(f))

    def fake(f, args, common_folder, session):
        if 'bad' in f:
            raise RuntimeError('boom')
        return dict(roi=np.zeros((37, 96), np.float32), tiv_ml=1.0)
    monkeypatch.setattr(embed_mod, '_embed_one', fake)
    monkeypatch.setattr(embed_mod, 'create_session', lambda *a: None)
    monkeypatch.setattr(embed_mod.lib_tool, 'get_model', lambda m: m)
    out = tmp_path / 'out'
    res = embed_mod.embed(files, str(out))
    assert [p.split('/')[-1] for p in res] == ['a_embed.npz']
    assert (out / 'bad_embedfail.log').exists()
    rows = list(csv.DictReader(open(out / embed_mod.SUMMARY_NAME)))
    assert [r['status'] for r in rows] == ['ok', 'failed']
    with pytest.raises(RuntimeError):
        embed_mod.embed(files[1], str(out))


def test_roi_means_rejects_a_grid_that_does_not_divide():
    with pytest.raises(ValueError):
        embed_mod.roi_means([np.zeros((1, 3, 4, 4), np.float32)], np.zeros((8, 8, 8)), labels=(2,))


def test_embed_save_failure_and_single_failure_reach_the_summary(tmp_path, monkeypatch):
    files = []
    for name in ('a', 'b'):
        f = tmp_path / f'{name}.nii.gz'
        nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.float32), np.eye(4)), str(f))
        files.append(str(f))
    monkeypatch.setattr(embed_mod, '_embed_one',
                        lambda f, *a: dict(roi=np.zeros((37, 96), np.float32), tiv_ml=1.0))
    monkeypatch.setattr(embed_mod, 'create_session', lambda *a: None)
    monkeypatch.setattr(embed_mod.lib_tool, 'get_model', lambda m: m)
    real_savez = np.savez

    def savez(path, **kw):
        if path.endswith('a_embed.npz'):
            raise OSError('disk full')
        real_savez(path, **kw)
    monkeypatch.setattr(embed_mod.np, 'savez', savez)
    out = tmp_path / 'out'
    res = embed_mod.embed(files, str(out))                    # a save error must not stop the batch
    assert [p.split('/')[-1] for p in res] == ['b_embed.npz']
    assert (out / 'a_embedfail.log').exists()
    with pytest.raises(OSError):
        embed_mod.embed(files[0], str(out))
    rows = list(csv.DictReader(open(out / embed_mod.SUMMARY_NAME)))
    assert [r['status'] for r in rows] == ['failed']           # a failed single input is in the summary too
