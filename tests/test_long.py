import importlib

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tigerbx.core import refine as core_refine

L = importlib.import_module('tigerbx.pipelines.long')


def test_halfway_squares_to_the_rigid_transform():
    R = np.eye(4)
    R[:3, :3] = Rotation.from_rotvec(np.radians([3.0, -2.0, 4.0])).as_matrix()
    R[:3, 3] = (2.0, -1.5, 0.7)
    H = L._halfway(R)
    assert np.allclose(H @ H, R, atol=1e-9) and np.allclose(H[3], [0, 0, 0, 1])


def test_region_masks():
    tseg = np.array([0, 2, 3, 4, 17, 24, 53, 41])
    m = L._region_masks(tseg)
    assert m['brain'].tolist() == [False, True, True, False, True, False, True, True]
    assert m['hippocampus'].tolist() == [False, False, False, False, True, False, True, False]
    assert m['ventricles'].sum() == 1


def test_refine_runs_the_model_calls_times():
    seen = []

    class Session:
        def get_inputs(self):
            return [type('I', (), {'name': n})() for n in ('moving', 'fixed', 'grid', 'identity')]

        def run(self, outputs, feeds):
            seen.append({k: v.shape for k, v in feeds.items()})
            return [feeds['grid'] + 0.5]

    g = core_refine.refine(np.zeros((4, 5, 6)), np.zeros((4, 5, 6)), np.zeros((3, 4, 5, 6)), session=Session(), calls=2)
    assert g.shape == (3, 4, 5, 6) and np.allclose(g, 1.0)
    assert seen == [{'moving': (1, 1, 4, 5, 6), 'fixed': (1, 1, 4, 5, 6), 'identity': (1, 3, 4, 5, 6),
                     'grid': (1, 3, 4, 5, 6)}] * 2


def test_long_needs_pairs():
    with pytest.raises(ValueError):
        L.long('a.nii.gz', None)
    with pytest.raises(ValueError):
        L.long(['a.nii.gz', 'b.nii.gz'], ['c.nii.gz'])
