import os

import nibabel as nib
import numpy as np

from tigerbx.pipelines import _common


def test_batch_skips_inputs_whose_output_exists(tmp_path):
    files = []
    for k in 'ab':
        f = str(tmp_path / f'{k}.nii.gz')
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4)), f)
        files.append(f)
    out = str(tmp_path / 'out')
    calls = []

    def one(f, ftemplate):
        calls.append(f)
        path = ftemplate.replace('@@@@', 'foo')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'w').write('x')
        return {}, {'foo': path}

    first = _common.batch('x', files, out, True, 0, one, exists='foo')
    assert len(calls) == 2 and len(first) == 2
    again = _common.batch('x', files, out, True, 0, one, exists='foo')
    assert len(calls) == 2 and again == []                     # both skipped, left out of the list
    assert len(_common.batch('x', files, out, True, 0, one)) == 2 and len(calls) == 4   # no exists: run again
