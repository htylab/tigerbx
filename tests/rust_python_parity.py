# Rust executable vs Python package parity on one T1w (manual check, not a pytest test).
#   python tests/rust_python_parity.py T1w.nii.gz /path/to/tigerbx_exe out_dir
# 1. modules: tigerbx.run('bmtd') vs `tigerbx_exe -b -m -t -d` (the models' own outputs): _tbxmask, _tseg, _dgm
# 2. seg pipeline: tigerbx.seg() vs `tigerbx_exe --seg` (deep GM from DGM12): _tseg
# All must be identical; on oblique inputs a few voxels on the edge of the field of view may differ (rust/README.md).
import os
import subprocess
import sys

import nibabel as nib
import numpy as np


def main(f, exe, out):
    import tigerbx
    py_dir, rs_dir = os.path.join(out, 'python'), os.path.join(out, 'rust')
    py_seg, rs_seg = os.path.join(out, 'python_seg'), os.path.join(out, 'rust_seg')
    tigerbx.run('bmtd', f, py_dir, verbose=0)
    tigerbx.seg(f, py_seg)
    subprocess.run([exe, f, '-o', rs_dir, '-b', '-m', '-t', '-d'], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([exe, f, '-o', rs_seg, '--seg'], check=True, stdout=subprocess.DEVNULL)
    stem = os.path.basename(f).replace('.nii.gz', '').replace('.nii', '')

    def load(d, k):
        return np.asarray(nib.load(os.path.join(d, f'{stem}_{k}.nii.gz')).dataobj).astype(np.int32)

    diff = {k: int((load(py_dir, k) != load(rs_dir, k)).sum()) for k in ('tbxmask', 'tseg', 'dgm')}
    diff['tseg (seg pipeline)'] = int((load(py_seg, 'tseg') != load(rs_seg, 'tseg')).sum())
    print(f'{stem}: differing voxels ' + '  '.join(f'_{k} {v}' for k, v in diff.items()))
    return not any(diff.values())


if __name__ == '__main__':
    sys.exit(0 if main(*sys.argv[1:4]) else 1)
