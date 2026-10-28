# Rust executable vs Python package parity on one T1w (manual check, not a pytest test).
#   python tests/rust_python_parity.py T1w.nii.gz /path/to/tigerbx_exe out_dir
# 1. modules: tigerbx.run('bmtd') vs `tigerbx_exe -b -m -t -d` (the models' own outputs): _tbxmask, _tseg, _dgm
# 2. seg pipeline: tigerbx.seg() vs `tigerbx_exe --seg` (deep GM from DGM12): _tseg
# 3. inspect: tigerbx.inspect(detail=True) vs `tigerbx_exe --inspect --detail`: every column of inspect_summary.csv
# All must be identical; on oblique inputs a few voxels on the edge of the field of view may differ (rust/README.md).
import csv
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
    py_ins, rs_ins = os.path.join(out, 'python_inspect'), os.path.join(out, 'rust_inspect')
    tigerbx.inspect(f, py_ins, detail=True)
    subprocess.run([exe, f, '-o', rs_ins, '--inspect', '--detail'], check=True, stdout=subprocess.DEVNULL)
    a, b = (list(csv.DictReader(open(os.path.join(d, 'inspect_summary.csv'), newline=''))) for d in (py_ins, rs_ins))
    cells = [k for k in a[0] if a[0][k] != b[0].get(k)]
    print(f'{stem}: inspect_summary.csv differing columns: {cells or "none"}')
    return not any(diff.values()) and not cells


if __name__ == '__main__':
    sys.exit(0 if main(*sys.argv[1:4]) else 1)
