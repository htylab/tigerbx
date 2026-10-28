# Rust executable vs Python package on one reverse phase-encoding pair (manual check, not a pytest test).
#   python tests/duope_parity.py FORWARD REVERSE /path/to/tigerbx_exe out_dir [--series S] [--pe X] [--readout T]
# Both write <input>_duope (+ _duope_disp) and duope_summary.csv; prints the displacement difference (voxels), the
# relative difference of every corrected image, and the summary values (phase axis, polarity, motion, joint step,
# series motion). Close, not bit for bit: mean displacement difference ~0.001-0.003 voxel (rust/README.md).
import argparse
import csv
import glob
import os
import subprocess

import nibabel as nib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('forward')
    ap.add_argument('reverse')
    ap.add_argument('exe')
    ap.add_argument('out')
    ap.add_argument('--series')
    ap.add_argument('--pe')
    ap.add_argument('--readout', type=float)
    a = ap.parse_args()
    import tigerbx
    from tigerbx.core import duope as core
    ro, po = os.path.join(a.out, 'rust'), os.path.join(a.out, 'python')
    extra = []
    for k in ('series', 'pe', 'readout'):
        if getattr(a, k) is not None:
            extra += [f'--{k}', str(getattr(a, k))]
    subprocess.run([a.exe, '--duope', a.forward, a.reverse, '-o', ro, '--save-disp'] + extra, check=True)
    tigerbx.duope(a.forward, a.reverse, po, series=a.series, pe=a.pe, readout=a.readout, save_disp=True)

    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(po, '*.nii.gz')))
    assert names == sorted(os.path.basename(p) for p in glob.glob(os.path.join(ro, '*.nii.gz'))), 'different outputs'
    f0 = nib.load(a.forward).get_fdata()
    f0 = f0.reshape(f0.shape[:3] + (-1,)).mean(-1)
    brain = core.head_mask(f0, f0) & (f0 > 0.3 * np.percentile(f0, 99))
    for n in names:
        x, y = nib.load(os.path.join(ro, n)).get_fdata(), nib.load(os.path.join(po, n)).get_fdata()
        if n.endswith('_disp.nii.gz'):
            e = np.abs(x - y)[brain]
            print(f'{n}: mean |rust - python| {e.mean():.4f} voxel, p99 {np.percentile(e, 99):.3f}')
        else:
            if x.shape[:3] == brain.shape:
                x, y = x.reshape(brain.shape + (-1,))[brain], y.reshape(brain.shape + (-1,))[brain]
            print(f'{n}: relative |rust - python| {np.abs(x - y).mean() / np.abs(y).mean():.5f}')
    keys = ('phase_axis', 'forward_polarity', 'forward_direction', 'motion_mm', 'motion_deg', 'joint',
            'series_motion_mm_deg')
    for label, d in (('rust', ro), ('python', po)):
        r = list(csv.DictReader(open(os.path.join(d, 'duope_summary.csv'))))[-1]
        print(f'{label:7s}', ', '.join(f'{k} {r[k]}' for k in keys))


if __name__ == '__main__':
    main()
