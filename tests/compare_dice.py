"""
Compare old vs new bx outputs using Dice score.

Usage:
    python tests/compare_dice.py [--old-dir DIR] [--new-dir DIR] [--mode api|cli|both]

Examples:
    python tests/compare_dice.py
    python tests/compare_dice.py --mode cli
    python tests/compare_dice.py --old-dir test_output/old --new-dir test_output/new
"""
import argparse
import os
import numpy as np
import nibabel as nib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files to compute per-label Dice (multi-class segmentation)
MULTILABEL_FILES = [
    'MNI152_T1_1mm_brain_aseg.nii.gz',
    'MNI152_T1_1mm_brain_dgm.nii.gz',
    'MNI152_T1_1mm_brain_dkt.nii.gz',
    'MNI152_T1_1mm_brain_syn.nii.gz',
    'MNI152_T1_1mm_brain_wmp.nii.gz',
]

# Files to compute binary Dice
BINARY_FILES = [
    'MNI152_T1_1mm_brain_wmh.nii.gz',
    'MNI152_T1_1mm_brain_tumor.nii.gz',
    'MNI152_T1_1mm_brain_tbetmask.nii.gz',
]

# Skipped (not segmentation)
SKIP_FILES = [
    'MNI152_T1_1mm_brain_cgw_pve0.nii.gz',
    'MNI152_T1_1mm_brain_cgw_pve1.nii.gz',
    'MNI152_T1_1mm_brain_cgw_pve2.nii.gz',
    'MNI152_T1_1mm_brain_ct.nii.gz',
    'MNI152_T1_1mm_brain_tbet.nii.gz',
]


def dice_binary(a, b):
    """Dice score for two binary arrays."""
    a = a > 0
    b = b > 0
    inter = np.sum(a & b)
    total = np.sum(a) + np.sum(b)
    if total == 0:
        return 1.0
    return 2 * inter / total


def dice_multilabel(a, b):
    """Per-label Dice, returns (mean_dice, label_dice_dict).
    Excludes label 0 (background).
    """
    labels = np.union1d(np.unique(a), np.unique(b))
    labels = labels[labels != 0]
    scores = {}
    for lbl in labels:
        ai = a == lbl
        bi = b == lbl
        inter = np.sum(ai & bi)
        total = np.sum(ai) + np.sum(bi)
        scores[int(lbl)] = 2 * inter / total if total > 0 else 1.0
    mean = float(np.mean(list(scores.values()))) if scores else float('nan')
    return mean, scores


def compare_dir(old_dir, new_dir, mode_label):
    print(f"\n{'='*60}")
    print(f"  [{mode_label}]  old: {old_dir}")
    print(f"{'='*60}")

    results = {}

    for fname in MULTILABEL_FILES:
        op = os.path.join(old_dir, fname)
        np_ = os.path.join(new_dir, fname)
        if not os.path.exists(op) or not os.path.exists(np_):
            print(f"  [SKIP] {fname} (missing)")
            continue
        a = np.round(nib.load(op).get_fdata()).astype(int)
        b = np.round(nib.load(np_).get_fdata()).astype(int)
        mean_dice, per_label = dice_multilabel(a, b)
        tag = fname.replace('MNI152_T1_1mm_brain_', '').replace('.nii.gz', '')
        print(f"  {tag:12s}  mean Dice = {mean_dice:.4f}  (over {len(per_label)} labels)")
        results[tag] = mean_dice

    for fname in BINARY_FILES:
        op = os.path.join(old_dir, fname)
        np_ = os.path.join(new_dir, fname)
        if not os.path.exists(op) or not os.path.exists(np_):
            print(f"  [SKIP] {fname} (missing)")
            continue
        a = nib.load(op).get_fdata()
        b = nib.load(np_).get_fdata()
        d = dice_binary(a, b)
        tag = fname.replace('MNI152_T1_1mm_brain_', '').replace('.nii.gz', '')
        print(f"  {tag:12s}  Dice = {d:.4f}")
        results[tag] = d

    if results:
        print(f"\n  Overall mean Dice = {np.mean(list(results.values())):.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare old/new bx outputs with Dice score.')
    parser.add_argument('--old-dir', default=os.path.join(ROOT, 'test_output', 'old', 'bx'),
                        help='old output bx directory (default: test_output/old/bx)')
    parser.add_argument('--new-dir', default=os.path.join(ROOT, 'test_output', 'new', 'bx'),
                        help='new output bx directory (default: test_output/new/bx)')
    parser.add_argument('--mode', choices=['api', 'cli', 'both'], default='both')
    args = parser.parse_args()

    modes = ['api', 'cli'] if args.mode == 'both' else [args.mode]
    for mode in modes:
        compare_dir(
            os.path.join(args.old_dir, mode),
            os.path.join(args.new_dir, mode),
            mode,
        )


if __name__ == '__main__':
    main()
