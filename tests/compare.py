"""
Compare test outputs between old and new tigerbx versions.
Usage: python tests/compare.py
       Expects test_output/old and test_output/new under the repo root.
       Populate them by running:
         python tests/run_inference.py <repo_root>/test_output/new   (new venv)
         python tests/run_inference.py <repo_root>/test_output/old   (old venv)
"""
import os
import sys
import numpy as np
import nibabel as nib

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_DIR = os.path.join(_REPO_ROOT, 'test_output', 'old')
NEW_DIR = os.path.join(_REPO_ROOT, 'test_output', 'new')
MODULES = ['bx', 'hlc', 'reg', 'nerve']
MODES = ['api', 'cli']


def compare_nifti(path_old, path_new):
    arr_old = nib.load(path_old).get_fdata()
    arr_new = nib.load(path_new).get_fdata()
    if arr_old.shape != arr_new.shape:
        return False, f"shape mismatch: {arr_old.shape} vs {arr_new.shape}"
    if not np.array_equal(arr_old, arr_new):
        max_diff = float(np.max(np.abs(arr_old.astype(float) - arr_new.astype(float))))
        n_diff = int(np.sum(arr_old != arr_new))
        return False, f"{n_diff} voxels differ, max_diff={max_diff:.6f}"
    return True, "identical"


def compare_npz(path_old, path_new):
    d_old = np.load(path_old, allow_pickle=True)
    d_new = np.load(path_new, allow_pickle=True)
    keys_old, keys_new = set(d_old.files), set(d_new.files)
    if keys_old != keys_new:
        return False, f"keys differ: {keys_old ^ keys_new}"
    for k in sorted(keys_old):
        a, b = d_old[k], d_new[k]
        if not np.array_equal(a, b):
            return False, f"key '{k}' differs"
    return True, "identical"


def compare_log(path_old, path_new):
    # QC logs use append mode; compare only unique lines to avoid run-count artefacts
    def unique_lines(p):
        return set(open(p).read().splitlines())
    lines_old = unique_lines(path_old)
    lines_new = unique_lines(path_new)
    if lines_old == lines_new:
        return True, f"identical content ({sorted(lines_old)})"
    return False, f"differs: {sorted(lines_old)} vs {sorted(lines_new)}"


def compare_file(old_path, new_path):
    fname = os.path.basename(old_path)
    if fname.endswith(('.nii', '.nii.gz')):
        return compare_nifti(old_path, new_path)
    if fname.endswith('.npz'):
        return compare_npz(old_path, new_path)
    if fname.endswith('.log'):
        return compare_log(old_path, new_path)
    return None, "skipped (unknown type)"


def compare_pair(dir_a, dir_b, label):
    print(f"\n[{label}]")
    if not os.path.isdir(dir_a):
        print(f"  MISSING: {dir_a}")
        return False
    if not os.path.isdir(dir_b):
        print(f"  MISSING: {dir_b}")
        return False

    files_a = set(os.listdir(dir_a))
    files_b = set(os.listdir(dir_b))
    missing = files_a - files_b
    extra = files_b - files_a
    if missing:
        print(f"  MISSING in B   : {sorted(missing)}")
    if extra:
        print(f"  EXTRA in B     : {sorted(extra)}")

    all_ok = not bool(missing)
    for fname in sorted(files_a & files_b):
        ok, msg = compare_file(
            os.path.join(dir_a, fname),
            os.path.join(dir_b, fname),
        )
        tag = '--' if ok is None else ('OK' if ok else 'FAIL')
        print(f"  [{tag}] {fname}: {msg}")
        if ok is False:
            all_ok = False
    return all_ok


def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


def main():
    # -- 1. API (new) vs CLI (new): validates tigerbx_cli wrapper --
    section("NEW: API vs CLI  (refactoring correctness)")
    wrap_ok = True
    for module in MODULES:
        ok = compare_pair(
            os.path.join(NEW_DIR, module, 'api'),
            os.path.join(NEW_DIR, module, 'cli'),
            f"new  {module}  api vs cli",
        )
        wrap_ok = wrap_ok and ok

    # -- 2. old vs new (same mode): detects regressions --
    section("OLD vs NEW  (regression check)")
    regress_ok = True
    for module in MODULES:
        for mode in MODES:
            ok = compare_pair(
                os.path.join(OLD_DIR, module, mode),
                os.path.join(NEW_DIR, module, mode),
                f"old vs new  {module}/{mode}",
            )
            regress_ok = regress_ok and ok

    # -- Summary --
    print(f"\n{'='*55}")
    print(f"[API vs CLI]   {'PASS -- CLI wrapper is correct' if wrap_ok else 'FAIL -- CLI wrapper differs from API'}")
    print(f"[Old vs New]   {'PASS -- no regressions' if regress_ok else 'FAIL -- see differences above'}")
    if not wrap_ok or not regress_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
