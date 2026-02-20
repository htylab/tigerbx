"""
Run tigerbx inference and compare outputs.

Usage:
    python tests/run_and_compare.py [MODULE ...] [options]

    MODULE   one or more of: bx  hlc  reg  nerve  all  (default: all)

Options:
    --output-dir DIR    where to save new test outputs
                        (default: <repo>/test_output/new)
    --old-dir DIR       reference outputs for regression check
                        (default: <repo>/test_output/old)
    --run-only          skip comparison step
    --compare-only      skip inference step

Examples:
    python tests/run_and_compare.py bx
    python tests/run_and_compare.py bx hlc
    python tests/run_and_compare.py all
    python tests/run_and_compare.py all --output-dir /tmp/test_new
    python tests/run_and_compare.py nerve --run-only
    python tests/run_and_compare.py all --compare-only
"""
import sys
import os
import argparse
import subprocess
import numpy as np
import nibabel as nib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, 'tigerbx', 'template', 'MNI152_T1_1mm_brain.nii.gz')
SCRIPTS_DIR = os.path.dirname(sys.executable)
TIGER_BIN = (
    os.path.join(SCRIPTS_DIR, 'tiger.exe')
    if sys.platform == 'win32'
    else os.path.join(SCRIPTS_DIR, 'tiger')
)

ALL_MODULES = ['bx', 'hlc', 'reg', 'nerve']

PASS = []
FAIL = []


# ── inference helpers ─────────────────────────────────────────────────────────

def section(label):
    print(f"\n{'='*55}\n  {label}\n{'='*55}")


def make_dir(*parts):
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path


def run_api(label, fn):
    section(f'[API] {label}')
    try:
        fn()
        PASS.append(f'API  {label}')
    except Exception as e:
        print(f'  ERROR: {e}')
        FAIL.append(f'API  {label}')


def run_cli(label, cmd):
    section(f'[CLI] {label}')
    try:
        subprocess.run(cmd, check=True)
        PASS.append(f'CLI  {label}')
    except Exception as e:
        print(f'  ERROR: {e}')
        FAIL.append(f'CLI  {label}')


# ── per-module inference ──────────────────────────────────────────────────────

def run_bx(output_dir, tigerbx):
    # b=bet  m=betmask  a=aseg  c=ct  C=cgw  d=dgm  S=syn  W=wmh  t=tumor  q=qc
    # Note: k=dkt and w=wmp are API-only; omitted to keep API/CLI outputs comparable
    api_dir = make_dir(output_dir, 'bx', 'api')
    cli_dir = make_dir(output_dir, 'bx', 'cli')
    run_api('bx — all outputs', lambda: tigerbx.run('bmacdCSWtq', TEMPLATE, api_dir))
    run_cli('bx — all outputs', [
        TIGER_BIN, 'bx',
        '-b', '-m', '-a', '-c', '-C', '-d', '-S', '-W', '-t', '-q',
        TEMPLATE, '-o', cli_dir,
    ])


def run_hlc(output_dir, tigerbx):
    # save=all -> m=betmask  b=bet  h=hlc  t=ct  c=csf  g=gm  w=wm
    api_dir = make_dir(output_dir, 'hlc', 'api')
    cli_dir = make_dir(output_dir, 'hlc', 'cli')
    run_api('hlc — save=all', lambda: tigerbx.hlc(TEMPLATE, api_dir, save='all'))
    run_cli('hlc — save all', [TIGER_BIN, 'hlc', '--save', 'all', TEMPLATE, '-o', cli_dir])


def run_reg(output_dir, tigerbx):
    # A=affine (C2FViT)  r=VMnet dense registration
    api_dir = make_dir(output_dir, 'reg', 'api')
    cli_dir = make_dir(output_dir, 'reg', 'cli')
    run_api('reg — affine + registration', lambda: tigerbx.reg('Ar', TEMPLATE, api_dir))
    run_cli('reg — -A -r', [TIGER_BIN, 'reg', '-A', '-r', TEMPLATE, '-o', cli_dir])


def run_nerve(output_dir, tigerbx):
    # e=encode (.npz)  p=save patches (.nii.gz)
    api_dir = make_dir(output_dir, 'nerve', 'api')
    cli_dir = make_dir(output_dir, 'nerve', 'cli')
    run_api('nerve — encode + save_patch', lambda: tigerbx.nerve('ep', TEMPLATE, api_dir))
    run_cli('nerve — -e -p', [TIGER_BIN, 'nerve', '-e', '-p', TEMPLATE, '-o', cli_dir])


MODULE_RUNNERS = {
    'bx':    run_bx,
    'hlc':   run_hlc,
    'reg':   run_reg,
    'nerve': run_nerve,
}


# ── comparison helpers ────────────────────────────────────────────────────────

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
        if not np.array_equal(d_old[k], d_new[k]):
            return False, f"key '{k}' differs"
    return True, "identical"


def compare_log(path_old, path_new):
    # QC logs use append mode; compare unique lines to avoid run-count artefacts
    def unique_lines(p):
        return set(open(p).read().splitlines())
    lines_old = unique_lines(path_old)
    lines_new = unique_lines(path_new)
    if lines_old == lines_new:
        return True, f"identical ({sorted(lines_old)})"
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
    extra   = files_b - files_a
    if missing:
        print(f"  MISSING in B: {sorted(missing)}")
    if extra:
        print(f"  EXTRA in B  : {sorted(extra)}")

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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog='run_and_compare.py',
        description='Run tigerbx inference and compare outputs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python tests/run_and_compare.py bx
  python tests/run_and_compare.py bx hlc
  python tests/run_and_compare.py all
  python tests/run_and_compare.py all --output-dir /tmp/test_new
  python tests/run_and_compare.py nerve --run-only
  python tests/run_and_compare.py all --compare-only
        """,
    )
    parser.add_argument(
        'modules',
        nargs='*',
        default=['all'],
        metavar='MODULE',
        help=f'modules to test: {", ".join(ALL_MODULES)}, or all (default: all)',
    )
    parser.add_argument(
        '--output-dir',
        default=os.path.join(ROOT, 'test_output', 'new'),
        metavar='DIR',
        help='output directory for new test results (default: <repo>/test_output/new)',
    )
    parser.add_argument(
        '--old-dir',
        default=os.path.join(ROOT, 'test_output', 'old'),
        metavar='DIR',
        help='reference directory for regression check (default: <repo>/test_output/old)',
    )
    parser.add_argument(
        '--run-only',
        action='store_true',
        help='skip comparison; only run inference',
    )
    parser.add_argument(
        '--compare-only',
        action='store_true',
        help='skip inference; only run comparison',
    )
    args = parser.parse_args()

    # Validate and resolve module list
    valid = set(ALL_MODULES) | {'all'}
    bad = [m for m in args.modules if m not in valid]
    if bad:
        parser.error(f"unknown module(s): {bad}. Choose from: {', '.join(sorted(valid))}")

    modules = ALL_MODULES if 'all' in args.modules else list(dict.fromkeys(args.modules))

    new_dir = args.output_dir
    old_dir = args.old_dir

    exit_code = 0

    # ── 1. Inference ──────────────────────────────────────────────────────────
    if not args.compare_only:
        import tigerbx

        print(f"\ntigerbx version : {tigerbx.__version__}")
        print(f"python          : {sys.executable}")
        print(f"tiger bin       : {TIGER_BIN}")
        print(f"output dir      : {new_dir}")
        print(f"modules         : {modules}")

        for module in modules:
            MODULE_RUNNERS[module](new_dir, tigerbx)

        print(f"\n{'='*55}")
        print('FILE SUMMARY')
        print('='*55)
        for module in modules:
            for mode in ['api', 'cli']:
                d = os.path.join(new_dir, module, mode)
                files = sorted(os.listdir(d)) if os.path.isdir(d) else []
                print(f"  {module}/{mode:3s}: {files}")

        print(f"\n{'='*55}")
        print(f"PASS ({len(PASS)}): {PASS}")
        print(f"FAIL ({len(FAIL)}): {FAIL}")

        if FAIL:
            exit_code = 1

    # ── 2. Comparison ─────────────────────────────────────────────────────────
    if not args.run_only:
        wrap_ok = True
        regress_ok = True

        section("NEW: API vs CLI  (refactoring correctness)")
        for module in modules:
            ok = compare_pair(
                os.path.join(new_dir, module, 'api'),
                os.path.join(new_dir, module, 'cli'),
                f"new  {module}  api vs cli",
            )
            wrap_ok = wrap_ok and ok

        section("OLD vs NEW  (regression check)")
        for module in modules:
            for mode in ['api', 'cli']:
                ok = compare_pair(
                    os.path.join(old_dir, module, mode),
                    os.path.join(new_dir, module, mode),
                    f"old vs new  {module}/{mode}",
                )
                regress_ok = regress_ok and ok

        print(f"\n{'='*55}")
        print(f"[API vs CLI]   {'PASS -- CLI wrapper is correct' if wrap_ok else 'FAIL -- CLI wrapper differs from API'}")
        print(f"[Old vs New]   {'PASS -- no regressions' if regress_ok else 'FAIL -- see differences above'}")

        if not wrap_ok or not regress_ok:
            exit_code = 1

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
