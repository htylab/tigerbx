"""
Comprehensive tigerbx test: all output types for bx / hlc / reg / nerve.
Usage: python tests/run_inference.py [output_dir]
       output_dir defaults to <repo_root>/test_output/new
"""
import sys
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, 'tigerbx', 'template', 'MNI152_T1_1mm_brain.nii.gz')
SCRIPTS_DIR = os.path.dirname(sys.executable)
TIGER_BIN = (
    os.path.join(SCRIPTS_DIR, 'tiger.exe')
    if sys.platform == 'win32'
    else os.path.join(SCRIPTS_DIR, 'tiger')
)

PASS = []
FAIL = []


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


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'test_output', 'new')

    import tigerbx
    print(f"\ntigerbx version : {tigerbx.__version__}")
    print(f"python          : {sys.executable}")
    print(f"tiger           : {TIGER_BIN}")
    print(f"output_dir      : {output_dir}")

    # ──────────────────────────────────────────────────────────
    # BX — all output types
    #   b  = bet brain
    #   m  = bet mask
    #   a  = aseg (43-region)
    #   c  = cortical thickness
    #   C  = CGW / FSL-style PVE (3 files: pve0/1/2)
    #   d  = deep gray matter
    #   k  = DKT parcellation
    #   S  = SynthSeg-like aseg
    #   w  = white matter parcellation
    #   W  = white matter hypo-intensity
    #   t  = tumor mask
    #   q  = QC log
    # ──────────────────────────────────────────────────────────
    bx_api = make_dir(output_dir, 'bx', 'api')
    bx_cli = make_dir(output_dir, 'bx', 'cli')

    run_api('bx — all outputs', lambda: tigerbx.run('bmacdCkSwWtq', TEMPLATE, bx_api))
    run_cli('bx — all outputs', [
        TIGER_BIN, 'bx',
        '-b', '-m', '-a', '-c', '-C', '-d', '-k', '-S', '-w', '-W', '-t', '-q',
        TEMPLATE, '-o', bx_cli,
    ])

    # ──────────────────────────────────────────────────────────
    # HLC — hierarchical label consolidation
    #   save=all -> mbhtcgw
    #   m = betmask, b = bet, h = hlc parcellation,
    #   t = cortical thickness, c = CSF, g = GM, w = WM
    # ──────────────────────────────────────────────────────────
    hlc_api = make_dir(output_dir, 'hlc', 'api')
    hlc_cli = make_dir(output_dir, 'hlc', 'cli')

    run_api('hlc — save=all', lambda: tigerbx.hlc(TEMPLATE, hlc_api, save='all'))
    run_cli('hlc — save all', [TIGER_BIN, 'hlc', '--save', 'all', TEMPLATE, '-o', hlc_cli])

    # ──────────────────────────────────────────────────────────
    # REG — registration to MNI space
    #   A = affine (C2FViT)
    #   r = dense registration (VoxelMorph)
    # ──────────────────────────────────────────────────────────
    reg_api = make_dir(output_dir, 'reg', 'api')
    reg_cli = make_dir(output_dir, 'reg', 'cli')

    run_api('reg — affine + registration', lambda: tigerbx.reg('Ar', TEMPLATE, reg_api))
    run_cli('reg — -A -r', [TIGER_BIN, 'reg', '-A', '-r', TEMPLATE, '-o', reg_cli])

    # ──────────────────────────────────────────────────────────
    # NERVE — hippocampus/amygdala VAE embedding
    #   e = encode latent vectors (.npz)
    #   p = save patches (.nii.gz)
    # ──────────────────────────────────────────────────────────
    nerve_api = make_dir(output_dir, 'nerve', 'api')
    nerve_cli = make_dir(output_dir, 'nerve', 'cli')

    run_api('nerve — encode + save_patch', lambda: tigerbx.nerve('ep', TEMPLATE, nerve_api))
    run_cli('nerve — -e -p', [TIGER_BIN, 'nerve', '-e', '-p', TEMPLATE, '-o', nerve_cli])

    # ──────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print('FILE SUMMARY')
    print('='*55)
    for module in ['bx', 'hlc', 'reg', 'nerve']:
        for mode in ['api', 'cli']:
            d = os.path.join(output_dir, module, mode)
            files = sorted(os.listdir(d)) if os.path.isdir(d) else []
            print(f"  {module}/{mode:3s}: {files}")

    print(f"\n{'='*55}")
    print(f"PASS ({len(PASS)}): {PASS}")
    print(f"FAIL ({len(FAIL)}): {FAIL}")
    if FAIL:
        sys.exit(1)


if __name__ == '__main__':
    main()
