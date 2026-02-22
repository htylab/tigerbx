"""
FreeSurfer full pipeline: recon-all → native-space aseg + wmparc NIfTI.

============================================================
 SETUP (one-time, on Linux/macOS only)
============================================================
1. Install system dependencies:
       sudo apt update && sudo apt install -y tcsh bc

2. Download & install FreeSurfer:
       https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
   Default install path: /usr/local/freesurfer

3. Run this script — it sets FREESURFER_HOME and PATH internally.
   No need to source environment.sh.

============================================================
 USAGE
============================================================
    python run_freesurfer.py \\
        --freesurfer-home /usr/local/freesurfer \\
        --data-dir      /data/T1_raw \\
        --recon-dir     /data/recon \\
        --output-dir    /data/nii_out \\
        --parallel      4 \\
        --omp-threads   4

    Total CPUs used ≈ --parallel × --omp-threads
    (e.g. 4 × 4 = 16 cores for a 16-core machine)

============================================================
 OUTPUT
============================================================
    <output_dir>/
    ├── aseg/     {subject}_aseg.nii.gz    ← subcortical labels, native space
    ├── wmparc/   {subject}_wmparc.nii.gz  ← white-matter parcellation, native space
    ├── recon_errors.txt                   ← subjects that failed recon-all (if any)
    └── convert_errors.txt                 ← subjects that failed NIfTI conversion (if any)
"""
import argparse
import glob
import logging
import os
import shutil
import subprocess
from multiprocessing import Pool
from os.path import abspath, basename, isdir, isfile, join
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TEMP_DIR = "./temp"


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def setup_freesurfer_env(freesurfer_home: str):
    """
    Set FreeSurfer environment variables in the current process so that
    all subprocess calls can find recon-all, mri_vol2vol, etc.
    """
    fs_home = abspath(freesurfer_home)
    if not isdir(fs_home):
        raise FileNotFoundError(
            f"FreeSurfer home not found: {fs_home}\n"
            "Download from: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall"
        )

    os.environ["FREESURFER_HOME"] = fs_home
    os.environ["SUBJECTS_DIR"] = os.environ.get("SUBJECTS_DIR", join(fs_home, "subjects"))
    os.environ["MNI_DIR"] = join(fs_home, "mni")
    os.environ["FSF_OUTPUT_FORMAT"] = "nii.gz"

    # Prepend FreeSurfer bin dirs to PATH
    bin_dirs = [
        join(fs_home, "bin"),
        join(fs_home, "mni", "bin"),
        join(fs_home, "tktools"),
    ]
    existing_path = os.environ.get("PATH", "")
    os.environ["PATH"] = ":".join(bin_dirs) + ":" + existing_path

    log.info(f"FreeSurfer home: {fs_home}")


# ---------------------------------------------------------------------------
# Shell helper
# ---------------------------------------------------------------------------

def run(cmd, check=True):
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed (rc={result.returncode}): {cmd}")


# ---------------------------------------------------------------------------
# Step 1: recon-all
# ---------------------------------------------------------------------------

def recon_all(args):
    ff, recon_dir, omp_threads = args
    name = basename(ff).replace("_T1w_raw.nii.gz", "").replace(".nii.gz", "")
    temp_path = join(TEMP_DIR, f"temp_{name}")
    done_path = join(TEMP_DIR, f"temp_{name}_done")
    error_path = join(TEMP_DIR, f"temp_{name}_error")
    completed_marker = join(recon_dir, name, "mri", "wmparc.mgz")

    if os.path.exists(done_path) or os.path.exists(completed_marker):
        return name, "skip"
    if os.path.exists(error_path):
        return name, "prev_error"

    try:
        os.makedirs(temp_path, exist_ok=True)
        run(f"recon-all -s {name} -i {ff} -sd {recon_dir} -all -openmp {omp_threads}")
        os.rename(temp_path, done_path)
        return name, "ok"
    except Exception as e:
        log.error(f"[recon-all] {name}: {e}")
        if os.path.exists(temp_path):
            os.rename(temp_path, error_path)
        return name, "error"


# ---------------------------------------------------------------------------
# Step 2: convert to native-space NIfTI
# ---------------------------------------------------------------------------

def convert(args):
    ff, output_dir = args
    name = basename(ff)
    rawavg   = join(ff, "mri", "rawavg.mgz")
    aseg_mgz = join(ff, "mri", "aseg.mgz")
    wmparc_mgz = join(ff, "mri", "wmparc.mgz")
    aseg_out   = join(output_dir, "aseg",   f"{name}_aseg.nii.gz")
    wmparc_out = join(output_dir, "wmparc", f"{name}_wmparc.nii.gz")

    for path in (rawavg, aseg_mgz, wmparc_mgz):
        if not isfile(path):
            log.warning(f"[convert] Missing {path}, skipping {name}")
            return name, "skip"

    try:
        if not isfile(aseg_out):
            run(
                f"mri_vol2vol"
                f" --mov {aseg_mgz} --targ {rawavg}"
                f" --o {aseg_out}"
                f" --regheader --interp nearest --no-save-reg"
            )
        if not isfile(wmparc_out):
            run(
                f"mri_vol2vol"
                f" --mov {wmparc_mgz} --targ {rawavg}"
                f" --o {wmparc_out}"
                f" --regheader --interp nearest --no-save-reg"
            )
        return name, "ok"
    except Exception as e:
        log.error(f"[convert] {name}: {e}")
        return name, "error"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(label, results, error_log_path):
    counts = {}
    errors = []
    for name, status in results:
        counts[status] = counts.get(status, 0) + 1
        if "error" in status:
            errors.append(name)
    log.info(f"{label} summary: {counts}")
    if errors:
        with open(error_log_path, "w") as f:
            f.write("\n".join(errors))
        log.warning(f"  {len(errors)} failed → {error_log_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="FreeSurfer pipeline: recon-all → native aseg + wmparc NIfTI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--freesurfer-home", default=os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer"),
                   help="Path to FreeSurfer installation (default: /usr/local/freesurfer or $FREESURFER_HOME)")
    p.add_argument("--data-dir",   required=True, help="Folder with input T1 .nii.gz files")
    p.add_argument("--recon-dir",  required=True, help="Output folder for recon-all results")
    p.add_argument("--output-dir", required=True, help="Output folder for NIfTI labels")
    p.add_argument("--parallel",      type=int, default=4, help="Number of parallel subjects (default: 4)")
    p.add_argument("--omp-threads",   type=int, default=4, help="OpenMP threads per recon-all (default: 4)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    setup_freesurfer_env(args.freesurfer_home)

    log.info(f"Parallel subjects : {args.parallel}")
    log.info(f"OpenMP per job    : {args.omp_threads}")
    log.info(f"Total est. CPUs   : {args.parallel * args.omp_threads}")

    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(args.recon_dir, exist_ok=True)
    os.makedirs(join(args.output_dir, "aseg"), exist_ok=True)
    os.makedirs(join(args.output_dir, "wmparc"), exist_ok=True)

    # Step 1: recon-all
    ffs = sorted(glob.glob(os.path.join(args.data_dir, "*.nii.gz")))
    log.info(f"\n=== Step 1: recon-all | {len(ffs)} subjects ===")
    with Pool(args.parallel) as p:
        recon_results = list(tqdm(
            p.imap(recon_all, [(ff, args.recon_dir, args.omp_threads) for ff in ffs]),
            total=len(ffs),
        ))
    summarize("recon-all", recon_results, join(args.output_dir, "recon_errors.txt"))

    # Step 2: convert
    subject_dirs = [
        ff for ff in sorted(glob.glob(os.path.join(args.recon_dir, "*")))
        if isdir(ff) and "fsaverage" not in basename(ff)
    ]
    log.info(f"\n=== Step 2: native NIfTI | {len(subject_dirs)} subjects ===")
    with Pool(args.parallel) as p:
        conv_results = list(tqdm(
            p.imap(convert, [(ff, args.output_dir) for ff in subject_dirs]),
            total=len(subject_dirs),
        ))
    summarize("convert", conv_results, join(args.output_dir, "convert_errors.txt"))

    if os.path.exists(TEMP_DIR) and not os.listdir(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    log.info("Done.")
