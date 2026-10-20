"""Shared IO helpers for TigerBx."""

import glob
import os
import warnings
from os.path import basename, dirname, join, relpath

import nibabel as nib
import numpy as np


# Postfixes tigerbx itself writes (save_nib / qc logs). A directory input that already contains outputs of an
# earlier run (the default output dir is the input folder) must not re-ingest them as new subjects.
OUTPUT_POSTFIXES = ('tbet', 'tbetmask', 'aseg', 'dgm', 'wmh', 'syn', 'ct', 'cgw_pve0', 'cgw_pve1', 'cgw_pve2',
                    'hlc', 'CSF', 'GM', 'WM', 'reg', 'rigid', 'Af', 'Fuse', 'SmoothedGM', 'gdm')


def _stem(f):
    return basename(f).replace('.nii.gz', '').replace('.nii', '')


def is_tigerbx_output(f, stems):
    """True if *f* is ``<stem>_<postfix>`` for a known tigerbx postfix and ``<stem>`` is also in *stems*
    (so a genuine input such as ``patient_ct.nii.gz`` is kept unless ``patient.nii.gz`` sits next to it)."""
    name = _stem(f)
    return any(name.endswith('_' + p) and name[:-len(p) - 1] in stems for p in OUTPUT_POSTFIXES)


def resolve_nifti_inputs(input_arg):
    """Resolve an input argument (dir, glob, or file list) to a sorted list of NIfTI paths.

    Directory / glob inputs skip files that look like tigerbx outputs (e.g. ``X_tbet.nii.gz``) with a warning;
    explicit file lists are used as given.
    """
    if not input_arg:
        return []
    if os.path.isdir(input_arg[0]):
        found = glob.glob(join(input_arg[0], '*.nii')) + glob.glob(join(input_arg[0], '*.nii.gz'))
    elif '*' in input_arg[0]:
        found = glob.glob(input_arg[0])
    else:
        return input_arg
    stems = {_stem(f) for f in found}
    skipped = [f for f in found if is_tigerbx_output(f, stems)]
    if skipped:
        warnings.warn(f'Skipping {len(skipped)} file(s) that look like tigerbx outputs '
                      f'(e.g. {basename(skipped[0])}); pass them explicitly to process them.', stacklevel=2)
    return sorted(f for f in found if f not in skipped)


def detect_common_folder(input_file_list):
    """Return commonpath when duplicate basenames exist, else None."""
    from os.path import commonpath
    base_ffs = [basename(f) for f in input_file_list]
    if len(base_ffs) != len(set(base_ffs)):
        return commonpath(input_file_list)
    return None


def resolve_inputs(input_arg):
    """Resolve input to file list and detect common folder for duplicate basenames."""
    input_file_list = resolve_nifti_inputs(input_arg)
    return input_file_list, detect_common_folder(input_file_list)


def _resolve_output_dir(output_dir, input_file):
    if output_dir is None:
        return os.path.dirname(os.path.abspath(input_file))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _apply_common_header(filename, input_file, common_folder):
    if common_folder is None:
        return filename
    header = relpath(dirname(input_file), common_folder).replace(os.sep, "_")
    return f"{header}_{filename}"


def get_template(f, output_dir, gz=None, common_folder=None):
    """Build output path template.

    gz=True/False: return '..._@@@@.nii[.gz]' pattern for save_nib.
    gz=None:       return stem (no extension) for custom suffix handling.
    """
    explicit_dir = output_dir is not None
    f_output_dir = _resolve_output_dir(output_dir, f)

    if gz is not None:
        ftemplate = basename(f).replace(".nii", "_@@@@.nii").replace(".npz", "_@@@@.nii.gz")
        if explicit_dir and common_folder is not None:
            ftemplate = _apply_common_header(ftemplate, f, common_folder)
        if gz and not ftemplate.endswith(".gz"):
            ftemplate += ".gz"
    else:
        ftemplate = basename(f)
        if explicit_dir and common_folder is not None:
            ftemplate = _apply_common_header(ftemplate, f, common_folder)
        ftemplate = ftemplate.replace(".nii.gz", "").replace(".nii", "")
        ftemplate = ftemplate.replace("_nerve.npz", "").replace(".npz", "")

    ftemplate = join(f_output_dir, ftemplate)
    return ftemplate, f_output_dir
def save_nib(data_nib, ftemplate, postfix):
    output_file = ftemplate.replace("@@@@", postfix)
    nib.save(data_nib, output_file)
    return output_file


def format_output_path(input_file, output_dir, postfix, suffix=".nii.gz"):
    base = basename(input_file).replace(".nii.gz", "").replace(".nii", "")
    return join(output_dir, f"{base}_{postfix}{suffix}")


def _pick_output_dtype(vol_out, reference_dtype):
    if np.size(vol_out) == 0 or not np.all(np.isfinite(vol_out)):
        return None

    data_min = np.min(vol_out)
    data_max = np.max(vol_out)
    if np.issubdtype(reference_dtype, np.integer):
        dtype_info = np.iinfo(reference_dtype)
        if data_min < dtype_info.min or data_max > dtype_info.max:
            return None
        return reference_dtype
    if np.issubdtype(reference_dtype, np.floating):
        dtype_info = np.finfo(reference_dtype)
        if data_min < dtype_info.min or data_max > dtype_info.max:
            return None
        return reference_dtype
    return reference_dtype


def set_output_zooms(result_img, reference_nib):
    out_ndim = len(result_img.shape)
    header_zooms = result_img.header.get_zooms()
    ref_zooms = reference_nib.header.get_zooms()
    out_zooms = tuple(ref_zooms[:out_ndim])
    if len(out_zooms) < out_ndim:
        out_zooms = out_zooms + tuple(header_zooms[len(out_zooms) : out_ndim])
    result_img.header.set_zooms(out_zooms)


def create_nifti_from_array(vol_out, reference_nib, dtype_override=None):
    dtype = dtype_override
    if dtype is None:
        dtype = _pick_output_dtype(vol_out, reference_nib.get_data_dtype())
    array = vol_out.astype(dtype) if dtype is not None else vol_out
    result = nib.Nifti1Image(array, reference_nib.affine)
    set_output_zooms(result, reference_nib)
    return result


def write_nifti_file(result_img, output_file, inmem=False):
    if not inmem:
        nib.save(result_img, output_file)
    return output_file, result_img


def write_gdm_nifti_like_input(input_file, output_dir, vol_out, inmem=False, postfix="gdmi", logger=None):
    if logger is None:
        class _NoopLogger:
            def warning(self, *args, **kwargs):
                return None
            def debug(self, *args, **kwargs):
                return None
        logger = _NoopLogger()

    if not os.path.isdir(output_dir):
        logger.warning("Output dir does not exist: %s", output_dir)
        return None, None

    output_file = format_output_path(input_file, output_dir, postfix, suffix=".nii.gz")
    logger.debug("Writing output file: %s", output_file)

    input_nib = nib.load(input_file)
    if postfix == "vdm":
        result = create_nifti_from_array(vol_out, input_nib, dtype_override=np.float32)
    else:
        result = create_nifti_from_array(vol_out, input_nib)

    return write_nifti_file(result, output_file, inmem=inmem)
