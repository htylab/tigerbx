"""Shared IO helpers for TigerBx."""

import glob
import os
import time
import warnings
from os.path import basename, dirname, join, relpath

import nibabel as nib
import numpy as np


# Postfixes tigerbx itself writes or wrote (save_nib). A directory input that already contains outputs of an
# earlier run (the default output dir is the input folder) must not re-ingest them as new subjects; removed outputs
# (wmh / syn / cgw_pve* of bx flags W / S / C, the HLC CSF / GM / WM) stay listed so older outputs are still skipped.
OUTPUT_POSTFIXES = ('tbx', 'tbxmask', 'tseg', 'dgm', 'wmh', 'syn', 'ct', 'cgw_pve0', 'cgw_pve1', 'cgw_pve2',
                    'pve_csf', 'pve_gm', 'pve_wm',
                    'hlc', 'CSF', 'GM', 'WM', 'reg', 'rigid', 'Af', 'Fuse', 'SmoothedGM', 'ModulatedGM', 'gdm', 'gdmi',
                    'affine', 'VXM', 'VFA', 'norm', 'duope', 'duope_disp', 'duope_field')   # reg/Af/Fuse: names used before 0.3


def _stem(f):
    return basename(f).replace('.nii.gz', '').replace('.nii', '')


def is_tigerbx_output(f, stems):
    """True if *f* is ``<stem>_<postfix>`` for a known tigerbx postfix and ``<stem>`` is also in *stems*
    (so a genuine input such as ``patient_ct.nii.gz`` is kept unless ``patient.nii.gz`` sits next to it)."""
    name = _stem(f)
    return any(name.endswith('_' + p) and name[:-len(p) - 1] in stems for p in OUTPUT_POSTFIXES)


def resolve_nifti_inputs(input_arg):
    """Resolve an input argument (dir, glob, or file list) to a sorted list of NIfTI paths.

    Directory / glob inputs skip files that look like tigerbx outputs (e.g. ``X_tbx.nii.gz``) with a warning;
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

    ftemplate = join(f_output_dir, ftemplate)
    return ftemplate, f_output_dir
def save_nib(data_nib, ftemplate, postfix):
    output_file = ftemplate.replace("@@@@", postfix)
    nib.save(data_nib, output_file)
    return output_file


def save_nib_volumes(header, affine, shape, volumes, ftemplate, postfix):
    """save_nib of a float32 image given as an iterable of its 3D volumes (in order), written as they come: the
    whole image is never in memory. header: the source header (dim_info, units, ... kept)."""
    output_file = ftemplate.replace("@@@@", postfix)
    img = nib.Nifti1Image(np.broadcast_to(np.float32(0), tuple(shape)), affine, header)   # header only, no data
    img.update_header()
    hdr = img.header
    hdr.set_data_dtype('<f4')
    hdr.set_slope_inter(1, 0)
    if hdr.endianness != '<':
        hdr = hdr.as_byteswapped('<')
    hdr.set_data_offset(352)
    with nib.openers.Opener(output_file, 'wb') as fh:
        fh.write(hdr.binaryblock)
        fh.write(b'\0' * 4)                                        # no extensions
        for v in volumes:
            fh.write(np.asarray(v, '<f4').tobytes(order='F'))
    return output_file


def load_nifti(path):
    """nib.load of an input image, with non-finite voxels (NaN / Inf, e.g. outside the head of template-space
    images) set to 0 and a warning: they would otherwise spread through the intensity scaling and the resampling
    (a tBX mask over the whole volume). Same in the executable (nifti_io::read_nifti)."""
    img = nib.load(path)
    if img.get_data_dtype().kind != 'f':
        return img
    data = np.asarray(img.dataobj)
    bad = ~np.isfinite(data)
    if not bad.any():
        return img
    warnings.warn(f'{basename(str(path))}: {int(bad.sum())} non-finite voxels (NaN / Inf) set to 0', RuntimeWarning)
    header = img.header.copy()
    header.set_slope_inter(1, 0)
    header.set_data_dtype(np.float32)
    return img.__class__(np.where(bad, 0, data).astype(np.float32), img.affine, header)


def write_fail_log(path, f, msg):
    """Best-effort ``<stem>_*fail.log`` (time, input, error) for batch pipelines; never raises."""
    try:
        os.makedirs(dirname(path), exist_ok=True)
        with open(path, 'w') as fh:
            fh.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")}\n{f}\n{msg}\n')
    except Exception as exc:                     # noqa: BLE001 - logging a failure must not stop a batch
        warnings.warn(f'could not write {path}: {exc}')


def remove_quietly(path):
    """Remove *path* if it exists (e.g. a stale fail log); never raises."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        warnings.warn(f'could not remove {path}: {exc}')
