"""Shared plumbing of the pipelines (input batches, the brain mask); each pipeline composes its own steps."""
import logging
import os
import time
import warnings

import nibabel as nib
import numpy as np
from tqdm import tqdm

from tigerbx.core.io import get_template, load_nifti, resolve_inputs

_logger = logging.getLogger('tigerbx')
_logger.addHandler(logging.NullHandler())


def batch(name, input, output, gz, verbose, one):
    """Run `one(f, ftemplate) -> (images, files)` on every input. A single input returns its images dict (errors
    raise); several return the list of files dicts of the successful ones (a failure is logged and skipped)."""
    files, common_folder = resolve_inputs(input if isinstance(input, list) else [input])
    results, failed = [], []
    for f in tqdm(files, desc=f'tigerbx-{name}', unit='file', disable=(verbose > 0 or len(files) < 2)):
        t = time.time()
        try:
            ftemplate, _ = get_template(f, output, gz, common_folder)
            images, saved = one(f, ftemplate)
        except Exception as exc:                 # noqa: BLE001 - one bad input must not stop a batch
            if len(files) == 1:
                raise
            _logger.warning('FAILED %s: %s: %s', os.path.basename(f), type(exc).__name__, exc)
            failed.append(f'{f}: {type(exc).__name__}: {exc}')
            continue
        if verbose >= 1:
            _logger.info('%s: %.1f s', os.path.basename(f), time.time() - t)
        results.append(images if len(files) == 1 else saved)
    if failed:
        # a warning, not only the log: without a logging handler a skipped input would go unnoticed
        more = f'\n  ... and {len(failed) - 10} more' if len(failed) > 10 else ''
        warnings.warn(f'tigerbx.{name}: {len(failed)} of {len(files)} inputs failed and were skipped:\n  '
                      + '\n  '.join(failed[:10]) + more, RuntimeWarning, stacklevel=3)
    return results[0] if len(files) == 1 and results else results


def brain(f, mask=None, gpu=False, tbx_model=None):
    """(input, brain, mask, tbx) of input file f. mask None: tBX (tbx = its _tbx image); a NIfTI path or image: that
    brain mask (non-zero; resampled nearest when on another grid); 'none': the input is already skull-stripped
    (mask = non-zero voxels). brain = input x mask in float32 on the input grid; tbx is None unless tBX ran."""
    from tigerbx.modules.bx import run as bx_run
    from tigerbx.core.resample import resample_to_img
    img = load_nifti(f)
    data = np.squeeze(np.asarray(img.dataobj, dtype=np.float32))
    tbx = None
    if mask is None:
        r = bx_run('bm' + ('g' if gpu else ''), f, output=None, model={'tbx': tbx_model} if tbx_model else None,
                   verbose=0, save_outputs=False)
        m, tbx = np.asarray(r['tbxmask'].dataobj) > 0, r['tbx']
    elif isinstance(mask, str) and mask.lower() == 'none':
        m = data != 0
    else:
        mn = nib.load(mask) if isinstance(mask, (str, os.PathLike)) else mask
        if mn.shape[:3] != data.shape or not np.allclose(mn.affine, img.affine, atol=1e-4):
            mn = resample_to_img(mn, img, interpolation='nearest')
        m = np.squeeze(np.asarray(mn.dataobj)) > 0
    if not m.any():
        raise RuntimeError(f'empty brain mask for {os.path.basename(str(f))!r}')
    header = img.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1, 0)
    b = nib.Nifti1Image(np.where(m, data, 0).astype(np.float32), img.affine, header)
    return img, b, nib.Nifti1Image(m.astype(np.uint8), img.affine), tbx


INTENSITIES = ('raw', 'minmax', 'gnorm')


def check_intensity(intensity, template):
    if intensity not in INTENSITIES:
        raise ValueError(f'intensity must be one of {INTENSITIES}, got {intensity!r}')
    if intensity == 'gnorm' and template is not None:
        raise ValueError("intensity='gnorm' uses the tissue priors of the bundled template: use template=None")


def scale(img, mask, intensity, verbose=0):
    """Intensities of a template-grid brain image (float32): 'raw' as sampled (the input's units), 'minmax' to
    [0, 1], 'gnorm' tissue-normalised (core.tissuenorm.tissue_norm: CSF 1/3, GM 2/3, WM 1; T1 only) inside the
    brain `mask` (bool; resampling rings outside the brain)."""
    if intensity == 'raw':
        return np.asarray(img, np.float32)
    if intensity == 'minmax':
        from tigerbx.lib_reg import min_max_norm
        return np.asarray(min_max_norm(img), np.float32)
    from tigerbx.core.tissuenorm import tissue_norm
    y, peaks = tissue_norm(img, mask=mask)
    if verbose >= 1:
        _logger.info('gnorm peaks CSF %.4f GM %.4f WM %.4f', *peaks)
    return np.asarray(y, np.float32)
