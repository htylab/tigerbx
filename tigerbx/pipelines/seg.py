import nibabel as nib
import numpy as np

from tigerbx.modules.bx import infer as bx_run
from tigerbx.core import cgw
from tigerbx.core.io import load_nifti, save_nib
from tigerbx.pipelines import _common


def seg(input=None, output=None, *, save_tbx=True, save_tissue=False, gpu=False, gz=True, model=None, resume=False,
        verbose=0):
    """Segmentation (executable ``--seg``): tBX -> TSEG -> ``_tseg`` (input grid; label numbers follow FreeSurfer)
    and ``_tbx`` (``save_tbx``). ``save_tissue=True`` adds the CSF / GM / WM partial-volume maps computed from the T1
    and this TSEG (``_pve_csf``, ``_pve_gm``, ``_pve_wm``; core.cgw, T1 only).

    model: {'tbx': ..., 'tseg': ...} overrides. resume: skip inputs whose ``_tseg`` is already there (continuing a
    batch; they are left out of the list, as ``tigerbx.run(continue_=True)``). Returns the images dict for a single
    input, otherwise a list of files dicts (a failing input is logged and skipped, a warning lists them).
    """
    def one(f, ftemplate):
        r = bx_run('bmt' + ('g' if gpu else ''), f, output=None, model=model, verbose=0, save_outputs=False)
        tseg = r['tseg']
        images, saved = {'tseg': tseg}, {'tseg': save_nib(tseg, ftemplate, 'tseg')}
        if save_tbx:
            images['tbx'], saved['tbx'] = r['tbx'], save_nib(r['tbx'], ftemplate, 'tbx')
        if save_tissue:
            img = load_nifti(f)
            mask = np.asarray(tseg.dataobj) > 0              # inside the labels: the CSF around the brain too
            pve = cgw.tissue_pve(np.squeeze(np.asarray(img.dataobj, np.float32)), tseg.dataobj, mask,
                                 img.header.get_zooms()[:3])
            for name, p in zip(cgw.NAMES, pve):
                key = f'pve_{name}'
                images[key] = nib.Nifti1Image(p, img.affine)
                saved[key] = save_nib(images[key], ftemplate, key)
        return images, saved

    return _common.batch('seg', input, output, gz, verbose, one, exists='tseg' if resume else None)
