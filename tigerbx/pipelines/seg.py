import nibabel as nib
import numpy as np

from tigerbx import lib_bx
from tigerbx.modules.bx import run as bx_run
from tigerbx.core import cgw
from tigerbx.core.io import load_nifti, save_nib
from tigerbx.pipelines import _common

# FreeSurfer number of DGM12 label k (1..12): L/R thalamus, caudate, putamen, pallidum, hippocampus, amygdala
DGM12_FS = np.array([10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54])
# neighbour order of the hole fill (ties go to the first): -x, +x, -y, +y, -z, +z
_STEPS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))


def merge_dgm(tseg, dgm):
    """TSEG with its 12 deep-GM structures taken from DGM12 (more accurate there). A voxel DGM12 labels gets its
    FreeSurfer number; a voxel TSEG gave a deep-GM label that DGM12 does not keep is a hole, filled layer by layer
    from the 6-connected neighbours outside the deep-GM structures (ties: the first of -x, +x, -y, +y, -z, +z).
    Same rule as the executable (rust/src/bx.rs merge_dgm). Returns the labels; holes left over get 0."""
    tseg = np.asarray(tseg).astype(np.int32)
    dgm = np.asarray(dgm).astype(np.int32)
    keep = (dgm > 0) & (dgm <= len(DGM12_FS))
    out = np.where(keep, DGM12_FS[np.clip(dgm - 1, 0, len(DGM12_FS) - 1)], tseg)
    hole = ~keep & np.isin(tseg, DGM12_FS)
    source = ~hole & ~np.isin(out, DGM12_FS)
    while hole.any():
        fill = np.zeros_like(out)
        got = np.zeros(out.shape, bool)
        for step in _STEPS:
            here = tuple(slice(max(-s, 0), n - max(s, 0)) for s, n in zip(step, out.shape))   # voxels p
            there = tuple(slice(max(s, 0), n - max(-s, 0)) for s, n in zip(step, out.shape))  # their p + step
            take = hole[here] & ~got[here] & source[there]
            fill[here][take] = out[there][take]
            got[here] |= take
        if not got.any():
            break
        out[got] = fill[got]
        hole &= ~got
        source |= got
    out[hole] = 0
    return out


def seg(input=None, output=None, *, merge_dgm=True, save_tbx=True, save_tissue=False, gpu=False, gz=True,
        model=None, verbose=0):
    """Segmentation (executable ``--seg``): tBX -> TSEG (+ DGM12) -> ``_tseg`` (input grid; label numbers follow
    FreeSurfer, 99 = Unlabeled-Intracranial) and ``_tbx`` (``save_tbx``). With ``merge_dgm`` (default) the 12
    deep-GM structures come from DGM12, more accurate there (seg18 0.8178 vs 0.8129); ``merge_dgm=False`` keeps
    the TSEG model output (``tigerbx.run('t')``). ``save_tissue=True`` adds the CSF / GM / WM partial-volume maps
    computed from the T1 and this TSEG (``_pve_csf``, ``_pve_gm``, ``_pve_wm``; core.cgw, T1 only).

    model: {'tbx': ..., 'tseg': ..., 'dgm': ...} overrides. Returns the images dict for a single input, otherwise a
    list of files dicts.
    """
    def one(f, ftemplate):
        r = bx_run(('bmtd' if merge_dgm else 'bmt') + ('g' if gpu else ''), f, output=None, model=model, verbose=0,
                   save_outputs=False)
        tseg = r['tseg']
        mask = np.asarray(r['tbxmask'].dataobj) > 0
        if merge_dgm:
            lab = _merge_dgm(r['tseg'].dataobj, r['dgm'].dataobj)
            lab[mask & (lab == 0)] = lib_bx.TSEG_UNLABELED_INTRACRANIAL
            lab[~mask] = 0
            tseg = nib.Nifti1Image(lab.astype(np.uint8), r['tseg'].affine, r['tseg'].header)
            tseg.header.set_data_dtype(np.uint8)
        images, saved = {'tseg': tseg}, {'tseg': save_nib(tseg, ftemplate, 'tseg')}
        if save_tbx:
            images['tbx'], saved['tbx'] = r['tbx'], save_nib(r['tbx'], ftemplate, 'tbx')
        if save_tissue:
            img = load_nifti(f)
            pve = cgw.tissue_pve(np.squeeze(np.asarray(img.dataobj, np.float32)), tseg.dataobj, mask,
                                 img.header.get_zooms()[:3])
            for name, p in zip(cgw.NAMES, pve):
                key = f'pve_{name}'
                images[key] = nib.Nifti1Image(p, img.affine)
                saved[key] = save_nib(images[key], ftemplate, key)
        return images, saved

    return _common.batch('seg', input, output, gz, verbose, one)


_merge_dgm = merge_dgm      # seg's merge_dgm argument shadows the function
