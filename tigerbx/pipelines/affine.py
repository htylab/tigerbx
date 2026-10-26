import nibabel as nib

from tigerbx.core import numreg, templatereg
from tigerbx.core.io import save_nib
from tigerbx.core.warpfile import save_warp
from tigerbx.pipelines import _common


def affine(input=None, output=None, *, mask=None, template=None, intensity='raw', save_warp_file=True,
           gpu=False, gz=True, model=None, verbose=0):
    """Affine-only template normalisation (executable ``--affine``): brain mask -> numreg affine -> the brain on
    the template grid. Keeps each brain's size and shape (e.g. ventricle volume), which a nonlinear warp removes.
    Any contrast (numreg uses mutual information).

    Outputs per input: ``_tbx`` (input grid; when tBX ran), ``_affine`` (template grid: the brain sampled once from
    the input grid, cubic B-spline) and ``_warp.npz`` (core.warpfile, affine_M only; tigerbx.transform applies it).

    mask: None runs tBX; a NIfTI path / image is the brain mask (non-zero); ``'none'``: the input is already
    skull-stripped. template: registration template (default: the bundled 1 mm template). intensity: ``'raw'``
    (the input's units, default), ``'minmax'`` ([0, 1]) or ``'gnorm'`` (T1 only: CSF 1/3, GM 2/3, WM 1; bundled
    template). model: {'tbx': ...} override. Returns the images dict for a single input, otherwise a list of files
    dicts.
    """
    _common.check_intensity(intensity, template)
    tpl = templatereg.template(template)
    tbx_model = (model or {}).get('tbx')

    def one(f, ftemplate):
        img, brain, mask_nib, tbx = _common.brain(f, mask, gpu, tbx_model)
        M = templatereg.affine(brain, tpl)
        moved = numreg.resample(brain, tpl, M, order=3)
        inside = None
        if intensity == 'gnorm':
            inside = numreg.resample(mask_nib, tpl, M, order=1, data=mask_nib.get_fdata()) > 0.5
        out = nib.Nifti1Image(_common.scale(moved, inside, intensity, verbose), tpl.affine, tpl.header)
        images, saved = {'affine': out, 'affine_M': M}, {}
        if tbx is not None:
            images['tbx'], saved['tbx'] = tbx, save_nib(tbx, ftemplate, 'tbx')
        saved['affine'] = save_nib(out, ftemplate, 'affine')
        if save_warp_file:
            saved['warp'] = save_warp(ftemplate.split('@@@@')[0] + 'warp.npz', tpl.affine, tpl.shape, img.affine,
                                      img.shape, affine_M=M)
        return images, saved

    return _common.batch('affine', input, output, gz, verbose, one)
