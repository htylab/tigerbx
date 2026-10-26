import nibabel as nib

from tigerbx.core import numreg, templatereg
from tigerbx.core.io import save_nib
from tigerbx.core.warpfile import save_warp
from tigerbx.pipelines import _common


def norm(input=None, output=None, *, mask=None, template=None, intensity='raw', save_affine=False,
         save_warp_file=True, gpu=False, gz=True, model=None, verbose=0):
    """Template normalisation (executable ``--norm``): brain mask -> numreg affine -> VFA -> the brain on the
    template grid, aligned voxel by voxel. VFA is a T1 model; other contrasts register less well.

    Outputs per input: ``_tbx`` (input grid; when tBX ran), ``_norm`` (template grid: the brain sampled once from
    the input grid at the affine + VFA warp composed, cubic B-spline), ``_warp.npz`` (core.warpfile: affine_M +
    vfa_warp; tigerbx.transform applies it to other images on the input grid) and, with ``save_affine=True``,
    ``_affine`` (as tigerbx.affine).

    mask: None runs tBX; a NIfTI path / image is the brain mask (non-zero); ``'none'``: the input is already
    skull-stripped. template: registration template (default: the bundled 1 mm template). intensity: ``'raw'`` (the input's units, default), ``'minmax'`` ([0, 1]) or ``'gnorm'`` (T1 only: CSF
    1/3, GM 2/3, WM 1; bundled template). model: {'tbx': ..., 'vfa': ...} overrides. Returns the images dict for a
    single input, otherwise a list of files dicts.
    """
    _common.check_intensity(intensity, template)
    tpl = templatereg.template(template)
    model = model or {}

    def one(f, ftemplate):
        img, brain, mask_nib, tbx = _common.brain(f, mask, gpu, model.get('tbx'))
        M = templatereg.affine(brain, tpl)
        warp = templatereg.vfa_warp(brain, tpl, M, model.get('vfa'), gpu)
        images, saved = {'affine_M': M, 'vfa_warp': warp}, {}
        if tbx is not None:
            images['tbx'], saved['tbx'] = tbx, save_nib(tbx, ftemplate, 'tbx')
        for name, w in (('affine', None), ('norm', warp)) if save_affine else (('norm', warp),):
            moved = numreg.resample(brain, tpl, M, warp=w, order=3)
            inside = None
            if intensity == 'gnorm':
                inside = numreg.resample(mask_nib, tpl, M, warp=w, order=1, data=mask_nib.get_fdata()) > 0.5
            images[name] = nib.Nifti1Image(_common.scale(moved, inside, intensity, verbose), tpl.affine, tpl.header)
            saved[name] = save_nib(images[name], ftemplate, name)
        if save_warp_file:
            saved['warp'] = save_warp(ftemplate.split('@@@@')[0] + 'warp.npz', tpl.affine, tpl.shape, img.affine,
                                      img.shape, affine_M=M, vfa_warp=warp)
        return images, saved

    return _common.batch('norm', input, output, gz, verbose, one)
