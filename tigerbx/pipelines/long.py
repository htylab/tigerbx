import csv
import os

import nibabel as nib
import numpy as np
from scipy import linalg, ndimage

from tigerbx.core import models, numreg, templatereg
from tigerbx.core.io import get_template, save_nib
from tigerbx.pipelines import _common

# TSEG regions reported (FreeSurfer numbers); 'brain' = every TSEG label except CSF and the ventricles
REGIONS = {
    'brain': None,
    'gm_cortex': [3, 42],
    'wm': [2, 41],
    'ventricles': [4, 5, 14, 15, 43, 44],
    'hippocampus': [17, 53],
    'amygdala': [18, 54],
    'thalamus': [10, 49],
    'caudate': [11, 50],
    'putamen': [12, 51],
    'cerebellum': [7, 8, 46, 47],
}
CSF = [4, 5, 14, 15, 24, 43, 44]
COLUMNS = ('region', 'change_pct', 'tseg_change_pct', 'tseg_baseline_ml', 'tseg_followup_ml')


def _region_masks(tseg):
    return {k: (tseg > 0) & ~np.isin(tseg, CSF) if labs is None else np.isin(tseg, labs) for k, labs in REGIONS.items()}


def _halfway(R):
    """Real square root of a rigid 4 x 4 world matrix."""
    H = np.real(linalg.sqrtm(R))
    H[3] = [0, 0, 0, 1]
    return H


def long(baseline=None, followup=None, output=None, *, refine=True, save_logjac=True, gpu=False, gz=True,
         model=None, verbose=0):
    """Longitudinal change between two T1 scans of one head (research use): per-region volume change (%) of the
    follow-up relative to the baseline, from the deformation between them.

    Steps (symmetric in the two scans): tBX + TSEG of each; rigid follow-up -> baseline (numreg, heads inside the
    tBX outer masks) and its halfway point, so both scans are resampled once (cubic) onto the template grid at the
    same distance from their own grids; tReg both ways, each refined (core.refine: local CC, 20 iterations, no
    folding; ``refine=True``); per region
    change % = 100 (sqrt(r1 r2) - 1) with r1 = the follow-up region volume over its baseline volume through the
    follow-up -> baseline deformation, r2 the same through the other one.

    Outputs per pair (named after the follow-up): ``_long.csv`` (region, change_pct, tseg_change_pct = the
    difference of the two TSEG volumes, tseg_baseline_ml, tseg_followup_ml) and, with ``save_logjac=True``,
    ``_logjac`` (template grid: log of the local volume ratio follow-up / baseline, symmetric).

    baseline, followup: file paths, or two lists of the same length (pairs in order). model: {'tbx', 'tseg',
    'treg', 'refine'} overrides. Returns the result dict for a single pair (regions, logjac), otherwise a list of
    files dicts.
    """
    pairs = list(zip(baseline, followup)) if isinstance(baseline, (list, tuple)) else [(baseline, followup)]
    if any(a is None or b is None for a, b in pairs) or (isinstance(baseline, (list, tuple)) and
                                                        len(baseline) != len(followup)):
        raise ValueError('baseline and followup: two files, or two lists of the same length')
    model = model or {}
    tpl = templatereg.template()
    results = []
    for fa, fb in pairs:
        ftemplate, _ = get_template(fb, output, gz)
        r = _pair(fa, fb, tpl, refine, gpu, model, verbose)
        saved = {'csv': ftemplate.replace('@@@@', 'long').split('.nii')[0] + '.csv'}
        os.makedirs(os.path.dirname(saved['csv']) or '.', exist_ok=True)
        with open(saved['csv'], 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(COLUMNS)
            for k, v in r['regions'].items():
                w.writerow([k] + [round(v[c], 4) for c in COLUMNS[1:]])
        if save_logjac:
            saved['logjac'] = save_nib(r['logjac'], ftemplate, 'logjac')
        results.append(r if len(pairs) == 1 else saved)
    return results[0] if len(pairs) == 1 else results


def _pair(fa, fb, tpl, refine, gpu, model, verbose):
    from tigerbx.core.onnx import create_session
    from tigerbx.core.refine import refine as refine_grid
    from tigerbx.lib_reg import min_max_norm
    from tigerbx.lib_tool import get_model
    from tigerbx.modules.bx import infer as bx_run
    seg_model = {k: model[k] for k in ('tbx', 'tseg') if k in model} or None
    sa, sb = (bx_run('bmxt' + ('g' if gpu else ''), f, output=None, model=seg_model, verbose=0, save_outputs=False)
              for f in (fa, fb))
    ia, ib = (nib.load(f) for f in (fa, fb))
    # rigid B -> A on the outer-masked heads (brain + CSF: the boundary is in the image, not the mask)
    oa = np.squeeze(np.asarray(ia.dataobj, np.float32)) * (np.asarray(sa['tbxoutermask'].dataobj) > 0)
    ob = np.squeeze(np.asarray(ib.dataobj, np.float32)) * (np.asarray(sb['tbxoutermask'].dataobj) > 0)
    R = numreg.register(ib, ia, moving_data=ob, fixed_data=oa, dof=6)            # A world -> B world
    H = _halfway(R)
    T = H @ templatereg.affine(sa['tbx'], tpl)                                   # template world -> halfway
    MA, MB = np.linalg.inv(H) @ T, H @ T
    u = ndimage.binary_dilation((numreg.apply(sa['tbxoutermask'], tpl, MA, order=0) > 0) |
                                (numreg.apply(sb['tbxoutermask'], tpl, MB, order=0) > 0), iterations=2)
    A = min_max_norm(numreg.resample(ia, tpl, MA, order=3, data=np.squeeze(ia.dataobj)) * u).astype(np.float32)
    B = min_max_norm(numreg.resample(ib, tpl, MB, order=3, data=np.squeeze(ib.dataobj)) * u).astype(np.float32)
    sess = create_session(get_model(templatereg.treg_model(model.get('treg'), gpu)), gpu)
    gAB = templatereg.treg_grid(A, tpl, gpu=gpu, fixed=B, session=sess)          # A(gAB(p)) ~ B(p)
    gBA = templatereg.treg_grid(B, tpl, gpu=gpu, fixed=A, session=sess)
    if refine:
        rs = create_session(get_model(model.get('refine') or models.REFINE), gpu)
        gAB, gBA = refine_grid(A, B, gAB, session=rs), refine_grid(B, A, gBA, session=rs)
    idg = templatereg.identity_grid(A.shape)
    jAB = templatereg.jacobian_det(np.asarray(gAB, np.float64) - idg)
    jBA = templatereg.jacobian_det(np.asarray(gBA, np.float64) - idg)
    ma = _region_masks(numreg.apply(sa['tseg'], tpl, MA, order=0).astype(int))
    mb = _region_masks(numreg.apply(sb['tseg'], tpl, MB, order=0).astype(int))
    ra, rb = (_region_masks(np.asarray(s['tseg'].dataobj).astype(int)) for s in (sa, sb))
    va, vb = (float(np.prod(s['tseg'].header.get_zooms()[:3])) / 1000 for s in (sa, sb))
    regions = {}
    for k in REGIONS:
        r1 = mb[k].sum() / max(jAB[mb[k]].sum(), 1e-9)
        r2 = jBA[ma[k]].sum() / max(ma[k].sum(), 1e-9)
        vola, volb = ra[k].sum() * va, rb[k].sum() * vb
        regions[k] = {'change_pct': float(100 * (np.sqrt(r1 * r2) - 1)) if ma[k].any() and mb[k].any() else np.nan,
                      'tseg_change_pct': float(100 * (volb / vola - 1)) if vola else np.nan,
                      'tseg_baseline_ml': float(vola), 'tseg_followup_ml': float(volb)}
    L = (np.log(np.clip(jBA, 1e-6, None)) - np.log(np.clip(jAB, 1e-6, None))) / 2
    if verbose >= 1:
        _common._logger.info('%s: brain %+.3f %%, folded %d', os.path.basename(fb), regions['brain']['change_pct'],
                             int((jAB <= 0).sum() + (jBA <= 0).sum()))
    return {'regions': regions, 'logjac': nib.Nifti1Image(L.astype(np.float32), tpl.affine), 'rigid': R}
