import json
import logging
import os
import shutil
import tempfile
import time

import nibabel as nib
import numpy as np
from tqdm import tqdm

from tigerbx import lib_bx, lib_tool
from tigerbx.core import numreg, templatereg
from tigerbx.core.io import load_nifti, resolve_inputs
from tigerbx.core.onnx import create_session
from tigerbx.modules.bx import run as bx_run
from tigerbx.pipelines import _common
from tigerbx.pipelines.seg import merge_dgm

_logger = logging.getLogger('tigerbx')
_logger.addHandler(logging.NullHandler())


def template(input=None, output=None, *, contrast='t1', iters=6, stat='median', mask=None, labels='auto',
             gpu=False, gz=True, model=None, verbose=0):
    """Build a population template from a group of images (any contrast) on the tigerbx template grid (1 mm,
    160 x 224 x 192, the bundled template's frame), so it can be used as ``template=`` of tigerbx.affine / norm.

    Each image: brain mask (tBX; ``mask='none'`` for skull-stripped inputs), numreg affine to the bundled template
    (mutual information: any contrast), intensity / brain median. Then
    ``iters`` rounds: VFA-register every subject to the current template, take the voxel-wise ``stat`` (``'median'``
    or ``'mean'``) of the warped images, and move it by the inverse of the mean displacement so the template stays
    in the mean frame. The last round samples every subject once from its input grid (affine + warp, cubic).

    Outputs in ``output`` (a directory): ``template.nii.gz`` (brain median scaled to the bundled template's),
    ``template_mask.nii.gz`` (majority of the subjects' brain masks), with ``contrast='t1'`` (``labels='auto'``)
    ``template_tseg.nii.gz`` (majority vote of the subjects' TSEG, deep GM from DGM12, inside the mask; so
    ``template=`` also finds its labels), and ``template_log.json``. VFA is a T1 model: other contrasts build,
    but a T2 template did not register T2 better than the bundled T1 template. Returns a dict of the template images.
    """
    if stat not in ('median', 'mean'):
        raise ValueError(f"stat must be 'median' or 'mean', got {stat!r}")
    if iters < 1:
        raise ValueError('iters must be >= 1')
    if mask is not None and not (isinstance(mask, str) and mask.lower() == 'none'):
        raise ValueError("template: mask must be None (tBX) or 'none' (skull-stripped inputs)")
    want_labels = (contrast.lower() == 't1') if labels == 'auto' else bool(labels)
    files, _ = resolve_inputs(input if isinstance(input, list) else [input])
    if len(files) < 2:
        raise ValueError(f'a template needs at least 2 inputs, got {len(files)}')
    os.makedirs(output, exist_ok=True)
    model = model or {}
    tpl = templatereg.template(None)
    shape = tpl.shape[:3]
    log = dict(n_inputs=len(files), contrast=contrast, iters=iters, stat=stat, labels=want_labels, rounds=[])
    tmp = tempfile.mkdtemp(prefix='.tigerbx_template_', dir=output)
    t0 = time.time()
    try:
        subs = _prepare(files, tpl, tmp, mask, want_labels, gpu, model, verbose, log)
        session = create_session(lib_tool.get_model(model.get('vfa') or templatereg.VFA_MODEL), gpu)
        cur = _stat(np.stack([s['affined'] for s in subs]), stat)            # round 0: affine-only average
        for k in range(1, iters + 1):
            last = k == iters
            warped, masks, tsegs, mean_w = [], [], [], np.zeros((3,) + shape, np.float32)
            for s in tqdm(subs, desc=f'tigerbx-template {k}/{iters}', unit='file', disable=verbose > 0):
                w = templatereg.vfa_grid(s['vfa_in'], tpl, fixed=cur, session=session) - templatereg.identity_grid(shape)
                mean_w += w / len(subs)
                if last:              # one cubic sample from the input grid; masks / labels from it too
                    brain = nib.load(s['brain'])
                    warped.append((numreg.resample(brain, tpl, s['M'], warp=w, order=3) / s['scale']).astype(np.float16))
                    m = nib.load(s['mask'])
                    masks.append(numreg.resample(m, tpl, s['M'], warp=w, order=1, data=m.get_fdata()) > 0.5)
                    if want_labels:
                        t = nib.load(s['tseg'])
                        tsegs.append(numreg.resample(t, tpl, s['M'], warp=w, order=0,
                                                     data=np.asarray(t.dataobj)).astype(np.uint8))
                else:
                    warped.append(templatereg.warp_on_grid(s['affined'], w).astype(np.float16))
            avg = _stat(np.stack(warped), stat)
            inv = templatereg.invert_warp(mean_w)
            prev, cur = cur, templatereg.warp_on_grid(avg, inv, order=3 if last else 1)
            r = dict(round=k, mean_disp_vox=float(np.sqrt((mean_w ** 2).sum(0)).mean()),
                     corr_prev=float(np.corrcoef(prev.ravel(), cur.ravel())[0, 1]), sec=round(time.time() - t0, 1))
            log['rounds'].append(r)
            if verbose >= 1:
                _logger.info('template round %d: mean displacement %.3f vox, corr to previous %.4f', k,
                             r['mean_disp_vox'], r['corr_prev'])
        brain_mask = templatereg.warp_on_grid(np.mean(np.stack(masks), 0), inv) > 0.5
        out = np.where(brain_mask, np.clip(cur, 0, None), 0).astype(np.float32)
        ref = np.asarray(tpl.dataobj, np.float32)
        out *= np.median(ref[ref > 0]) / max(float(np.median(out[brain_mask])), 1e-6)
        res = {'template': nib.Nifti1Image(out, tpl.affine), 'mask': nib.Nifti1Image(brain_mask.astype(np.uint8), tpl.affine)}
        if want_labels:
            lab = _vote(np.stack(tsegs), np.mean(np.stack(masks), 0) > 0.5)
            lab = templatereg.warp_on_grid(lab.astype(np.float32), inv, order=0).astype(np.uint8)
            lab[~brain_mask] = 0
            res['tseg'] = nib.Nifti1Image(lab, tpl.affine)
        ext = '.nii.gz' if gz else '.nii'
        for key, name in (('template', 'template'), ('mask', 'template_mask'), ('tseg', 'template_tseg')):
            if key in res:
                nib.save(res[key], os.path.join(output, name + ext))
        log['sec'] = round(time.time() - t0, 1)
        with open(os.path.join(output, 'template_log.json'), 'w') as fh:
            json.dump(log, fh, indent=1)
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _prepare(files, tpl, tmp, mask, want_labels, gpu, model, verbose, log):
    """Per input: brain / mask (/ TSEG) on the input grid in `tmp`, numreg affine M, the VFA input and the affine
    image on the template grid (intensity / brain median)."""
    subs, failed = [], []
    for i, f in enumerate(tqdm(files, desc='tigerbx-template prep', unit='file', disable=verbose > 0)):
        try:
            if want_labels and mask is None:          # one call for tBX + TSEG + DGM12 (the seg pipeline's labels)
                r = bx_run('bmtd' + ('g' if gpu else ''), f, output=None, model=model or None, verbose=0,
                           save_outputs=False)
                m = np.asarray(r['tbxmask'].dataobj) > 0
                lab = merge_dgm(r['tseg'].dataobj, r['dgm'].dataobj)
                lab[m & (lab == 0)] = lib_bx.TSEG_UNLABELED_INTRACRANIAL
                lab[~m] = 0
                img = load_nifti(f)
                data = np.squeeze(np.asarray(img.dataobj, np.float32))
                brain = nib.Nifti1Image(np.where(m, data, 0).astype(np.float32), img.affine)
                mask_nib = nib.Nifti1Image(m.astype(np.uint8), img.affine)
                tseg = nib.Nifti1Image(lab.astype(np.uint8), img.affine)
            else:
                img, brain, mask_nib, _ = _common.brain(f, mask, gpu, model.get('tbx'))
                tseg = None
                if want_labels:
                    r = bx_run('t' + ('g' if gpu else ''), f, output=None, model=model or None, verbose=0,
                               save_outputs=False)
                    tseg = nib.Nifti1Image(np.asarray(r['tseg'].dataobj).astype(np.uint8), img.affine)
            d = np.asarray(brain.dataobj)
            scale = float(np.median(d[d > 0]))
            M = templatereg.affine(brain, tpl)
            s = dict(file=f, M=M, scale=scale, brain=os.path.join(tmp, f'{i}_brain.nii'),
                     mask=os.path.join(tmp, f'{i}_mask.nii'))
            nib.save(brain, s['brain'])
            nib.save(mask_nib, s['mask'])
            if tseg is not None:
                s['tseg'] = os.path.join(tmp, f'{i}_tseg.nii')
                nib.save(tseg, s['tseg'])
            s['vfa_in'] = templatereg.vfa_input(brain, tpl, M)
            s['affined'] = (numreg.resample(brain, tpl, M, order=1) / scale).astype(np.float16)
            subs.append(s)
        except Exception as exc:                     # noqa: BLE001 - one bad input must not stop the template
            _logger.warning('FAILED %s: %s: %s', os.path.basename(f), type(exc).__name__, exc)
            failed.append(os.path.basename(f))
    log['n_used'], log['failed'] = len(subs), failed
    if len(subs) < 2:
        raise RuntimeError(f'only {len(subs)} input(s) could be prepared')
    return subs


def _stat(stack, stat, slab=16):
    """Voxel-wise median / mean of a (N, X, Y, Z) stack, in slabs along X (float32 out)."""
    out = np.empty(stack.shape[1:], np.float32)
    for i in range(0, stack.shape[1], slab):
        x = stack[:, i:i + slab].astype(np.float32)
        out[i:i + slab] = np.median(x, 0) if stat == 'median' else x.mean(0)
    return out


def _vote(stack, mask, slab=16):
    """Majority label per voxel among the non-zero labels (inside `mask`; a subject missing the region does not vote
    for background there)."""
    labels = [int(v) for v in np.unique(stack) if v > 0]
    out = np.zeros(stack.shape[1:], np.uint8)
    for i in range(0, stack.shape[1], slab):
        x = stack[:, i:i + slab]
        counts = np.stack([(x == v).sum(0) for v in labels])
        win = np.asarray(labels, np.uint8)[counts.argmax(0)]
        out[i:i + slab] = np.where(mask[i:i + slab] & (counts.max(0) > 0), win, 0)
    return out
