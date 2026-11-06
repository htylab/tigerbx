from os.path import basename
import warnings
import numpy as np
import nibabel as nib
from tigerbx.core.onnx import create_session, has_labels_output, predict, predict_labels
from tigerbx.core.io import load_nifti
from tigerbx.core.resample import reorder_img, resample_voxel

from tigerbx import lib_tool

label_all = dict()
label_all['tseg'] = (2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,30,31,41,42,43,
                44,46,47,49,50,51,52,53,54,58,60,62,63,77,85,251,252,253,254,255)
# whole-brain segmentation (models.TSEG): label number of model channel k (also the model's 'labels' metadata).
# Channel 98 (dura, venous sinuses, vessels) is written as 0, as SynthSeg / FreeSurfer have no such label.
label_all['tseg_v1'] = (0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47,
                       49, 50, 51, 52, 53, 54, 58, 60, 0)
label_all['dkt'] = ( 1002, 1003,
               1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015,
               1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026,
               1027, 1028, 1029, 1030, 1031, 1034, 1035, 2002, 2003, 2005, 2006,
               2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
               2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028,
               2029, 2030, 2031, 2034, 2035)

label_all['wmp'] = (  251,  252,  253,  254,  255, 3001, 3002, 3003, 3005, 3006, 3007,
                     3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018,
                     3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029,
                     3030, 3031, 3032, 3033, 3034, 3035, 4001, 4002, 4003, 4005, 4006,
                     4007, 4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017,
                     4018, 4019, 4020, 4021, 4022, 4023, 4024, 4025, 4026, 4027, 4028,
                     4029, 4030, 4031, 4032, 4033, 4034, 4035)


SEG_MODES = ('tbx', 'tbxdual', 'tseg', 'dkt', 'dgm12', 'wmp', 'seg3')


def get_mode(model_ff):
    """Parse ``<prefix>_<mode>_<version>_<tag>.onnx``; <mode> selects normalisation, softmax/sigmoid and the label
    LUT, so a wrong name silently changes preprocessing (see doc/models.md)."""
    parts = basename(model_ff).split('_')
    if len(parts) < 4 or parts[1] not in SEG_MODES:
        raise ValueError(
            f'Model filename {basename(model_ff)!r} must look like <prefix>_<mode>_<version>_<tag>.onnx '
            f'with <mode> in {SEG_MODES}, e.g. mprage_tseg_v0011_opt.onnx (see doc/models.md).')
    seg_mode, version, model_str = parts[1:4]
    return seg_mode, version, model_str

def getLarea(input_mask):
    from scipy import ndimage
    labeled_mask, cc_num = ndimage.label(input_mask)
    if cc_num > 0:
        mask = (labeled_mask == (np.bincount(
            labeled_mask.flat)[1:].argmax() + 1))
    else:
        mask = input_mask
    return mask

def _pad16(x):
    return np.pad(x, [(0, 0), (0, 0)] + [(0, (-n) % 16) for n in x.shape[-3:]])


def _cpu_if_gpu_fails(fn, model_ff, session, shape):
    """fn(session) on the GPU session; if it fails there, again on a CPU session of the same model. A large input
    (an oblique image resampled onto a RAS grid: 40+ M voxels) overflows the int32 index of the CUDA Concat kernels;
    the CPU runs it, with the same result (about 20 GB at the peak)."""
    if session is None or 'CUDAExecutionProvider' not in session.get_providers():
        return fn(session)
    try:
        return fn(session)
    except Exception as exc:                                                # noqa: BLE001 - ORT raises its own types
        warnings.warn(f'{basename(model_ff)} failed on the GPU for a {"x".join(map(str, shape))} input '
                      f'({type(exc).__name__}); running it on the CPU', RuntimeWarning, stacklevel=3)
    return fn(create_session(model_ff, False))


def _tbxdual_labels(model_ff, image, GPU, patch, session):
    """Dual brain mask model (mode tbxdual, two independent sigmoid channels: outer = brain with the surrounding CSF,
    inner = brain without it): 0 background, 1 outer only, 2 inner. The input is padded at the end of each axis to a
    multiple of 16; outer and inner each keep their largest connected component, the inner one inside the outer."""
    shape = image.shape[-3:]
    image = _pad16(image)
    if patch:
        logits = predict(model_ff, image, GPU, mode='patch', session=session)[0]
        outer = logits[0] >= 0
        lab = outer.astype(np.uint8) + (outer & (logits[1] >= 0))
    else:
        def labels(sess):
            if has_labels_output(sess):
                return predict_labels(sess, image)[0]
            logits = predict(model_ff, image, GPU, session=sess)[0]
            outer = logits[0] >= 0
            return outer.astype(np.uint8) + (outer & (logits[1] >= 0))
        lab = _cpu_if_gpu_fails(labels, model_ff, session, image.shape[-3:])
    lab = np.asarray(lab).reshape(image.shape[-3:])[:shape[0], :shape[1], :shape[2]]
    outer = getLarea(lab >= 1)
    inner = getLarea((lab == 2) & outer)
    return outer.astype(np.uint8) + inner.astype(np.uint8)


def run(model_ff, input_nib, GPU, patch=False, session=None):

    seg_mode, _, model_str = get_mode(model_ff)

    data = lib_tool.read_nib(input_nib)

    image = data[None, ...][None, ...]
    mx = np.max(image)
    if mx > 0:
        image = image / mx

    # Only the label of each voxel is needed: tBX thresholds sigmoid(logit) at 0.5, i.e. logit >= 0, and the
    # segmentation models take the argmax, which softmax does not change. A model with a ``labels`` output
    # (added when the model is optimised) computes that itself, so the logits never leave ONNX Runtime. Patch
    # inference blends logits across patches, so it always uses them.
    if not patch and session is None:
        session = create_session(model_ff, GPU)
    if seg_mode == 'tbxdual':                  # scaled in float32, as the model's reference inference
        image = np.nan_to_num(data.astype(np.float32))[None, None]
        if image.max() > 0:
            image = image / image.max()
        return nib.Nifti1Image(_tbxdual_labels(model_ff, image, GPU, patch, session).astype(int),
                               input_nib.affine, input_nib.header)
    if patch:
        logits = predict(model_ff, image, GPU, mode='patch', session=session)[0, ...]
        mask_pred = (logits[0] >= 0) if seg_mode == 'tbx' else np.argmax(logits, axis=0)
    else:
        def labels(sess):
            if has_labels_output(sess):
                return predict_labels(sess, image)[0]
            logits = predict(model_ff, image, GPU, session=sess)[0, ...]
            return (logits[0] >= 0) if seg_mode == 'tbx' else np.argmax(logits, axis=0)
        mask_pred = _cpu_if_gpu_fails(labels, model_ff, session, image.shape[-3:])
    if seg_mode == 'tbx':
        mask_pred = getLarea(mask_pred)

    if seg_mode in ['tseg', 'dkt', 'wmp']:
        labels = label_all[seg_mode]
        lut = np.zeros(len(labels) + 2, dtype=np.int32)
        for ii, lbl in enumerate(labels):
            lut[ii + 1] = lbl
        mask_pred = lut[mask_pred]



    mask_pred = mask_pred.astype(int)


    output_nib = nib.Nifti1Image(
        mask_pred, input_nib.affine, input_nib.header)

    return output_nib



def read_tbx_input(input_file, input_nib=None):
    if input_nib is None:
        input_nib = load_nifti(input_file)
    zoom = input_nib.header.get_zooms()[:3]
    needs_1mm = (max(zoom) > 1.1 or min(zoom) < 0.9)

    if needs_1mm:
        input_nib = resample_voxel(input_nib, (1, 1, 1), interpolation='continuous')
    return reorder_img(input_nib, resample='continuous')


def tseg(session, input_nib, mask_nib):
    """Whole-brain segmentation (models.TSEG) on the input grid, uint8 FreeSurfer numbers, zero outside `mask_nib` (the
    tBX outer mask). The input is the brain inside that mask, resampled to 1 mm only when a voxel size is outside 0.9-1.1 mm and reordered (read_tbx_input),
    cropped to its non-zero voxels + 16, scaled by the crop's maximum and padded at the end of each axis to a multiple
    of 16; the labels go back to the input grid (nearest)."""
    from tigerbx.core.resample import resample_to_img
    x = np.nan_to_num(np.asarray(input_nib.dataobj, np.float32).squeeze(), nan=0, posinf=0, neginf=0)
    x = x * (np.asarray(mask_nib.dataobj) > 0)
    g = read_tbx_input(None, input_nib=nib.Nifti1Image(x, input_nib.affine, input_nib.header))
    v = g.get_fdata().astype(np.float32)
    nz = np.nonzero(v > 0)
    lo = [max(0, int(t.min()) - 16) for t in nz]
    hi = [min(n, int(t.max()) + 17) for n, t in zip(v.shape, nz)]
    v = v[tuple(slice(p, q) for p, q in zip(lo, hi))]
    v = np.clip(v / v.max(), 0, 1)
    xp = np.pad(v, [(0, (-n) % 16) for n in v.shape])[None, None]
    if has_labels_output(session):
        idx = predict_labels(session, xp)[0]
    else:                                      # GPU session: the labels output is removed from the graph
        idx = np.argmax(predict(None, xp, True, session=session)[0], axis=0)
    idx = np.asarray(idx).reshape(xp.shape[2:])[:v.shape[0], :v.shape[1], :v.shape[2]]
    lab = np.asarray(label_all['tseg_v1'], np.uint8)[idx]
    aff = g.affine.copy()
    aff[:3, 3] = g.affine[:3, :3] @ np.array(lo) + g.affine[:3, 3]
    out = resample_to_img(nib.Nifti1Image(lab, aff), input_nib, interpolation='nearest')
    out = np.asarray(out.dataobj).astype(np.uint8) * (np.asarray(mask_nib.dataobj) > 0)   # nothing outside the mask
    out = nib.Nifti1Image(out.astype(np.uint8), input_nib.affine, input_nib.header)
    out.header.set_data_dtype(np.uint8)
    return out
