from os.path import basename
import numpy as np
import nibabel as nib
from tigerbx.core.onnx import create_session, has_labels_output, predict, predict_labels
from tigerbx.core.resample import reorder_img, resample_voxel

from tigerbx import lib_tool

label_all = dict()
label_all['aseg43'] = (2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,30,31,41,42,43,
                44,46,47,49,50,51,52,53,54,58,60,62,63,77,85,251,252,253,254,255)
# aseg output only (not a model channel): voxels inside the tBET mask that no aseg structure claims (sulcal /
# extra-cerebral CSF, pial partial volume, meninges, vessels), so that TIV = (aseg > 0).sum() * voxel volume
ASEG_UNLABELED_INTRACRANIAL = 99
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

#nib.Nifti1Header.quaternion_threshold = -100


SEG_MODES = ('bet', 'aseg43', 'dkt', 'dgm12', 'wmp', 'seg3')


def get_mode(model_ff):
    """Parse ``<prefix>_<mode>_<version>_<tag>.onnx``; <mode> selects normalisation, softmax/sigmoid and the label
    LUT, so a wrong name silently changes preprocessing (see doc/models.md)."""
    parts = basename(model_ff).split('_')
    if len(parts) < 4 or parts[1] not in SEG_MODES:
        raise ValueError(
            f'Model filename {basename(model_ff)!r} must look like <prefix>_<mode>_<version>_<tag>.onnx '
            f'with <mode> in {SEG_MODES}, e.g. mprage_aseg43_v0011_allseg.onnx (see doc/models.md).')
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

def run(model_ff, input_nib, GPU, patch=False, session=None):

    seg_mode, _, model_str = get_mode(model_ff)

    data = lib_tool.read_nib(input_nib)

    image = data[None, ...][None, ...]
    mx = np.max(image)
    if mx > 0:
        image = image / mx

    # Only the label of each voxel is needed: tBET thresholds sigmoid(logit) at 0.5, i.e. logit >= 0, and the
    # segmentation models take the argmax, which softmax does not change. A model with a ``labels`` output
    # (tigerbx.dev.onnx_optimize --labels) computes that itself, so the logits never leave ONNX Runtime. Patch
    # inference blends logits across patches, so it always uses them.
    if not patch and session is None:
        session = create_session(model_ff, GPU)
    if not patch and has_labels_output(session):
        mask_pred = predict_labels(session, image)[0]
    else:
        if patch:
            logits = predict(model_ff, image, GPU, mode='patch', session=session)[0, ...]
        else:
            logits = predict(model_ff, image, GPU, session=session)[0, ...]
        mask_pred = (logits[0] >= 0) if seg_mode == 'bet' else np.argmax(logits, axis=0)
    if seg_mode == 'bet':
        mask_pred = getLarea(mask_pred)

    if seg_mode in ['aseg43', 'dkt', 'wmp']:
        labels = label_all[seg_mode]
        lut = np.zeros(len(labels) + 2, dtype=np.int32)
        for ii, lbl in enumerate(labels):
            lut[ii + 1] = lbl
        mask_pred = lut[mask_pred]



    mask_pred = mask_pred.astype(int)


    output_nib = nib.Nifti1Image(
        mask_pred, input_nib.affine, input_nib.header)

    return output_nib



def read_bet_input(input_file, input_nib=None):
    if input_nib is None:
        input_nib = nib.load(input_file)
    zoom = input_nib.header.get_zooms()[:3]
    needs_1mm = (max(zoom) > 1.1 or min(zoom) < 0.9)

    if needs_1mm:
        input_nib = resample_voxel(input_nib, (1, 1, 1), interpolation='continuous')
    return reorder_img(input_nib, resample='continuous')
