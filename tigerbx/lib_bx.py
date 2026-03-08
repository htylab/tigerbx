from os.path import basename
import numpy as np
import nibabel as nib
from scipy.special import softmax
from tigerbx.core.onnx import predict
from tigerbx.core.resample import reorder_img, resample_voxel
from tigerbx.model_registry import get_model_spec, is_registry_model

from tigerbx import lib_tool

label_all = dict()
#nib.Nifti1Header.quaternion_threshold = -100

for _key in ('aseg', 'dgm', 'wmh', 'syn', 'syn2'):
    label_all[_key] = get_model_spec(_key)['labels']
label_all['aseg43'] = label_all['aseg']
label_all['dgm12'] = label_all['dgm']
label_all['synthseg'] = label_all['syn']


def get_mode(model_ff):
    seg_mode, version, model_str = basename(model_ff).split('_')[1:4]  # aseg43, bet

    legacy_alias = {'aseg43': 'aseg', 'dgm12': 'dgm', 'synthseg': 'syn'}
    seg_mode = legacy_alias.get(seg_mode, seg_mode)

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

def logit_to_prob(logits, seg_mode=None, n_classes=None):
    label_num = dict()
    label_num['bet'] = 2
    label_num['aseg'] = 44
    label_num['aseg43'] = 44
    label_num['dgm'] = 13
    label_num['dgm12'] = 13
    label_num['seg3'] = 4
    label_num['wmh'] = 2
    label_num['syn'] = 33
    label_num['synthseg'] = 33
    label_num['syn2'] = 33

    if n_classes is None:
        if seg_mode not in label_num:
            raise KeyError(f'Unknown seg_mode without n_classes: {seg_mode}')
        n_classes = label_num[seg_mode]

    #so far we only use sigmoid in tBET
    if n_classes > logits.shape[0]:
        #sigmoid
        th = 0.5
        from scipy.special import expit
        prob = expit(logits)
    else:
        #softmax mode
        #print(logits.shape)
        prob = softmax(logits, axis=0)
    return prob

def _normalise_image(image, input_norm):
    if input_norm == 'minmax':
        rng = np.max(image) - np.min(image)
        if rng > 0:
            return (image - np.min(image)) / rng
        return image
    if input_norm == 'max':
        mx = np.max(image)
        if mx > 0:
            return image / mx
        return image
    raise ValueError(f'Unsupported input_norm: {input_norm}')


def run(model_ff, input_nib, GPU, patch=False, session=None, spec=None):

    if spec is None:
        seg_mode, _, model_str = get_mode(model_ff)
        if is_registry_model(seg_mode):
            spec = get_model_spec(seg_mode)
            input_norm = spec.get('input_norm', 'max')
            labels = spec.get('labels')
            n_classes = spec.get('n_classes')
            compact_to_final_label = spec.get('compact_to_final_label')
        else:
            input_norm = 'minmax' if seg_mode in ('syn', 'syn2') else 'max'
            labels = label_all.get(seg_mode)
            n_classes = None
            compact_to_final_label = None
    else:
        seg_mode = spec['task']
        input_norm = spec.get('input_norm', 'max')
        labels = spec.get('labels')
        n_classes = spec.get('n_classes')
        compact_to_final_label = spec.get('compact_to_final_label')

    data = lib_tool.read_nib(input_nib)

    image = data[None, ...][None, ...]
    image = _normalise_image(image, input_norm)

    if patch:
        logits = predict(model_ff, image, GPU, mode='patch', session=session)[0, ...]
    else:
        logits = predict(model_ff, image, GPU, session=session)[0, ...]

    if n_classes is not None and logits.shape[0] != n_classes:
        raise ValueError(
            f'Model/spec mismatch for {model_ff}: logits have {logits.shape[0]} channels, '
            f'but spec expects {n_classes}.'
        )

    prob = logit_to_prob(logits, seg_mode=seg_mode, n_classes=n_classes)

    if seg_mode =='bet': #sigmoid 1 channel
        th = 0.5
        mask_pred = np.ones(prob[0, ...].shape)
        mask_pred[prob[0, ...] < th] = 0
        mask_pred = getLarea(mask_pred)
    else:
        mask_pred = np.argmax(prob, axis=0)


    if compact_to_final_label is not None:
        lut = np.asarray(compact_to_final_label, dtype=np.int32)
        if np.max(mask_pred) >= len(lut):
            raise ValueError(
                f'Predicted compact class index {np.max(mask_pred)} exceeds LUT size {len(lut)}'
            )
        mask_pred = lut[mask_pred]
    elif labels is not None:
        lut = np.zeros(len(labels) + 2, dtype=np.int32)
        for ii, lbl in enumerate(labels):
            lut[ii + 1] = lbl
        mask_pred = lut[mask_pred]

    mask_pred = mask_pred.astype(int)


    output_nib = nib.Nifti1Image(
        mask_pred, input_nib.affine, input_nib.header)

    return output_nib, prob



def read_bet_input(input_file, input_nib=None):
    if input_nib is None:
        input_nib = nib.load(input_file)
    zoom = input_nib.header.get_zooms()[:3]
    needs_1mm = (max(zoom) > 1.1 or min(zoom) < 0.9)

    if needs_1mm:
        input_nib = resample_voxel(input_nib, (1, 1, 1), interpolation='continuous')
    return reorder_img(input_nib, resample='continuous')
