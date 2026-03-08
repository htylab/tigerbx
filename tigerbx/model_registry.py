import copy


_ASEG_LABELS = (
    2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24,
    26, 28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54,
    58, 60, 62, 63, 77, 85, 251, 252, 253, 254, 255,
)

_DGM_LABELS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
)

_WMH_LABELS = (1,)

_SYN_LABELS = (
    2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24,
    26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60,
)

_SYN_LABEL_NAMES = {
    0: 'Background',
    2: 'L-Cerebral-WM',
    3: 'L-Cerebral-Ctx',
    4: 'L-Lat-Ventricle',
    5: 'L-Inf-Lat-Vent',
    7: 'L-Cereb-WM',
    8: 'L-Cereb-Ctx',
    10: 'L-Thalamus',
    11: 'L-Caudate',
    12: 'L-Putamen',
    13: 'L-Pallidum',
    14: '3rd-Ventricle',
    15: '4th-Ventricle',
    16: 'Brain-Stem',
    17: 'L-Hippocampus',
    18: 'L-Amygdala',
    24: 'CSF',
    26: 'L-Accumbens',
    28: 'L-VentralDC',
    41: 'R-Cerebral-WM',
    42: 'R-Cerebral-Ctx',
    43: 'R-Lat-Ventricle',
    44: 'R-Inf-Lat-Vent',
    46: 'R-Cereb-WM',
    47: 'R-Cereb-Ctx',
    49: 'R-Thalamus',
    50: 'R-Caudate',
    51: 'R-Putamen',
    52: 'R-Pallidum',
    53: 'R-Hippocampus',
    54: 'R-Amygdala',
    58: 'R-Accumbens',
    60: 'R-VentralDC',
}

_SYN_COMPACT_TO_FINAL = (0,) + _SYN_LABELS


MODEL_SPECS = {
    'aseg': {
        'model': 'mprage_aseg43_v007_16ksynth.onnx',
        'task': 'aseg',
        'labels': _ASEG_LABELS,
        'n_classes': 44,
        'input_norm': 'max',
        'use_tbet111_crop': True,
        'apply_brainmask': True,
        'output_key': 'aseg',
        'output_suffix': 'aseg',
        'background_label': 0,
        'compact_to_final_label': (0,) + _ASEG_LABELS,
    },
    'dgm': {
        'model': 'mprage_dgm12_v002_mix6.onnx',
        'task': 'dgm',
        'labels': _DGM_LABELS,
        'n_classes': 13,
        'input_norm': 'max',
        'use_tbet111_crop': True,
        'apply_brainmask': True,
        'output_key': 'dgm',
        'output_suffix': 'dgm',
        'background_label': 0,
        'compact_to_final_label': (0,) + _DGM_LABELS,
    },
    'wmh': {
        'model': 'mprage_wmh_v002_betr111.onnx',
        'task': 'wmh',
        'labels': _WMH_LABELS,
        'n_classes': 2,
        'input_norm': 'max',
        'use_tbet111_crop': True,
        'apply_brainmask': True,
        'output_key': 'wmh',
        'output_suffix': 'wmh',
        'background_label': 0,
        'compact_to_final_label': (0, 1),
    },
    'syn': {
        'model': 'mprage_synthseg_v003_r111.onnx',
        'task': 'syn',
        'labels': _SYN_LABELS,
        'n_classes': 33,
        'input_norm': 'minmax',
        'use_tbet111_crop': True,
        'apply_brainmask': True,
        'output_key': 'syn',
        'output_suffix': 'syn',
        'background_label': 0,
        'label_names': _SYN_LABEL_NAMES,
        'compact_to_final_label': _SYN_COMPACT_TO_FINAL,
    },
    'syn2': {
        'model': 'mprage_syn2_v001_r111.onnx',
        'task': 'syn2',
        'labels': _SYN_LABELS,
        'n_classes': 33,
        'input_norm': 'minmax',
        'use_tbet111_crop': True,
        'apply_brainmask': True,
        'output_key': 'syn2',
        'output_suffix': 'syn2',
        'background_label': 0,
        'label_names': _SYN_LABEL_NAMES,
        'compact_to_final_label': _SYN_COMPACT_TO_FINAL,
    },
}


def is_registry_model(key):
    return key in MODEL_SPECS


def get_default_model_names():
    return {key: spec['model'] for key, spec in MODEL_SPECS.items()}


def get_model_spec(key, model_name=None):
    if key not in MODEL_SPECS:
        return None
    spec = copy.deepcopy(MODEL_SPECS[key])
    if model_name is not None:
        spec['model'] = model_name
    return spec
