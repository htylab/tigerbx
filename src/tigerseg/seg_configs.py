import os.path


def get_mode0_config(modelpath=os.getcwd()):
    """Returns the Brain Tumor Segmentation mode configuration."""
    config = dict()
    config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.
    config["labels"] = (2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,30,31,41,42,43,44,46,47,49,50,51,52,53,54,58,60,62,63,77,85,251,252,253,254,255)
    config["labels_name"] = {2: 'Left-Cerebral-White-Matter', 3: 'Left-Cerebral-Cortex', 4: 'Left-Lateral-Ventricle', 5: 'Left-Inf-Lat-Vent', 7: 'Left-Cerebellum-White-Matter',
            8: 'Left-Cerebellum-Cortex', 10: 'Left-Thalamus', 11: 'Left-Caudate', 12: 'Left-Putamen', 13: 'Left-Pallidum', 14: '3rd-Ventricle',
            15: '4th-Ventricle', 16: 'Brain-Stem', 17: 'Left-Hippocampus', 18: 'Left-Amygdala', 24: 'CSF', 26: 'Left-Accumbens-area',
            28: 'Left-VentralDC', 30: 'Left-vessel', 31: 'Left-choroid-plexus', 41: 'Right-Cerebral-White-Matter', 42: 'Right-Cerebral-Cortex',
            43: 'Right-Lateral-Ventricle', 44: 'Right-Inf-Lat-Vent', 46: 'Right-Cerebellum-White-Matter', 47: 'Right-Cerebellum-Cortex',
            49: 'Right-Thalamus', 50: 'Right-Caudate', 51: 'Right-Putamen', 52: 'Right-Pallidum', 53: 'Right-Hippocampus', 54: 'Right-Amygdala',
            58: 'Right-Accumbens-area', 60: 'Right-VentralDC', 62: 'Right-vessel', 63: 'Right-choroid-plexus', 72: '5th-Ventricle', 77: 'WM-hypointensities',
            78: 'Left-WM-hypointensities', 79: 'Right-WM-hypointensities', 80: 'non-WM-hypointensities', 81: 'Left-non-WM-hypointensities',
            82: 'Right-non-WM-hypointensities', 85: 'Optic-Chiasm', 251: 'CC_Posterior', 252: 'CC_Mid_Posterior ', 253: 'CC_Central', 254: 'CC_Mid_Anterior',
            255: 'CC_Anterior',}
    config['threshold'] = 0.5
    config["mri_types"] = '1' #T1w
    config["preprocessing_mode"] = 0
    config["postprocessing_mode"] = 0
    config['model_file'] = os.path.join(modelpath, "unet_model.onnx")
    config['model_url'] = 'https://github.com/JENNSHIUAN/myfirstpost/releases/download/0.0.1/unet_model.onnx'
    config['example_url'] = 'https://github.com/JENNSHIUAN/myfirstpost/releases/download/0.0.1/example.nii.gz'
    return config


def get_mode1_config(modelpath=os.getcwd(), mri_type='fc12'):
    """Returns the Brain Tumor Segmentation mode configuration."""
    config = dict()
    config["image_shape"] = None # no resize
    config["labels"] = (0,1,2,4)
    config["labels_name"] = None
    config['threshold'] = None # For Softmax
    mri_type_order = {'f':1,'c':2, '1':3, '2':4} # Multi image
    mri_types = ''.join(set(mri_type.lower())) * (4//len(mri_type))
    if len(mri_types)!=4: raise ValueError('Get wrong MRI image type')
    mri_types = ''.join(sorted(mri_types, key=lambda x: mri_type_order[x]))
    config["mri_types"] = mri_types
    config["preprocessing_mode"] = 0
    config["postprocessing_mode"] = None
    config['model_file'] = os.path.join(modelpath, f"brats2021model_{mri_types}.onnx")
    config['model_url'] = 'https://github.com/JENNSHIUAN/myfirstpost/releases/download/0.0.1/unet_model.onnx'
    config['example_url'] = 'https://github.com/JENNSHIUAN/myfirstpost/releases/download/0.0.1/example.nii.gz'
    return config


def get_mode2_config(modelpath=os.getcwd(), ):
    """Returns the Nasopharyngeal Carcinoma Segmentation mode configuration."""
    config = dict()
    config["image_shape"] = (224, 224, 35)
    config["labels"] = (0,1,2,3,4)
    config["labels_name"] = None
    config['threshold'] = None
    config["mri_types"] = 'c'
    config["preprocessing_mode"] = 0
    config["postprocessing_mode"] = 1
    config['model_file'] = os.path.join(modelpath, f"NPC_model.onnx")
    config['model_url'] = 'https://github.com/JENNSHIUAN/myfirstpost/releases/download/0.0.1/unet_model.onnx'
    config['example_url'] = 'https://github.com/JENNSHIUAN/myfirstpost/releases/download/0.0.1/example.nii.gz'
    return config

def get_mode3_config(modelpath=os.getcwd(), ):
    """Returns the Brain Extraction mode configuration."""
    config = dict()
    config["image_shape"] = (128, 128, 128)
    config["labels"] = (0,1)
    config["labels_name"] = {0: 'Background',1:'Brain Region'}
    config['threshold'] = None
    config["mri_types"] = '1'
    config["preprocessing_mode"] = 0
    config["postprocessing_mode"] = 0
    config['model_file'] = os.path.join(modelpath, f"BET_model.onnx")
    config['model_url'] = 'https://github.com/JENNSHIUAN/myfirstpost/releases/download/0.0.1/unet_model.onnx'
    config['example_url'] = 'https://github.com/JENNSHIUAN/myfirstpost/releases/download/0.0.1/example.nii.gz'
    return config