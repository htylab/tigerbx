import os

import nibabel as nib
import numpy as np
import logging
from tensorflow.keras.models import load_model
from .contrib import InstanceNormalization
from .metrics import (dice_coefficient, dice_coefficient_loss, dice_coef, dice_coef_loss,
                            weighted_dice_coefficient_loss, weighted_dice_coefficient)


def get_prediction_labels(prediction, threshold=0.5, labels=None):
    n_samples = prediction.shape[0]
    label_arrays = []
    for sample_number in range(n_samples):
        label_data = np.argmax(prediction[sample_number], axis=0) + 1
        label_data[np.max(prediction[sample_number], axis=0) < threshold] = 0
        if labels:
            argmax_value = np.unique(label_data).tolist()[1:]
            argmax_value.reverse()
            for value in argmax_value:
                label_data[label_data == value] = labels[value - 1]
        label_arrays.append(np.array(label_data, dtype=np.uint8))
    return label_arrays



def prediction_to_image(prediction, affine, label_map=False, threshold=0.5, labels=None):
    if prediction.shape[1] == 1:
        data = prediction[0, 0]
        if label_map:
            label_map_data = np.zeros(prediction[0, 0].shape, np.int8)
            if labels:
                label = labels[0]
            else:
                label = 1
            label_map_data[data > threshold] = label
            data = label_map_data
    elif prediction.shape[1] > 1:

        label_map_data = get_prediction_labels(prediction, threshold=threshold, labels=labels)
        data = label_map_data[0]
            
    else:
        raise RuntimeError("Invalid prediction array shape: {0}".format(prediction.shape))

    return nib.Nifti1Image(data, affine)


def multi_class_prediction(prediction, affine):
    prediction_images = []
    for i in range(prediction.shape[1]):
        prediction_images.append(nib.Nifti1Image(prediction[0, i], affine))
    return prediction_images


def run_validation_case(output_dir, model, data_file, 
                        output_label_map=False, threshold=0.5, labels=None, overlap=16, permute=False,
                        test=False, output_basename="prediction.nii.gz"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    affine = data_file.affine
    test_data = data_file.get_fdata()
    test_data = test_data[np.newaxis,np.newaxis, :, : , : ]

    prediction = predict(model, test_data, permute=permute)

    prediction_image = prediction_to_image(prediction, affine, label_map=output_label_map, threshold=threshold,
                                           labels=labels)

    prediction_image.to_filename(os.path.join(output_dir, output_basename))



def predict(model, data, permute=False):
    return model.predict(data)



def load_old_model(model_file):
    logging.info("Loading pre-trained model")
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                      'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss,
                      'weighted_dice_coefficient': weighted_dice_coefficient,
                      'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss}
    try:
        custom_objects["InstanceNormalization"] = InstanceNormalization
    except ImportError:
        pass
    try:
        return load_model(model_file, custom_objects=custom_objects)
    except ValueError as error:
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            raise error