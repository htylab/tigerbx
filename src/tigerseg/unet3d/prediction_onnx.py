import os

import nibabel as nib
import numpy as np
import logging
import onnxruntime as ort


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


def run_case(output_dir, model, data_file, output_label_map=False,
             threshold=0.5, labels=None, output_basename="prediction.nii.gz"):

    os.makedirs(output_dir, exist_ok=True)

    affine = data_file.affine
    test_data = data_file.get_fdata()
    test_data = np.expand_dims(np.stack([test_data]*model.get_inputs()[0].shape[1]), axis=0)

    prediction = predict(model, test_data)

    prediction_image = prediction_to_image(prediction, affine, label_map=output_label_map, threshold=threshold,
                                           labels=labels)

    prediction_image.to_filename(os.path.join(output_dir, output_basename))



def predict(model, data, permute=False):
    if model.get_inputs()[0].type=='tensor(float)':
        return np.squeeze(model.run(None, {model.get_inputs()[0].name: data.astype('float32')}, ), axis=0)
    else:
        return np.squeeze(model.run(None, {model.get_inputs()[0].name: data.astype('float64')}, ), axis=0)


def load_onnx_model(model_file, only_CPU=False):
    logging.info("Loading pre-trained model")

    try:
        if only_CPU:
            return ort.InferenceSession(model_file, providers=['CPUExecutionProvider'])
        else:
            return ort.InferenceSession(model_file, providers=['CUDAExecutionProvider'])

    except ValueError as error:
        raise error
