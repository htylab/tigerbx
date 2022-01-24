import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import time
import logging
import warnings
import urllib.request
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img

from tigerseg.unet3d.utils.utils import read_image, get_input_image, walk_input_dir, read_image_by_mri_type,postprocessing


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)
model_url = 'https://github.com/JENNSHIUAN/myfirstpost/releases/download/0.0.1/unet_model.h5'
example_url = 'https://github.com/JENNSHIUAN/myfirstpost/releases/download/0.0.1/example.nii.gz'


def apply(input=None, output=None, modelpath=os.getcwd(), only_CPU=False, permute=False):

    import tensorflow as tf
    from tigerseg.unet3d.prediction import run_validation_case, load_old_model

    start = time.time()
    model_file = os.path.join(modelpath,'unet_model.h5')
    if not os.path.exists(model_file):
        logging.info(f'Downloading model files....')
        model_file,header  = urllib.request.urlretrieve(model_url, model_file)
    if only_CPU:
        logging.info(only_CPU)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if tf.test.gpu_device_name():
        logging.info('GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        logging.info("Now you are using CPU version of TF")


    config = dict()
    config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.
    config["labels"] = (2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,30,31,41,42,43,44,46,47,49,50,51,52,53,54,58,60,62,63,77,85,251,252,253,254,255)
    config['model_file'] = model_file
    config["permute"] = bool(permute)

    if not input :
        logging.info('Downloading example files....')
        input , header= urllib.request.urlretrieve(example_url, os.path.join(os.getcwd(),'example.nii.gz'))

    if not output :
        logging.info('Didn\'t set the output path. The result will be saved to the input path.')
        output = os.path.join(os.getcwd(), 'output')
        if not os.path.isdir(output):
            os.mkdir(output)

    input_dir = get_input_image(input)
    model = load_old_model(config["model_file"])

    for data_file in input_dir:
        output_name = "result_{subject}".format(subject=os.path.split(data_file)[1])
        single_file = read_image(data_file, image_shape=(128,128,128), crop=False, interpolation='linear')

        run_validation_case(output_dir=output,
                            model=model,
                            data_file=single_file,
                            output_label_map=True,
                            labels=config["labels"],
                            threshold=0.5,
                            overlap=16,
                            permute=False,
                            output_basename=output_name,
                            test=False)
        logging.info(f'{output_name} is finished.')
        prediction_filename = os.path.join(output,output_name)
        ref = nib.load(data_file)
        pred = nib.load(prediction_filename)
        pred_resampled = resample_to_img(pred, ref, interpolation="nearest")
        nib.save(pred_resampled, prediction_filename)

    end = time.time()
    logging.info(f'Save result to: {output}')
    logging.info('Total cost was:%.2f secs' % (end - start))




def onnx_apply(input=None,output=None,modelpath=os.getcwd(),only_CPU=False,seg_mode=0,mri_type='fc12',report_enabled=False):
    
    import onnxruntime as ort
    from tigerseg.unet3d.prediction_onnx import run_case, load_onnx_model


    start = time.time()

    config = dict()
    if seg_mode==0:
        logging.info("You are using Subcortical Brain Segmentation mode.")
        config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.
        config["labels"] = (2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,30,31,41,42,43,44,46,47,49,50,51,52,53,54,58,60,62,63,77,85,251,252,253,254,255)
        config['threshold'] = 0.5
        config["mri_types"] = '1' #T1w
        config["preprocessing_mode"] = 0
        config["postprocessing_mode"] = 0 if report_enabled else None
        model_file = os.path.join(modelpath, "unet_model.onnx")
        config['model_file'] = model_file
        model_url = 'https://github.com/JENNSHIUAN/myfirstpost/releases/download/0.0.1/unet_model.onnx'
        example_url = 'https://github.com/JENNSHIUAN/myfirstpost/releases/download/0.0.1/example.nii.gz'

    elif seg_mode==1:
        logging.info("You are using Brain Tumor Segmentation mode.")
        config["image_shape"] = None # no resize
        config["labels"] = (0,1,2,4)
        config['threshold'] = None # For Softmax

        mri_type_order = {'f':1,'c':2, '1':3, '2':4} # Multi image
        mri_types = ''.join(set(mri_type.lower())) * (4//len(mri_type))
        if len(mri_types)!=4: raise ValueError('Get wrong MRI image type')
        mri_types = ''.join(sorted(mri_types, key=lambda x: mri_type_order[x]))
        config["mri_types"] = mri_types 
        config["preprocessing_mode"] = 0
        config["postprocessing_mode"] = None

        model_file = os.path.join(modelpath, f"brats2021model_{mri_types}.onnx")
        config['model_file'] = model_file
        model_url = 'https://github.com/JENNSHIUAN/myfirstpost/releases/download/0.0.1/unet_model.onnx'
        example_url = 'https://github.com/JENNSHIUAN/myfirstpost/releases/download/0.0.1/example.nii.gz'

    elif seg_mode==2:
        logging.info("You are using Nasopharyngeal Carcinoma Segmentation mode.")
        config["image_shape"] = (224,224,35)
        config["labels"] = (0,1,2,3,4)
        config['threshold'] = None
        config["mri_types"] = 'c'
        config["preprocessing_mode"] = 0
        config["postprocessing_mode"] = 1 if report_enabled else None
        model_file = os.path.join(modelpath, f"NPC_model.onnx")
        config['model_file'] = model_file
        model_url = 'https://github.com/JENNSHIUAN/myfirstpost/releases/download/0.0.1/unet_model.onnx'
        example_url = 'https://github.com/JENNSHIUAN/myfirstpost/releases/download/0.0.1/example.nii.gz'

    else:
        logging.info("Mode selection error.")
        raise ValueError('seg_mode value error')


    
    if not os.path.exists(model_file):
        logging.info(f'Downloading model files....')
        model_file,header  = urllib.request.urlretrieve(model_url, model_file)

    if only_CPU:
        model = load_onnx_model(config["model_file"], only_CPU=True)
        logging.info("Now you are using CPU version of onnx")
    elif ort.get_device() == "GPU":
        model = load_onnx_model(config["model_file"], only_CPU=False)
        logging.info('GPU Device: {}'.format(ort.get_device()))
    else:
        model = load_onnx_model(config["model_file"], only_CPU=True)
        logging.info("Didn\'t find GPU. Now you are using CPU version of onnx")



    if not input :
        logging.info('Downloading example files....')
        input, header= urllib.request.urlretrieve(example_url, os.path.join(os.getcwd(),'example.nii.gz'))

    if not output :
        logging.info('Didn\'t set the output path. The result will be saved to the input path.')
        output = os.path.join(os.getcwd(), 'output')
        os.makedirs(output, exist_ok=True)

    if report_enabled:
        import pandas as pd
        report = pd.DataFrame([])


    niigz_dirs = walk_input_dir(input)

    for data_dir in niigz_dirs:
        # ouput_dir = data_dir.replace(input, output)
        ouput_dir = os.path.join(data_dir[:len(input)].replace(input, output), data_dir[len(input)+1:])
        os.makedirs(ouput_dir, exist_ok=True)

        if len(config["mri_types"])==1:  # Sigle image in a case
            data_files = get_input_image(data_dir)
            for data_file in data_files:
                try:
                    output_name = "result_{subject}".format(subject=os.path.split(data_file)[1])
                    single_file = read_image(data_file, image_shape=config["image_shape"], preprocessing_mode=config["preprocessing_mode"], crop=False, interpolation='linear')

                    run_case(output_dir=ouput_dir,
                                    model=model,
                                    data_files=single_file,
                                    output_label_map=True,
                                    labels=config["labels"],
                                    threshold=config['threshold'],
                                    output_basename=output_name)
                    logging.info(f'{output_name} is finished.')
                    prediction_filename = os.path.join(ouput_dir, output_name)
                    ref = nib.load(data_file)
                    pred = nib.load(prediction_filename)
                    pred_resampled = resample_to_img(pred, ref, interpolation="nearest")
                    nib.save(pred_resampled, prediction_filename)

                    if report_enabled:
                        report.loc[data_file, 'Segmentation Result'] = True
                    if config["postprocessing_mode"] is not None:
                        postprocessing(pred_resampled, mode=config["postprocessing_mode"],report=report,index=data_file)
                        
                except:
                    logging.info(f'{data_file} failed.')
                    if report_enabled:
                        report.loc[data_file, 'Segmentation Result'] = False

        
        else:  # Multi image in a case
            try:
                data_files = get_input_image(data_dir)
                output_name = "result_{subject}.nii.gz".format(subject=os.path.split(data_dir)[1])
                multi_files = read_image_by_mri_type(data_dir, image_shape=config["image_shape"], preprocessing_mode=config["preprocessing_mode"], 
                                                    mri_types=config["mri_types"], crop=False, interpolation='linear')

                run_case(output_dir=ouput_dir,
                                model=model,
                                data_files=multi_files,
                                output_label_map=True,
                                labels=config["labels"],
                                threshold=config['threshold'],
                                output_basename=output_name)
                logging.info(f'{output_name} is finished.')
                prediction_filename = os.path.join(ouput_dir, output_name)
                ref = nib.load(data_files[0])
                pred = nib.load(prediction_filename)
                pred_resampled = resample_to_img(pred, ref, interpolation="nearest")
                nib.save(pred_resampled, prediction_filename)
                
                if report_enabled:
                    report.loc[data_dir, 'Segmentation Result'] = True
                if config["postprocessing_mode"] is not None:
                    postprocessing(pred_resampled, mode=config["postprocessing_mode"],report=report,index=data_dir)

            except:
                    logging.info(f'{data_dir} failed.')
                    if report_enabled:
                        report.loc[data_dir, 'Segmentation Result'] = False



    end = time.time()
    logging.info(f'Save result to: {output}')
    if report_enabled:
        report.to_csv(os.path.join(output, 'report.csv'))
    logging.info('Total cost was:%.2f secs' % (end - start))

