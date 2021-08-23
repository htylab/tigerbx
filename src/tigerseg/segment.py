import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import time
import shutil
import argparse
import logging
import warnings
import urllib.request
import nibabel as nib
import numpy as np
import tensorflow as tf
from nilearn.image import resample_to_img
from distutils.util import strtobool
from tigerseg.unet3d.training import load_old_model
from tigerseg.unet3d.prediction import run_validation_case
from tigerseg.unet3d.utils.utils import read_image, get_input_image
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)
model_url = 'https://github.com/JENNSHIUAN/myfirstpost/releases/download/0.0.1/unet_model.h5'
example_url = 'https://github.com/JENNSHIUAN/myfirstpost/releases/download/0.0.1/example.nii.gz'

WORKING_PATH = '/NFS/weng/3Dsegmentation/tigerseg'
prediction_dir = os.path.join(WORKING_PATH,'src','tigerseg','result')



def apply(input=None,output=prediction_dir,modelpath=os.getcwd(),only_CPU=False,permute=False):

    start = time.time()
    model_file = os.path.join(modelpath,model_url.split('/')[-1])
    if not os.path.exists(model_file):
        logging.info(f'Downloading model files....')
        model_file,header  = urllib.request.urlretrieve(model_url, model_file)
    if only_CPU:
        logging.info(only_CPU)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if tf.test.gpu_device_name():
        logging.info('GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        logging.info("Now you are uainde CPU version of TF")


    config = dict()
    config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.
    config["labels"] = (2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,30,31,41,42,43,44,46,47,49,50,51,52,53,54,58,60,62,63,77,85,251,252,253,254,255)
    config['model_file'] = model_file
    config["permute"] = bool(permute)

    if not input :
        logging.info('Downloading example files....')
        input , header= urllib.request.urlretrieve(example_url, os.path.join(os.getcwd(),'example.nii.gz'))

    input_dir = get_input_image(input)
    model = load_old_model(config["model_file"])

    for data_file in input_dir:
        output_name = "result_{subject}".format(subject=data_file.split('/')[-1])
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
        prediction_filename = os.path.join(os.path.join(output,output_name))
        ref = nib.load(data_file)
        pred = nib.load(prediction_filename)
        pred_resampled = resample_to_img(pred, ref, interpolation="nearest")
        nib.save(pred_resampled,prediction_filename)

    end = time.time()
    logging.info(f'Save result to: {output}')
    logging.info('Total cost was:%.2f secs' % (end - start))
