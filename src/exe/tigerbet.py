import sys
import os
import argparse
import time
import tigerseg.segment
import tigerseg.methods.mprage
from distutils.util import strtobool
import glob

def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')


def main():

    default_model = 'mprage_v0004_bet_full.onnx'
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', metavar='INPUT_FILE', default=None, type=str, nargs='+', help='Path to the input image, can be a folder for the specific format(nii.gz)')
    parser.add_argument('-o', '--output', metavar='OUTPUT_DIR', default=None, type=path, help='Filepath for output segmentation, default: the directory of input files')
    parser.add_argument('--model', default=default_model, type=str, help='specifies the modelname')
    parser.add_argument('--GPU',default='False',type = strtobool, help='True: GPU, False: CPU, default: False, CPU')
    parser.add_argument('--report',default='True',type = strtobool, help='Produce additional reports')
    args = parser.parse_args()
    

    if args.input is not None:
        input_file_list = []
        for arg in args.input:
            input_file_list += glob.glob(arg)
    else:
        input_file_list = glob.glob('*.nii') + glob.glob('*.nii.gz')


    output_dir = args.output
    model_name = args.model
    

    seg_method = os.path.basename(model_name).split('_')[0]

    print('Total nii files:', len(input_file_list))

    output_file_list = []

    for f in input_file_list:

        print('Predicting:', f)
        t = time.time()

            
        input_data = tigerseg.methods.mprage.read_file(model_name, f)

        mask = tigerseg.segment.apply(model_name, input_data,  GPU=False)

        if output_dir is not None:
            output_file = tigerseg.methods.mprage.write_file(model_name, f, output_dir, mask)
            output_file_list.append(output_file)

        else:
            output_file = tigerseg.methods.mprage.write_file(model_name, f, os.path.dirname(os.path.abspath(f)), mask)
            output_file_list.append(output_file)


        print('Processing time: %d seconds' %  (time.time() - t))





if __name__ == "__main__":
    main()
    os.system('pause')