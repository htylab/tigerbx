import argparse
from tigerbx import nerve_nerme


def setup_parser(parser):
    parser.add_argument('input', type=str, nargs='+', help='Path to the input image(s); can be a folder containing images in the specific format (nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output segmentation (default: the directory of input files)')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
    parser.add_argument('--model', default=None, type=str, help='Specifying the model name')
    parser.add_argument('--method', default='NERVE', type=str, help='Specifying the model name')
    parser.add_argument('-p', '--save_patch', action='store_true', help='Saving patches')
    parser.add_argument('-d', '--decode', action='store_true', help='Decode patches')
    parser.add_argument('-e', '--encode', action='store_true', help='Encode patches')
    parser.add_argument('-v', '--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('-s', '--sigma', action='store_true', help='Using variable reconstruction')


def run_args(args):
    nerve_nerme.run_args(args)
