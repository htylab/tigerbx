import argparse
from tigerbx import bx


def setup_parser(parser):
    parser.add_argument('input', type=str, nargs='+', help='Path to the input image(s); can be a folder containing images in the specific format (nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output segmentation (default: the directory of input files)')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
    parser.add_argument('-m', '--betmask', action='store_true', help='Producing BET mask')
    parser.add_argument('-a', '--aseg', action='store_true', help='Producing ASEG mask')
    parser.add_argument('-b', '--bet', action='store_true', help='Producing BET images')
    parser.add_argument('-c', '--ct', action='store_true', help='Producing cortical thickness map')
    parser.add_argument('-C', '--cgw', action='store_true', help='Producing FSL-style PVE segmentation')
    parser.add_argument('-d', '--dgm', action='store_true', help='Producing deep GM mask')
    parser.add_argument('-S', '--syn', action='store_true', help='Producing ASEG mask using SynthSeg-like method')
    parser.add_argument('-W', '--wmh', action='store_true', help='Producing white matter hypo-intensity mask')
    parser.add_argument('-t', '--tumor', action='store_true', help='Producing tumor mask')
    parser.add_argument('-q', '--qc', action='store_true', help='Saving QC score (pay attention to results with QC scores less than 50)')
    parser.add_argument('-z', '--gz', action='store_true', help='Forcing storing in nii.gz format')
    parser.add_argument('-p', '--patch', action='store_true', help='patch inference')
    parser.add_argument('--silent', action='store_true', help='Silent mode')
    parser.add_argument('--model', default=None, type=str, help='Specifying the model name')
    parser.add_argument('--clean_onnx', action='store_true', help='Clean onnx models')


def run_args(args):
    bx.run_args(args)
