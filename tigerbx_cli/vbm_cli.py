import argparse
import logging
import importlib


def setup_parser(parser):
    parser.add_argument('input', type=str, nargs='+', help='Path to the input image, can be a folder for the specific format(nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output image, default: the directory of input files')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
    parser.add_argument('--model', default=None, type=str, help='Registration model overrides (rigid/affine/reg only)')
    parser.add_argument('-z', '--gz', action='store_true', help='Forcing storing in nii.gz format')
    parser.add_argument('--reg-plan', default='AF', help='Registration plan for DeepVBM (default: AF)')
    parser.add_argument('-T', '--template', type=str, help='The template filename(default is MNI152)')
    parser.add_argument('--save_displacement', action='store_true', help='Flag to save the displacement field')
    parser.add_argument('--affine_type', choices=['C2FViT', 'ANTs'], default='C2FViT', help='Specify affine transformation type')
    parser.add_argument('--verbose', type=int, default=1, metavar='N',
                        help='Verbosity: 0=quiet (tqdm only), 1=progress (default), 2=debug')


def run_args(args):
    verbose = getattr(args, 'verbose', 1)
    level = logging.DEBUG if verbose >= 2 else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    handler.setLevel(level)
    logger = logging.getLogger('tigerbx')
    logger.addHandler(handler)
    logger.setLevel(level)

    args.vbm = True
    vbm_module = importlib.import_module('tigerbx.pipeline.vbm')
    vbm_module.run_args(args)
