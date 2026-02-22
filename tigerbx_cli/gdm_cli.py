import argparse
import logging
from tigerbx import gdmi


def setup_parser(parser):
    parser.add_argument('input', type=str, nargs='+', help='Path to the input image, can be a folder for the specific format(nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output image, default: the directory of input files')
    parser.add_argument('-b0', '--b0_index', default=None, type=str, help='The index of b0 slice or the .bval file, default: 0 (the first slice)')
    parser.add_argument('-n', '--no_resample', action='store_true', help="Don't resample to 1.7x1.7x1.7mm3")
    parser.add_argument('-m', '--dmap', action='store_true', help='Producing the virtual displacement map')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
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
    gdmi.run_args(args)
