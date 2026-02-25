import argparse
import logging
from tigerbx.hlc import run_args as hlc_run_args


def setup_parser(parser):
    parser.add_argument('input', type=str, nargs='+', help='Path to the input image(s); can be a folder containing images in the specific format (nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output segmentation (default: the directory of input files)')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
    parser.add_argument('--save', default='h', type=str, help='Selected outputs mbhtcgw, default:h')
    parser.add_argument('--model', default=None, type=str, help='Specifying the model name')
    parser.add_argument('-z', '--gz', action='store_true', help='Forcing storing in nii.gz format')
    parser.add_argument('-p', '--patch', action='store_true', help='patch inference')
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
    hlc_run_args(args)
