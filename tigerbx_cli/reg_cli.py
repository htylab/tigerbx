import argparse
import logging
from tigerbx import reg_vbm


def setup_parser(parser):
    parser.add_argument('input', type=str, nargs='+', help='Path to the input image, can be a folder for the specific format(nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output image, default: the directory of input files')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
    parser.add_argument('--model', default=None, type=str, help='Specifying the model name')
    parser.add_argument('-z', '--gz', action='store_true', help='Forcing storing in nii.gz format')
    parser.add_argument('-b', '--bet', action='store_true', help='Producing BET images')
    parser.add_argument('-A', '--affine', action='store_true', help='Affining images to template')
    parser.add_argument('-r', '--registration', action='store_true', help='Registering images to template(VoxelMorph)')
    parser.add_argument('-s', '--syn', action='store_true', help='Registering images to template(SyN)')
    parser.add_argument('-S', '--syncc', action='store_true', help='Registering images to template(SyNCC)')
    parser.add_argument('-F', '--fusemorph', action='store_true', help='Registering images to template(FuseMorph)')
    parser.add_argument('-T', '--template', type=str, help='The template filename(default is MNI152)')
    parser.add_argument('-R', '--rigid', action='store_true', help='Rigid transforms images to template')
    parser.add_argument('-v', '--vbm', action='store_true', help='vbm analysis')
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
    reg_vbm.run_args(args)
