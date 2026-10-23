import logging
import importlib


def setup_parser(parser):
    parser.add_argument('input', type=str, nargs='+', help='Input T1w image(s), folder or glob')
    parser.add_argument('-o', '--output', default=None, help='Output folder, default: next to each input')
    parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--model', default=None, type=str, help='Embedding encoder ONNX (default: mprage_embed_v001_vfa)')
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
    importlib.import_module('tigerbx.embed').run_args(args)
