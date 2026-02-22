import sys
import os
from os.path import join
import glob
from scipy.io import savemat
import nibabel as nib
import numpy as np
import time
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm

from tigerbx import lib_tool
from tigerbx import lib_gdm

_logger = logging.getLogger('tigerbx')
_logger.addHandler(logging.NullHandler())

def gdm(input, output=None, b0_index=0, dmap=False, no_resample=False, GPU=False, verbose=0):

    from types import SimpleNamespace as Namespace
    args = Namespace()

    args.b0_index = str(b0_index)
    args.dmap = dmap
    args.no_resample = no_resample
    args.gpu = GPU
    args.verbose = verbose

    if not isinstance(input, list):
        input = [input]
    args.input = input
    args.output = output

    return run_args(args)


def run_args(args):

    verbose = getattr(args, 'verbose', 0)

    def printer(*msg):
        if verbose >= 1:
            _logger.info(' '.join(str(x) for x in msg))

    def _dbg(*msg):
        if verbose >= 2:
            _logger.debug(' '.join(str(x) for x in msg))

    input_file_list = args.input
    if os.path.isdir(args.input[0]):
        input_file_list = glob.glob(join(args.input[0], '*.nii'))
        input_file_list += glob.glob(join(args.input[0], '*.nii.gz'))

    elif '*' in args.input[0]:
        input_file_list = glob.glob(args.input[0])

    output_dir = args.output

    if args.b0_index is None:
        b0_index = 0
    elif os.path.exists(args.b0_index.replace('.bval', '') + '.bval'):
        b0_index = lib_gdm.get_b0_slice(args.b0_index.replace('.bval', '') + '.bval')
    else:
        b0_index = int(args.b0_index)

    resample = (not args.no_resample)

    printer('Total nii files:', len(input_file_list))

    model_name = lib_tool.get_model('vdm_unet3d_v002')

    result_all = []
    _pbar = tqdm(input_file_list, desc='tigerbx-gdm', unit='file', disable=(verbose > 0))
    for f in _pbar:

        _pbar.set_postfix_str(os.path.basename(f))
        printer('Predicting:', f)
        t = time.time()
        input_data = lib_gdm.read_file(model_name, f)
        gdmi, gdmap = lib_gdm.run(model_name, input_data, b0_index, GPU=args.gpu, resample=resample)

        if output_dir is None:
            f_output_dir = os.path.dirname(os.path.abspath(f))
        else:
            f_output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

        result_dict = {}
        fn, _ = lib_gdm.write_file(model_name, f, f_output_dir, gdmi)
        result_dict['gdmi'] = fn

        if args.dmap:
            fn, _ = lib_gdm.write_file(model_name, f, f_output_dir, gdmap, postfix='gdm')
            result_dict['gdm'] = fn

        printer('Processing time: %d seconds' % (time.time() - t))
        result_all.append(result_dict)

    if len(input_file_list) == 1:
        return result_all[0]
    return result_all
