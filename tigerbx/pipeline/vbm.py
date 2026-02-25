import os
from types import SimpleNamespace
import time
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm

import tigerbx.reg as reg
from tigerbx.bx import run as bx_run
from tigerbx.core.io import save_nib, resolve_inputs

_logger = logging.getLogger('tigerbx')
_logger.addHandler(logging.NullHandler())


def _build_vbm_ftemplate(ftemplate):
    dir_path, filename = os.path.split(ftemplate)
    prefix = filename.split('_@@@@')[0]
    new_dir_path = os.path.join(dir_path, prefix)
    os.makedirs(new_dir_path, exist_ok=True)
    return os.path.join(dir_path, prefix, filename)


def vbm(input=None, output=None, model=None, template=None, reg_plan='AF',
        gpu=False, gz=False, save_displacement=False, affine_type='C2FViT',
        verbose=0):
    if not isinstance(input, list):
        input = [input]

    args = SimpleNamespace()
    args.input = input
    args.output = output
    args.model = model
    args.gpu = gpu
    args.gz = gz
    args.reg_plan = reg_plan or 'AF'
    args.vbm = True
    args.template = template
    args.save_displacement = save_displacement
    args.affine_type = affine_type
    args.verbose = verbose
    return run_args(args)


def run_args(args):
    verbose = getattr(args, 'verbose', 0)

    def printer(*msg):
        if verbose >= 1:
            _logger.info(' '.join(str(x) for x in msg))

    def _dbg(*msg):
        if verbose >= 2:
            _logger.debug(' '.join(str(x) for x in msg))

    run_d = vars(args)
    if not run_d.get('reg_plan'):
        run_d['reg_plan'] = 'AF'
    run_d['vbm'] = True
    run_d.setdefault('aseg', False)

    input_file_list, common_folder = resolve_inputs(args.input)
    printer('Reg plan:', run_d['reg_plan'])
    printer('Total nii files:', len(input_file_list))

    result_all = []
    is_single = len(input_file_list) == 1
    _pbar = tqdm(input_file_list, desc='tigerbx-vbm', unit='file', disable=(verbose > 0))
    for count, f in enumerate(_pbar, 1):
        _pbar.set_postfix_str(os.path.basename(f))
        printer(f'{count} Preprocessing :', os.path.basename(f))
        t = time.time()

        result_dict, result_filedict, reg_ctx_obj = reg.run_case(
            args, f, common_folder=common_folder,
            save_reg_output=False, verbose=0)
        if reg_ctx_obj is None:
            raise RuntimeError('Missing RegistrationContext from tigerbx.reg for VBM pipeline.')
        ftemplate = reg_ctx_obj.ftemplate
        vbm_ftemplate = _build_vbm_ftemplate(ftemplate)

        if 'QC' in result_dict:
            printer('QC score:', result_dict['QC'])

        # CGW is a VBM-specific prerequisite; fetch it directly via bx.
        cgw_result = bx_run('mC', f, output=None, verbose=0, save_outputs=False)
        cgw_list = cgw_result.get('cgw')
        if not cgw_list or len(cgw_list) < 2:
            raise RuntimeError('VBM requires CGW outputs (GM PVE missing).')
        gm_pve_nib = cgw_list[1]
        _dbg(vbm_ftemplate)

        reg_gm = reg.apply_warp(gm_pve_nib, reg_ctx_obj, interpolation='linear')
        result_dict['Reg_GM'] = reg_gm

        mod_gm = reg.modulate(reg_gm, reg_ctx_obj)
        result_dict['Modulated_GM'] = mod_gm

        fwhm_value = 7.065
        smooth_gm = reg.smooth(mod_gm, fwhm=fwhm_value)
        fn = save_nib(smooth_gm, vbm_ftemplate, 'SmoothedGM')
        result_dict['Smoothed_GM'] = smooth_gm
        result_filedict['Smoothed_GM'] = fn

        printer('Processing time: %d seconds' % (time.time() - t))
        if is_single:
            result_all = result_dict
        else:
            result_all.append(result_filedict)
    return result_all
