"""Registration module.

Provides the public ``reg()`` API, plan parsing, execution orchestration,
and core registration method implementations.
"""

import os
import time
import logging
import warnings
from os.path import basename
from dataclasses import dataclass
from types import SimpleNamespace as Namespace

import nibabel as nib
import numpy as np
from tqdm import tqdm

from tigerbx import lib_reg
from tigerbx import lib_tool
from tigerbx.bx import run as bx_run
from tigerbx.core.io import get_template, save_nib, resolve_inputs
from tigerbx.core.onnx import predict
from tigerbx.core.resample import reorder_img

warnings.simplefilter(action='ignore', category=FutureWarning)

_logger = logging.getLogger('tigerbx')
_logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Registration context (dataclass)
# ---------------------------------------------------------------------------

@dataclass
class RegistrationContext:
    ftemplate: str
    input_nib: nib.Nifti1Image
    template_nib: nib.Nifti1Image | None
    affine_nib: nib.Nifti1Image | None
    affine_matrix: object
    init_flow: object
    warp: object
    pad_width: object
    displacement_dict: dict | None


# ---------------------------------------------------------------------------
# Registration methods
# ---------------------------------------------------------------------------

def _apply_affine_to_volume(volume_data, volume_nib, state, run_d, gpu,
                            interpolation='nearestNeighbor'):
    """Apply affine transform to a volume using C2FViT or ANTs.

    Parameters
    ----------
    volume_data : np.ndarray
        3-D float32 array (already reordered).
    volume_nib : nib.Nifti1Image
        NIfTI image for affine metadata (used by ANTs path).
    state : dict
        Registration state with 'affine_matrix', 'init_flow', 'pad_width',
        'displacement_dict'.
    run_d : dict
        Run dictionary (needs 'affine_type').
    gpu : bool
        Whether to use GPU for ONNX inference.
    interpolation : str
        'nearestNeighbor' or 'linear'.  Selects the ONNX model variant
        (near vs bili) for the C2FViT path, and the ANTs interpolation mode.
    """
    if run_d['affine_type'] != 'ANTs':
        if interpolation == 'nearestNeighbor':
            model = lib_tool.get_model('mprage_affinetransform_v002_near.onnx')
        else:
            model = lib_tool.get_model('mprage_affinetransform_v002_bili.onnx')
        vol, _ = lib_reg.pad_to_shape(volume_data, (256, 256, 256))
        vol, _ = lib_reg.crop_image(vol, target_shape=(256, 256, 256))
        vol = np.expand_dims(np.expand_dims(vol, axis=0), axis=1)
        affine_matrix = np.expand_dims(state['affine_matrix'], axis=0)
        output = predict(model, [vol, state['init_flow'], affine_matrix],
                         GPU=gpu, mode='affine_transform')
        result = np.squeeze(output[0])
        result = lib_reg.remove_padding(result, state['pad_width'])
    else:
        ants_interp = 'linear' if interpolation != 'nearestNeighbor' else 'nearestNeighbor'
        ants_vol, _ = lib_reg.get_ants_info(volume_data, volume_nib.affine)
        result = lib_reg.ants_transform(
            ants_vol, state['displacement_dict'],
            interpolation=ants_interp, mode='affine')
    return result


def _run_bx_prereq(f, run_d, gpu, verbose):
    """Fetch BET/ASEG/CGW prerequisites via bx.run without writing outputs."""
    bx_flags = ['m']  # reg pipeline always needs BET mask + QC
    if run_d.get('bet'):
        bx_flags.append('b')
    if run_d.get('aseg'):
        bx_flags.append('a')
    if gpu:
        bx_flags.append('g')

    return bx_run(''.join(bx_flags), f, output=None,
                  verbose=max(0, verbose - 1), save_outputs=False)


def _init_displacement_dict():
    return {
        "init_flow": None,
        "rigid_matrix": None,
        "affine_matrix": None,
        "reference_info": None,
        "dense_warp": None,
        "SyN_dense_warp": None,
        "SyNCC_dense_warp": None,
        "Fuse_dense_warp": None
    }


def _prepare_registration_state(input_nib, tbetmask_nib, run_d):
    displacement_dict = _init_displacement_dict()
    state = {
        'displacement_dict': displacement_dict,
        'pad_width': None,
        'affine_matrix': None,
        'init_flow': None,
        'warp': None,
        'affine_nib': None,
        'template_nib': None,
    }

    bet = lib_tool.read_nib(input_nib) * lib_tool.read_nib(tbetmask_nib)
    bet = bet.astype(input_nib.dataobj.dtype)
    bet_nib = nib.Nifti1Image(bet, input_nib.affine, input_nib.header)

    bet_nib = reorder_img(bet_nib, resample='continuous')
    state['bet_nib'] = bet_nib
    state['ori_affine'] = bet_nib.affine
    state['bet_data'] = bet_nib.get_fdata()

    template_nib = lib_reg.get_template(run_d['template'])
    template_nib = reorder_img(template_nib, resample='continuous')
    state['template_nib'] = template_nib
    state['fixed_affine'] = template_nib.affine
    return state


def _run_rigid_method(state, run_d, args, omodel, result_dict, result_filedict, ftemplate):
    template_nib = state['template_nib']
    template_data = template_nib.get_fdata()
    bet_data_R, _ = lib_reg.pad_to_shape(state['bet_data'], (256, 256, 256))
    bet_data_R, _ = lib_reg.crop_image(bet_data_R, target_shape=(256, 256, 256))
    template_data, pad_width = lib_reg.pad_to_shape(template_data, (256, 256, 256))

    moving = bet_data_R.astype(np.float32)[None, ...][None, ...]
    moving = lib_reg.min_max_norm(moving)
    if run_d['template'] == None:
        template_data = np.clip(template_data, a_min=2500, a_max=np.max(template_data))
    fixed = template_data.astype(np.float32)[None, ...][None, ...]
    fixed = lib_reg.min_max_norm(fixed)

    model_ff = lib_tool.get_model(omodel['rigid'])
    output = predict(model_ff, [moving, fixed], GPU=args.gpu, mode='reg')
    rigided, rigid_matrix, init_flow = np.squeeze(output[0]), np.squeeze(output[1]), output[2]

    state['displacement_dict']["init_flow"] = init_flow
    state['displacement_dict']["rigid_matrix"] = rigid_matrix
    state['init_flow'] = init_flow
    state['pad_width'] = pad_width

    rigided = lib_reg.remove_padding(rigided, pad_width)
    rigid_nib = nib.Nifti1Image(rigided, state['fixed_affine'])
    fn = save_nib(rigid_nib, ftemplate, 'rigid')
    result_dict['rigid'] = rigid_nib
    result_filedict['rigid'] = fn


def _run_affine_method(state, run_d, args, omodel, result_dict, result_filedict, ftemplate):
    template_nib = state['template_nib']
    template_data = template_nib.get_fdata()
    bet_data = state['bet_data']

    if run_d['affine_type'] != 'ANTs':
        bet_data, _ = lib_reg.pad_to_shape(bet_data, (256, 256, 256))
        bet_data, _ = lib_reg.crop_image(bet_data, target_shape=(256, 256, 256))
        template_data, pad_width = lib_reg.pad_to_shape(template_data, (256, 256, 256))
        state['pad_width'] = pad_width

    moving = bet_data.astype(np.float32)[None, ...][None, ...]
    moving = lib_reg.min_max_norm(moving)
    if run_d['template'] == None:
        template_data = np.clip(template_data, a_min=2500, a_max=np.max(template_data))
    fixed = template_data.astype(np.float32)[None, ...][None, ...]
    fixed = lib_reg.min_max_norm(fixed)

    if run_d['affine_type'] == 'C2FViT':
        model_ff = lib_tool.get_model(omodel['affine'])
        output = predict(model_ff, [moving, fixed], GPU=args.gpu, mode='reg')
        affined, affine_matrix, init_flow = np.squeeze(output[0]), np.squeeze(output[1]), output[2]
        initflow_nib = nib.Nifti1Image(init_flow, state['ori_affine'])
        state['displacement_dict']["init_flow"] = init_flow
        result_dict['init_flow'] = initflow_nib
        state['displacement_dict']["affine_matrix"] = affine_matrix
        state['init_flow'] = init_flow
        state['affine_matrix'] = affine_matrix
    elif run_d['affine_type'] == 'ANTs':
        ants_fixed, reference_info = lib_reg.get_ants_info(template_data, state['fixed_affine'])
        ants_moving, _ = lib_reg.get_ants_info(bet_data, state['ori_affine'])
        affined, affine_matrix = lib_reg.apply_ANTs_reg(ants_moving, ants_fixed, 'Affine')
        affined = lib_reg.min_max_norm(affined)
        state['displacement_dict'].update(reference_info)
        state['displacement_dict'].update(affine_matrix)
        state['affine_matrix'] = affine_matrix

    result_dict['Affine_matrix'] = state['affine_matrix']
    if run_d['affine_type'] != 'ANTs':
        affined = lib_reg.remove_padding(affined, state['pad_width'])
    affine_nib = nib.Nifti1Image(affined, state['fixed_affine'])
    state['affine_nib'] = affine_nib

    if run_d['affine']:
        fn = save_nib(affine_nib, ftemplate, 'Af')
        result_dict['Af'] = affine_nib
        result_filedict['Af'] = fn


def _run_vmnet_registration_method(state, args, omodel, result_dict, result_filedict, ftemplate, save_reg_output):
    template_nib = state['template_nib']
    template_data = template_nib.get_fdata()

    fixed_image = template_data.astype(np.float32)[None, ...][None, ...]
    fixed_image = lib_reg.min_max_norm(fixed_image)

    Af_data = state['affine_nib'].get_fdata()
    moving_image = Af_data.astype(np.float32)[None, ...][None, ...]

    model_ff = lib_tool.get_model(omodel['reg'])
    output = predict(model_ff, [moving_image, fixed_image], GPU=args.gpu, mode='reg')
    moved, warp = np.squeeze(output[0]), np.squeeze(output[1])
    moved_nib = nib.Nifti1Image(moved, state['fixed_affine'], template_nib.header)
    warp_nib = nib.Nifti1Image(warp, state['fixed_affine'], template_nib.header)

    if save_reg_output:
        fn = save_nib(moved_nib, ftemplate, 'reg')
        result_filedict['reg'] = fn
    result_dict['reg'] = moved_nib

    state['displacement_dict']["dense_warp"] = warp
    result_dict['dense_warp'] = warp_nib
    state['warp'] = warp


def _run_ants_nonlinear_method(state, ants_reg_str, result_dict, ftemplate):
    template_nib = state['template_nib']
    template_data = template_nib.get_fdata()
    ants_fixed, reference_info = lib_reg.get_ants_info(template_data, template_nib.affine)
    bet_data = state['bet_nib'].get_fdata()
    ants_moving, _ = lib_reg.get_ants_info(bet_data, state['bet_nib'].affine)

    if ants_reg_str == 'syn':
        moved, ants_dict = lib_reg.apply_ANTs_reg(ants_moving, ants_fixed, 'SyN')
    else:
        moved, ants_dict = lib_reg.apply_ANTs_reg(ants_moving, ants_fixed, 'SyNCC')

    moved = lib_reg.min_max_norm(moved)
    state['displacement_dict'].update(reference_info)
    state['displacement_dict'].update(ants_dict)
    moved_nib = nib.Nifti1Image(moved, state['fixed_affine'], template_nib.header)
    save_nib(moved_nib, ftemplate, ants_reg_str)
    result_dict[ants_reg_str] = moved_nib
    result_dict[ants_reg_str + '_dense_warp'] = ants_dict


def _run_fusemorph_method(state, run_d, args, omodel, result_dict, result_filedict, ftemplate):
    template_nib = state['template_nib']
    model_transform = lib_tool.get_model('mprage_transform_v002_near.onnx')
    model_transform_bili = lib_tool.get_model('mprage_transform_v002_bili.onnx')

    template_data = template_nib.get_fdata()
    fixed_image = template_data.astype(np.float32)[None, ...][None, ...]
    fixed_image = lib_reg.min_max_norm(fixed_image)

    template_seg_nib = lib_reg.get_template_seg(run_d['template'])
    template_seg_nib = reorder_img(template_seg_nib, resample='continuous')
    template_seg_data = template_seg_nib.get_fdata()
    fixed_seg_image = template_seg_data.astype(np.float32)[None, ...][None, ...]

    Af_data = state['affine_nib'].get_fdata()
    moving_image = Af_data.astype(np.float32)[None, ...][None, ...]

    moving_seg_nib = result_dict['aseg']
    moving_seg_nib = reorder_img(moving_seg_nib, resample='nearest')
    moving_seg_data = moving_seg_nib.get_fdata().astype(np.float32)
    moving_seg = _apply_affine_to_volume(
        moving_seg_data, moving_seg_nib, state, run_d, args.gpu,
        interpolation='nearestNeighbor')

    model_ff = lib_tool.get_model(omodel['reg'])
    moving_image_current = moving_image
    moving_seg_current = np.expand_dims(np.expand_dims(moving_seg, axis=0), axis=1)
    warps = []

    for _ in range(1, 4):
        output = predict(model_ff, [moving_image_current, fixed_image], GPU=args.gpu, mode='reg')
        moved, warp = output[0], output[1]
        warps.append(warp)
        output = predict(model_transform, [moving_seg_current, warp], GPU=args.gpu, mode='reg')
        moving_image_current = moved
        moving_seg_current = output[0]

    _, _, best_warp = lib_reg.optimize_fusemorph(warps, moving_seg, model_transform, fixed_seg_image, args)
    output = predict(model_transform_bili, [moving_image, best_warp], GPU=args.gpu, mode='reg')

    moved = np.squeeze(output[0])
    warp = np.squeeze(best_warp)
    moved_nib = nib.Nifti1Image(moved, state['fixed_affine'], template_nib.header)
    warp_nib = nib.Nifti1Image(warp, state['fixed_affine'], template_nib.header)

    fn = save_nib(moved_nib, ftemplate, 'Fuse')
    result_filedict['Fuse'] = fn
    result_dict['Fuse'] = moved_nib

    state['displacement_dict']["Fuse_dense_warp"] = warp
    result_dict['dense_warp'] = warp_nib
    state['warp'] = warp


def _run_registration_suite(input_nib, tbetmask_nib, ftemplate, run_d, args, omodel,
                            result_dict, result_filedict, save_reg_output, plan_steps):
    if not plan_steps:
        return None

    state = _prepare_registration_state(input_nib, tbetmask_nib, run_d)

    for step in plan_steps:
        if step == 'R':
            _run_rigid_method(state, run_d, args, omodel, result_dict, result_filedict, ftemplate)
            continue

        if step == 'A':
            _run_affine_method(state, run_d, args, omodel, result_dict, result_filedict, ftemplate)
            continue

        if step == 'V':
            if state['affine_nib'] is None:
                raise ValueError("Reg plan step 'V' requires a prior 'A' step.")
            _run_vmnet_registration_method(
                state, args, omodel, result_dict, result_filedict, ftemplate, save_reg_output)
            continue

        if step == 'N':
            _run_ants_nonlinear_method(state, 'syn', result_dict, ftemplate)
            continue

        if step == 'C':
            _run_ants_nonlinear_method(state, 'syncc', result_dict, ftemplate)
            continue

        if step == 'F':
            if state['affine_nib'] is None:
                raise ValueError("Reg plan step 'F' requires a prior 'A' step.")
            if 'aseg' not in result_dict:
                raise ValueError("Reg plan step 'F' requires Aseg from bx.run.")
            _run_fusemorph_method(state, run_d, args, omodel, result_dict, result_filedict, ftemplate)
            continue

        raise ValueError(f"Unknown registration step: {step}")

    if run_d['save_displacement']:
        np.savez(ftemplate.replace('@@@@.nii.gz', 'warp') + '.npz', **state['displacement_dict'])

    return RegistrationContext(
        ftemplate=ftemplate,
        input_nib=input_nib,
        template_nib=state['template_nib'],
        affine_nib=state['affine_nib'],
        affine_matrix=state['affine_matrix'],
        init_flow=state['init_flow'],
        warp=state['warp'],
        pad_width=state['pad_width'],
        displacement_dict=state['displacement_dict'],
    )


# ---------------------------------------------------------------------------
# Plan parsing & runner helpers
# ---------------------------------------------------------------------------

_REG_STEP_TO_FLAG = {
    'R': 'rigid',
    'A': 'affine',
    'V': 'registration',  # VMnet
    'N': 'syn',           # ANTs SyN
    'C': 'syncc',         # ANTs SyNCC
    'F': 'fusemorph',
}
_REG_STEPS = set(_REG_STEP_TO_FLAG.keys())


def _coerce_plan_string(plan):
    if plan is None:
        raise ValueError("Missing registration plan. Example: 'AF' or 'AC'.")
    if isinstance(plan, (list, tuple)):
        plan = ''.join(str(x) for x in plan)
    plan = str(plan).upper()
    for sep in (' ', ',', '+', '-', '_'):
        plan = plan.replace(sep, '')
    if not plan:
        raise ValueError("Registration plan is empty. Example: 'AF' or 'AC'.")
    return plan


def _parse_reg_plan(plan):
    plan = _coerce_plan_string(plan)
    steps = list(plan)
    invalid = [s for s in steps if s not in _REG_STEPS]
    if invalid:
        raise ValueError(
            f"Invalid reg plan step(s): {''.join(invalid)}. Allowed steps: R, A, V, N, C, F.")
    if len(set(steps)) != len(steps):
        raise ValueError(f"Duplicate reg plan steps are not supported: {plan}")

    seen = set()
    for idx, step in enumerate(steps):
        if step == 'R' and len(steps) > 1 and idx != 0:
            raise ValueError("Step 'R' must be the first step when combined with other steps.")
        if step in {'V', 'F'} and 'A' not in seen:
            raise ValueError(f"Step '{step}' requires a prior 'A' step in reg plan (e.g. 'AV', 'AF').")
        seen.add(step)
    return steps


def _get_plan_from_args(args):
    plan = getattr(args, 'plan', None)
    if plan is None:
        plan = getattr(args, 'reg_plan', None)
    return _parse_reg_plan(plan)


def _build_run_dict(args):
    run_d = dict(vars(args))
    plan_steps = _get_plan_from_args(args)
    run_d['plan_steps'] = tuple(plan_steps)
    run_d['plan'] = ''.join(plan_steps)

    for _, flag_name in _REG_STEP_TO_FLAG.items():
        run_d[flag_name] = False
    for step in plan_steps:
        run_d[_REG_STEP_TO_FLAG[step]] = True

    run_d.setdefault('bet', False)
    run_d.setdefault('aseg', False)
    run_d.setdefault('save_displacement', False)
    run_d.setdefault('affine_type', 'C2FViT')
    run_d.setdefault('template', None)
    if run_d['fusemorph']:
        run_d['aseg'] = True
    return run_d


def _build_omodel(args_model):
    """Registration-only model overrides (BX-owned models are excluded)."""
    allowed_keys = {'reg', 'affine', 'rigid'}
    omodel = {
        'reg': 'mprage_reg_v003_train.onnx',
        'affine': 'mprage_affine_v002_train.onnx',
        'rigid': 'mprage_rigid_v002_train.onnx',
    }
    if isinstance(args_model, dict):
        for mm in args_model.keys():
            if mm in allowed_keys:
                omodel[mm] = args_model[mm]
    elif isinstance(args_model, str):
        import ast
        model_dict = ast.literal_eval(args_model)
        for mm in model_dict.keys():
            if mm in allowed_keys:
                omodel[mm] = model_dict[mm]
    return omodel


def _collect_results(result_all, result_dict, result_filedict, is_single):
    """Append per-case results; return dict directly for single-file input."""
    if is_single:
        return result_dict
    result_all.append(result_filedict)
    return result_all


# ---------------------------------------------------------------------------
# Public API & entry points
# ---------------------------------------------------------------------------

def _normalize_interpolation(interpolation: str | None) -> str:
    if interpolation is None:
        return 'linear'
    interp = str(interpolation)
    if interp in {'linear', 'nearestNeighbor'}:
        return interp

    interp_lower = interp.lower()
    if interp_lower in {'linear', 'trilinear', 'bili', 'bilinear', 'continuous'}:
        return 'linear'
    if interp_lower in {'nearest', 'nearestneighbor', 'nearest_neighbor', 'nn'}:
        return 'nearestNeighbor'
    raise ValueError(f"Unknown interpolation: {interpolation!r}. Use 'linear' or 'nearestNeighbor'.")


def _coerce_dense_warp_cdhw(warp: object) -> np.ndarray:
    warp_arr = np.asarray(warp)
    if warp_arr.ndim != 4:
        raise ValueError(
            f"Dense warp must be 4-D with 3 channels; got shape {warp_arr.shape}."
        )
    if warp_arr.shape[0] == 3:
        return warp_arr
    if warp_arr.shape[-1] == 3:
        return np.moveaxis(warp_arr, -1, 0)
    raise ValueError(
        f"Dense warp must have 3 channels in axis 0 or -1; got shape {warp_arr.shape}."
    )


def apply_warp(volume_nib: nib.Nifti1Image, reg_ctx: RegistrationContext,
               interpolation: str = 'linear') -> nib.Nifti1Image:
    """Apply registration (affine + dense warp) to an image volume.

    Parameters
    ----------
    volume_nib : nib.Nifti1Image
        Input image in native space.
    reg_ctx : RegistrationContext
        Registration context returned by :func:`run_case`.
    interpolation : str
        'linear' or 'nearestNeighbor'.

    Returns
    -------
    nib.Nifti1Image
        Registered image in template space.
    """
    if reg_ctx is None:
        raise ValueError("apply_warp requires a RegistrationContext (got None).")
    if reg_ctx.template_nib is None:
        raise ValueError("apply_warp requires reg_ctx.template_nib (template metadata missing).")
    if reg_ctx.warp is None:
        raise ValueError("apply_warp requires reg_ctx.warp (dense warp missing).")
    if reg_ctx.displacement_dict is None:
        raise ValueError("apply_warp requires reg_ctx.displacement_dict (registration state missing).")

    interpolation = _normalize_interpolation(interpolation)
    resample_mode = 'nearest' if interpolation == 'nearestNeighbor' else 'continuous'

    # Registration is performed on reordered images; match that convention here.
    volume_nib = reorder_img(volume_nib, resample=resample_mode)
    volume_data = volume_nib.get_fdata().astype(np.float32)
    if volume_data.ndim != 3:
        raise ValueError(f"apply_warp expects a 3-D NIfTI image; got shape {volume_data.shape}.")

    affine_type = 'ANTs' if reg_ctx.init_flow is None else 'C2FViT'
    run_d = {'affine_type': affine_type}
    state = {
        'affine_matrix': reg_ctx.affine_matrix,
        'init_flow': reg_ctx.init_flow,
        'pad_width': reg_ctx.pad_width,
        'displacement_dict': reg_ctx.displacement_dict,
    }

    affined = _apply_affine_to_volume(
        volume_data, volume_nib, state, run_d, None,
        interpolation=interpolation,
    )

    model_name = (
        'mprage_transform_v002_near.onnx'
        if interpolation == 'nearestNeighbor'
        else 'mprage_transform_v002_bili.onnx'
    )
    model_transform = lib_tool.get_model(model_name)

    affined = np.expand_dims(np.expand_dims(affined, axis=0), axis=1)
    warp = np.expand_dims(_coerce_dense_warp_cdhw(reg_ctx.warp), axis=0)
    output = predict(model_transform, [affined, warp], GPU=None, mode='reg')
    moved = np.squeeze(output[0])

    template_nib = reg_ctx.template_nib
    return nib.Nifti1Image(moved, template_nib.affine, template_nib.header)


def modulate(volume_nib: nib.Nifti1Image, reg_ctx: RegistrationContext) -> nib.Nifti1Image:
    """Modulate a registered image by the Jacobian determinant of the warp.

    Parameters
    ----------
    volume_nib : nib.Nifti1Image
        Already-registered image in template space.
    reg_ctx : RegistrationContext
        Registration context returned by :func:`run_case`.

    Returns
    -------
    nib.Nifti1Image
        Modulated image in template space.
    """
    if reg_ctx is None:
        raise ValueError("modulate requires a RegistrationContext (got None).")
    if reg_ctx.warp is None:
        raise ValueError("modulate requires reg_ctx.warp (dense warp missing).")

    warp_cdhw = _coerce_dense_warp_cdhw(reg_ctx.warp)
    warp_dhwc = warp_cdhw.transpose(1, 2, 3, 0)
    jac = lib_reg.jacobian_determinant(warp_dhwc)

    volume = volume_nib.get_fdata().astype(np.float32)
    if volume.shape != jac.shape:
        raise ValueError(
            f"Volume shape {volume.shape} does not match Jacobian shape {jac.shape}."
        )
    modulated = volume * jac
    return nib.Nifti1Image(modulated, volume_nib.affine, volume_nib.header)


def smooth(volume_nib: nib.Nifti1Image, fwhm: float) -> nib.Nifti1Image:
    """Gaussian smoothing of a NIfTI image (FWHM in voxel units)."""
    if fwhm is None:
        raise ValueError("smooth requires fwhm (got None).")
    if fwhm <= 0:
        raise ValueError(f"smooth requires fwhm > 0 (got {fwhm}).")

    volume = volume_nib.get_fdata().astype(np.float32)
    smoothed = lib_reg.apply_gaussian_smoothing(volume, fwhm=float(fwhm))
    return nib.Nifti1Image(smoothed, volume_nib.affine, volume_nib.header)


def run_case(args, f, *, common_folder=None,
             save_reg_output=True, verbose=None):
    """Run a single registration case and return registration context.

    This is the explicit wrapper used by higher-level pipelines (e.g. VBM) to
    compose registration functionality without going through CLI-style `run_args`.
    """
    if verbose is None:
        verbose = getattr(args, 'verbose', 0)

    def _dbg(*msg):
        if verbose >= 2:
            _logger.debug(' '.join(str(x) for x in msg))

    run_d = _build_run_dict(args)
    plan_steps = list(run_d['plan_steps'])

    omodel = _build_omodel(getattr(args, 'model', None))
    result_dict = {}
    result_filedict = {}

    ftemplate, _ = get_template(f, args.output, args.gz, common_folder)

    bx_result = _run_bx_prereq(f, run_d, args.gpu, verbose)
    tbetmask_nib = bx_result['tbetmask']
    qc_score = bx_result['QC']
    input_nib = nib.load(f)
    tbet_nib = bx_result.get('tbet')
    if tbet_nib is None and run_d.get('bet', False):
        tbet_arr = lib_tool.read_nib(input_nib) * lib_tool.read_nib(tbetmask_nib)
        if lib_tool.check_dtype(tbet_arr, input_nib.dataobj.dtype):
            tbet_arr = tbet_arr.astype(input_nib.dataobj.dtype)
        tbet_nib = nib.Nifti1Image(tbet_arr, input_nib.affine, input_nib.header)

    result_dict['QC'] = qc_score
    result_dict['reg_plan'] = ''.join(plan_steps)

    if run_d.get('bet', False):
        fn = save_nib(tbet_nib, ftemplate, 'tbet')
        result_dict['tbet'] = tbet_nib
        result_filedict['tbet'] = fn

    if run_d.get('aseg', False):
        result_nib = bx_result['aseg']
        if not run_d['fusemorph']:
            fn = save_nib(result_nib, ftemplate, 'aseg')
            result_filedict['aseg'] = fn
        result_dict['aseg'] = result_nib

    reg_ctx = _run_registration_suite(
        input_nib, tbetmask_nib, ftemplate, run_d, args, omodel,
        result_dict, result_filedict, save_reg_output, plan_steps)

    return result_dict, result_filedict, reg_ctx


def run_args(args):
    if getattr(args, 'vbm', False):
        raise ValueError("VBM pipeline is not supported in tigerbx.reg.run_args; "
                         "use tigerbx.vbm or tigerbx.pipeline.vbm.run_args().")

    verbose = getattr(args, 'verbose', 0)

    def printer(*msg):
        if verbose >= 1:
            _logger.info(' '.join(str(x) for x in msg))

    plan_steps = _get_plan_from_args(args)
    input_file_list, common_folder = resolve_inputs(args.input)

    # Validate/normalize model overrides early (shared helper also used by run_case).
    _build_omodel(getattr(args, 'model', None))

    printer('Reg plan:', ''.join(plan_steps))
    printer('Total nii files:', len(input_file_list))
    result_all = []
    is_single = len(input_file_list) == 1
    _pbar = tqdm(input_file_list, desc='tigerbx-reg', unit='file', disable=(verbose > 0))
    for count, f in enumerate(_pbar, 1):
        _pbar.set_postfix_str(os.path.basename(f))
        printer(f'{count} Preprocessing :', os.path.basename(f))
        t = time.time()
        result_dict, result_filedict, _ = run_case(
            args, f, common_folder=common_folder,
            save_reg_output=True, verbose=verbose)

        if 'QC' in result_dict:
            printer('QC score:', result_dict['QC'])

        printer('Processing time: %d seconds' % (time.time() - t))
        result_all = _collect_results(result_all, result_dict, result_filedict, is_single)
    return result_all


def reg(plan, input=None, output=None, model=None, template=None, gpu=False, gz=False,
        save_displacement=False, affine_type='C2FViT', verbose=0):
    args = Namespace()
    if not isinstance(input, list):
        input = [input]
    args.plan = plan
    args.input = input
    args.output = output
    args.model = model
    args.gpu = gpu
    args.gz = gz
    args.template = template
    args.save_displacement = save_displacement
    args.affine_type = affine_type
    args.verbose = verbose
    return run_args(args)
