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
from tigerbx.modules.bx import run as bx_run
from tigerbx.core.io import get_template, load_nifti, save_nib, resolve_inputs
from tigerbx.core.onnx import predict
from tigerbx.core.deform import jacobian_determinant
from tigerbx.core.resample import reorder_img
from tigerbx.core import models, numreg, templatereg
from tigerbx.core.warp import affine_transform, dense_transform, init_shift
from tigerbx.core.warpfile import save_warp

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
    """Apply affine transform to a volume using C2FViT or numreg.

    Parameters
    ----------
    volume_data : np.ndarray
        3-D float32 array (already reordered).
    volume_nib : nib.Nifti1Image
        NIfTI image for affine metadata (used by the numreg path).
    state : dict
        Registration state with 'affine_matrix', 'init_flow', 'pad_width',
        'displacement_dict'.
    run_d : dict
        Run dictionary (needs 'affine_type').
    gpu : bool
        Unused (kept for call compatibility; the C2FViT resampling runs in numpy).
    interpolation : str
        'nearestNeighbor' or 'linear', for every affine type.
    """
    if run_d['affine_type'] == 'numreg':
        ref = state['displacement_dict']['reference_info']
        ref = Namespace(affine=np.asarray(ref['affine']), shape=tuple(ref['shape']))
        result = numreg.apply(volume_nib, ref, state['displacement_dict']['affine_matrix']['M'],
                              order=0 if interpolation == 'nearestNeighbor' else 1, data=volume_data)
    else:
        vol, _ = lib_reg.pad_to_shape(volume_data, (256, 256, 256))
        vol, _ = lib_reg.crop_image(vol, target_shape=(256, 256, 256))
        result = affine_transform(vol, state['init_flow'], state['affine_matrix'], interpolation)
        result = lib_reg.remove_padding(result, state['pad_width'])
    return result


def _run_bx_prereq(f, run_d, gpu, verbose):
    """Fetch tBX / TSEG prerequisites via bx.run without writing outputs."""
    bx_flags = ['m']  # reg pipeline always needs the tBX mask
    if run_d.get('tbx'):
        bx_flags.append('b')
    if run_d.get('tseg'):
        bx_flags.append('t')
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
        "VFA_dense_warp": None,
    }


def _prepare_registration_state(input_nib, tbxmask_nib, run_d):
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

    tbx = lib_tool.read_nib(input_nib) * lib_tool.read_nib(tbxmask_nib)
    # float: casting back to the on-disk dtype overflows int16 images with a large scl_slope
    tbx = tbx.astype(np.float32)
    tbx_nib = nib.Nifti1Image(tbx, input_nib.affine, input_nib.header)
    # the input grid: outputs are sampled once from it (reordering an oblique / anisotropic image interpolates)
    state['native_nib'] = tbx_nib

    tbx_nib = reorder_img(tbx_nib, resample='continuous')
    state['tbx_nib'] = tbx_nib
    state['ori_affine'] = tbx_nib.affine
    state['tbx_data'] = tbx_nib.get_fdata()

    template_nib = lib_reg.get_template(run_d['template'])
    template_nib = reorder_img(template_nib, resample='continuous')
    state['template_nib'] = template_nib
    state['fixed_affine'] = template_nib.affine
    return state


def _run_rigid_method(state, run_d, args, omodel, result_dict, result_filedict, ftemplate, save_reg_output=True):
    """Step 'R': rigid (rotation + translation) registration to the template with numreg (dof=6)."""
    template_nib = state['template_nib']
    M = numreg.register(state['tbx_nib'], template_nib, moving_data=state['tbx_data'], dof=6)
    rigided = lib_reg.min_max_norm(numreg.apply(state['tbx_nib'], template_nib, M, order=1, data=state['tbx_data']))
    state['displacement_dict']['rigid_matrix'] = {'M': M}
    state['displacement_dict']['reference_info'] = {'affine': template_nib.affine,
                                                    'shape': tuple(template_nib.shape[:3])}
    rigid_nib = nib.Nifti1Image(rigided, state['fixed_affine'])
    if save_reg_output:
        result_filedict['rigid'] = save_nib(rigid_nib, ftemplate, 'rigid')
    result_dict['rigid'] = rigid_nib


def _run_affine_method(state, run_d, args, omodel, result_dict, result_filedict, ftemplate, save_reg_output=True):
    template_nib = state['template_nib']
    template_data = template_nib.get_fdata()
    tbx_data = state['tbx_data']

    if run_d['affine_type'] == 'C2FViT':
        tbx_data, _ = lib_reg.pad_to_shape(tbx_data, (256, 256, 256))
        tbx_data, _ = lib_reg.crop_image(tbx_data, target_shape=(256, 256, 256))
        template_data, pad_width = lib_reg.pad_to_shape(template_data, (256, 256, 256))
        state['pad_width'] = pad_width

    moving = tbx_data.astype(np.float32)[None, ...][None, ...]
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
    elif run_d['affine_type'] == 'numreg':
        # unclipped template: numreg builds the fixed mask from voxels above the background value
        M = numreg.register(state['tbx_nib'], template_nib, moving_data=tbx_data)
        # both sampled once from the input grid: _affine cubic (sharper), the VFA input trilinear (as VFA expects)
        native = state['native_nib']
        affined = lib_reg.min_max_norm(numreg.resample(native, template_nib, M, order=3))
        vfa_input = lib_reg.min_max_norm(numreg.resample(native, template_nib, M, order=1))
        affine_matrix = {'affine_matrix': {'M': M}}
        state['displacement_dict']['reference_info'] = {'affine': template_nib.affine,
                                                        'shape': tuple(template_nib.shape[:3])}
        state['displacement_dict'].update(affine_matrix)
        state['affine_matrix'] = affine_matrix
    else:
        raise ValueError(f"Unknown affine_type: {run_d['affine_type']!r} (use 'numreg' or 'C2FViT').")

    result_dict['Affine_matrix'] = state['affine_matrix']
    if run_d['affine_type'] == 'C2FViT':
        affined = lib_reg.remove_padding(affined, state['pad_width'])
    affine_nib = nib.Nifti1Image(affined, state['fixed_affine'])
    state['affine_nib'] = affine_nib if run_d['affine_type'] == 'C2FViT' else \
        nib.Nifti1Image(vfa_input, state['fixed_affine'])        # the input of the deformable step

    if save_reg_output:
        result_filedict['affine'] = save_nib(affine_nib, ftemplate, 'affine')
    result_dict['affine'] = affine_nib


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
        fn = save_nib(moved_nib, ftemplate, 'VXM')
        result_filedict['VXM'] = fn
    result_dict['VXM'] = moved_nib

    state['displacement_dict']["dense_warp"] = warp
    result_dict['dense_warp'] = warp_nib
    state['warp'] = warp


def _run_vfa_registration_method(state, args, omodel, result_dict, result_filedict, ftemplate, save_reg_output):
    """Deformable step 'D': VFA. The ONNX model takes [moving, fixed] (min-max normalised inside the graph) and
    returns a voxel sampling grid; warp = grid - identity. After numreg the output is the input sampled once at
    the affine + warp (cubic); after C2FViT the affine image warped with dense_transform like VMnet's."""
    template_nib = state['template_nib']
    Af_data = state['affine_nib'].get_fdata().astype(np.float32)
    grid = templatereg.vfa_grid(Af_data, template_nib, omodel['vfa'], args.gpu)
    identity = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float32) for n in grid.shape[1:]], indexing='ij'))
    warp = grid - identity
    M = (state['affine_matrix'] or {}).get('affine_matrix', {}).get('M') if state['init_flow'] is None else None
    if M is not None:
        # numreg: affine + warp composed, one cubic sample from the input grid (not the affine image warped again)
        moved = lib_reg.min_max_norm(numreg.resample(state['native_nib'], template_nib, M, warp=warp, order=3))
    else:
        moved = dense_transform(Af_data, warp, 'linear')
    moved_nib = nib.Nifti1Image(moved, state['fixed_affine'], template_nib.header)
    warp_nib = nib.Nifti1Image(warp, state['fixed_affine'], template_nib.header)

    if save_reg_output:
        fn = save_nib(moved_nib, ftemplate, 'VFA')
        result_filedict['VFA'] = fn
    result_dict['VFA'] = moved_nib

    state['displacement_dict']["VFA_dense_warp"] = warp
    result_dict['dense_warp'] = warp_nib
    state['warp'] = warp


def warp_file_path(ftemplate):
    """<stem>_warp.npz for an output template <stem>_@@@@.nii[.gz]."""
    return ftemplate.split('@@@@')[0] + 'warp.npz'


def save_warp_file(path, disp, template_nib, input_nib):
    """displacement_dict of a registration -> plain-array _warp.npz (core.warpfile)."""
    aff, rig = disp.get('affine_matrix'), disp.get('rigid_matrix')
    c2f = disp.get('init_flow') is not None
    return save_warp(path, template_nib.affine, template_nib.shape, input_nib.affine, input_nib.shape,
                     affine_M=aff['M'] if isinstance(aff, dict) else None,
                     rigid_M=rig['M'] if isinstance(rig, dict) else None,
                     vfa_warp=disp.get('VFA_dense_warp'), vxm_warp=disp.get('dense_warp'),
                     c2fvit_matrix=aff if c2f else None,
                     c2fvit_init_shift=init_shift(disp['init_flow']) if c2f else None)


def _run_registration_suite(input_nib, tbxmask_nib, ftemplate, run_d, args, omodel,
                            result_dict, result_filedict, save_reg_output, plan_steps):
    if not plan_steps:
        return None

    state = _prepare_registration_state(input_nib, tbxmask_nib, run_d)

    for step in plan_steps:
        if step == 'R':
            _run_rigid_method(state, run_d, args, omodel, result_dict, result_filedict, ftemplate, save_reg_output)
            continue

        if step == 'A':
            _run_affine_method(state, run_d, args, omodel, result_dict, result_filedict, ftemplate, save_reg_output)
            continue

        if step == 'V':
            if state['affine_nib'] is None:
                raise ValueError("Reg plan step 'V' requires a prior 'A' step.")
            _run_vmnet_registration_method(
                state, args, omodel, result_dict, result_filedict, ftemplate, save_reg_output)
            continue

        if step == 'D':
            if state['affine_nib'] is None:
                raise ValueError("Reg plan step 'D' requires a prior 'A' step.")
            _run_vfa_registration_method(
                state, args, omodel, result_dict, result_filedict, ftemplate, save_reg_output)
            continue

        raise ValueError(f"Unknown registration step: {step}")

    if run_d['save_displacement']:
        save_warp_file(warp_file_path(ftemplate), state['displacement_dict'],
                       state['template_nib'], input_nib)

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
    'D': 'vfa',           # VFA (deformable)
}
_REG_STEPS = set(_REG_STEP_TO_FLAG.keys())


def _coerce_plan_string(plan):
    if plan is None:
        raise ValueError("Missing registration plan. Example: 'AD' or 'AV'.")
    if isinstance(plan, (list, tuple)):
        plan = ''.join(str(x) for x in plan)
    plan = str(plan).upper()
    for sep in (' ', ',', '+', '-', '_'):
        plan = plan.replace(sep, '')
    if not plan:
        raise ValueError("Registration plan is empty. Example: 'AD' or 'AV'.")
    return plan


def _parse_reg_plan(plan):
    plan = _coerce_plan_string(plan)
    steps = list(plan)
    if 'F' in steps:
        raise ValueError("Reg plan step 'F' (FuseMorph) was removed; use 'AD' (affine + VFA) instead.")
    for s in ('N', 'C'):
        if s in steps:
            raise ValueError(f"Reg plan step '{s}' (ANTs SyN / SyNCC) was removed; use 'AD' (affine + VFA) instead.")
    invalid = [s for s in steps if s not in _REG_STEPS]
    if invalid:
        raise ValueError(
            f"Invalid reg plan step(s): {''.join(invalid)}. Allowed steps: R, A, V, D.")
    if len(set(steps)) != len(steps):
        raise ValueError(f"Duplicate reg plan steps are not supported: {plan}")

    seen = set()
    for idx, step in enumerate(steps):
        if step == 'R' and len(steps) > 1 and idx != 0:
            raise ValueError("Step 'R' must be the first step when combined with other steps.")
        if step in {'V', 'D'} and 'A' not in seen:
            raise ValueError(f"Step '{step}' requires a prior 'A' step in reg plan (e.g. 'AD', 'AV').")
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

    run_d.setdefault('tbx', False)
    run_d.setdefault('tseg', False)
    run_d.setdefault('save_displacement', False)
    run_d.setdefault('affine_type', 'numreg')
    run_d.setdefault('template', None)
    if run_d['affine_type'] == 'ANTs':      # ANTs affine was replaced by numreg (same accuracy, no antspyx)
        warnings.warn("affine_type='ANTs' was removed; using 'numreg' instead.", FutureWarning, stacklevel=3)
        run_d['affine_type'] = 'numreg'
    return run_d


def _build_omodel(args_model):
    """Registration-only model overrides (BX-owned models are excluded)."""
    allowed_keys = {'reg', 'vfa', 'affine'}
    omodel = {
        'reg': models.VMNET,
        'vfa': models.VFA,
        'affine': models.AFFINE_NET,
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
    template_nib = reg_ctx.template_nib
    if reg_ctx.init_flow is None:
        # numreg: affine + warp composed, one sample from the image's own grid (nearest or trilinear), not
        # reorder -> affine -> warp (three interpolations; reg60 label Dice 0.866 vs 0.852)
        volume_data = np.squeeze(np.asarray(volume_nib.dataobj, dtype=np.float32))
        if volume_data.ndim != 3:
            raise ValueError(f"apply_warp expects a 3-D NIfTI image; got shape {volume_data.shape}.")
        moved = numreg.resample(volume_nib, template_nib, reg_ctx.displacement_dict['affine_matrix']['M'],
                                warp=_coerce_dense_warp_cdhw(reg_ctx.warp),
                                order=0 if interpolation == 'nearestNeighbor' else 1, data=volume_data)
        return nib.Nifti1Image(moved, template_nib.affine, template_nib.header)

    resample_mode = 'nearest' if interpolation == 'nearestNeighbor' else 'continuous'
    # C2FViT: registration is performed on reordered images; match that convention here.
    volume_nib = reorder_img(volume_nib, resample=resample_mode)
    volume_data = volume_nib.get_fdata().astype(np.float32)
    if volume_data.ndim != 3:
        raise ValueError(f"apply_warp expects a 3-D NIfTI image; got shape {volume_data.shape}.")

    if reg_ctx.init_flow is not None:
        affine_type = 'C2FViT'
    else:
        affine_type = 'numreg'
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

    moved = dense_transform(affined, _coerce_dense_warp_cdhw(reg_ctx.warp), interpolation)
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
    jac = jacobian_determinant(warp_dhwc)

    volume = volume_nib.get_fdata().astype(np.float32)
    if volume.shape != jac.shape:
        raise ValueError(
            f"Volume shape {volume.shape} does not match Jacobian shape {jac.shape}."
        )
    modulated = volume * jac
    return nib.Nifti1Image(modulated, volume_nib.affine, volume_nib.header)


def smooth(volume_nib: nib.Nifti1Image, fwhm: float) -> nib.Nifti1Image:
    """Gaussian smoothing of a NIfTI image (FWHM in mm, converted per axis with the voxel size)."""
    if fwhm is None:
        raise ValueError("smooth requires fwhm (got None).")
    if fwhm <= 0:
        raise ValueError(f"smooth requires fwhm > 0 (got {fwhm}).")

    volume = volume_nib.get_fdata().astype(np.float32)
    zooms = np.asarray(volume_nib.header.get_zooms()[:3], dtype=float)
    smoothed = lib_reg.apply_gaussian_smoothing(volume, fwhm=float(fwhm) / zooms)
    return nib.Nifti1Image(smoothed, volume_nib.affine, volume_nib.header)


def run_case(args, f, *, common_folder=None,
             save_reg_output=True, verbose=None, bx_result=None):
    """Run a single registration case and return registration context.

    This is the explicit wrapper used by higher-level pipelines (e.g. VBM) to
    compose registration functionality without going through CLI-style `run_args`.
    bx_result: an existing ``tigerbx.run`` result for *f* with at least ``tbxmask`` (e.g. from
    ``run('mC')``), so the caller's tBX is reused instead of running tBX again.
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

    if bx_result is None or (run_d.get('tseg') and 'tseg' not in bx_result):
        bx_result = _run_bx_prereq(f, run_d, args.gpu, verbose)
    tbxmask_nib = bx_result['tbxmask']
    input_nib = load_nifti(f)
    tbx_nib = bx_result.get('tbx')
    if tbx_nib is None and run_d.get('tbx', False):
        tbx_arr = lib_tool.read_nib(input_nib) * lib_tool.read_nib(tbxmask_nib)
        if lib_tool.check_dtype(tbx_arr, input_nib.dataobj.dtype):
            tbx_arr = tbx_arr.astype(input_nib.dataobj.dtype)
        tbx_nib = nib.Nifti1Image(tbx_arr, input_nib.affine, input_nib.header)

    result_dict['reg_plan'] = ''.join(plan_steps)

    if run_d.get('tbx', False):
        fn = save_nib(tbx_nib, ftemplate, 'tbx')
        result_dict['tbx'] = tbx_nib
        result_filedict['tbx'] = fn

    if run_d.get('tseg', False):
        result_nib = bx_result['tseg']
        fn = save_nib(result_nib, ftemplate, 'tseg')
        result_filedict['tseg'] = fn
        result_dict['tseg'] = result_nib

    reg_ctx = _run_registration_suite(
        input_nib, tbxmask_nib, ftemplate, run_d, args, omodel,
        result_dict, result_filedict, save_reg_output, plan_steps)

    return result_dict, result_filedict, reg_ctx


def run_args(args):
    if getattr(args, 'vbm', False):
        raise ValueError("VBM pipeline is not supported in tigerbx.reg.run_args; "
                         "use tigerbx.vbm or tigerbx.pipelines.vbm.run_args().")

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

        printer('Processing time: %d seconds' % (time.time() - t))
        result_all = _collect_results(result_all, result_dict, result_filedict, is_single)
    return result_all


def reg(plan, input=None, output=None, model=None, template=None, gpu=False, gz=False,
        save_displacement=False, affine_type='numreg', verbose=0):
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
