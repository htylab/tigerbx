import os
import re
from typing import List, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter

PATCH_SIZE_ENV = "TIGERBX_PATCH_SIZE"
DEFAULT_PATCH_SIZE = (128, 128, 128)
MIN_CROP_SIZE = (160, 160, 160)
ORT_INTRA_THREADS_ENV = "TIGERBX_ORT_INTRA_THREADS"
ORT_INTER_THREADS_ENV = "TIGERBX_ORT_INTER_THREADS"
DEFAULT_ORT_INTRA_THREADS_RATIO = 0.7
DEFAULT_ORT_INTER_THREADS = 1


def _parse_patch_size(value: str):
    value = value.strip()
    if not value:
        raise ValueError("empty patch size")

    if value.isdigit():
        size = int(value)
        return (size, size, size)

    parts = [p for p in re.split(r"[x, ]+", value) if p]
    if len(parts) != 3:
        raise ValueError(f"patch size must be like '128' or '128,128,128' (got {value!r})")
    return tuple(int(p) for p in parts)


def _resolve_patch_size(patch_size):
    if patch_size is None:
        env = os.environ.get(PATCH_SIZE_ENV)
        if env:
            patch_size = _parse_patch_size(env)
        else:
            patch_size = DEFAULT_PATCH_SIZE
    elif isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)

    if len(patch_size) != 3:
        raise ValueError("patch_size must be an int or a 3-tuple like (128, 128, 128)")

    patch_size = tuple(int(s) for s in patch_size)
    if not all(s > 0 for s in patch_size):
        raise ValueError(f"patch_size must be > 0 (got {patch_size})")
    if not all(s < MIN_CROP_SIZE[i] for i, s in enumerate(patch_size)):
        raise ValueError(
            f"patch_size must be < {MIN_CROP_SIZE} to match MIN_CROP (got {patch_size})."
        )
    return patch_size


def cpu_count():
    """Return available CPUs, preferring affinity-limited count when available."""
    try:
        # Linux cpuset/affinity aware (and much simpler than legacy heuristics).
        count = len(os.sched_getaffinity(0))
        if count > 0:
            return count
    except Exception:
        pass

    count = os.cpu_count()
    if count is None or count < 1:
        return 1
    return count


def _get_env_positive_int(name):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer (got {raw!r})") from exc
    if value < 1:
        raise ValueError(f"{name} must be >= 1 (got {raw!r})")
    return value


def _session_data_type(session):
    if session.get_inputs()[0].type == "tensor(float)":
        return "float32"
    return "float64"


def create_session(model_ff, GPU):
    """Build an ort.InferenceSession for the given model path."""
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    so = ort.SessionOptions()
    intra_threads = _get_env_positive_int(ORT_INTRA_THREADS_ENV)
    if intra_threads is None:
        intra_threads = max(int(cpu_count() * DEFAULT_ORT_INTRA_THREADS_RATIO), 1)
    inter_threads = _get_env_positive_int(ORT_INTER_THREADS_ENV)
    if inter_threads is None:
        inter_threads = DEFAULT_ORT_INTER_THREADS

    so.intra_op_num_threads = intra_threads
    so.inter_op_num_threads = inter_threads
    so.log_severity_level = 3
    if GPU and ort.get_device() == "GPU":
        return ort.InferenceSession(
            model_ff,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            sess_options=so,
        )
    return ort.InferenceSession(
        model_ff,
        providers=["CPUExecutionProvider"],
        sess_options=so,
    )


def predict(
    model,
    data,
    GPU,
    mode=None,
    patch_size=None,
    tile_step_size=0.5,
    gaussian=True,
    session=None,
):
    if session is None:
        session = create_session(model, GPU)

    data_type = _session_data_type(session)
    if mode == "reg":
        input_names = [input.name for input in session.get_inputs()]
        inputs = {input_names[0]: data[0], input_names[1]: data[1]}
        return session.run(None, inputs)
    if mode == "affine_transform":
        input_names = [input.name for input in session.get_inputs()]
        inputs = {input_names[0]: data[0], input_names[1]: data[1], input_names[2]: data[2]}
        return session.run(None, inputs)
    if mode == "encode":
        mu, sigma = session.run(None, {session.get_inputs()[0].name: data.astype(data_type)})
        return mu, sigma

    if mode == "decode":
        result = session.run(None, {session.get_inputs()[0].name: data.astype(data_type)})
        return result[0]

    if mode == "patch":
        patch_size = _resolve_patch_size(patch_size)
        input_shape = session.get_inputs()[0].shape
        if input_shape is not None and len(input_shape) >= 5:
            expected_spatial = input_shape[-3:]
            if all(isinstance(s, int) for s in expected_spatial):
                if tuple(expected_spatial) != tuple(patch_size):
                    raise ValueError(
                        f"Model expects fixed spatial dims {tuple(expected_spatial)}, "
                        f"but patch_size is {tuple(patch_size)}. "
                        f"Set {PATCH_SIZE_ENV} or pass patch_size=({expected_spatial[0]},{expected_spatial[1]},{expected_spatial[2]})"
                    )

        try:
            logits = patch_inference_3d_lite(
                session,
                data.astype(data_type),
                patch_size=patch_size,
                tile_step_size=tile_step_size,
                gaussian=gaussian,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Patch inference failed (patch_size={patch_size}, tile_step_size={tile_step_size}, gaussian={gaussian})."
            ) from exc

        return logits

    return session.run(None, {session.get_inputs()[0].name: data.astype(data_type)})[0]


def predict_single_output(session, data):
    data_type = _session_data_type(session)
    return session.run(None, {session.get_inputs()[0].name: data.astype(data_type)})[0]


def encode_latent(enc_sess, vol):
    return enc_sess.run(None, {enc_sess.get_inputs()[0].name: vol.astype(np.float32)})


def decode_latent(dec_sess, latent):
    return dec_sess.run(None, {dec_sess.get_inputs()[0].name: latent})[0]


def patch_inference_3d_lite(
    session,
    vol_d: np.ndarray,
    patch_size: Tuple[int, ...] = (128,) * 3,
    tile_step_size: float = 0.5,
    gaussian=True,
):
    patches, point_list = img_to_patches(vol_d, patch_size, tile_step_size)
    gaussian_map = compute_gaussian(patch_size) if gaussian else None
    patch_logits_shape = session.run(None, {session.get_inputs()[0].name: patches[0]})[0].shape
    prob_tensor = np.zeros(((patch_logits_shape[1],) + vol_d.shape[-3:]))
    weight_tensor = np.zeros(vol_d.shape[-3:])
    if gaussian:
        weight_patch = gaussian_map
    else:
        weight_patch = np.ones(patch_size, dtype=weight_tensor.dtype)
    for p in point_list:
        weight_tensor[p[0] : p[0] + patch_size[0], p[1] : p[1] + patch_size[1], p[2] : p[2] + patch_size[2]] += weight_patch
    for patch, p in zip(patches, point_list):
        logits = session.run(None, {session.get_inputs()[0].name: patch})[0]
        output_patch = logits.squeeze(0) * gaussian_map if gaussian else logits.squeeze(0)
        prob_tensor[:, p[0] : p[0] + patch_size[0], p[1] : p[1] + patch_size[1], p[2] : p[2] + patch_size[2]] += output_patch
    prob_tensor = prob_tensor / weight_tensor
    return prob_tensor[np.newaxis, :]


def compute_steps_for_sliding_window(
    image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float
) -> List[List[int]]:
    assert all(i >= j for i, j in zip(image_size, tile_size)), "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, "step_size must be larger than 0 and smaller or equal to 1"

    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]
    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
        steps.append(steps_here)
    return steps


def compute_gaussian(
    tile_size: Union[Tuple[int, ...], List[int]],
    sigma_scale: float = 1.0 / 8,
    value_scaling_factor: float = 1,
    dtype=np.float16,
) -> np.ndarray:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode="constant", cval=0)

    gaussian_importance_map /= np.max(gaussian_importance_map) / value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.astype(dtype)
    mask = gaussian_importance_map == 0
    gaussian_importance_map[mask] = np.min(gaussian_importance_map[~mask])
    return gaussian_importance_map


def img_to_patches(vol_d: np.ndarray, patch_size: Tuple[int, ...], tile_step_size: float):
    steps = compute_steps_for_sliding_window(vol_d.shape[-3:], patch_size, tile_step_size)
    slice_list = []
    point_list = [[i, j, k] for i in steps[0] for j in steps[1] for k in steps[2]]
    for p in point_list:
        slice_input = vol_d[:, :, p[0] : p[0] + patch_size[0], p[1] : p[1] + patch_size[1], p[2] : p[2] + patch_size[2]]
        slice_list.append(slice_input)
    return np.concatenate([s[np.newaxis, ...] for s in slice_list], axis=0), point_list

