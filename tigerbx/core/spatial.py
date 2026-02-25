import numpy as np


def pad_to_shape(img, target_shape):
    """Pad with zeros to the requested shape and return pad metadata."""
    padding = [(max(0, t - s)) for s, t in zip(img.shape, target_shape)]
    pad_width = [(p // 2, p - (p // 2)) for p in padding]
    padded_img = np.pad(img, pad_width, mode="constant", constant_values=0)
    return padded_img, pad_width


def crop_image(image, target_shape):
    """Crop centrally to the requested shape and return the crop slices."""
    current_shape = image.shape
    crop_slices = []

    for i in range(len(target_shape)):
        start = (current_shape[i] - target_shape[i]) // 2
        end = start + target_shape[i]
        crop_slices.append(slice(start, end))

    cropped_image = image[tuple(crop_slices)]
    return cropped_image, crop_slices


def remove_padding(padded_img, pad_width):
    """Remove padding created by :func:`pad_to_shape`."""
    slices = [slice(p[0], -p[1] if p[1] != 0 else None) for p in pad_width]
    cropped_img = padded_img[tuple(slices)]
    return cropped_img


def resize_with_pad_or_crop(image, image_size):
    slices = tuple()
    padding = []
    for i, size in enumerate(image_size):
        if image.shape[i] > size:
            d = image.shape[i] - size
            slices += (slice(int(d // 2), int(image.shape[i]) - int(d / 2 + 0.5)),)
            padding.append((0, 0))
        elif image.shape[i] < size:
            d = size - image.shape[i]
            slices += (slice(0, image.shape[i]),)
            padding.append((int(d // 2), int(d / 2 + 0.5)))
        else:
            slices += (slice(0, image.shape[i]),)
            padding.append((0, 0))

    image_resize = image.copy()
    image_resize = image_resize[slices]
    image_resize = np.pad(
        image_resize,
        padding,
        "constant",
        constant_values=np.zeros((len(image_size), 2)),
    )

    return image_resize


def _expand_bounds_to_min_size(start: int, end: int, min_size: int, axis_size: int):
    current = end - start + 1
    if current >= min_size:
        return start, end
    if axis_size < min_size:
        raise ValueError(f"axis_size ({axis_size}) must be >= min_size ({min_size})")

    center = (start + end) // 2
    new_start = center - (min_size // 2)
    if new_start < 0:
        new_start = 0
    new_end = new_start + min_size - 1
    if new_end >= axis_size:
        new_end = axis_size - 1
        new_start = new_end - min_size + 1

    return new_start, new_end


def crop_cube(ABC, tbetmask_image, padding=16, min_size=None):
    """
    Crop the 3D region with mask > 0 and optionally extend to a minimum size.

    Input array shape is (X, Y, Z). Returns the cropped cube and boundary list xyz6,
    ordered as [x_min, x_max, y_min, y_max, z_min, z_max] (inclusive bounds).
    """
    non_zero = np.where(tbetmask_image > 0)
    if len(non_zero[0]) == 0:
        raise ValueError("No region with signal > 0 found in the image")

    x_min, x_max = int(np.min(non_zero[0])), int(np.max(non_zero[0]))
    y_min, y_max = int(np.min(non_zero[1])), int(np.max(non_zero[1]))
    z_min, z_max = int(np.min(non_zero[2])), int(np.max(non_zero[2]))

    X, Y, Z = ABC.shape
    x_min = max(0, x_min - padding)
    x_max = min(X - 1, x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(Y - 1, y_max + padding)
    z_min = max(0, z_min - padding)
    z_max = min(Z - 1, z_max + padding)

    if min_size is not None:
        if len(min_size) != 3:
            raise ValueError("min_size must be a 3-tuple like (x, y, z)")
        x_min, x_max = _expand_bounds_to_min_size(x_min, x_max, int(min_size[0]), X)
        y_min, y_max = _expand_bounds_to_min_size(y_min, y_max, int(min_size[1]), Y)
        z_min, z_max = _expand_bounds_to_min_size(z_min, z_max, int(min_size[2]), Z)

    xyz6 = [x_min, x_max, y_min, y_max, z_min, z_max]
    cube = ABC[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
    return cube, xyz6


def restore_result(ABC_shape, result, xyz6):
    """
    Place a processed cube back into a zero array with the original size.

    ABC_shape: shape of the original image (X, Y, Z)
    result: processed cube (must match the shape implied by xyz6)
    xyz6: [x_min, x_max, y_min, y_max, z_min, z_max] (inclusive bounds)
    """
    output = np.zeros(ABC_shape, dtype=result.dtype)
    x_min, x_max, y_min, y_max, z_min, z_max = xyz6
    output[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1] = result
    return output
