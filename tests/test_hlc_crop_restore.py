import numpy as np

from tigerbx.lib_crop import crop_cube, restore_result


def test_crop_restore_roundtrip_no_padding():
    rng = np.random.default_rng(0)
    image = rng.normal(size=(11, 7, 13)).astype(np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)

    mask[2:6, 1:4, 4:10] = 1

    cube, xyz6 = crop_cube(image, mask, padding=0)
    restored = restore_result(image.shape, cube, xyz6)

    x_min, x_max, y_min, y_max, z_min, z_max = xyz6
    assert np.array_equal(
        restored[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1],
        cube,
    )

    outside = restored.copy()
    outside[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1] = 0
    assert np.all(outside == 0)


def test_crop_padding_clips_to_bounds():
    image = np.ones((10, 9, 8), dtype=np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[0, 0, 0] = 1

    cube, xyz6 = crop_cube(image, mask, padding=5)
    x_min, x_max, y_min, y_max, z_min, z_max = xyz6

    assert (x_min, y_min, z_min) == (0, 0, 0)
    assert cube.shape == (x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1)

    restored = restore_result(image.shape, np.full(cube.shape, 2, dtype=cube.dtype), xyz6)
    assert np.all(restored[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1] == 2)
