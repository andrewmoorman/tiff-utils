import os
import pytest

def assert_tiff(path, shape, subresolutions=4):
    import numpy as np
    from tiff_utils import get_image_reader, count_levels

    # assert pyramidal structure
    levels = subresolutions + 1
    for level in range(levels):
        with get_image_reader(path, level=level) as f:
            img = f.read(T=0, C=0)
            assert f.X == img.shape[1]
            assert f.Y == img.shape[0]
            assert f.X <= shape[0]
            assert f.Y <= shape[1]
            assert len(f.shape) == 5
            assert len(img.shape) == 2
            assert np.max(img) <= 255
            assert np.min(img) >= 0
            shape = (f.X, f.Y)

    # assert pyramidal depth
    def try_read():
        with get_image_reader(path, level=levels) as f:
            return f.read(T=0, C=0)
    pytest.raises(ValueError, try_read)

    # assert levels counter
    assert count_levels(path) == levels

def test_convert(path_dice, path_temp):
    from tiff_utils import convert_to_tiff

    # convert
    subresolutions = 3
    convert_to_tiff(
        path_dice,
        path_temp,
        subresolutions=subresolutions,
    )
    assert os.path.exists(path_temp)

    # read
    assert_tiff(path_temp, shape=(640, 480), subresolutions=subresolutions)


def test_store(path_temp, random_img):
    from tiff_utils import write_tiff

    # save
    subresolutions = 5
    write_tiff(random_img, path_temp, subresolutions=subresolutions)
    assert_tiff(path_temp, shape=random_img.shape, subresolutions=subresolutions)
