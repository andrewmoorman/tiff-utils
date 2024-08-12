import os
import pytest

@pytest.fixture
def path_dice():
    path = "tests/data/dice.png"
    assert os.path.exists(path)
    yield path

@pytest.fixture
def path_temp():
    path = "temp.tiff"
    yield path
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture
def random_img():
    import numpy as np
    shape = (900, 900)
    img = np.random.randint(0, 255, shape, dtype=np.uint8)
    return img
