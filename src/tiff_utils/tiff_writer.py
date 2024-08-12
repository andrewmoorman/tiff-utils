"""
Write an array to a temporary TIFF file, that is then converted using `convert_to_tiff`.
"""

import tempfile
import os
import tifffile
from tiff_utils import convert_to_tiff

def write_tiff(array, path, **kwargs):
    with tempfile.NamedTemporaryFile(suffix=".tiff") as f:
        tifffile.imwrite(f.name, array)
        convert_to_tiff(f.name, path, **kwargs)
        assert os.path.exists(path)
