"""
TIFF Converter

This module provides a function to convert images to TIFF format with specified
subresolutions and grid sizes. It uses the tifffile package and an appropriate
image reader for the input file format.

Functions:
    convert_to_tiff: Converts an image to TIFF format with specified
    subresolutions and grid size.

Usage:
    Convert an image to TIFF format:

        convert_to_tiff(read_path, write_path, subresolutions=7, grid_size=1)
"""

import tifffile
import os
import typing
import subprocess
from tiff_utils.image_readers import get_image_reader
import math
import warnings
from ome_types import to_xml
from ome_types.model import TiffData


def convert_to_tiff(
    read_path: typing.Union[str, os.PathLike],
    write_path: typing.Union[str, os.PathLike],
    subresolutions: int = 4,
    grid_size: int = 1,
    max_workers: int = None,
    **kwargs,
):
    """
    Convert an image to a TIFF format with specified subresolutions and grid
    size.

    Parameters
    ----------
    read_path : typing.Union[str, os.PathLike]
        The path to the input image
    write_path : typing.Union[str, os.PathLike]
        The path to save the output TIFF file
    subresolutions : int, optional
        The number of subresolutions to create, by default 4
    grid_size : int, optional
        Grid size for writing the image, by default 1. Final tile size used is
        grid_size * 1024
    max_workers: int, optional
        Number of cores to use when compressing image, by default 90% of
        available cores
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module='tifffile.tifffile.TiffWriter',
        )
    with get_image_reader(read_path) as reader:
        with tifffile.TiffWriter(write_path, bigtiff=True, ome=True) as writer:
            # Setup
            tile_size = 1024 * grid_size
            shape = reader.shape
            if max_workers is None:
                n_cpu = subprocess.run('nproc', capture_output=True).stdout
                max_workers = math.floor(int(n_cpu) * 0.9)
            kwargs = dict(
                tile=(tile_size, tile_size),
                dtype=reader.dtype,
                compression='jpeg2000',
                maxworkers=max_workers,
                metadata={'axes': 'TCZYX'},
            )
            # Oth level
            writer.write(
                reader.get_tiles(tile_size),
                shape=shape,
                subifds=subresolutions,
                **kwargs,
            )
            # Pyramidal sub-levels
            for n in range(1, subresolutions + 1):
                s = 2 ** -n
                sub_shape = shape[:-2] + tuple(int(d * s) for d in shape[-2:])
                writer.write(
                    reader.get_tiles(tile_size, s=s),
                    shape=sub_shape,
                    subfiletype=0,
                    **kwargs,
                )
        # Write metadata
        ome = reader.ome_metadata
        tifffile.tiffcomment(write_path, str.encode(to_xml(ome)))
