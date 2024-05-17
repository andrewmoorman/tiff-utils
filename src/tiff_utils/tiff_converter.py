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
    
        convert_to_tiff(input_path, output_path, subresolutions=7, grid_size=1)
"""

import tifffile
import os
import typing
import subprocess
from image_readers import get_image_reader


def convert_to_tiff(
    read_path: typing.Union[str, os.PathLike],
    write_path: typing.Union[str, os.PathLike],
    subresolutions: int = 4,
    grid_size: int = 1,
    **kwargs,
):
    """
    Convert an image to a TIFF format with specified subresolutions and grid
    size.

    Parameters
    ----------
    read_path : typing.Union[str, os.PathLike]
        The path to the input image.
    write_path : typing.Union[str, os.PathLike]
        The path to save the output TIFF file.
    subresolutions : int, optional
        The number of subresolutions to create, by default 4.
    grid_size : int, optional
        Grid size for processing the image, by default 1.
    """
    tile_size = 1024 * grid_size
    with get_image_reader(read_path) as reader:
        axes = 'TCZYX'
        shape = reader.shape
        with tifffile.TiffWriter(write_path, bigtiff=True) as writer:
            n_cpu = int(subprocess.run('nproc', capture_output=True).stdout)
            kwargs = dict(
                tile=(tile_size, tile_size),
                dtype=reader.dtype,
                compression='jpeg2000',
                maxworkers=int(n_cpu * 0.9),
            )
            # Oth level
            writer.write(
                reader.get_tiles(tile_size),
                shape=shape,
                subifds=subresolutions,
                description=str.encode(reader.ome_metadata),
                metadata={'axes': 'TCZYX'},
                **kwargs,
            )
            # Pyramidal sub-levels
            for n in range(1, subresolutions+1):
                s = 2 ** -n
                sub_shape = shape[:-2] + tuple(int(d*s) for d in shape[-2:])
                writer.write(
                    reader.get_tiles(tile_size, s=s),
                    shape=sub_shape,
                    subfiletype=1,
                    **kwargs,
                )
