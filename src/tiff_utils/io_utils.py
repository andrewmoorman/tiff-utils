import tifffile
import os
import typing
import numpy as np
import re


def count_levels(path):
    """Count the number of pyramidal levels in a TIFF image"""
    with tifffile.TiffFile(path, mode='r') as f:
        return len(f.series[0].levels)

def select_tiff_level(
    file_path: typing.Union[str, os.PathLike],
    max_mem: typing.Union[int, str] = None,
    max_shape: tuple[int] = None,
    by_dims: tuple[str] = None,
) -> int:
    """
    Return the largest pyramidal level with either:
        a) a size in memory less than or equal to 'max_mem', or
        b) a shape less than or equal to 'max_shape'
    Exactly one of 'max_mem' or 'max_shape' must be supplied by the user.

    The 'max_mem' argument can be an integer (in bytes) or any numerical value
    and one of B, KB, MB, GB, TB units.

    The 'max_shape' argument can be a tuple of integers. If the length of
    'max_shape' is less than the total number of dimensions or the number of
    dimensions requested, subset using the shapes of the LAST n dimensions.

    Optionally, limit request to a subset of dimensions.

    Parameters
    ----------
    file_path : typing.Union[str, os.PathLike]
        The path to the TIFF image
    max_mem : typing.Union[int, str], optional
        Maximum size in memory of the level to return. Either an integer
        representing the number of bytes or numerical value and unit (str)
        are accepted, by default None
    max_shape : tuple[int], optional
        Maximum shape of array to return, by default None
    by_dims : tuple[str], optional
        Subset calculation to only these dimensions, by default None

    Returns
    -------
    int
        The largest pyramidal level meeting the supplied arguments

    Raises
    ------
    ValueError
        If no level meets the requirements, or if incorrect arguments were
        provided
    """
    # Internal helper
    def parse_size(size : str):
        units = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}
        size = size.upper()
        if not re.match(r' ', size):
            size = re.sub(r'([KMGT]?B)', r' \1', size)
        val, unit = map(str.strip, size.split())
        return int(float(val) * units[unit])
    # Select by size in memory
    if max_shape is None:
        if max_mem is None:
            msg = (
                "Exactly one of 'max_mem' or 'max_size' must be provided, but "
                "neither argument was supplied"
            )
            raise ValueError(msg)
        else:
            if isinstance(max_mem, str):
                max_mem = parse_size(max_mem)
            with tifffile.TiffFile(file_path, mode='r') as tf:
                for n, level in enumerate(tf.series[0].levels):
                    if by_dims is None:
                        by_dims = level.axes
                    mask = [d in list(by_dims) for d in level.axes]
                    shape = np.array(level.shape)[mask]
                    n_bytes = np.prod(shape) * level.dtype.itemsize
                    if n_bytes <= max_mem:
                        return n
            msg = (
                f"No level found matching memory request of {max_mem}B for "
                f"dimensions {''.join(list(by_dims))}. Smallest level size is "
                f"{n_bytes}B"
            )
            raise ValueError(msg)
    # Select by shape of array
    else:
        if max_mem is not None:
            msg = (
                "Exactly one of 'max_mem' or 'max_size' must be provided, but "
                "both arguments were supplied"
            )
            raise ValueError(msg)
        else:
            with tifffile.TiffFile(file_path, mode='r') as tf:
                for n, level in enumerate(tf.series[0].levels):
                    if by_dims is None:
                        by_dims = level.axes
                    mask = [d in list(by_dims) for d in level.axes]
                    shape = np.array(level.shape)[mask]
                    if len(shape) < len(max_shape):
                        msg = (
                            f"{len(max_shape)} maximum dimension sizes were "
                            f"provided to select array with {len(shape)} "
                            "dimensions"
                        )
                        raise ValueError(msg)
                    if all(shape[-len(max_shape):] <= np.array(max_shape)):
                        return n
            msg = (
                f"No level found matching shape request of {max_shape} for "
                f"dimensions {''.join(list(by_dims[-len(max_shape):]))}. "
                f"Smallest level size is {shape[-len(max_shape):]}"
            )
            raise ValueError(msg)
