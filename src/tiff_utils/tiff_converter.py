import bfio
import tifffile
import os
import typing
import pydantic
import itertools
import cv2
from tqdm import tqdm
import numpy as np
import zarr
from abc import ABC, abstractmethod


_valid_image_formats = dict()


# Decorator to register new image formats below
def _register_image_format(file_type):
    def decorator(fn):
        # Multiple synonymous file extensions, e.g., .tif/.tiff
        if isinstance(file_type, list):
            for ft in file_type:
                _valid_image_formats[ft] = fn
        # One file extension
        else:
            _valid_image_formats[file_type] = fn
        return fn
    return decorator


# Abstract creator class
class AbstractReader(ABC):
    """
    An abstract class defining the key behaviors for reading dimensions from
    an image, e.g., retrieving channels or image slices. Its purpose
    is to enable factory method-style file loading, so all core utilities
    should make use of other file format-specific libraries.

    All implementing classes should be used as a context manager with a read
    function like this:

        with AbstractReader('path/to/file') as reader:
            ...
            reader.read(X=x, Y=y, ...)

    Attributes
    ----------
    file_path : typing.Union[str, os.PathLike]
        Path of the file to load
    level : int, optional
        Pyramidal level of the image, by default None
    """

    def __init__(
        self,
        file_path: typing.Union[str, os.PathLike],
        level: int = None,
    ):
        self.file_path = file_path
        self.level = level

    def __enter__(self):
        """
        Handle entrance to the context manager
        """
        pass

    def __close__(self):
        """
        Handle exit from context manager
        """
        pass

    @abstractmethod
    def read(self, level, T, C, Z, Y, X) -> np.ndarray:
        """
        Reads an ndarray or ndarray-like object from self.file_path given
        slicing parameters defined in kwargs

        Returns
        -------
        np.ndarray
            Sliced array
        """
        pass


class ContextManagerNotSetError(Exception):
    """
    Exception raise when attempting to read outside of the Context Manager
    setting

    Attributes:
        message -- explanation of the error
    """

    message = (
        "Context manager is closed or has not been set. The current "
        "implementation only supports context manager style use when reading. "
        "Please use: \n"
        "\twith AbstractReader('path/to/file') as reader: \n"
        "\t\treader.read(...)"
    )

    def __init__(self, message=message):
        super().__init__(message)


@_register_image_format(['.tif', '.tiff'])
class TiffReader(AbstractReader):
    """
    Very lightweight TIFF reader using the tifffile python package and its Zarr
    implementation for TIFF files
    """
    def __init__(self, file_path: str | os.PathLike, level: int = None):
        super().__init__(file_path)
        self._store = None

    def __enter__(self):
        """
        Custom context manager for handling TIFF Zarr stores using tifffile.
        
        Note: Requested pyramidal level is only checked when entering the
        context manager, not during object creation; in practice, these always
        happen in the same line.
        """
        if self.level is not None:
            with tifffile.TiffFile(self.file_path, mode='r') as tf:
                n_levels = len(tf.series[0].levels)
            if self.level >= n_levels:
                msg = (
                    f"Cannot open image: Level {self.level} not found in "
                    "image file"
                )
                raise ValueError(msg)
        self._store = tifffile.imread(self.file_path, aszarr=True)
        return self

    def __close__(self):
        if self._store is not None:
            self._store.close()
            self._store = None

    def read(
        self,
        C: int = None,
        Z: int = None,
        Y: typing.Union[int, tuple[int, int], slice] = None,
        X: typing.Union[int, tuple[int, int], slice] = None,
    ):
        # Error #1: Zarr store not yet set up
        if self._store is None:
            raise ContextManagerNotSetError()
        # Error #2: Slicing dimension doesn't exist in file
        with tifffile.TiffFile(self.file_path, mode='r') as tf:
            axes = tf.series[0].levels[self.level].axes
        kwargs = dict(zip('CZYX', [C, Z, Y, X]))
        for name, arg in kwargs.items():
            if arg is None:
                kwargs.pop(name)
            elif name not in axes:
                msg = (
                    f"Value for dimension '{name}' was given, but '{name}'"
                    f" does not exist in {os.path.basename(self.filepath)}"
                )
                raise ValueError(msg)
        # Read sub-array
        z = zarr.open(self._store, mode='r')
        reordered_args = []
        z[0 if self.level is None else self.level]


def convert_to_tiff(
    read_path: typing.Union[str, os.PathLike],
    write_path: typing.Union[str, os.PathLike],
    subresolutions: int = 7,
    grid_size: int = 1,
    **kwargs,
):
    """
    _summary_

    _extended_summary_

    Parameters
    ----------
    read_path : typing.Union[str, os.PathLike]
        _description_
    write_path : typing.Union[str, os.PathLike]
        _description_
    subresolutions : int, optional
        _description_, by default 7
    """
    tile_size = 1024 * grid_size
    with get_image_reader(read_path) as reader:
        
        def get_tiles(br, tile_size, s=1):
            tile_size = int(tile_size / s)
            # TODO: Add shape property to AbstractReader
            tiles = itertools.product(
                range(br.T),
                range(br.C),
                range(br.Z),
                range(0, br.Y, tile_size),
                range(0, br.X, tile_size)
            )
            for tile in tqdm(tiles):
                t, c, z, y, x = tile
                x = x, min(br.X, x + tile_size)
                y = y, min(br.Y, y + tile_size)
                img = br.read(X=x, Y=y, Z=z, C=c, T=t)
                if s < 1:
                    img = cv2.resize(
                        img,
                        dsize=None,
                        fx=s,
                        fy=s,
                        interpolation=cv2.INTER_AREA
                    )
                yield img

        axes = 'TCZYX'
        shape = [br._DIMS[a] for a in axes]
        metadata = {
            'Axes': axes,
            'PhysicalSizeX': br.physical_size_x[0],
            'PhysicalSizeXUnit': br.physical_size_x[1]._value_,
            'PhysicalSizeY': br.physical_size_y[0],
            'PhysicalSizeYUnit': br.physical_size_x[1]._value_,
            'Channel': {'Name': br.channel_names},
        }
        with tifffile.TiffWriter(write_path, bigtiff=True) as writer:
            kwargs = dict(
                tile=(tile_size, tile_size),
                dtype=br.dtype,
                compression='jpeg2000',
            )
            # Oth level
            writer.write(
                get_tiles(br, tile_size),
                shape=shape,
                metadata=metadata,
                subifds=subresolutions,
                **kwargs,
            )
            # Pyramidal sub-levels
            for n in range(1, subresolutions+1):
                s = 2 ** -n
                sub_shape = shape[:-2] + [int(d*s) for d in shape[-2:]]
                writer.write(
                    get_tiles(br, tile_size, s=s),
                    shape=sub_shape,
                    subfiletype=1,
                    **kwargs
                )


# Factory method
def get_image_reader(
    file_path: typing.Union[str, os.PathLike],
) -> typing.Type[AbstractReader]:
    """
    Factory pattern implementation of image file loading for various supported
    slide formats. Its return type supports basic behaviors for reading a slide
    image, e.g., retrieving channels or image slices.

    Currently supported formats are .tiff and all Bioformat file types.

    Parameters
    ----------
    file_path : typing.Union[str, os.PathLike]
        The file path of the image to load

    Returns
    -------
    typing.Type[AbstractReader]
        A subclass of AbstractReader according to the extension of the file
        path

    Raises
    ------
    ValueError
        If the file type isn't supported or the path provided is not a file
        (e.g., a directory)
    """
    _, extension = os.path.splitext(file_path)
    if extension not in _valid_image_formats:
        try:
            return BioFormatsReader(file_path)  # default reader
        except:
            raise ValueError("Unsupported file format")
    return _valid_image_formats[extension](file_path)