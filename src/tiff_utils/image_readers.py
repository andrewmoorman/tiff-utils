"""
Image Readers

This module provides abstract and concrete classes for reading various image
formats. It supports TIFF and BioFormats file types and includes a factory
method to select the appropriate reader based on the file extension.

Classes:
    AbstractReader: An abstract base class defining the interface for image
    readers.
    ContextManagerNotSetError: An exception raised when attempting to read
    outside of the context manager.
    TiffReader: A reader for TIFF files using the tifffile package.
    BioFormatsReader: A reader for BioFormats files using the bfio package.

Functions:
    _register_image_format: A decorator for registering new image formats with
    their handlers.
    get_image_reader: A factory method for selecting the appropriate image
    reader based on the file extension.

Usage:
    Use the AbstractReader class as a context manager to read image data:

        with AbstractReader('path/to/file') as reader:
            img_data = reader.read(X=x, Y=y, ...)
"""

import tifffile
import bfio
import os
import typing
import itertools
import cv2
from tqdm.auto import tqdm
import numpy as np
import zarr
from abc import ABC, abstractmethod
from collections import defaultdict
from ome_types.model import OME, Image, Pixels, Channel, TiffData, UnitsLength
from ome_types import from_xml
from typing import Union, Literal


_valid_image_formats = dict()
_not_supported_formats = ["jpg", "JPG"]

# Decorator to register new image formats below
def _register_image_format(file_type):
    """
    Register a new image format with the appropriate handler.

    Parameters
    ----------
    file_type : str or list of str
        The file extension(s) to register.
    """
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

# TODO: Add 'resize' instance method using Dask to resize large arrays without
# exceeding memory limitations
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

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Handle exit from context manager
        """
        pass

    @abstractmethod
    def read(self, level, T, C, Z, Y, X, pixels) -> np.ndarray:
        """
        Reads an ndarray or ndarray-like object from self.file_path given
        slicing parameters defined in kwargs

        If 'pixels' is True, coordinates are assumed to be in image/pixel
        space, otherwise in physical space

        Returns
        -------
        np.ndarray
            Sliced array

        Raises
        ------
        ContextManagerNotSetError
            If attempting to read file without setting Context Manager
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int]:
        """
        A tuple of T, C, Z, Y, X dimension sizes of the image, always
        in that order

        Returns value 1 for dimensions which are not named in the image file.
        This can only be T, C, Z; image must always have X and Y dimensions.

        Returns
        -------
        tuple[int]
            Shape of image in T, C, Z, Y, X order
        """
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """
        The numpy data type of the image

        Returns e.g., np.dtype('uint16') for 16-bit int array

        Returns
        -------
        np.dtype
            data type of the image
        """
        pass

    @property
    @abstractmethod
    def channels(self) -> list[str]:
        """
        The names of each channel in the image

        Returns e.g., ['DAPI', 'Cy5', 'RFP']

        Returns
        -------
        list[str]
            The names of channels
        """
        pass

    @property
    def T(self) -> int: return self.shape[0]

    @property
    def C(self) -> int: return self.shape[1]

    @property
    def Z(self) -> int: return self.shape[2]

    @property
    def Y(self) -> int: return self.shape[3]

    @property
    def X(self) -> int: return self.shape[4]

    @abstractmethod
    def _get_physical_size(
        self,
        dimension: str,
    ) -> typing.Union[tuple[float, UnitsLength], None]:
        """
        Retrieve the physical size of a given dimension (value and unit)

        Returns e.g., (0.25, <UnitsLength.MICROMETER: 'µm'>)

        Parameters
        ----------
        dimension : str
            The dimension to retrieve the physical size for ('X', 'Y', or 'Z').

        Returns
        -------
        typing.Union[tuple[float, UnitsLength], None]
            Physical size and unit of the given dimension, or None if not
            available.
        """
        pass

    @property
    def physical_size_z(self) -> typing.Union[tuple[float, UnitsLength], None]:
        return self._get_physical_size('Z')

    @property
    def physical_size_y(self) -> typing.Union[tuple[float, UnitsLength], None]:
        return self._get_physical_size('Y')

    @property
    def physical_size_x(self) -> typing.Union[tuple[float, UnitsLength], None]:
        return self._get_physical_size('X')

    def get_tiles(self, tile_size: int, s: int=1):
        """
        Generate tiles of the image for writing to disk.

        Parameters
        ----------
        tile_size : int
            Size of the tiles to generate.
        s : int, optional
            Scale factor for resizing tiles, by default 1

        Yields
        ------
        np.ndarray
            Image tiles.
        """
        tile_size = int(tile_size / s)
        tiles = list(itertools.product(
            range(self.T),
            range(self.C),
            range(self.Z),
            range(0, self.Y, tile_size),
            range(0, self.X, tile_size),
        ))
        with tqdm(total=len(tiles)) as pbar:
            for tile in tiles:
                t, c, z, y, x = tile
                x = slice(x, min(self.X, x + tile_size))
                y = slice(y, min(self.Y, y + tile_size))
                img = self.read(X=x, Y=y, Z=z, C=c, T=t)
                if s < 1:
                    img = cv2.resize(
                        img,
                        dsize=None,
                        fx=s,
                        fy=s,
                        interpolation=cv2.INTER_AREA,
                    )
                pbar.update(1)
                yield img

    def convert_slice(
        self,
        value: typing.Union[int, tuple[int, int], slice],
        axis: str,
    ):
        """
        Converts a given value to pixel coordinates based on the specified axis

        This method converts an integer, a tuple of integers, or a slice object to
        pixel coordinates using the units per pixel (upp) of the specified axis

        Parameters
        ----------
        value : typing.Union[int, tuple[int, int], slice]
            The value to be converted to pixel coordinates. It can be an integer,
            a tuple of two integers, or a slice object
        axis : str
            The axis for which the conversion is to be done. It should be either
            'x' or 'y'

        Returns
        -------
        typing.Union[int, tuple[int, int], slice, None]
            The converted value in pixel coordinates

        Raises
        ------
        ValueError
            If the axis is not 'x' or 'y', or if the value is not an integer, tuple,
            or slice, a ValueError is raised.
        """
        # Determine units per pixel based on the specified axis
        if axis.lower() == 'x':
            upp = self.physical_size_x[0]
        elif axis.lower() == 'y':
            upp = self.physical_size_y[0]
        else:
            msg = f"Parameter 'axis' must be one of X/x or Y/y, not {axis}"
            raise ValueError(msg)
        # Convert value to pixel coordinates
        if value is None:
            return value
        elif isinstance(value, int):
            return int(value / upp)
        elif isinstance(value, tuple):
            return int(value[0] / upp), int(value[1] / upp)
        elif isinstance(value, slice):
            fn = lambda v: int(v / upp) if v is not None else v
            return slice(
                fn(value.start),
                fn(value.stop),
                fn(value.step),
            )
        # Raise ValueError if value type is not supported
        else:
            msg = f'Unable to convert {value} to pixel coordinates'
            raise ValueError(msg)

    @property
    def ome_metadata(self) -> str:
        """
        Generate minimal OME metadata for the given image data as an XML string

        Returns
        -------
        str: An OME object in XML format containing the minimal metadata.

        Example
        -------
        >>> with AbstractReader('path/to/file') as reader:
        >>>     ome_metadata = reader.ome_metadata(image_data)
        """
        # Create minimal OME metadata
        ome = OME()
        ome_image = Image(
            id='Image:0',
            pixels=Pixels(
                id='Pixels:0',
                dimension_order='XYZCT',
                type=str(self.dtype),
                size_x=self.X,
                size_y=self.Y,
                size_z=self.Z,
                size_c=self.C,
                size_t=self.T,
                physical_size_x=self.physical_size_x[0],
                physical_size_x_unit=self.physical_size_x[1],
                physical_size_y=self.physical_size_y[0],
                physical_size_y_unit=self.physical_size_y[1],
                physical_size_z=self.physical_size_z[0],
                physical_size_z_unit=self.physical_size_z[1],
                channels=[
                    Channel(
                        id=f'Channel:{i}',
                        samples_per_pixel=1,
                        name=name,
                    ) for i, name in enumerate(self.channels)
                ],
                tiff_data_blocks=[TiffData(plane_count=self.C*self.Z*self.T)]
            )
        )
        ome.images.append(ome_image)
        return ome


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
        "Please use: \n\n"
        "   with AbstractReader('path/to/file') as reader: \n"
        "       reader.read(...)"
    )

    def __init__(self, message=message):
        super().__init__(message)


@_register_image_format(['.tif', '.tiff'])
class TiffReader(AbstractReader):
    """
    Very lightweight TIFF reader using the tifffile python package and its Zarr
    implementation for TIFF files
    """
    def __init__(
        self,
        file_path: typing.Union[str, os.PathLike],
        level: int = 0,
    ):
        super().__init__(file_path, level)
        self._store = None

    def __enter__(self):
        """
        Custom context manager for handling TIFF Zarr stores using tifffile.

        Note: Requested pyramidal level is only checked when entering the
        context manager, not during object creation; in practice, these always
        happen in the same line.
        """
        if self.level is not None:
            with tifffile.TiffFile(
                self.file_path,
                is_ome=False if self._is_multifile else None,
            ) as tf:
                n_levels = len(tf.series[0].levels)
            if self.level >= n_levels:
                msg = (
                    f"Cannot open image: Level {self.level} not found in "
                    "image file"
                )
                raise ValueError(msg)
        self._store = tifffile.imread(
            self.file_path,
            aszarr=True,
            level=self.level,
            is_ome=False if self._is_multifile else None,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._store is not None:
            self._store.close()
            self._store = None

    def read(
        self,
        T: int = None,
        C: int = None,
        Z: int = None,
        Y: typing.Union[int, tuple[int, int], slice] = None,
        X: typing.Union[int, tuple[int, int], slice] = None,
        pixels: bool = True,
    ):
        # Error #1: Zarr store not yet set up
        if self._store is None:
            raise ContextManagerNotSetError()

        # Optionally convert coordinates from physical space to image space
        if not pixels:
            X = self.convert_slice(X, axis='x')
            Y = self.convert_slice(Y, axis='y')

        # Filter slices
        kwargs = dict(zip(range(5), [T,C,Z,Y,X]))
        # Remove if no slice requested
        kwargs = filter(lambda x: x[1] is not None, kwargs.items())
        # Remove if empty axis sliced correctly
        kwargs = filter(lambda x: x[1] != 0 or self.shape[x[0]] != 1, kwargs)
        # Map keys to names of axes
        kwargs = map(lambda x: ('TCZYX'[x[0]], x[1]), kwargs)
        # Map tuples to slices
        kwargs = map(
            lambda x: (
                x[0],
                slice(*x[1]) if isinstance(x[1], tuple) else x[1]
            ),
            kwargs,
        )
        kwargs = dict(kwargs)

        # Error #2: Empty axis sliced incorrectly
        with tifffile.TiffFile(
            self.file_path,
            is_ome=False if self._is_multifile else None,
        ) as tf:
            axes = tf.series[0].levels[self.level].axes
            axes = axes.replace('S', 'C')
        extra = set(kwargs.keys()).difference(axes)
        if any(extra):
            keys = ', '.join(extra)
            msg = (
                f"Value(s) for dimension(s) '{keys}' were given, but '{keys}' "
                f"does not exist in {os.path.basename(self.file_path)}"
            )
            raise ValueError(msg)

        # Slice image
        slices = defaultdict(lambda: slice(None))
        slices.update(kwargs)
        z = zarr.open(self._store, mode='r', )
        return z[tuple(slices[ax] for ax in axes)]

    @property
    def shape(self):
        shape = {'T': 1, 'C': 1, 'Z': 1}  # default shape is 1
        with tifffile.TiffFile(
            self.file_path,
            is_ome=False if self._is_multifile else None,
        ) as tf:
            level = tf.series[0].levels[self.level]
            shape.update(dict(zip(
                level.axes.replace('S', 'C'),
                level.shape
            )))
        return tuple([shape[a] for a in 'TCZYX'])

    @property
    def dtype(self):
        with tifffile.TiffFile(
            self.file_path,
            is_ome=False if self._is_multifile else None,
        ) as tf:
            dtype = tf.series[0].levels[self.level].dtype
        return dtype

    @property
    def channels(self):
        with tifffile.TiffFile(
            self.file_path,
            is_ome=False if self._is_multifile else None,
        ) as tf:
            if tf.is_ome:
                ome_xml = from_xml(tf.ome_metadata)
                channels = ome_xml.images[0].pixels.channels
                return [c.name for c in channels]
            else:
                return ['' for _ in range(self.C)]

    def _get_physical_size(self, dim):
        with tifffile.TiffFile(
            self.file_path,
            #is_ome=False if self._is_multifile else None,
        ) as tf:
            if tf.is_ome:
                ome_xml = from_xml(tf.ome_metadata)
                key = f'physical_size_{dim.lower()}'
                try:
                    pixels_dict = ome_xml.images[0].pixels.model_dump()
                    return pixels_dict[key], pixels_dict[key + '_unit']
                except AttributeError:
                    pass
        return (None, UnitsLength.MICROMETER)

    @property
    def _is_multifile(self) -> Union[bool, None]:
        """
        Whether file path opens a multi-file TIFF

        Returns
        -------
        bool
            True file is multi-file TIFF else False
        """
        with tifffile.TiffFile(self.file_path) as tf:
            if tf.is_ome:
                ome = from_xml(tf.ome_metadata)
                n_blocks = len(ome.images[0].pixels.tiff_data_blocks)
                return n_blocks > 1
            else:
                return False


class BioFormatsReader(AbstractReader):
    """
    Very lightweight BioFormats reader which effectively just wraps the bfio
    implementation of BioReader
    """
    def __init__(
        self,
        file_path: typing.Union[str, os.PathLike],
        level: int = None,
    ):
        super().__init__(file_path, level)
        self._reader = None

    def __enter__(self):
        """
        Simple wrapper for bfio.BioReader context manager
        """
        self._reader = bfio.BioReader(
            file_path=self.file_path,
            backend='bioformats',
            level=self.level,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Simple wrapper for bfio.BioReader context manager
        """
        self._reader.close()
        self._reader = None

    def read(
        self,
        T: typing.Union[int, tuple[int, int], slice] = None,
        C: typing.Union[int, tuple[int, int], slice] = None,
        Z: typing.Union[int, tuple[int, int], slice] = None,
        Y: typing.Union[int, tuple[int, int], slice] = None,
        X: typing.Union[int, tuple[int, int], slice] = None,
        pixels: bool = True,
    ):
        # Error: Reader not set up
        if self._reader is None:
            raise ContextManagerNotSetError()
        # For interface consistency, T and C should be provided as ints if
        # slicing one item; however, BioReader requires them to be lists
        if isinstance(C, int): C = [C]
        if isinstance(T, int): T = [T]
        X, Y, Z, C, T = map(
            lambda a: (a.start, a.stop) if isinstance(a, slice) else a,
            [X, Y, Z, C, T],
        )
        # Optionally convert coordinates from physical space to image space
        if not pixels:
            X = self.convert_slice(X, axis='x')
            Y = self.convert_slice(Y, axis='y')
        return self._reader.read(X=X, Y=Y, Z=Z, C=C, T=T)

    @property
    def shape(self):
        # Error: Reader not set up
        if self._reader is None:
            raise ContextManagerNotSetError()
        else:
            shape = self._reader._DIMS
            return tuple([shape[a] for a in 'TCZYX'])

    @property
    def dtype(self):
        # Error: Reader not set up
        if self._reader is None:
            raise ContextManagerNotSetError()
        else:
            return self._reader.dtype

    @property
    def channels(self):
        # Error: Reader not set up
        if self._reader is None:
            raise ContextManagerNotSetError()
        else:
            return self._reader.channel_names

    def _get_physical_size(self, dim):
        # Error: Reader not set up
        if self._reader is None:
            raise ContextManagerNotSetError()
        else:
            attr = f'physical_size_{dim.lower()}'
            return self._reader.__getattribute__(attr)


# Factory method
def get_image_reader(
    file_path: typing.Union[str, os.PathLike],
    **kwargs,
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
    if extension in _not_supported_formats:
        raise ValueError(f"Unsupported file format: {extension}.")
    if extension not in _valid_image_formats:
        try:
            return BioFormatsReader(file_path, **kwargs)  # default reader
        except:
            raise ValueError("Unsupported file format")
    return _valid_image_formats[extension](file_path, **kwargs)


def read_image(
    file_path: typing.Union[str, os.PathLike],
    level: int=0,
    **kwargs,
) -> np.ndarray:
    """
    Read an image file and return the image data as a numpy array.

    Parameters
    ----------
    file_path : typing.Union[str, os.PathLike]
        The file path of the image to load

    Returns
    -------
    np.ndarray
        The image data as a numpy array
    """
    with get_image_reader(file_path, level=level) as reader:
        return reader.read(**kwargs)
