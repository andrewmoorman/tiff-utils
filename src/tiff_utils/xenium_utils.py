from enum import Enum
from pydantic import DirectoryPath, validate_call
import dask.dataframe as dd
from collections.abc import Iterable
from typing import Optional, Tuple


class SegmentationType(str, Enum):
    cell = 'cell'
    nucleus = 'nucleus'


class MorphologyImage(str, Enum):
    DAPI = 'morphology_focus_0000.ome.tif',
    boundary = 'morphology_focus_0001.ome.tif',
    interior_rna = 'morphology_focus_0002.ome.tif',
    interior_protein = 'morphology_focus_0003.ome.tif'


@validate_call
def plot_10x_segmentation_boundaries(
    xenium_dir: DirectoryPath,
    x: Tuple[int, int],
    y: Tuple[int, int],
    segmentation_type: SegmentationType = SegmentationType.cell,
):

    # Load 10x images
    
    
    # Load boundaries parquet file
    boundaries = dd.read_parquet(
        xenium_dir / f'{segmentation_type}_boundaries.parquet'
    )
    boundaries.loc[
        boundaries.vertex_x.between(*x) & \
        boundaries.vertex_y.between(*y)]
    ]
    
    
    