from enum import Enum
from pydantic import DirectoryPath, validate_call
from typing import Optional, Tuple
import numpy as np
import dask.dataframe as dd
from skimage.util import img_as_ubyte
from image_readers import get_image_reader
from matplotlib.patches import Polygon


class SegmentationType(str, Enum):
    """
    _summary_
    """
    cell = 'cell'
    nucleus = 'nucleus'


class MorphologyImage(str, Enum):
    """
    _summary_
    """
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
    ax=None,
):
    """
    _summary_

    _extended_summary_

    Parameters
    ----------
    xenium_dir : DirectoryPath
        _description_
    x : Tuple[int, int]
        _description_
    y : Tuple[int, int]
        _description_
    segmentation_type : SegmentationType, optional
        _description_, by default SegmentationType.cell
    ax : _type_, optional
        _description_, by default None
    """
    def preprocess(img):
        vmin, vmax = np.quantile(img, [0.001, 0.999])
        if vmax > vmin:
            img = np.clip(img, vmin, vmax)
            img = (img - vmin) / (vmax - vmin)
        return img_as_ubyte(img)
    
    # Load 10X DAPI image
    path_dapi = xenium_dir/f'morphology_focus/{MorphologyImage.dapi}'
    with get_image_reader(path_dapi, level=0) as reader:
        img_dapi = reader.read(X=x, Y=y, pixels=False)
    
    # Load 10X membrane image
    path_boundary = xenium_dir/f'morphology_focus/{MorphologyImage.interior_rna}'
    with get_image_reader(path_boundary, level=0) as reader:
        img_boundary = reader.read(X=x, Y=y, pixels=False)

    # Compose image
    img_dapi = preprocess(img_dapi)
    # Cyan DAPI image
    img_dapi = np.stack(
        [np.zeros(img_dapi.shape).astype(int)] + [img_dapi]*2,
        axis=2
    )
    img = img_dapi
    if segmentation_type == SegmentationType.cell:
        img_boundary = preprocess(img_boundary)
        # Red membrane image
        img_boundary = np.stack(
            [img_boundary] + [np.zeros(img_boundary.shape).astype(int)]*2,
            axis=2
        )
        img += img_boundary
    
    # Load boundaries parquet file
    boundaries = dd.read_parquet(
        xenium_dir / f'{segmentation_type}_boundaries.parquet'
    )
    boundaries = boundaries.loc[
        boundaries.vertex_x.between(*x) & \
        boundaries.vertex_y.between(*y)
    ].compute()
    # Convert boundaries to image space
    boundaries['vertex_x'] -= x[0]
    boundaries['vertex_x'] /= (x[1] - x[0])
    boundaries['vertex_x'] *= img.shape[1]
    
    boundaries['vertex_y'] -= y[0]
    boundaries['vertex_y'] /= (y[1] - y[0])
    boundaries['vertex_y'] *= img.shape[0]

    # Plot image
    ax.imshow(img)
    styles = {'fill': False, 'color': 'w', 'lw': 0.25, 'alpha': 0.5}
    for cell, group in boundaries.groupby('cell_id'):
        coords = group[['vertex_x', 'vertex_y']].values
        p = Polygon(coords, **styles)
        ax.add_patch(p)
    ax.axis('Off')
