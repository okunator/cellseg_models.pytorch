from typing import Dict

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import shapes
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon, shape

__all__ = ["inst2gdf", "sem2gdf", "smooth_polygon"]


def smooth_polygon(polygon: Polygon, sigma: float = 0.7):
    """Smooth a shapely polygon using a Gaussian filter."""
    x, y = polygon.exterior.xy

    x_smooth = gaussian_filter(x, sigma=sigma)
    y_smooth = gaussian_filter(y, sigma=sigma)

    smoothed_polygon = Polygon(zip(x_smooth, y_smooth))

    return smoothed_polygon


def _smooth(polygon, sigma, apply_smoothing):
    return smooth_polygon(polygon, sigma) if apply_smoothing else polygon


def inst2gdf(
    inst_map: np.ndarray,
    type_map: np.ndarray = None,
    xoff: int = None,
    yoff: int = None,
    apply_smoothing: bool = True,
    sigma: float = 0.7,
    class_dict: Dict[int, str] = None,
    min_size: int = 15,
) -> gpd.GeoDataFrame:
    """Convert an instance map to a GeoDataFrame.

    adapted form https://github.com/corteva/geocube/blob/master/geocube/vector.py

    Parameters:
        inst_map (np.ndarray):
            An instance map. Shape (H, W).
        type_map (np.ndarray, default=None):
            A type map. Shape (H, W). If provided, the types will be included in the
            resulting GeoDataFrame in column 'class_name'.
        xoff (int, default=None):
            The x offset. Optional.
        yoff (int, default=None):
            The y offset. Optional.
        apply_smoothing (bool, default=True)::
            Whether to apply gaussian smoothing to the polygons.
        sigma (float, default=0.7):
            The standard deviation of the Gaussian filter. Defaults to 0.7. Only used if
            apply_smoothing is True.
        class_dict (Dict[int, str], default=None):
            A dictionary mapping class indices to class names. If None, the class indices
            will be used. e.g. {1: 'neoplastic', 2: 'immune'}.
        min_size (int, default=15):
            The minimum size (in pixels) of the polygons to include in the GeoDataFrame.


    resurns:
        gpd.GeoDataFrame: A GeoDataFrame of the instance map.
        Containing columns 'id', 'class_name' and 'geometry'.
    """

    if type_map is None:
        type_map = inst_map > 0

    types = np.unique(type_map)[1:]

    if class_dict is None:
        class_dict = {int(i): int(i) for i in types}

    inst_maps_per_type = []
    for t in types:
        mask = type_map == t
        vectorized_data = (
            (value, class_dict[int(t)], _smooth(shape(polygon), sigma, apply_smoothing))
            for polygon, value in shapes(
                inst_map,
                mask=mask,
            )
        )

        res = gpd.GeoDataFrame(
            vectorized_data,
            columns=["id", "class_name", "geometry"],
        )
        res["id"] = res["id"].astype(int)
        inst_maps_per_type.append(res)

    res = pd.concat(inst_maps_per_type)
    res = res.loc[res.area > min_size].reset_index(drop=True)

    if xoff is not None:
        res["geometry"] = res["geometry"].translate(xoff, 0)

    if yoff is not None:
        res["geometry"] = res["geometry"].translate(0, yoff)

    return res


def sem2gdf(
    sem_map: np.ndarray,
    xoff: int = None,
    yoff: int = None,
    apply_smoothing: bool = False,
    sigma: float = 0.7,
    class_dict: Dict[int, str] = None,
    min_size: int = 15,
) -> gpd.GeoDataFrame:
    """Convert an instance map to a GeoDataFrame.

    adapted form https://github.com/corteva/geocube/blob/master/geocube/vector.py

    Parameters:
        sem_map (np.ndarray):
            A semantic segmentation map. Shape (H, W).
        xoff (int, default=None):
            The x offset. Optional.
        yoff (int, default=None):
            The y offset. Optional.
        apply_smoothing (bool, default=None):
            Whether to apply gaussian smoothing to the polygons. Defaults to False.
        sigma (float, default=None):
            The standard deviation of the Gaussian filter. Defaults to 0.7. Only used if
            apply_smoothing is True.
        class_dict (Dict[int, str], default=None):
            A dictionary mapping class indices to class names. If None, the class indices
            will be used. e.g. {1: 'neoplastic', 2: 'immune'}.
        min_size (int, default=15):
            The minimum size (in pixels) of the polygons to include in the GeoDataFrame.

    resurns:
        gpd.GeoDataFrame: A GeoDataFrame of the semantic segmenation map.
        Containing columns 'id', 'class_name' and 'geometry'.
    """
    if class_dict is None:
        class_dict = {int(i): int(i) for i in np.unique(sem_map)[1:]}

    vectorized_data = (
        (value, _smooth(shape(polygon), sigma, apply_smoothing))
        for polygon, value in shapes(
            sem_map,
            mask=sem_map > 0,
        )
    )

    res = gpd.GeoDataFrame(
        vectorized_data,
        columns=["id", "geometry"],
    )
    res["id"] = res["id"].astype(int)
    res = res.loc[res.area > min_size].reset_index(drop=True)
    res["class_name"] = res["id"].map(class_dict)
    res = res[["id", "class_name", "geometry"]]  # reorder columns

    if xoff is not None:
        res["geometry"] = res["geometry"].translate(xoff, 0)

    if yoff is not None:
        res["geometry"] = res["geometry"].translate(0, yoff)

    return res
