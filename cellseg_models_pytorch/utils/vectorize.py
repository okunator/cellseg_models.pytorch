from typing import Any, Callable, Dict

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import shapes
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, shape

__all__ = ["inst2gdf", "sem2gdf", "gaussian_smooth"]


def gaussian_filter_2d(x: np.ndarray, y: np.ndarray, sigma: float = 0.7):
    x_smooth = gaussian_filter(x, sigma=sigma)
    y_smooth = gaussian_filter(y, sigma=sigma)

    return x_smooth, y_smooth


def gaussian_smooth(obj: Any, sigma: float = 0.7):
    """Smooth a shapely (multi)polygon|(multi)linestring using a Gaussian filter."""
    if isinstance(obj, Polygon):
        x, y = obj.exterior.xy
        x_smooth, y_smooth = gaussian_filter_2d(x, y, sigma=sigma)
        smoothed = Polygon(zip(x_smooth, y_smooth))
    elif isinstance(obj, MultiPolygon):
        smoothed = MultiPolygon(
            [gaussian_smooth(poly.exterior, sigma=sigma) for poly in obj.geoms]
        )
    elif isinstance(obj, LineString):
        x, y = obj.xy
        x_smooth, y_smooth = gaussian_filter_2d(x, y, sigma=sigma)
        smoothed = LineString(zip(x_smooth, y_smooth))
    elif isinstance(obj, MultiLineString):
        smoothed = MultiLineString(
            [
                LineString(
                    zip(*gaussian_filter_2d(line.xy[0], line.xy[1], sigma=sigma))
                )
                for line in obj.geoms
            ]
        )
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")

    return smoothed


# adapted form https://github.com/corteva/geocube/blob/master/geocube/vector.py
def inst2gdf(
    inst_map: np.ndarray,
    type_map: np.ndarray = None,
    xoff: int = None,
    yoff: int = None,
    class_dict: Dict[int, str] = None,
    min_size: int = 15,
    smooth_func: Callable = None,
) -> gpd.GeoDataFrame:
    """Convert an instance map to a GeoDataFrame.

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
        class_dict (Dict[int, str], default=None):
            A dictionary mapping class indices to class names. If None, the class indices
            will be used. e.g. {1: 'neoplastic', 2: 'immune'}.
        min_size (int, default=15):
            The minimum size (in pixels) of the polygons to include in the GeoDataFrame.
        smooth_func (Callable, default=None):
            A function to smooth the polygons. The function should take a shapely Polygon
            as input and return a shapely Polygon.


    returns:
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
            (value, class_dict[int(t)], shape(polygon))
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

    if smooth_func is not None:
        res["geometry"] = res["geometry"].apply(smooth_func)

    return res


def sem2gdf(
    sem_map: np.ndarray,
    xoff: int = None,
    yoff: int = None,
    class_dict: Dict[int, str] = None,
    min_size: int = 15,
    smooth_func: Callable = None,
) -> gpd.GeoDataFrame:
    """Convert an instance map to a GeoDataFrame.

    Parameters:
        sem_map (np.ndarray):
            A semantic segmentation map. Shape (H, W).
        xoff (int, default=None):
            The x offset. Optional.
        yoff (int, default=None):
            The y offset. Optional.
        class_dict (Dict[int, str], default=None):
            A dictionary mapping class indices to class names. If None, the class indices
            will be used. e.g. {1: 'neoplastic', 2: 'immune'}.
        min_size (int, default=15):
            The minimum size (in pixels) of the polygons to include in the GeoDataFrame.
        smooth_func (Callable, default=None):
            A function to smooth the polygons. The function should take a shapely Polygon
            as input and return a shapely Polygon.

    returns:
        gpd.GeoDataFrame: A GeoDataFrame of the semantic segmenation map.
        Containing columns 'id', 'class_name' and 'geometry'.
    """
    if class_dict is None:
        class_dict = {int(i): int(i) for i in np.unique(sem_map)[1:]}

    vectorized_data = (
        (value, shape(polygon))
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

    if smooth_func is not None:
        res["geometry"] = res["geometry"].apply(smooth_func)

    return res
