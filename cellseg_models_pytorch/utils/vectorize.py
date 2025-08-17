from typing import Any, Callable, Dict

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import shapes
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
    shape,
)

__all__ = ["inst2gdf", "sem2gdf", "gaussian_smooth"]


def _uniform_filter_2d(x: np.ndarray, y: np.ndarray, size: int = 5):
    """Fastest smoothing option - uniform (box) filter."""
    x_smooth = uniform_filter1d(x, size=size, mode="nearest")
    y_smooth = uniform_filter1d(y, size=size, mode="nearest")
    return x_smooth, y_smooth


def _gaussian_filter_2d(x: np.ndarray, y: np.ndarray, sigma: float = 0.7):
    x_smooth = gaussian_filter1d(x, sigma=sigma, mode="nearest")
    y_smooth = gaussian_filter1d(y, sigma=sigma, mode="nearest")

    return x_smooth, y_smooth


def gaussian_smooth(obj: Any, sigma: float = 0.7):
    """Smooth a shapely (multi)polygon|(multi)linestring using a Gaussian filter."""
    if isinstance(obj, Polygon):
        x, y = obj.exterior.xy
        x_smooth, y_smooth = _gaussian_filter_2d(x, y, sigma=sigma)
        smoothed = Polygon(zip(x_smooth, y_smooth))
    elif isinstance(obj, MultiPolygon):
        smoothed = MultiPolygon(
            [gaussian_smooth(poly.exterior, sigma=sigma) for poly in obj.geoms]
        )
    elif isinstance(obj, LineString):
        x, y = obj.xy
        x_smooth, y_smooth = _gaussian_filter_2d(x, y, sigma=sigma)
        smoothed = LineString(zip(x_smooth, y_smooth))
    elif isinstance(obj, MultiLineString):
        smoothed = MultiLineString(
            [
                LineString(
                    zip(*_gaussian_filter_2d(line.xy[0], line.xy[1], sigma=sigma))
                )
                for line in obj.geoms
            ]
        )
    elif isinstance(obj, GeometryCollection):
        smoothed = GeometryCollection(
            [
                _gaussian_filter_2d(
                    geom.exterior.xy[0], geom.exterior.xy[1], sigma=sigma
                )
                for geom in obj.geoms
                if isinstance(geom, (Polygon, LineString))
            ]
        )
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")

    return smoothed


def uniform_smooth(obj: Any, size: int = 10):
    """Smooth a shapely (multi)polygon|(multi)linestring using a uniform filter."""
    if isinstance(obj, Polygon):
        x, y = obj.exterior.xy
        x_smooth, y_smooth = _uniform_filter_2d(x, y, size=size)
        smoothed = Polygon(zip(x_smooth, y_smooth))
    elif isinstance(obj, MultiPolygon):
        smoothed = MultiPolygon(
            [uniform_smooth(poly.exterior, size=size) for poly in obj.geoms]
        )
    elif isinstance(obj, LineString):
        x, y = obj.xy
        x_smooth, y_smooth = _uniform_filter_2d(x, y, size=size)
        smoothed = LineString(zip(x_smooth, y_smooth))
    elif isinstance(obj, MultiLineString):
        smoothed = MultiLineString(
            [
                LineString(zip(*_uniform_filter_2d(line.xy[0], line.xy[1], size=size)))
                for line in obj.geoms
            ]
        )
    elif isinstance(obj, GeometryCollection):
        smoothed = GeometryCollection(
            [
                _uniform_filter_2d(geom.exterior.xy[0], geom.exterior.xy[1], size=size)
                for geom in obj.geoms
                if isinstance(geom, (Polygon, LineString))
            ]
        )
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")

    return smoothed


def inst2gdf(
    inst_map: np.ndarray,
    type_map: np.ndarray = None,
    xoff: int = None,
    yoff: int = None,
    class_dict: Dict[int, str] = None,
    min_size: int = 15,
    smooth_func: Callable = uniform_smooth,
) -> gpd.GeoDataFrame:
    """Convert an instance segmentation raster mask to a GeoDataFrame.

    Note:
        This function should be applied to nuclei instance segmentation masks. Nuclei
        types can be provided with the `type_map` and `class_dict` arguments if needed.

    Parameters:
        inst_map (np.ndarray):
            An instance segmentation mask. Shape (H, W).
        type_map (np.ndarray):
            A type segmentation mask. Shape (H, W). If provided, the types will be
            included in the resulting GeoDataFrame in column 'class_name'.
        xoff (int):
            The x offset. Optional. The offset is used to translate the geometries
            in the GeoDataFrame. If None, no translation is applied.
        yoff (int):
            The y offset. Optional. The offset is used to translate the geometries
            in the GeoDataFrame. If None, no translation is applied.
        class_dict (Dict[int, str]):
            A dictionary mapping class indices to class names.
            e.g. {1: 'neoplastic', 2: 'immune'}. If None, the class indices will be used.
        min_size (int):
            The minimum size (in pixels) of the polygons to include in the GeoDataFrame.
        smooth_func (Callable):
            A function to smooth the polygons. The function should take a shapely Polygon
            as input and return a shapely Polygon. Defaults to `uniform_smooth`, which
            applies a uniform filter. `histolytics.utils._filters` also provides
            `gaussian_smooth` and `median_smooth` for smoothing.

    returns:
        gpd.GeoDataFrame:
            A GeoDataFrame of the raster instance mask. Contains columns:

                - 'id' - the numeric pixel value of the instance mask,
                - 'class_name' - the name or index of the instance class (requires `type_map` and `class_dict`),
                - 'geometry' - the geometry of the polygon.

    Examples:
        >>> from histolytics.utils.raster import inst2gdf
        >>> from histolytics.data import hgsc_cancer_inst_mask, hgsc_cancer_type_mask
        >>> # load raster masks
        >>> inst_mask = hgsc_cancer_inst_mask()
        >>> type_mask = hgsc_cancer_type_mask()
        >>> # convert to GeoDataFrame
        >>> gdf = inst2gdf(inst_mask, type_mask)
        >>> print(gdf.head(3))
                uid  class_name                                           geometry
            0  135           1  POLYGON ((405.019 0.45, 405.43 1.58, 406.589 2...
            1  200           1  POLYGON ((817.01 0.225, 817.215 0.804, 817.795...
            2    0           1  POLYGON ((1394.01 0.45, 1394.215 1.58, 1394.79...
    """
    # handle empty masks
    if inst_map.size == 0 or np.max(inst_map) == 0:
        return gpd.GeoDataFrame(columns=["uid", "class_name", "geometry"])

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
            for polygon, value in shapes(inst_map, mask=mask)
        )

        res = gpd.GeoDataFrame(
            vectorized_data,
            columns=["uid", "class_name", "geometry"],
        )
        res["uid"] = res["uid"].astype(int)
        inst_maps_per_type.append(res)

    res = pd.concat(inst_maps_per_type)

    # filter out small geometries
    res = res.loc[res.area > min_size].reset_index(drop=True)

    # translate geometries if offsets are provided
    if xoff is not None or yoff is not None:
        res["geometry"] = res["geometry"].translate(
            xoff if xoff is not None else 0, yoff if yoff is not None else 0
        )

    # smooth geometries if a smoothing function is provided
    if smooth_func is not None:
        res["geometry"] = res["geometry"].apply(smooth_func)

    return res


def sem2gdf(
    sem_map: np.ndarray,
    xoff: int = None,
    yoff: int = None,
    class_dict: Dict[int, str] = None,
    min_size: int = 15,
    smooth_func: Callable = uniform_smooth,
) -> gpd.GeoDataFrame:
    """Convert an semantic segmentation raster mask to a GeoDataFrame.

    Note:
        This function should be applied to semantic tissue segmentation masks.

    Parameters:
        sem_map (np.ndarray):
            A semantic segmentation mask. Shape (H, W).
        xoff (int):
            The x offset. Optional. The offset is used to translate the geometries
            in the GeoDataFrame. If None, no translation is applied.
        yoff (int):
            The y offset. Optional. The offset is used to translate the geometries
            in the GeoDataFrame. If None, no translation is applied.
        class_dict (Dict[int, str]):
            A dictionary mapping class indices to class names.
            e.g. {1: 'neoplastic', 2: 'immune'}. If None, the class indices will be used.
        min_size (int):
            The minimum size (in pixels) of the polygons to include in the GeoDataFrame.
        smooth_func (Callable):
            A function to smooth the polygons. The function should take a shapely Polygon
            as input and return a shapely Polygon. Defaults to `uniform_smooth`, which
            applies a uniform filter. `histolytics.utils._filters` also provides
            `gaussian_smooth` and `median_smooth` for smoothing.
    returns:
        gpd.GeoDataFrame:
            A GeoDataFrame of the raster semantic mask. Contains columns:

                - 'id' - the numeric pixel value of the semantic mask,
                - 'class_name' - the name of the class (same as id if class_dict is None),
                - 'geometry' - the geometry of the polygon.

    Examples:
        >>> from histolytics.utils.raster import sem2gdf
        >>> from histolytics.data import hgsc_cancer_type_mask
        >>> # load semantic mask
        >>> type_mask = hgsc_cancer_type_mask()
        >>> # convert to GeoDataFrame
        >>> gdf = sem2gdf(type_mask)
        >>> print(gdf.head(3))
                uid  class_name                                           geometry
            0   2           2  POLYGON ((850.019 0.45, 850.431 1.58, 851.657 ...
            1   2           2  POLYGON ((1194.01 0.225, 1194.215 0.795, 1194....
            2   1           1  POLYGON ((405.019 0.45, 405.43 1.58, 406.589 2...
    """
    # Handle empty semantic mask
    if sem_map.size == 0 or np.max(sem_map) == 0:
        return gpd.GeoDataFrame(columns=["uid", "class_name", "geometry"])

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
        columns=["uid", "geometry"],
    )
    res["uid"] = res["uid"].astype(int)
    res = res.loc[res.area > min_size].reset_index(drop=True)
    res["class_name"] = res["uid"].map(class_dict)
    res = res[["uid", "class_name", "geometry"]]  # reorder columns

    if xoff is not None or yoff is not None:
        res["geometry"] = res["geometry"].translate(
            xoff if xoff is not None else 0, yoff if yoff is not None else 0
        )

    if smooth_func is not None:
        res["geometry"] = res["geometry"].apply(smooth_func)

    return res
