from typing import Union

import geopandas as gpd
import numpy as np

__all__ = ["get_objs"]


def get_objs(
    area: gpd.GeoDataFrame,
    objects: gpd.GeoDataFrame,
    predicate: str = "intersects",
    **kwargs,
) -> Union[gpd.GeoDataFrame, None]:
    # NOTE, gdfs need to have same crs, otherwise warning flood.
    inds = objects.geometry.sindex.query(area.geometry, predicate=predicate, **kwargs)
    objs: gpd.GeoDataFrame = objects.iloc[np.unique(inds)]

    return objs
