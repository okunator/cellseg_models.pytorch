import geopandas as gpd
import pytest
from shapely.geometry import Polygon, Point
from cellseg_models_pytorch.utils.spatial_ops import get_objs

def test_get_objs_intersects():
    area = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])])

    objects = gpd.GeoDataFrame(geometry=[
        Point(1, 1),
        Point(3, 3),
        Point(1, 3),
        Point(3, 1)
    ])

    result = get_objs(area, objects, predicate="intersects")

    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 1
    assert result.iloc[0].geometry == Point(1, 1)

def test_get_objs_contains():
    area = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])])

    objects = gpd.GeoDataFrame(geometry=[
        Point(1, 1),
        Point(3, 3),
        Point(1, 3),
        Point(3, 1)
    ])

    result = get_objs(area, objects, predicate="contains")

    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 4

def test_get_objs_no_match():
    area = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])])

    objects = gpd.GeoDataFrame(geometry=[
        Point(2, 2),
        Point(3, 3)
    ])

    result = get_objs(area, objects, predicate="intersects")

    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0
