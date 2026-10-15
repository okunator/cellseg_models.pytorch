import geopandas as gpd
import pytest
from shapely.geometry import box
from shapely.ops import unary_union

from cellseg_models_pytorch.wsi.inst_merger import InstMerger


@pytest.mark.parametrize(
    "coordinates,geometries,expected",
    [
        ([(0, 0, 100, 100)], [box(20, 20, 30, 30)], [box(20, 20, 30, 30)]),
        (
            [
                (0, 0, 100, 100),
                (100, 0, 100, 100),
                (0, 100, 100, 100),
                (100, 100, 100, 100),
            ],
            [
                box(20, 20, 30, 30),
                box(120, 20, 130, 30),
                box(20, 120, 30, 130),
                box(120, 120, 130, 130),
            ],
            [
                box(20, 20, 30, 30),
                box(120, 20, 130, 30),
                box(20, 120, 30, 130),
                box(120, 120, 130, 130),
            ],
        ),
        (
            [(0, 0, 100, 100), (100, 0, 100, 100)],
            [
                box(90, 20, 100, 30),
                box(100, 20, 115, 30),
                box(20, 20, 30, 30),
                box(180, 20, 190, 30),
            ],
            [box(90, 20, 115, 30), box(20, 20, 30, 30), box(180, 20, 190, 30)],
        ),
        (
            [
                (0, 0, 100, 100),
                (100, 0, 100, 100),
                (0, 100, 100, 100),
                (100, 100, 100, 100),
            ],
            [
                box(90, 90, 100, 100),
                box(100, 90, 110, 100),
                box(90, 100, 100, 110),
                box(100, 100, 110, 110),
            ],
            [box(90, 90, 110, 110)],
        ),
        (
            [(0, 0, 100, 100), (100, 0, 100, 100)],
            [box(90, 20, 100, 30), box(100, 34, 110, 44)],
            [box(90, 20, 100, 30), box(100, 34, 110, 44)],
        ),
        (
            [(0, 0, 100, 100), (100, 0, 100, 100)],
            [box(90, 20, 100, 30), box(90, 30, 100, 40)],
            [box(90, 20, 100, 30), box(90, 30, 100, 40)],
        ),
        (
            [(0, 0, 100, 100), (200, 0, 100, 100)],
            [box(90, 20, 100, 30), box(200, 20, 210, 30)],
            [box(90, 20, 100, 30), box(200, 20, 210, 30)],
        ),
    ],
)
def test_merge_preserves_interiors_and_joins_only_neighboring_tile_fragments(
    coordinates, geometries, expected
):
    source = gpd.GeoDataFrame(
        {"class_name": [1] * len(geometries), "geometry": geometries},
        index=[0] * len(geometries),
    )
    actual = InstMerger(source, coordinates).merge(simplify_level=0)
    assert len(actual) == len(expected)
    assert all(
        any(geometry.equals(wanted) for geometry in actual.geometry)
        for wanted in expected
    )
    assert unary_union(actual.geometry).equals(unary_union(geometries))
    assert actual.uid.is_unique
    assert actual.crs is None
    assert len(source) == len(geometries)


def test_merge_empty_and_existing_output(tmp_path):
    source = gpd.GeoDataFrame({"class_name": [], "geometry": []})
    merger = InstMerger(source, [(0, 0, 100, 100)])
    assert merger.merge().empty
    path = tmp_path / "nuclei.parquet"
    merger.merge(path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        merger.merge(path)
    assert path.read_bytes() == before


def test_merge_class_comes_from_largest_fragment():
    source = gpd.GeoDataFrame(
        {
            "class_name": [3, 2],
            "geometry": [box(90, 20, 100, 30), box(100, 20, 120, 30)],
        }
    )
    actual = InstMerger(source, [(0, 0, 100, 100), (100, 0, 100, 100)]).merge(
        simplify_level=0
    )
    assert actual.class_name.to_list() == [2]
