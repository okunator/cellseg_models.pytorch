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


def test_abutting_disconnected_fragments_do_not_become_one_multipart_nucleus():
    source = gpd.GeoDataFrame(
        {"class_name": [1, 1]},
        geometry=[box(90, 20, 99.8, 30), box(100.2, 20, 115, 30)],
    )
    actual = InstMerger(source, [(0, 0, 100, 100), (100, 0, 100, 100)]).merge(
        simplify_level=0
    )
    assert len(actual) == 2
    assert (actual.geom_type == "Polygon").all()
    assert actual.uid.is_unique


@pytest.mark.parametrize("axis", ["horizontal", "vertical", "junction"])
def test_overlap_keeps_a_complete_contour_instead_of_disconnected_fragments(axis):
    from cellseg_models_pytorch.utils.vectorize import gaussian_smooth

    coordinates = [(0, 0, 100, 100), (60, 0, 100, 100)]
    complete = box(85, 35, 115, 65)
    if axis == "vertical":
        coordinates = [(0, 0, 100, 100), (0, 60, 100, 100)]
        complete = box(35, 85, 65, 115)
    elif axis == "junction":
        coordinates = [
            (0, 0, 100, 100),
            (60, 0, 100, 100),
            (0, 60, 100, 100),
            (60, 60, 100, 100),
        ]
        complete = box(85, 85, 115, 115)
    fragments = [
        gaussian_smooth(complete.intersection(box(x, y, x + w, y + h)))
        for x, y, w, h in coordinates
    ]
    source = gpd.GeoDataFrame(
        {"class_name": [1] * len(fragments)},
        geometry=fragments,
    )
    actual = InstMerger(
        source, coordinates, tile_indices=list(range(len(source)))
    ).merge(simplify_level=0)
    assert len(actual) == 1
    assert actual.geometry.iloc[0].geom_type == "Polygon"
    assert actual.geometry.iloc[0].is_valid
    assert actual.geometry.iloc[0].equals(fragments[-1])
    # Every output pixel comes from the selected original prediction.
    assert any(actual.geometry.iloc[0].equals(fragment) for fragment in fragments)


def test_overlap_matching_cannot_fuse_two_nuclei_from_the_same_tile():
    source = gpd.GeoDataFrame(
        {"class_name": [1, 2, 3]},
        geometry=[box(62, 20, 96, 34), box(62, 34, 96, 48), box(66, 20, 100, 48)],
    )
    actual = InstMerger(
        source, [(0, 0, 100, 100), (60, 0, 100, 100)], tile_indices=[0, 0, 1]
    ).merge(simplify_level=0)
    assert len(actual) == 2
    assert actual.uid.is_unique


def test_overlap_keeps_the_context_winners_holes_and_attributes():
    from shapely.geometry import Polygon

    complete = Polygon(
        [(85, 20), (115, 20), (115, 60), (85, 60)],
        holes=[[(95, 30), (95, 40), (105, 40), (105, 30)]],
    )
    source = gpd.GeoDataFrame(
        {"class_name": [1, 3], "score": [0.2, 0.9]},
        geometry=[complete.intersection(box(0, 0, 100, 100)), complete],
    )
    actual = InstMerger(
        source, [(0, 0, 100, 100), (60, 0, 100, 100)], tile_indices=[0, 1]
    ).merge(simplify_level=0)
    assert len(actual) == 1
    assert actual.geometry.iloc[0].equals(complete)
    assert len(actual.geometry.iloc[0].interiors) == 1
    assert actual.class_name.tolist() == [3]
    assert actual.score.tolist() == [0.9]


@pytest.mark.parametrize("indices", [None, [0], [0, 2], [-1, 1], [0.5, 1.0]])
def test_overlap_requires_valid_source_tile_indices(indices):
    source = gpd.GeoDataFrame(
        {"class_name": [1, 1]},
        geometry=[box(80, 20, 90, 30)] * 2,
    )
    with pytest.raises(ValueError, match="tile_indices"):
        InstMerger(source, [(0, 0, 100, 100), (60, 0, 100, 100)], tile_indices=indices)


def test_overlap_ownership_discards_unmatched_edge_fragments():
    source = gpd.GeoDataFrame(
        {"class_name": [1]},
        geometry=[box(92, 30, 99.8, 60)],
    )
    actual = InstMerger(
        source, [(0, 0, 100, 100), (60, 0, 100, 100)], tile_indices=[0]
    ).merge(simplify_level=0)
    assert actual.empty


def test_overlap_ownership_keeps_unmatched_outer_edge_nuclei():
    source = gpd.GeoDataFrame(
        {"class_name": [1]},
        geometry=[box(0, 30, 8, 60)],
    )
    actual = InstMerger(
        source, [(0, 0, 100, 100), (60, 0, 100, 100)], tile_indices=[0]
    ).merge(simplify_level=0)
    assert len(actual) == 1
    assert actual.geometry.iloc[0].equals(source.geometry.iloc[0])


def test_overlapping_predictions_with_centroid_jitter_are_still_deduplicated():
    source = gpd.GeoDataFrame(
        {"class_name": [1, 2]},
        geometry=[box(60, 20, 97, 60), box(64, 20, 99, 60)],
    )
    actual = InstMerger(
        source, [(0, 0, 100, 100), (60, 0, 100, 100)], tile_indices=[0, 1]
    ).merge(simplify_level=0)
    assert len(actual) == 1
    assert actual.geometry.iloc[0].equals(source.geometry.iloc[1])
    assert actual.class_name.tolist() == [2]
