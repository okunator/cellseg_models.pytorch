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
    clipped = [
        complete.intersection(box(x, y, x + w, y + h)) for x, y, w, h in coordinates
    ]
    fragments = [gaussian_smooth(fragment) for fragment in clipped]
    source = gpd.GeoDataFrame(
        {
            "class_name": [1] * len(fragments),
            "is_partial": [not fragment.equals(complete) for fragment in clipped],
        },
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


@pytest.mark.parametrize("better_view", [0, 1])
def test_one_versus_many_uses_a_coherent_view_without_combining_alternatives(
    better_view,
):
    source = gpd.GeoDataFrame(
        {"class_name": [1, 2, 3], "is_partial": [False] * 3},
        geometry=[box(70, 20, 90, 35), box(70, 35, 90, 50), box(70, 20, 90, 50)],
    )
    coordinates = [(0, 0, 120, 100), (60, 0, 100, 100)]
    if better_view == 1:
        coordinates.reverse()
    actual = InstMerger(source, coordinates, tile_indices=[0, 0, 1]).merge(
        simplify_level=0
    )
    expected = source.iloc[:2] if better_view == 0 else source.iloc[2:]
    assert actual.class_name.tolist() == expected.class_name.tolist()
    assert all(a.equals(b) for a, b in zip(actual.geometry, expected.geometry))
    assert actual.uid.is_unique


def test_overlap_keeps_the_context_winners_holes_and_attributes():
    from shapely.geometry import Polygon

    complete = Polygon(
        [(64, 20), (99, 20), (99, 60), (64, 60)],
        holes=[[(75, 30), (75, 40), (82, 40), (82, 30)]],
    )
    source = gpd.GeoDataFrame(
        {"class_name": [1, 3], "score": [0.2, 0.9]},
        geometry=[box(60, 20, 97, 60), complete],
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


def test_overlap_keeps_unmatched_edge_prediction_without_a_replacement():
    source = gpd.GeoDataFrame(
        {"class_name": [1]},
        geometry=[box(92, 30, 99.8, 60)],
    )
    actual = InstMerger(
        source, [(0, 0, 100, 100), (60, 0, 100, 100)], tile_indices=[0]
    ).merge(simplify_level=0)
    assert len(actual) == 1
    assert actual.geometry.iloc[0].equals(source.geometry.iloc[0])


def test_overlap_keeps_unmatched_outer_edge_nuclei():
    source = gpd.GeoDataFrame(
        {"class_name": [1]},
        geometry=[box(0, 30, 8, 60)],
    )
    actual = InstMerger(
        source, [(0, 0, 100, 100), (60, 0, 100, 100)], tile_indices=[0]
    ).merge(simplify_level=0)
    assert len(actual) == 1
    assert actual.geometry.iloc[0].equals(source.geometry.iloc[0])


@pytest.mark.parametrize("indices", [[0, 1], [1, 0]])
def test_overlapping_predictions_with_centroid_jitter_are_still_deduplicated(indices):
    source = gpd.GeoDataFrame(
        {"class_name": [1, 2]},
        geometry=[box(60, 20, 97, 60), box(64, 20, 99, 60)],
    )
    actual = InstMerger(
        source, [(0, 0, 100, 100), (60, 0, 100, 100)], tile_indices=indices
    ).merge(simplify_level=0)
    assert len(actual) == 1
    assert actual.geometry.iloc[0].equals(source.geometry.iloc[1])
    assert actual.class_name.tolist() == [2]


def test_complete_unmatched_prediction_survives_outside_nominal_ownership():
    source = gpd.GeoDataFrame(
        {"class_name": [3], "score": [0.9]}, geometry=[box(82, 30, 90, 45)]
    )
    actual = InstMerger(
        source, [(0, 0, 100, 100), (60, 0, 100, 100)], tile_indices=[0]
    ).merge(simplify_level=0)
    assert len(actual) == 1
    assert actual.geometry.iloc[0].equals(source.geometry.iloc[0])
    assert actual.class_name.tolist() == [3]
    assert actual.score.tolist() == [0.9]


def test_tiny_interior_prediction_cannot_replace_a_complete_nucleus():
    source = gpd.GeoDataFrame(
        {"class_name": [3, 1], "score": [0.9, 0.2]},
        geometry=[box(60, 20, 98, 60), box(82, 30, 86, 40)],
    )
    actual = InstMerger(
        source, [(0, 0, 100, 100), (60, 0, 100, 100)], tile_indices=[0, 1]
    ).merge(simplify_level=0)
    assert len(actual) == 1
    complete = actual.loc[actual.class_name == 3].iloc[0]
    assert complete.geometry.equals(source.geometry.iloc[0])
    assert complete.geometry.area == 1520
    assert complete.score == 0.9


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("same_tile", [False, True])
@pytest.mark.parametrize("simplify,precision", [(0, None), (0.1, 1)])
def test_distinct_overlapping_nuclei_keep_original_contours_without_intersection_objects(
    count, same_tile, simplify, precision
):
    geometries = [
        box(65 + 7 * i, 20 + 7 * i, 85 + 7 * i, 40 + 7 * i) for i in range(count)
    ]
    source = gpd.GeoDataFrame(
        {"class_name": list(range(1, count + 1)), "source_id": list(range(count))},
        geometry=geometries,
    )
    actual = InstMerger(
        source,
        [(0, 0, 120, 100)] * count,
        tile_indices=[0] * count if same_tile else list(range(count)),
    ).merge(simplify_level=simplify, precision=precision)
    assert len(actual) == count
    assert actual.source_id.tolist() == list(range(count))
    assert all(a.equals(b) for a, b in zip(actual.geometry, geometries))
    assert actual.uid.is_unique and actual.is_valid.all()
    assert actual.geometry.iloc[0].intersection(actual.geometry.iloc[1]).area > 0


@pytest.mark.parametrize("order", [[0, 1, 2, 3], [3, 2, 0, 1], [1, 0, 3, 2]])
def test_split_alternatives_below_half_iou_are_selected_independently_of_row_order(
    order,
):
    geometries = [
        box(70, 20, 90, 50),
        box(68, 20, 91, 34),
        box(68, 36, 91, 50),
        box(100, 65, 105, 70),
    ]
    source = gpd.GeoDataFrame(
        {"class_name": [1, 2, 3, 4], "is_partial": [False] * 4}, geometry=geometries
    ).iloc[order]
    owners = [0, 1, 1, 0]
    actual = InstMerger(
        source,
        [(0, 0, 110, 100), (40, 0, 110, 100)],
        tile_indices=[owners[i] for i in order],
    ).merge(simplify_level=0)
    assert set(actual.class_name) == {2, 3, 4}
    for label in (2, 3, 4):
        assert (
            actual.loc[actual.class_name == label]
            .geometry.iloc[0]
            .equals(geometries[label - 1])
        )


def test_an_existing_intersection_prediction_is_redundant_to_two_complete_nuclei():
    first, second = box(70, 20, 90, 45), box(80, 20, 100, 45)
    source = gpd.GeoDataFrame(
        {"class_name": [1, 2, 3], "source_id": [10, 20, 30]},
        geometry=[first, second, first.intersection(second)],
    )
    actual = InstMerger(
        source, [(0, 0, 120, 100), (60, 0, 120, 100)], tile_indices=[0, 0, 1]
    ).merge(simplify_level=0)
    assert actual.source_id.tolist() == [10, 20]
    assert actual.geometry.iloc[0].equals(first)
    assert actual.geometry.iloc[1].equals(second)
    assert actual.geometry.iloc[0].intersection(actual.geometry.iloc[1]).area == 250


def test_conflict_selection_does_not_discard_a_transitively_connected_nucleus():
    source = gpd.GeoDataFrame(
        {"class_name": [1, 2, 3], "is_partial": [False] * 3},
        geometry=[box(70, 30, 90, 50), box(75, 30, 95, 50), box(80, 30, 100, 50)],
    )
    actual = InstMerger(
        source,
        [(-10, 0, 130, 100), (70, 0, 100, 100), (75, 0, 100, 100)],
        tile_indices=[0, 1, 2],
    ).merge(simplify_level=0)
    assert actual.class_name.tolist() == [1, 3]
    assert actual.geometry.iloc[0].equals(source.geometry.iloc[0])
    assert actual.geometry.iloc[1].equals(source.geometry.iloc[2])


def test_containment_within_one_source_tile_does_not_remove_an_instance():
    source = gpd.GeoDataFrame(
        {"class_name": [1, 2]}, geometry=[box(70, 20, 90, 45), box(75, 25, 80, 30)]
    )
    actual = InstMerger(
        source, [(0, 0, 120, 100), (60, 0, 120, 100)], tile_indices=[0, 0]
    ).merge(simplify_level=0)
    assert len(actual) == 2
    assert all(a.equals(b) for a, b in zip(actual.geometry, source.geometry))


def test_complete_view_is_not_excluded_by_a_larger_partial_alternative():
    source = gpd.GeoDataFrame(
        {"class_name": [1, 2], "is_partial": [True, False]},
        geometry=[box(70, 20, 100, 50), box(70, 20, 90, 50)],
    )
    actual = InstMerger(
        source, [(0, 0, 100, 100), (60, 0, 100, 100)], tile_indices=[0, 1]
    ).merge(simplify_level=0)
    assert actual.class_name.tolist() == [2]
    assert actual.geometry.iloc[0].equals(source.geometry.iloc[1])


def test_complete_nuclei_collectively_replace_a_large_partial_alternative():
    source = gpd.GeoDataFrame(
        {"class_name": [1, 2, 3], "is_partial": [True, False, False]},
        geometry=[box(70, 20, 100, 50), box(70, 20, 90, 34), box(70, 36, 90, 50)],
    )
    actual = InstMerger(
        source, [(0, 0, 100, 100), (60, 0, 100, 100)], tile_indices=[0, 1, 1]
    ).merge(simplify_level=0)
    assert actual.class_name.tolist() == [2, 3]
    assert all(a.equals(b) for a, b in zip(actual.geometry, source.geometry.iloc[1:]))


def test_complete_alternative_can_replace_a_partial_contour_below_half_iou():
    source = gpd.GeoDataFrame(
        {"class_name": [1, 2], "is_partial": [True, False]},
        geometry=[box(70, 20, 100, 50), box(67, 20, 85, 50)],
    )
    actual = InstMerger(
        source, [(0, 0, 100, 100), (60, 0, 100, 100)], tile_indices=[0, 1]
    ).merge(simplify_level=0)
    assert actual.class_name.tolist() == [2]
    assert actual.geometry.iloc[0].equals(source.geometry.iloc[1])


def test_rounding_may_mix_overlapping_and_abutting_tiles():
    from cellseg_models_pytorch.wsi.tiles import get_tile_coordinates

    coordinates = get_tile_coordinates((20, 200), width=100, height=10, overlap=0.05)
    source = gpd.GeoDataFrame(
        {"class_name": [1, 1]},
        geometry=[box(20, 7, 30, 10), box(20, 10, 30, 13)],
    )
    actual = InstMerger(source, coordinates, tile_indices=[0, 2]).merge(
        simplify_level=0
    )
    assert len(actual) == 1
    assert actual.geometry.iloc[0].equals(box(20, 7, 30, 13))


def test_mixed_grid_joins_abutting_fragments_after_overlap_deduplication():
    coordinates = [
        (0, 0, 100, 100),
        (60, 0, 100, 100),
        (0, 100, 100, 100),
        (60, 100, 100, 100),
    ]
    complete = box(85, 85, 115, 115)
    source = gpd.GeoDataFrame(
        {"class_name": [1, 2, 3, 4]},
        geometry=[
            complete.intersection(box(x, y, x + w, y + h)) for x, y, w, h in coordinates
        ],
    )
    actual = InstMerger(source, coordinates, tile_indices=[0, 1, 2, 3]).merge(
        simplify_level=0
    )
    assert len(actual) == 1
    assert actual.geometry.iloc[0].equals(complete)
    assert actual.geometry.iloc[0].geom_type == "Polygon"
    assert actual.class_name.tolist() == [2]


@pytest.mark.parametrize("flags", [[False, None], [0, 1], ["False", "True"]])
def test_overlap_rejects_invalid_border_flags(flags):
    source = gpd.GeoDataFrame(
        {"class_name": [1, 1], "is_partial": flags},
        geometry=[box(80, 20, 90, 30)] * 2,
    )
    with pytest.raises(ValueError, match="is_partial"):
        InstMerger(source, [(0, 0, 100, 100), (60, 0, 100, 100)], tile_indices=[0, 1])


def test_complete_tiny_detection_does_not_replace_a_larger_partial_contour():
    source = gpd.GeoDataFrame(
        {"class_name": [3, 1], "is_partial": [True, False]},
        geometry=[box(70, 20, 100, 60), box(82, 30, 86, 40)],
    )
    actual = InstMerger(
        source, [(0, 0, 100, 100), (60, 0, 100, 100)], tile_indices=[0, 1]
    ).merge(simplify_level=0)
    assert len(actual) == 2
    assert actual.geometry.iloc[0].equals(source.geometry.iloc[0])
