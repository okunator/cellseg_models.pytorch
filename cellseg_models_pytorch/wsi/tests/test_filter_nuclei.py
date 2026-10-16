import geopandas as gpd
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from cellseg_models_pytorch.wsi import inst_merger
from cellseg_models_pytorch.wsi.inst_merger import InstMerger, filter_nuclei


@pytest.mark.parametrize("pixel_size", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("angle", [0, 35, 90])
def test_middle_filter_preserves_thin_stromal_nuclei_and_small_thicker_nuclei(
    pixel_size, angle
):
    # A long thin stromal contour has enough area; a small contour has enough thickness.
    physical = [box(0, 0, 1, 20), box(3, 0, 5.2, 3.4), box(7, 0, 8, 4)]
    source = gpd.GeoDataFrame(
        {"uid": [17, 31, 90], "class_name": [2, 2, 0], "score": [0.9, 0.8, 0.7]},
        geometry=[
            translate(
                scale(rotate(g, angle), xfact=1 / pixel_size, yfact=1 / pixel_size),
                xoff=100_000,
                yoff=-100_000,
            )
            for g in physical
        ],
        index=[101, 203, 305],
    ).rename_geometry("contour")
    before = source.copy()
    # Rotation roundoff is irrelevant here; neither retained contour is near both limits.
    actual = filter_nuclei(source, pixel_size=pixel_size)
    expected = source.iloc[:2]
    assert_geodataframe_equal(actual, expected)
    assert actual.geometry.to_wkb().tolist() == expected.geometry.to_wkb().tolist()
    assert_geodataframe_equal(source, before)


@pytest.mark.parametrize(
    "level,expected",
    [("conservative", [10, 20, 30]), ("middle", [20, 30]), ("strong", [30])],
)
def test_filter_levels_require_both_area_and_thickness_below_their_limits(
    level, expected
):
    source = gpd.GeoDataFrame(
        {"uid": [10, 20, 30], "class_name": [2, 2, 2]},
        geometry=[box(0, 0, 1, 8), box(3, 0, 4, 14), box(6, 0, 9, 8)],
    )
    assert filter_nuclei(source, pixel_size=1, level=level).uid.tolist() == expected


@pytest.mark.parametrize(
    "geometry,keep",
    [
        (box(0, 0, 4, 48), True),  # Area is exactly 12 µm², thickness below 1.2 µm.
        (box(0, 0, 8, 12), True),  # Thickness is exactly 1.2 µm, area below 12 µm².
        (box(0, 0, 4, 44), False),
        (box(0, 0, 8, 10), False),
    ],
)
def test_middle_filter_uses_strict_physical_thresholds(geometry, keep):
    source = gpd.GeoDataFrame(geometry=[geometry])
    actual = filter_nuclei(source, pixel_size=0.25)
    assert len(actual) == int(keep)
    if keep:
        assert actual.geometry.iloc[0].wkb == geometry.wkb


@pytest.mark.parametrize("pixel_size", [0, -0.25, np.nan, np.inf, -np.inf])
def test_filter_rejects_invalid_calibration(pixel_size):
    with pytest.raises(ValueError, match="pixel_size"):
        filter_nuclei(gpd.GeoDataFrame(geometry=[]), pixel_size=pixel_size)


def test_filter_rejects_unknown_level():
    with pytest.raises(ValueError, match="filter level"):
        filter_nuclei(gpd.GeoDataFrame(geometry=[]), pixel_size=0.25, level="unknown")


def test_empty_filtered_result_preserves_schema_and_crs():
    source = gpd.GeoDataFrame(
        {"uid": [7], "class_name": [0]}, geometry=[box(0, 0, 1, 4)], crs="EPSG:3857"
    )
    empty = source.iloc[:0]
    assert_geodataframe_equal(filter_nuclei(source, pixel_size=1), empty)
    assert_geodataframe_equal(filter_nuclei(empty, pixel_size=1), empty)


@pytest.mark.parametrize("export", [False, True])
def test_merge_filters_after_joining_fragments_and_preserves_retained_ids(
    tmp_path, export
):
    geometries = [box(20, 4, 24, 20), box(12, 12, 16, 40), box(12, 40, 16, 68)]
    source = gpd.GeoDataFrame(
        {"class_name": [0, 2, 2], "source_id": ["small", "top", "bottom"]},
        geometry=geometries,
    )
    merger = InstMerger(
        source, [(0, 0, 40, 40), (0, 40, 40, 40)], tile_indices=[0, 0, 1]
    )
    unfiltered = merger.merge(simplify_level=0)
    assert len(unfiltered) == 2
    assert filter_nuclei(source.iloc[1:], pixel_size=0.25).empty
    destination = tmp_path / "nuclei.parquet" if export else None
    actual = merger.merge(
        destination, simplify_level=0, filter_fragments=True, pixel_size=0.25
    )
    if export:
        assert actual is None
        actual = gpd.read_parquet(destination)
    assert actual.uid.tolist() == [1]
    assert_geodataframe_equal(actual, unfiltered.iloc[[1]])
    assert actual.geometry.iloc[0].equals(unary_union(geometries[1:]))


def test_disabled_filter_does_no_filter_work(monkeypatch):
    def unexpected_filter(*args, **kwargs):
        pytest.fail("A disabled filter must not measure or select polygons")

    monkeypatch.setattr(inst_merger, "filter_nuclei", unexpected_filter)
    source = gpd.GeoDataFrame({"class_name": [2]}, geometry=[box(10, 10, 11, 14)])
    actual = InstMerger(source, [(0, 0, 40, 40)]).merge(simplify_level=0)
    assert actual.geometry.iloc[0].wkb == source.geometry.iloc[0].wkb


@pytest.mark.parametrize(
    "kwargs", [{}, {"pixel_size": 0}, {"pixel_size": 0.25, "filter_level": "unknown"}]
)
def test_invalid_enabled_filter_leaves_existing_data_untouched(tmp_path, kwargs):
    source = gpd.GeoDataFrame({"class_name": [2]}, geometry=[box(10, 10, 11, 14)])
    before = source.copy()
    destination = tmp_path / "nuclei.parquet"
    with pytest.raises(ValueError):
        InstMerger(source, [(0, 0, 40, 40)]).merge(
            destination, filter_fragments=True, **kwargs
        )
    assert not destination.exists()
    assert_geodataframe_equal(source, before)


def test_observed_elongated_stromal_contour_is_retained_unchanged():
    """Retain a prediction matching a stromal annotation in the CC0 PanopTILs data."""
    # Dataset d40293bc, region_0658, prediction 110; WSI-style 0.1-pixel simplification.
    geometry = Polygon(
        [
            (47.99032254456237, 415.6454068980835),
            (47.78486436730554, 417.3838010420999),
            (47.20539965263339, 418.61643305639393),
            (46.79454182274318, 420.3933029236672),
            (46.20545817725683, 421.83162721371184),
            (45.79454182274318, 424.1975807264714),
            (45.20545817725683, 426.25251364678036),
            (44.20545817725683, 433.73986898487107),
            (43.22481308813211, 450.9719717441484),
            (43.22481308813211, 469.940404435302),
            (43.79465887199005, 477.54101092746555),
            (44.20545817725683, 480.14743771058),
            (45.20545817725683, 481.79436624887285),
            (46.82357418905609, 483.20534112801),
            (47.850748026093356, 483.775186911868),
            (50.15898795396773, 483.775186911868),
            (52.57928914080183, 482.79448329811976),
            (53.79448329811974, 481.57940619004876),
            (54.20545817725683, 480.41085782989023),
            (54.79454182274318, 479.5890836454863),
            (55.20545817725683, 478.4109163545137),
            (55.79454182274318, 477.58902512086297),
            (56.20545817725684, 476.401238899076),
            (56.79454182274318, 475.37383096354506),
            (57.20545817725684, 473.5969610962718),
            (57.79454182274318, 471.95317862897025),
            (58.20545817725683, 468.99814147072436),
            (58.79454182274317, 466.3263072970341),
            (59.77518691186789, 454.70008226936136),
            (59.77518691186789, 440.30965371069976),
            (58.79454182274317, 429.46829305033265),
            (57.79454182274318, 424.8316857383353),
            (57.20545817725684, 423.39341997291405),
            (56.79448329811974, 421.63578796726927),
            (55.57940619004871, 420.20557522650375),
            (54.41085782989022, 419.7945418227432),
            (53.58914217010979, 419.20545817725684),
            (52.4205938099513, 418.7945418227432),
            (51.20551670188027, 417.7945418227432),
            (50.79454182274317, 417.20545817725684),
            (50.20545817725683, 416.7945418227432),
            (48.79460034736661, 415.2151356326945),
            (48.21513563269447, 415.00967745543767),
            (47.99032254456237, 415.6454068980835),
        ]
    )
    source = gpd.GeoDataFrame(
        {"uid": [110], "class_name": [2]}, geometry=[geometry], index=[110]
    )
    actual = filter_nuclei(source, pixel_size=0.25)
    assert_geodataframe_equal(actual, source)
    assert actual.geometry.iloc[0].wkb == geometry.wkb
