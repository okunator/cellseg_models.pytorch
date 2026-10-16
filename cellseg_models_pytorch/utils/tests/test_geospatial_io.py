import geopandas as gpd
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import Polygon, box

from cellseg_models_pytorch.utils import FileHandler
from cellseg_models_pytorch.utils.vectorize import gaussian_smooth, inst2gdf


def test_vectorization_keeps_smoothed_narrow_contours_valid():
    instances = (np.random.default_rng(14).random((8, 8)) > 0.4).astype(np.int32)
    actual = inst2gdf(instances, min_size=0, smooth_func=gaussian_smooth)
    assert not actual.empty
    assert actual.is_valid.all()
    assert (actual.area > 0).all()
    assert set(actual.uid) == {1}
    assert set(actual.class_name) == {1}


@pytest.mark.parametrize("suffix", ["parquet", "feather", "geojson"])
def test_geometry_export_round_trip(tmp_path, suffix):
    expected = gpd.GeoDataFrame(
        {
            "uid": [101, 505],
            "class_name": ["tumour", "immune"],
            "score": [0.75, np.nan],
        },
        geometry=[
            Polygon(
                [(0, 0), (8, 0), (8, 8), (0, 8)],
                holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
            ),
            box(10, 1, 14, 5),
        ],
        crs="EPSG:4326",
    )
    path = tmp_path / f"instances.{suffix}"
    FileHandler.gdf_to_file(expected, path)
    readers = {
        "parquet": gpd.read_parquet,
        "feather": gpd.read_feather,
        "geojson": gpd.read_file,
    }
    actual = readers[suffix](path)
    assert set(actual.columns) == set(expected.columns)
    actual = actual[expected.columns]
    # GeoJSON carries numeric values, not NumPy integer widths.
    assert_geodataframe_equal(
        actual, expected, normalize=True, check_dtype=suffix != "geojson"
    )
    assert actual["uid"].dtype.kind in "iu"
    assert [g.normalize().wkb for g in actual.geometry] == [
        g.normalize().wkb for g in expected.geometry
    ]
    assert actual.geometry.iloc[0].area == 60
    assert len(actual.geometry.iloc[0].interiors) == 1


def test_vectorization_preserves_labels_and_pixel_coordinates():
    instances = np.zeros((20, 24), dtype=np.int32)
    instances[2:8, 3:9] = 101
    instances[10:16, 12:18] = 505
    types = np.zeros_like(instances)
    types[instances == 101] = 1
    types[instances == 505] = 2
    actual = inst2gdf(
        instances,
        types,
        xoff=10,
        yoff=20,
        class_dict={1: "tumour", 2: "immune"},
        min_size=0,
        smooth_func=None,
    ).sort_values("uid")
    assert actual["uid"].tolist() == [101, 505]
    assert actual["class_name"].tolist() == ["tumour", "immune"]
    np.testing.assert_array_equal(actual.geometry.area.to_numpy(), [36, 36])
    np.testing.assert_array_equal(
        actual.geometry.bounds.to_numpy(), [[13, 22, 19, 28], [22, 30, 28, 36]]
    )


def test_alpha_shape_preserves_instance_outline():
    from libpysal.cg import alpha_shape_auto

    points = np.array([[0, 0], [4, 0], [8, 0], [8, 4], [8, 8], [4, 8], [0, 8], [0, 4]])
    actual = alpha_shape_auto(points, step=2)
    assert actual.is_valid
    assert actual.equals(box(0, 0, 8, 8))


@pytest.mark.parametrize("class_id", [0, 2])
@pytest.mark.parametrize("background", [False, True])
def test_vectorization_keeps_foreground_even_without_background_class(
    class_id, background
):
    instances = np.ones((12, 12), dtype=np.int32)
    types = np.full_like(instances, class_id)
    if background:
        instances[:2] = 0
    actual = inst2gdf(instances, types, min_size=0, smooth_func=None)
    assert actual.uid.to_list() == [1]
    assert actual.class_name.to_list() == [class_id]
    assert actual.geometry.area.sum() == (120 if background else 144)


@pytest.mark.parametrize("empty", [False, True])
def test_vectorization_records_border_contact_before_smoothing(empty):
    instances = np.zeros((100, 100), dtype=np.int32)
    if not empty:
        instances[35:65, 85:] = 1
        instances[20:30, 20:30] = 2

    actual = inst2gdf(
        instances, xoff=50, yoff=100, min_size=0, smooth_func=gaussian_smooth
    ).sort_values("uid")
    assert actual.is_partial.dtype == bool
    assert actual.is_partial.tolist() == ([] if empty else [True, False])
    if not empty:
        # Smoothing removes geometric border contact but must not erase its origin.
        assert actual.geometry.iloc[0].bounds[2] < 149
