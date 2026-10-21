import numpy as np
import pytest

from cellseg_models_pytorch.wsi.tiles import get_overlap_area, get_overlap_index


@pytest.mark.parametrize("kind", ["list", "int32", "int64", "float32", "float64"])
def test_overlap_helpers_preserve_input_order_values_and_dtypes(kind: str) -> None:
    bounds = [
        (18, 20, 2, 6),
        (10, 20, 8, 6),
        (6, 18, 8, 6),
        (8, 18, 12, 10),
        (10, 26, 8, 2),
        (30, 40, 2, 2),
    ]
    coordinates = bounds if kind == "list" else np.array(bounds, dtype=kind)
    original = np.array(coordinates, copy=True)

    indices = get_overlap_index((10, 20, 8, 6), coordinates)
    areas = get_overlap_area((10, 20, 8, 6), coordinates)

    np.testing.assert_array_equal(indices, [1, 2, 3])
    np.testing.assert_array_equal(areas, [0, 48, 16, 48, 0, 0])
    assert indices.dtype == np.dtype(np.intp)
    assert areas.dtype == original.dtype
    np.testing.assert_array_equal(coordinates, original)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_overlap_helpers_preserve_fractional_array_coordinates(dtype: str) -> None:
    coordinates = np.array([(9.5, 19.5, 1, 1)], dtype=dtype)
    original = coordinates.copy()

    np.testing.assert_array_equal(get_overlap_index((10, 20, 8, 6), coordinates), [0])
    areas = get_overlap_area((10, 20, 8, 6), coordinates)
    np.testing.assert_array_equal(areas, [0.25])
    assert areas.dtype == coordinates.dtype
    np.testing.assert_array_equal(coordinates, original)


def test_overlap_helpers_accept_empty_coordinate_arrays() -> None:
    coordinates = np.empty((0, 4), dtype=np.int32)

    indices = get_overlap_index((0, 0, 10, 10), coordinates)
    areas = get_overlap_area((0, 0, 10, 10), coordinates)

    assert indices.shape == areas.shape == (0,)
    assert indices.dtype == np.dtype(np.intp)
    assert areas.dtype == coordinates.dtype
