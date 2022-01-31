import numpy as np
import pytest

from cellseg_models_pytorch.utils import (
    binarize,
    bounding_box,
    center_crop,
    fill_holes_semantic,
    fix_duplicates,
    get_inst_centroid,
    get_inst_types,
    get_type_instances,
    label_semantic,
    one_hot,
    remap_label,
    remove_1px_boundary,
    remove_debris_binary,
    remove_debris_instance,
    remove_debris_semantic,
    remove_small_objects,
    soft_type_flatten,
    type_map_flatten,
)


@pytest.fixture(scope="package")
def inst_map() -> np.ndarray:
    inst_map = np.array(
        [
            [2, 2, 2, 1, 1, 1, 1, 0, 5, 5],
            [2, 2, 2, 1, 1, 1, 1, 0, 5, 5],
            [2, 2, 0, 0, 1, 1, 1, 0, 5, 5],
            [2, 0, 0, 0, 0, 0, 0, 4, 4, 0],
            [0, 0, 3, 3, 3, 0, 4, 4, 4, 0],
            [0, 3, 3, 3, 3, 0, 4, 4, 4, 0],
            [0, 3, 3, 3, 0, 0, 4, 4, 4, 0],
            [0, 3, 3, 0, 0, 0, 4, 4, 0, 0],
            [0, 0, 0, 0, 0, 0, 9, 9, 0, 0],
            [0, 8, 8, 8, 0, 0, 9, 9, 9, 0],
            [0, 8, 8, 8, 0, 0, 9, 9, 9, 0],
        ],
        dtype=int,
    )
    return inst_map


@pytest.fixture(scope="package")
def type_map() -> np.ndarray:
    type_map = np.array(
        [
            [2, 2, 2, 1, 1, 1, 1, 0, 3, 3],
            [2, 2, 2, 1, 1, 1, 1, 0, 3, 3],
            [2, 2, 0, 0, 1, 1, 1, 0, 3, 3],
            [2, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 3, 3, 3, 0, 1, 1, 1, 0],
            [0, 3, 3, 3, 3, 0, 1, 1, 1, 0],
            [0, 3, 3, 3, 0, 0, 1, 1, 1, 0],
            [0, 3, 3, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 2, 2, 2, 0, 0, 1, 1, 1, 0],
            [0, 2, 2, 2, 0, 0, 1, 1, 1, 0],
        ],
        dtype=int,
    )
    return type_map


@pytest.fixture(scope="package")
def sem_map() -> np.ndarray:
    sem_map = np.array(
        [
            [2, 2, 2, 1, 1, 1, 1, 3, 3, 3],
            [2, 2, 2, 1, 1, 1, 1, 3, 3, 3],
            [2, 1, 2, 2, 1, 1, 1, 3, 3, 3],
            [2, 1, 0, 2, 3, 3, 3, 1, 1, 1],
            [2, 0, 2, 2, 3, 3, 1, 1, 1, 0],
            [2, 2, 2, 2, 3, 3, 1, 2, 1, 1],
            [2, 2, 2, 2, 3, 3, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 3, 1, 1, 0, 0],
            [3, 3, 3, 0, 3, 3, 1, 1, 0, 0],
            [3, 2, 2, 2, 3, 3, 1, 1, 1, 1],
            [3, 2, 3, 2, 3, 3, 1, 1, 1, 1],
        ],
        dtype=int,
    )
    return sem_map


def test_remove_small_objects(inst_map):
    expected = np.array(
        [
            [2, 2, 2, 1, 1, 1, 1, 0, 0, 0],
            [2, 2, 2, 1, 1, 1, 1, 0, 0, 0],
            [2, 2, 0, 0, 1, 1, 1, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 4, 4, 0],
            [0, 0, 3, 3, 3, 0, 4, 4, 4, 0],
            [0, 3, 3, 3, 3, 0, 4, 4, 4, 0],
            [0, 3, 3, 3, 0, 0, 4, 4, 4, 0],
            [0, 3, 3, 0, 0, 0, 4, 4, 0, 0],
            [0, 0, 0, 0, 0, 0, 9, 9, 0, 0],
            [0, 0, 0, 0, 0, 0, 9, 9, 9, 0],
            [0, 0, 0, 0, 0, 0, 9, 9, 9, 0],
        ],
        dtype=int,
    )
    observed = remove_small_objects(inst_map, min_size=7)
    np.testing.assert_array_equal(observed, expected)


def test_binarize(inst_map):
    b = binarize(inst_map)
    assert np.max(b) == 1


def test_fix_duplicates():
    inst_map = np.array(
        [
            [2, 2, 2, 1, 1, 1, 1, 0, 0, 0],
            [2, 2, 2, 1, 1, 1, 1, 0, 0, 0],
            [2, 2, 0, 0, 1, 1, 1, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 2, 2, 0],
            [0, 0, 3, 3, 3, 0, 2, 2, 2, 0],
            [0, 3, 3, 3, 3, 0, 2, 2, 2, 0],
            [0, 3, 3, 3, 0, 0, 2, 2, 2, 0],
            [0, 3, 3, 0, 0, 0, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 9, 9, 0, 0],
            [0, 0, 0, 0, 0, 0, 9, 9, 9, 0],
            [0, 0, 0, 0, 0, 0, 9, 9, 9, 0],
        ],
        dtype=int,
    )

    expected = np.array(
        [
            [2, 2, 2, 1, 1, 1, 1, 0, 0, 0],
            [2, 2, 2, 1, 1, 1, 1, 0, 0, 0],
            [2, 2, 0, 0, 1, 1, 1, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 11, 11, 0],
            [0, 0, 3, 3, 3, 0, 11, 11, 11, 0],
            [0, 3, 3, 3, 3, 0, 11, 11, 11, 0],
            [0, 3, 3, 3, 0, 0, 11, 11, 11, 0],
            [0, 3, 3, 0, 0, 0, 11, 11, 0, 0],
            [0, 0, 0, 0, 0, 0, 9, 9, 0, 0],
            [0, 0, 0, 0, 0, 0, 9, 9, 9, 0],
            [0, 0, 0, 0, 0, 0, 9, 9, 9, 0],
        ],
        dtype=int,
    )

    observed = fix_duplicates(inst_map)
    np.testing.assert_array_equal(observed, expected)


def test_remove_1px_boundary(inst_map):
    expected = np.array(
        [
            [2, 2, 0, 0, 1, 1, 0, 0, 0, 5],
            [2, 2, 0, 0, 1, 1, 0, 0, 0, 5],
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
            [0, 0, 3, 3, 0, 0, 0, 4, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 4, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
            [0, 0, 8, 0, 0, 0, 0, 9, 0, 0],
        ],
        dtype=int,
    )
    observed = remove_1px_boundary(inst_map)
    np.testing.assert_array_equal(observed, expected)


def test_remap_label(inst_map):
    expected = np.array(
        [
            [2, 2, 2, 1, 1, 1, 1, 0, 5, 5],
            [2, 2, 2, 1, 1, 1, 1, 0, 5, 5],
            [2, 2, 0, 0, 1, 1, 1, 0, 5, 5],
            [2, 0, 0, 0, 0, 0, 0, 4, 4, 0],
            [0, 0, 3, 3, 3, 0, 4, 4, 4, 0],
            [0, 3, 3, 3, 3, 0, 4, 4, 4, 0],
            [0, 3, 3, 3, 0, 0, 4, 4, 4, 0],
            [0, 3, 3, 0, 0, 0, 4, 4, 0, 0],
            [0, 0, 0, 0, 0, 0, 7, 7, 0, 0],
            [0, 6, 6, 6, 0, 0, 7, 7, 7, 0],
            [0, 6, 6, 6, 0, 0, 7, 7, 7, 0],
        ],
        dtype=int,
    )
    observed = remap_label(inst_map)
    np.testing.assert_array_equal(observed, expected)


def test_center_crop(inst_map):
    expected = np.array(
        [
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 4],
            [3, 3, 3, 0, 4, 4],
            [3, 3, 3, 0, 4, 4],
            [3, 3, 0, 0, 4, 4],
            [3, 0, 0, 0, 4, 4],
        ],
        dtype=int,
    )
    observed = center_crop(inst_map, 6, 6)
    np.testing.assert_array_equal(observed, expected)


def test_bounding_box(inst_map):
    expected = [3, 8, 6, 9]
    observed = bounding_box(inst_map == 4)
    np.testing.assert_array_equal(observed, expected)


def test_get_inst_centroid(inst_map):
    centroids = get_inst_centroid(inst_map)
    assert centroids.shape == (7, 2)
    assert centroids.dtype == "float64"


def test_get_inst_types(inst_map, type_map):
    expected = np.array([[1], [2], [3], [1], [3], [2], [1]], dtype=int)
    observed = get_inst_types(inst_map, type_map)
    np.testing.assert_array_equal(observed, expected)


def test_get_type_instances(inst_map, type_map):
    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 5, 5],
            [0, 0, 0, 0, 0, 0, 0, 0, 5, 5],
            [0, 0, 0, 0, 0, 0, 0, 0, 5, 5],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 3, 3, 0, 0, 0, 0, 0],
            [0, 3, 3, 3, 3, 0, 0, 0, 0, 0],
            [0, 3, 3, 3, 0, 0, 0, 0, 0, 0],
            [0, 3, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    observed = get_type_instances(inst_map, type_map, 3)
    np.testing.assert_array_equal(observed, expected)


def test_one_hot(type_map):
    onehot = one_hot(type_map, 3)
    assert onehot.shape == (11, 10, 4)
    assert onehot.dtype == "float64"


def test_type_map_flatten(type_map):
    onehot = one_hot(type_map, 3)
    observed = type_map_flatten(onehot)
    np.testing.assert_array_equal(observed, type_map)


def test_soft_type_flatten(type_map):
    expected = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    onehot = one_hot(type_map, 3)
    observed = soft_type_flatten(onehot)
    np.testing.assert_array_equal(observed, expected)


def test_remove_debris_binary(inst_map):
    expected = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        ],
        dtype=int,
    )
    binary = binarize(inst_map)
    observed = remove_debris_binary(binary)
    np.testing.assert_array_equal(observed, expected)


def test_remove_debris_instance(inst_map):
    expected = np.array(
        [
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 4, 4, 0],
            [0, 0, 3, 3, 3, 0, 4, 4, 4, 0],
            [0, 3, 3, 3, 3, 0, 4, 4, 4, 0],
            [0, 3, 3, 3, 0, 0, 4, 4, 4, 0],
            [0, 3, 3, 0, 0, 0, 4, 4, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    observed = remove_debris_instance(inst_map)
    np.testing.assert_array_equal(observed, expected)


def test_remove_debris_semantic(sem_map):
    expected = np.array(
        [
            [2, 2, 2, 1, 1, 1, 1, 3, 3, 3],
            [2, 2, 2, 1, 1, 1, 1, 3, 3, 3],
            [2, 2, 2, 2, 1, 1, 1, 3, 3, 3],
            [2, 2, 0, 2, 3, 3, 3, 1, 1, 1],
            [2, 0, 2, 2, 3, 3, 1, 1, 1, 0],
            [2, 2, 2, 2, 3, 3, 1, 1, 1, 1],
            [2, 2, 2, 2, 3, 3, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 3, 1, 1, 0, 0],
            [3, 3, 3, 0, 3, 3, 1, 1, 0, 0],
            [3, 2, 2, 2, 3, 3, 1, 1, 1, 1],
            [3, 2, 3, 2, 3, 3, 1, 1, 1, 1],
        ],
        dtype=int,
    )
    observed = remove_debris_semantic(sem_map, min_size=4)
    np.testing.assert_array_equal(observed, expected)


def test_fill_holes_semantic(sem_map):
    expected = np.array(
        [
            [2, 2, 2, 1, 1, 1, 1, 3, 3, 3],
            [2, 2, 2, 1, 1, 1, 1, 3, 3, 3],
            [2, 1, 2, 2, 1, 1, 1, 3, 3, 3],
            [2, 1, 2, 2, 3, 3, 3, 1, 1, 1],
            [2, 2, 2, 2, 3, 3, 1, 1, 1, 1],
            [2, 2, 2, 2, 3, 3, 1, 2, 1, 1],
            [2, 2, 2, 2, 3, 3, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 3, 1, 1, 0, 0],
            [3, 3, 3, 3, 3, 3, 1, 1, 0, 0],
            [3, 2, 2, 2, 3, 3, 1, 1, 1, 1],
            [3, 2, 3, 2, 3, 3, 1, 1, 1, 1],
        ],
        dtype=int,
    )
    observed = fill_holes_semantic(sem_map, min_size=3)
    np.testing.assert_array_equal(observed, expected)


def test_label_semantic(sem_map):
    expected = np.array(
        [
            [7, 7, 7, 1, 1, 1, 1, 4, 4, 4],
            [7, 7, 7, 1, 1, 1, 1, 4, 4, 4],
            [7, 2, 7, 7, 1, 1, 1, 4, 4, 4],
            [7, 2, 0, 7, 5, 5, 5, 3, 3, 3],
            [7, 0, 7, 7, 5, 5, 3, 3, 3, 0],
            [7, 7, 7, 7, 5, 5, 3, 8, 3, 3],
            [7, 7, 7, 7, 5, 5, 3, 3, 3, 3],
            [5, 5, 5, 5, 5, 5, 3, 3, 0, 0],
            [5, 5, 5, 0, 5, 5, 3, 3, 0, 0],
            [5, 9, 9, 9, 5, 5, 3, 3, 3, 3],
            [5, 9, 6, 9, 5, 5, 3, 3, 3, 3],
        ],
        dtype=int,
    )
    observed = label_semantic(sem_map)
    np.testing.assert_array_equal(observed, expected)
