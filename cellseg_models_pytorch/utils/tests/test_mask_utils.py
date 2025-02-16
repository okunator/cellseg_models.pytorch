import numpy as np
import pytest
import logging

from cellseg_models_pytorch.utils import (
    FileHandler,
    binarize,
    bounding_box,
    center_crop,
    draw_stuff_contours,
    draw_thing_contours,
    fill_holes_semantic,
    fix_duplicates,
    get_inst_centroid,
    get_inst_types,
    get_type_instances,
    label_semantic,
    majority_vote_sequential,
    med_filt_sequential,
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


@pytest.mark.parametrize("func", [draw_stuff_contours, draw_thing_contours])
@pytest.mark.parametrize("fill_contours", [True, False])
@pytest.mark.parametrize(
    "classes", [None, {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6}]
)
@pytest.mark.parametrize(
    "colors",
    [
        None,
        {
            "a": (1.0, 2.0, 3.0),
            "b": (1.0, 2.0, 3.0),
            "c": (1.0, 2.0, 3.0),
            "d": (1.0, 2.0, 3.0),
            "e": (1.0, 2.0, 3.0),
            "f": (1.0, 2.0, 3.0),
            "g": (1.0, 2.0, 3.0),
        },
    ],
)
def test_conoturs(img_patch_dir, mask_patch_dir, func, fill_contours, classes, colors):
    im_path = sorted(img_patch_dir.glob("*"))[0]
    mask_path = sorted(mask_patch_dir.glob("*"))[0]

    img = FileHandler.read_img(im_path)
    masks = FileHandler.read_mat(mask_path)
    cont = func(
        masks["inst_map"],
        img,
        masks["type_map"],
        fill_contours=fill_contours,
        classes=classes,
        colors=colors,
    )

    assert cont.shape == img.shape
    assert cont.dtype == img.dtype


def test_med_filter():
    im = np.random.randint(0, 255, size=(3, 32, 32), dtype="uint8")
    imf = med_filt_sequential(im)

    assert imf.shape == im.shape
    assert imf.dtype == im.dtype


def test_majority_vote(inst_map, type_map):
    tmap = majority_vote_sequential(type_map, inst_map)

    assert tmap.shape == type_map.shape
    assert tmap.dtype == type_map.dtype
