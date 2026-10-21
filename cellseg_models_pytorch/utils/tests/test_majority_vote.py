import numpy as np
import pytest

from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.postproc import POSTPROC_LOOKUP
from cellseg_models_pytorch.utils import majority_vote_sequential


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int32", "int64"])
def test_majority_vote_preserves_values_ties_dtype_and_inputs(dtype: str) -> None:
    instances = np.array(
        [
            [0, 101, 101, 101, 0],
            [0, 101, 101, 101, 0],
            [0, 505, 505, 505, 0],
            [0, 505, 505, 505, 0],
        ],
        dtype=np.int32,
    )
    classes = np.array(
        [[9, 1, 2, 2, 9], [9, 2, 2, 1, 9], [9, 3, 4, 3, 9], [9, 4, 3, 4, 9]],
        dtype=dtype,
    )
    original_instances = instances.copy()
    original_classes = classes.copy()

    actual = majority_vote_sequential(classes, instances)

    np.testing.assert_array_equal(
        actual,
        [[0, 2, 2, 2, 0], [0, 2, 2, 2, 0], [0, 3, 3, 3, 0], [0, 3, 3, 3, 0]],
    )
    assert actual.dtype == classes.dtype
    np.testing.assert_array_equal(classes, original_classes)
    np.testing.assert_array_equal(instances, original_instances)


@pytest.mark.parametrize(
    ("labels", "expected"),
    [([0, 0, 0, 2, 3, 3], 3), ([0, 0, 2, 2], 2), ([0, 0, 0, 0], 0)],
)
def test_majority_vote_preserves_class_zero_fallback(
    labels: list[int], expected: int
) -> None:
    instances = np.array([[0] + [101] * len(labels)], dtype=np.int32)
    classes = np.array([[9] + labels], dtype=np.int32)

    actual = majority_vote_sequential(classes, instances)

    np.testing.assert_array_equal(actual, [[0] + [expected] * len(labels)])


@pytest.mark.parametrize("shape", [(0, 0), (2, 3)])
def test_majority_vote_preserves_empty_and_background_only_maps(
    shape: tuple[int, int],
) -> None:
    instances = np.zeros(shape, dtype=np.int32)
    classes = np.full(shape, 9, dtype=np.uint8)

    actual = majority_vote_sequential(classes, instances)

    np.testing.assert_array_equal(actual, np.zeros(shape, dtype=np.uint8))
    assert actual.dtype == classes.dtype
    np.testing.assert_array_equal(classes, np.full(shape, 9, dtype=np.uint8))


@pytest.mark.parametrize("dtype", ["uint8", "int64"])
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize(
    ("instance_ids", "class_ids", "voted"),
    [
        ([[1, 1], [1, 1]], [[1, 2], [2, 2]], [[2, 2], [2, 2]]),
        (
            [[101, 101, 505, 505], [101, 101, 505, 505]],
            [[1, 2, 3, 4], [2, 2, 4, 4]],
            [[2, 2, 4, 4], [2, 2, 4, 4]],
        ),
    ],
)
def test_majority_vote_covers_all_instances_with_or_without_background(
    instance_ids: list[list[int]],
    class_ids: list[list[int]],
    voted: list[list[int]],
    background: bool,
    dtype: str,
) -> None:
    instances = np.array(instance_ids, dtype=np.int32)
    classes = np.array(class_ids, dtype=dtype)
    expected = np.array(voted, dtype=dtype)
    if background:
        instances = np.pad(instances, 1)
        classes = np.pad(classes, 1, constant_values=9)
        expected = np.pad(expected, 1)
    original_instances = instances.copy()
    original_classes = classes.copy()

    actual = majority_vote_sequential(classes, instances)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == classes.dtype
    np.testing.assert_array_equal(classes, original_classes)
    np.testing.assert_array_equal(instances, original_instances)


def test_postprocessing_votes_on_instance_without_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reconstruct(inst_map: np.ndarray, aux_map: np.ndarray) -> np.ndarray:
        return np.ones(inst_map.shape, dtype=np.int32)

    monkeypatch.setitem(POSTPROC_LOOKUP, "baseline", reconstruct)
    classes = np.array([[1, 2], [2, 2]], dtype=np.int64)

    result = PostProcessor("baseline").postproc_inst(
        np.ones((2, 2), dtype=np.float32),
        np.zeros((2, 2, 2), dtype=np.float32),
        classes,
    )

    assert result is not None
    instances, actual = result
    np.testing.assert_array_equal(instances, np.ones((2, 2), dtype=np.int32))
    np.testing.assert_array_equal(actual, [[2, 2], [2, 2]])
    assert actual.dtype == np.int32
    np.testing.assert_array_equal(classes, [[1, 2], [2, 2]])
