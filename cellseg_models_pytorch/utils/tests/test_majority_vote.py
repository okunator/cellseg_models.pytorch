import numpy as np
import pytest

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
