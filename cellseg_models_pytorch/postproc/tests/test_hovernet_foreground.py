"""Keep nucleus separation independent of the positive cell-type numbers."""

import numpy as np
import pytest
import torch

from cellseg_models_pytorch.decoders.multitask_decoder import SoftInstanceOutput
from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.postproc import post_proc_hovernet
from cellseg_models_pytorch.transforms import gen_hv_maps


@pytest.fixture
def touching_nuclei() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:96, :96]
    instances = np.zeros((96, 96), dtype=np.int32)
    instances[(x - 36) ** 2 + (y - 48) ** 2 < 18**2] = 1
    instances[((x - 61) ** 2 + (y - 48) ** 2 < 18**2) & (instances == 0)] = 2
    return instances, gen_hv_maps(instances)


@pytest.mark.parametrize("enhance", [False, True])
@pytest.mark.parametrize("classes", [(1, 1), (2, 2), (2, 6), (37, 4)])
def test_hovernet_separation_ignores_positive_class_numbers(
    touching_nuclei: tuple[np.ndarray, np.ndarray],
    classes: tuple[int, int],
    enhance: bool,
) -> None:
    instances, hv = touching_nuclei
    labels = np.array([0, *classes], dtype=np.int32)[instances]
    original_labels, original_hv = labels.copy(), hv.copy()
    expected = post_proc_hovernet((instances > 0).astype(np.int32), hv, enhance=enhance)
    actual = post_proc_hovernet(labels, hv, enhance=enhance)

    assert actual.dtype == np.int32
    assert np.unique(actual).tolist() == [0, 1, 2]
    assert actual[48, 36] != actual[48, 61]
    np.testing.assert_array_equal(actual > 0, instances > 0)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(labels, original_labels)
    np.testing.assert_array_equal(hv, original_hv)


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.int32, np.float32])
@pytest.mark.parametrize("empty", [False, True])
def test_hovernet_binary_foreground_inputs(
    touching_nuclei: tuple[np.ndarray, np.ndarray], dtype: type, empty: bool
) -> None:
    instances, hv = touching_nuclei
    foreground = np.zeros_like(instances) if empty else (instances > 0).astype(np.int32)
    expected = post_proc_hovernet(foreground, hv)
    actual = post_proc_hovernet(foreground.astype(dtype), hv)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.int32
    assert np.unique(actual).size == (1 if empty else 3)


@pytest.mark.parametrize("mode", ["serial", "parallel", "async"])
@pytest.mark.parametrize("binary_head", [False, True])
def test_hovernet_postprocessor_preserves_classes_and_separates_objects(
    touching_nuclei: tuple[np.ndarray, np.ndarray], mode: str, binary_head: bool
) -> None:
    instances, hv = touching_nuclei
    count = 1 if mode == "serial" else 2
    classes = np.stack(
        [
            np.array([0, 2, 6], dtype=np.int32)[instances],
            np.array([0, 6, 2], dtype=np.int32)[instances],
        ]
    )[:count]
    types = torch.from_numpy(classes.copy())
    maps = torch.from_numpy(np.stack([hv] * count).astype(np.float32))
    binary = torch.from_numpy((classes > 0).astype(np.float32)).unsqueeze(1)
    prediction = {
        "nuc": SoftInstanceOutput(
            type_map=types, aux_map=maps, binary_map=binary if binary_head else None
        ),
        "cyto": None,
        "tissue": None,
    }
    processor = PostProcessor("hovernet")
    if mode == "serial":
        output = processor.postproc_serial(prediction)
    elif mode == "parallel":
        output = processor.postproc_parallel(
            prediction, n_jobs=2, start_method="threading"
        )
    else:
        output = processor.postproc_parallel_async(
            prediction, n_jobs=2, start_method="threading"
        )
    expected = post_proc_hovernet((instances > 0).astype(np.int32), hv)
    assert np.unique(expected).tolist() == [0, 1, 2]
    assert output["cyto"] is output["tissue"] is None
    assert len(output["nuc"]) == count
    for index, (objects, labels) in enumerate(output["nuc"]):
        np.testing.assert_array_equal(objects, expected)
        assert objects.dtype == labels.dtype == np.int32
        assert np.unique(labels).tolist() == [0, 2, 6]
        assert not labels[objects == 0].any()
        for object_id in (1, 2):
            mask = objects == object_id
            voted_class = np.bincount(classes[index][mask]).argmax()
            np.testing.assert_array_equal(labels[mask], voted_class)
    np.testing.assert_array_equal(types.numpy(), classes)
    np.testing.assert_array_equal(
        maps.numpy(), np.stack([hv] * count).astype(np.float32)
    )
