"""Check tensor conversion and batch reconstruction without real checkpoints."""

import numpy as np
import pytest
import torch
from scipy.ndimage import label

from cellseg_models_pytorch.decoders.multitask_decoder import (
    SoftInstanceOutput,
    SoftSemanticOutput,
)
from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.postproc import POSTPROC_LOOKUP

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]


def reconstruct_fixture(inst_map: np.ndarray, aux_map: np.ndarray) -> np.ndarray:
    assert inst_map.dtype in (np.int32, np.float32)
    assert aux_map.dtype == np.float32
    np.testing.assert_array_equal(
        aux_map, (inst_map > 0)[None] * np.array([0.25, 0.5])[:, None, None]
    )
    return label(inst_map > 0)[0]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["f4", "i4"])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_bfloat16_conversion_preserves_values_and_leaves_tensor_untouched(
    device, dtype, requires_grad
):
    values = np.array([[-2.5, -0.0, 0.125], [1.5, 127.5, 65536]], dtype=np.float32)
    tensor = torch.tensor(values, dtype=torch.bfloat16, device=device).T.requires_grad_(
        requires_grad
    )
    before = tensor.detach().clone()
    assert not tensor.is_contiguous()

    actual = PostProcessor("hovernet")._to_ndarray(tensor, dtype)

    assert actual.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(actual, values.T.astype(dtype))
    if dtype == "f4":
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(values.T))
    # Returned arrays must not let CPU reconstruction overwrite the input tensor.
    actual[0, 0] = 99
    torch.testing.assert_close(tensor, before, rtol=0, atol=0)
    assert tensor.device.type == device and tensor.dtype == torch.bfloat16
    assert tensor.requires_grad == requires_grad and tensor.grad is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "tensor_dtype,numpy_dtype",
    [
        (torch.float16, np.float16),
        (torch.float32, np.float32),
        (torch.float64, np.float64),
        (torch.int64, np.int64),
        (torch.bool, np.bool_),
    ],
)
@pytest.mark.parametrize("dtype", ["f4", "i4", "f8"])
def test_existing_conversion_preserves_values_and_copy_semantics(
    device, tensor_dtype, numpy_dtype, dtype
):
    values = np.array(
        [16777217, -16777219, 3, 0]
        if tensor_dtype == torch.int64
        else [1 + 2**-30, -2.5, 0.125, 0],
        dtype=numpy_dtype,
    )
    tensor = torch.tensor(values, device=device, dtype=tensor_dtype)
    tensor.requires_grad_(tensor.is_floating_point())
    before = tensor.detach().clone()

    actual = PostProcessor("hovernet")._to_ndarray(tensor, dtype)

    np.testing.assert_array_equal(actual, values.astype(dtype))
    assert actual.dtype == np.dtype(dtype)
    actual[0] = 0
    torch.testing.assert_close(tensor, before, rtol=0, atol=0)
    assert tensor.dtype == tensor_dtype and tensor.device.type == device
    assert tensor.grad is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("binary_head", [False, True])
@pytest.mark.parametrize("mode", ["serial", "parallel", "parallel_async"])
def test_bfloat16_batch_reconstruction_preserves_instances_and_classes(
    monkeypatch, device, binary_head, mode
):
    instances = np.zeros((2, 6, 8), dtype=np.int32)
    instances[0, 1:3, 1:3] = 1
    instances[0, 3:5, 5:7] = 2
    instances[1, 2:4, 2:5] = 1
    classes = np.where(instances > 0, 2, 0).astype(np.int32)
    classes[instances == 2] = 3
    foreground = torch.tensor(instances > 0, device=device, dtype=torch.bfloat16)
    auxiliary = torch.stack((foreground * 0.25, foreground * 0.5), dim=1)
    binary = foreground[:, None] if binary_head else None
    soft = SoftInstanceOutput(
        type_map=torch.tensor(classes, device=device, dtype=torch.int64),
        aux_map=auxiliary.requires_grad_(),
        binary_map=binary,
    )

    monkeypatch.setitem(POSTPROC_LOOKUP, "baseline", reconstruct_fixture)
    processor = PostProcessor("baseline")
    method = getattr(processor, f"postproc_{mode}")
    kwargs = {} if mode == "serial" else {"n_jobs": 2, "start_method": "threading"}

    actual = method({"nuc": soft, "cyto": soft, "tissue": None}, **kwargs)

    assert actual["tissue"] is None
    for head in ("nuc", "cyto"):
        assert len(actual[head]) == 2
        for index, (actual_instances, actual_classes) in enumerate(actual[head]):
            np.testing.assert_array_equal(actual_instances, instances[index])
            np.testing.assert_array_equal(actual_classes, classes[index])
            assert actual_instances.dtype == actual_classes.dtype == np.int32
    torch.testing.assert_close(
        soft.type_map, torch.tensor(classes, device=device, dtype=torch.int64)
    )
    assert soft.aux_map.dtype == torch.bfloat16 and soft.aux_map.grad is None
    assert soft.aux_map.requires_grad and soft.aux_map.device.type == device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("binary_head", [False, True])
def test_bfloat16_tissue_conversion_preserves_class_labels(device, binary_head):
    classes = np.array([[[0, 1, 2], [3, 2, 0]]], dtype=np.int32)
    tensor = torch.tensor(classes, device=device, dtype=torch.bfloat16)
    soft = SoftSemanticOutput(
        type_map=tensor, binary_map=tensor if binary_head else None
    )

    actual = PostProcessor("hovernet")._prepare_tissue_maps(soft)

    assert len(actual) == 1 and actual[0].dtype == np.int32
    np.testing.assert_array_equal(actual[0], classes[0])
