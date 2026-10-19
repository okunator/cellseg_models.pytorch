"""Check fixed unpooling values, derivatives and checkpoint buffers."""

import pytest
import torch

from cellseg_models_pytorch.modules.base_modules import Up
from cellseg_models_pytorch.modules.upsample import FixedUnpool

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("scale_factor", [1, 2, 3])
def test_values_and_input_gradients_match_block_replication(
    device, dtype, autocast, scale_factor
):
    # Transpose a rectangular input to also exercise noncontiguous features.
    values = (torch.arange(24, device=device, dtype=dtype) / 8 - 1).reshape(2, 2, 2, 3)
    values = values.transpose(2, 3).requires_grad_()
    layer = FixedUnpool(scale_factor).to(device)
    saved = layer.unpool_mat.clone()
    amp_dtype = (
        torch.bfloat16 if device == "cpu" or dtype == torch.bfloat16 else torch.float16
    )
    with torch.autocast(device, dtype=amp_dtype, enabled=autocast):
        actual = layer(values)
    # CPU autocast lowers contractions; CUDA preserves matching operand dtypes.
    expected_dtype = (
        amp_dtype if autocast and device == "cpu" and dtype != torch.float64 else dtype
    )
    assert actual.dtype == expected_dtype
    assert actual.device == values.device
    expected = (
        values.detach()
        .repeat_interleave(scale_factor, 2)
        .repeat_interleave(scale_factor, 3)
    )
    torch.testing.assert_close(actual, expected.to(expected_dtype), rtol=0, atol=0)
    weights = ((torch.arange(actual.numel(), device=device) % 7 - 3) / 4).reshape_as(
        actual
    )
    actual.backward(weights.to(actual.dtype))
    # Each input receives all output gradients from its replicated block.
    expected_gradient = (
        weights.double()
        .reshape(
            *values.shape[:2],
            values.shape[2],
            scale_factor,
            values.shape[3],
            scale_factor,
        )
        .sum((3, 5))
    )
    torch.testing.assert_close(values.grad, expected_gradient.to(dtype), rtol=0, atol=0)
    assert layer.unpool_mat.dtype == torch.float32
    torch.testing.assert_close(layer.unpool_mat, saved, rtol=0, atol=0)
    assert set(layer.state_dict()) == {"unpool_mat"}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("autocast", [False, True])
def test_float64_gradients_match_finite_differences(device, autocast):
    layer = FixedUnpool(2).to(device)
    values = torch.tensor(
        [[[[0.1, -0.3], [0.7, 1.2]]]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
        assert torch.autograd.gradcheck(layer, (values,))


@pytest.mark.parametrize("device", DEVICES)
def test_checkpoint_matrix_values_are_used_without_mutating_the_buffer(device):
    layer = FixedUnpool(2).to(device)
    matrix = torch.tensor([[0.25, -0.5], [0.75, 2.0]], device=device)
    layer.load_state_dict({"unpool_mat": matrix}, strict=True)
    values = torch.tensor(
        [[[[1.5, -2.0]]]], dtype=torch.bfloat16, device=device, requires_grad=True
    )
    actual = layer(values)
    expected = torch.tensor(
        [[[[0.375, -0.75, -0.5, 1.0], [1.125, 3.0, -1.5, -4.0]]]],
        dtype=torch.bfloat16,
        device=device,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    torch.testing.assert_close(
        values.grad, torch.full_like(values, 2.5), rtol=0, atol=0
    )
    torch.testing.assert_close(layer.state_dict()["unpool_mat"], matrix, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_default_upsampling_wrapper_accepts_autocast_convolution_features(device):
    convolution = torch.nn.Conv2d(2, 3, 1, bias=False).to(device)
    with torch.no_grad():
        convolution.weight.fill_(0.25)
    values = torch.ones(1, 2, 2, 3, device=device, requires_grad=True)
    amp_dtype = torch.bfloat16 if device == "cpu" else torch.float16
    with torch.autocast(device, dtype=amp_dtype):
        features = convolution(values)
        actual = Up("fixed-unpool", 2).to(device)(features)
    assert actual.dtype == amp_dtype
    torch.testing.assert_close(actual, torch.full_like(actual, 0.5), rtol=0, atol=0)
    actual.float().sum().backward()
    torch.testing.assert_close(
        values.grad, torch.full_like(values, 3.0), rtol=0, atol=0
    )
    torch.testing.assert_close(
        convolution.weight.grad,
        torch.full_like(convolution.weight, 24.0),
        rtol=0,
        atol=0,
    )
