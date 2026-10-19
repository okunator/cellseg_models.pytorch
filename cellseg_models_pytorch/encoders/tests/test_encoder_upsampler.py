"""Check transformer feature upsampling without downloading encoder weights."""

import pytest
import torch
import torch.nn.functional as F

from cellseg_models_pytorch.encoders.encoder_upsampler import (
    EncoderUpsampler,
    FeatUpsampleBlock,
)

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("upsample_weight", [1, 8])
def test_float16_autocast_preserves_large_features_and_gradients(
    device, input_dtype, upsample_weight
):
    layer = FeatUpsampleBlock(1).to(device)
    with torch.no_grad():
        layer.up.weight.fill_(upsample_weight)
        layer.up.bias.zero_()
        layer.conv_block.weight.fill_(1)
        layer.conv_block.bias.zero_()
    features = torch.full(
        (1, 1, 2, 3), 10000, device=device, dtype=input_dtype, requires_grad=True
    )
    reference_input = features.detach().double().requires_grad_()
    reference_params = [
        p.detach().double().requires_grad_() for p in layer.parameters()
    ]
    expected = F.conv2d(
        F.conv_transpose2d(reference_input, *reference_params[:2], stride=2),
        *reference_params[2:],
        padding=1,
    )
    # Values between convolutions can exceed float16 even when the final model does not.
    assert expected.max() > torch.finfo(torch.float16).max
    with torch.autocast(device, dtype=torch.float16):
        actual = layer(features)
        assert torch.is_autocast_enabled(device)
    assert not torch.is_autocast_enabled(device)
    assert torch.isfinite(actual).all()
    assert actual.dtype == torch.float32 and actual.device == features.device
    torch.testing.assert_close(actual.double(), expected, rtol=0, atol=0)
    (actual.mean() / 100000).backward()
    (expected.mean() / 100000).backward()
    for actual_value, reference_value in zip(
        [features, *layer.parameters()], [reference_input, *reference_params]
    ):
        assert torch.isfinite(actual_value.grad).all()
        torch.testing.assert_close(
            actual_value.grad.double(), reference_value.grad, rtol=0.003, atol=1e-7
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_non_autocast_values_and_gradients_match_original_operations(device, dtype):
    torch.manual_seed(8)
    layer = FeatUpsampleBlock(2, 3).to(device=device, dtype=dtype)
    features = torch.randn(1, 2, 2, 3, device=device, dtype=dtype, requires_grad=True)
    actual = layer(features)
    expected = layer.conv_block(layer.up(features))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameters = (features, *layer.parameters())
    actual_gradients = torch.autograd.grad(actual.square().mean(), parameters)
    expected_gradients = torch.autograd.grad(expected.square().mean(), parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "weight_dtype,amp_dtype",
    [
        (torch.float32, torch.bfloat16),
        (torch.float64, torch.float16),
        (torch.float16, torch.float16),
    ],
)
def test_other_precision_modes_keep_original_operations(
    device, weight_dtype, amp_dtype
):
    torch.manual_seed(10)
    layer = FeatUpsampleBlock(2, 3).to(device=device, dtype=weight_dtype)
    features = torch.randn(1, 2, 2, 3, device=device, dtype=weight_dtype)
    with torch.autocast(device, dtype=amp_dtype):
        actual = layer(features)
        expected = layer.conv_block(layer.up(features))
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("autocast", [False, True])
def test_upsampler_preserves_checkpoint_layout_and_restores_autocast(device, autocast):
    torch.manual_seed(12)
    info = tuple({"num_chs": 1, "reduction": 16} for _ in range(3))
    layer = EncoderUpsampler(info, (1, 1, 1)).to(device)
    # This is the original layer layout used by existing checkpoints.
    expected_keys = {"bottleneck.weight", "bottleneck.bias"}
    for name, count in (("up1", 1), ("up2", 2)):
        for index in range(count):
            for component in ("up", "conv_block"):
                for parameter in ("weight", "bias"):
                    expected_keys.add(
                        f"up_blocks.{name}.{index}.{component}.{parameter}"
                    )
    assert set(layer.state_dict()) == expected_keys
    restored = EncoderUpsampler(info, (1, 1, 1)).to(device)
    restored.load_state_dict(layer.state_dict(), strict=True)
    features = tuple(torch.ones(1, 1, 2, 2, device=device) for _ in info)
    with torch.autocast(device, dtype=torch.float16, enabled=autocast):
        actual = layer(features)
        reloaded = restored(features)
        assert torch.is_autocast_enabled(device) == autocast
    for index, (value, expected) in enumerate(zip(actual, reloaded)):
        assert value.shape == (1, 1, 2 ** (index + 1), 2 ** (index + 1))
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert not torch.is_autocast_enabled(device)


@pytest.mark.parametrize("device", DEVICES)
def test_float16_autocast_restored_after_invalid_input(device):
    layer = FeatUpsampleBlock(2).to(device)
    with torch.autocast(device, dtype=torch.float16):
        with pytest.raises(RuntimeError):
            layer(torch.ones(1, 3, 2, 2, device=device))
        assert torch.is_autocast_enabled(device)
    assert not torch.is_autocast_enabled(device)
