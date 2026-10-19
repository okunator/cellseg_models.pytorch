"""Check smoothing against independent spatial weights and loss formulas."""

import math

import pytest
import torch
import torch.nn.functional as F

from cellseg_models_pytorch.losses import Loss
from cellseg_models_pytorch.losses.weighted_base_loss import WeightedBaseLoss


def _spatial_reference(target, kernel_size, sigma):
    """Apply a replicate-padded Gaussian stencil, balancing center and neighbors."""
    if kernel_size == 1:
        return target
    center = kernel_size // 2
    distances = {
        (y, x): (y - center) ** 2 + (x - center) ** 2
        for y in range(kernel_size)
        for x in range(kernel_size)
        if (y, x) != (center, center)
    }
    nearest = min(distances.values())
    weights = {
        point: math.exp(-(distance - nearest) / (2 * sigma**2))
        for point, distance in distances.items()
    }
    total = sum(weights.values())
    padded = F.pad(target, [center] * 4, mode="replicate")
    height, width = target.shape[-2:]
    result = 0.5 * target
    for (y, x), weight in weights.items():
        result = (
            result + (0.5 * weight / total) * padded[..., y : y + height, x : x + width]
        )
    return result


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize(
    "kernel_size,sigma", [(3, 0.15), (3, 0.1), (3, 0.05), (5, 3.0), (1, 3.0)]
)
def test_spatial_targets_and_target_gradients_match_stencil(
    device, autocast, kernel_size, sigma
):
    target = torch.zeros(1, 2, 3, 3, device=device)
    target[:, 0] = 1
    target[:, :, 1, 1] = target.new_tensor([0, 1])
    target.requires_grad_()
    reference = target.detach().double().requires_grad_()
    expected = _spatial_reference(reference, kernel_size, sigma)
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with torch.autocast(device, dtype=amp_dtype, enabled=autocast):
        actual = WeightedBaseLoss().apply_svls_to_target(
            target, 2, kernel_size=kernel_size, sigma=sigma
        )
    assert actual.dtype == torch.float32
    assert actual.device == target.device
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual.sum(1), torch.ones_like(actual[:, 0]))
    if kernel_size > 1:
        torch.testing.assert_close(actual[0, :, 1, 1], target.new_tensor([0.5, 0.5]))
    # Unequal output weights expose the smoothing operator's backward distribution.
    weights = torch.linspace(-1, 2, target.numel(), device=device).reshape_as(target)
    (gradient,) = torch.autograd.grad((actual * weights).sum(), target)
    (expected_gradient,) = torch.autograd.grad(
        (expected * weights.double()).sum(), reference
    )
    torch.testing.assert_close(
        gradient.double(), expected_gradient, rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(target, reference.detach().float())


def _loss_reference(name, values, soft_target):
    """Compute losses directly from soft targets, without the library helpers."""
    if name == "bce":
        return F.binary_cross_entropy_with_logits(values, soft_target)
    if name == "mse":
        return (values - soft_target).square().mean()
    log_probabilities = values.log_softmax(1)
    probabilities = log_probabilities.exp()
    if name == "ce":
        return -(soft_target * log_probabilities).sum(1).mean()
    if name == "focal":
        return (
            -(0.5 * (1 - probabilities).square() * soft_target * log_probabilities)
            .sum(1)
            .mean()
        )
    if name == "sce":
        forward = -(soft_target * log_probabilities).sum(1)
        reverse = -(probabilities * soft_target.clamp_min(1e-4).log()).sum(1)
        return (0.5 * forward + reverse).mean()
    intersection = (probabilities * soft_target).sum((2, 3))
    if name == "dice":
        return (1 - 2 * intersection / (probabilities + soft_target).sum((2, 3))).mean()
    if name == "iou":
        union = (probabilities + soft_target).sum((2, 3)) - intersection
        return (1 - intersection / union).mean()
    assert name == "tversky"
    extra = (probabilities * (1 - soft_target)).sum((2, 3))
    missed = ((1 - probabilities) * soft_target).sum((2, 3))
    return (1 - intersection / (intersection + 0.7 * extra + 0.3 * missed)).mean()


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("sigma", [0.15, 3.0])
@pytest.mark.parametrize(
    "name", ["ce", "bce", "focal", "sce", "mse", "dice", "iou", "tversky"]
)
def test_combined_smoothing_loss_values_and_gradients_match_reference(
    device, autocast, dtype, sigma, name
):
    labels = torch.tensor([[[0, 0, 1], [0, 2, 1], [2, 2, 1]]], device=device)
    target = F.one_hot(labels, 3).permute(0, 3, 1, 2)
    soft_target = 0.8 * _spatial_reference(target.double(), 3, sigma)
    soft_target = soft_target + (0.1 if name == "bce" else 0.2 / 3)
    values = torch.linspace(-1.5, 1.5, 27, device=device, dtype=dtype).reshape(
        1, 3, 3, 3
    )
    values.requires_grad_()
    reference = values.detach().double().requires_grad_()
    expected = _loss_reference(name, reference, soft_target)
    target_input = target.to(dtype) if name == "bce" else labels
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with torch.autocast(device, dtype=amp_dtype, enabled=autocast):
        actual = Loss(name, apply_ls=True, apply_svls=True)(
            values, target_input, kernel_size=3, sigma=sigma, label_smoothing=0.2
        )
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert actual.dtype == expected_dtype
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-7)
    (gradient,) = torch.autograd.grad(actual, values)
    (expected_gradient,) = torch.autograd.grad(expected, reference)
    assert torch.isfinite(gradient).all()
    # Half gradients round once when accumulated into the original input tensor.
    tolerance = (
        0.01 if dtype == torch.bfloat16 else (0.002 if dtype == torch.float16 else 1e-5)
    )
    torch.testing.assert_close(
        gradient.double(), expected_gradient, rtol=tolerance, atol=1e-7
    )
