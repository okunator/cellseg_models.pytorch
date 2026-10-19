"""Check regression loss reductions, weighting and derivatives independently."""

import math

import pytest
import torch

from cellseg_models_pytorch.losses import MAE, MSE, Loss
from cellseg_models_pytorch.losses.tests.test_ce_focal import _spatial_targets


def _weighted_reference(prediction, target, name, mask, edges, regularize):
    """Return independently calculated weighted losses and prediction derivatives."""
    channels = prediction.shape[1]
    labels = torch.stack([target == c for c in range(channels)], 1).to(prediction.dtype)
    difference = prediction - labels
    if name == "mse":
        values = difference.square().mean(1) * mask
        derivative = 2 * difference * mask.unsqueeze(1) / channels
    else:
        denominator = mask.mean() + 1e-7
        values = difference.abs().mean(1) * mask / denominator + 0.2 * (
            1 - mask
        ) * prediction.abs().mean(1)
        derivative = (
            difference.sign() * mask.unsqueeze(1) / denominator
            + 0.2 * (1 - mask).unsqueeze(1) * prediction.sign()
        ) / channels
    if regularize:
        values = values + 0.005 * prediction.square().mean(1)
        derivative = derivative + 0.01 * prediction / channels
    factors = 2**edges
    if name == "mse":
        factors = factors * torch.where(target == 0, 1.5, 0.5)
    return (values * factors).mean(), derivative * factors.unsqueeze(1) / target.numel()


@pytest.mark.parametrize("loss_class", [MSE, MAE])
def test_default_bchw_values_and_prediction_target_gradients(loss_class):
    prediction = (
        torch.linspace(-1.2, 1.6, 24, dtype=torch.float64)
        .reshape(2, 2, 2, 3)
        .requires_grad_()
    )
    target = (
        torch.linspace(0.7, -0.8, 24, dtype=torch.float64)
        .reshape_as(prediction)
        .requires_grad_()
    )
    difference = prediction.detach() - target.detach()
    expected = (
        difference.square().mean() if loss_class is MSE else difference.abs().mean()
    )
    gradient = (
        2 * difference if loss_class is MSE else difference.sign()
    ) / prediction.numel()
    result = loss_class()(prediction, target)
    torch.testing.assert_close(result, expected)
    result.backward()
    torch.testing.assert_close(prediction.grad, gradient)
    torch.testing.assert_close(target.grad, -gradient)


@pytest.mark.parametrize("name", ["mse", "mae"])
@pytest.mark.parametrize("mask_kind", ["partial", "zero", "full"])
@pytest.mark.parametrize("regularize", [False, True])
def test_masks_weights_and_regularization_match_analytic_values_and_gradients(
    name, mask_kind, regularize
):
    prediction = (
        torch.linspace(-1.2, 1.6, 12, dtype=torch.float64)
        .reshape(2, 2, 1, 3)
        .requires_grad_()
    )
    target = torch.tensor([[[0, 1, 0]], [[1, 0, 1]]])
    mask = torch.tensor([[[1.0, 0.25, 0.0]], [[0.5, 0.0, 1.0]]], dtype=torch.float64)
    if mask_kind == "zero":
        mask.zero_()
    elif mask_kind == "full":
        mask.fill_(1)
    edges = torch.tensor([[[0.0, 1.0, 2.0]], [[2.0, 0.0, 1.0]]], dtype=torch.float64)
    expected, gradient = _weighted_reference(
        prediction.detach(), target, name, mask, edges, regularize
    )
    target_input = (
        target if name == "mse" else torch.stack([target == 0, target == 1], 1).double()
    )
    result = Loss(
        name,
        alpha=0.2,
        apply_mask=True,
        apply_sd=regularize,
        edge_weight=2,
        class_weights=torch.tensor([1.5, 0.5], dtype=torch.float64)
        if name == "mse"
        else None,
    )(prediction, target_input, mask=mask, target_weight=edges)
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)
    result.backward()
    torch.testing.assert_close(prediction.grad, gradient, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("loss_class", [MSE, MAE])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("regularize", [False, True])
def test_opposite_float16_limits_have_finite_loss_and_gradients(
    loss_class, dtype, regularize
):
    prediction = (
        torch.tensor([65504.0, -65504.0] * 4, dtype=dtype)
        .reshape(1, 2, 2, 2)
        .requires_grad_()
    )
    target = -prediction.detach()
    difference = prediction.detach().double() - target.double()
    expected = (
        difference.square().mean() if loss_class is MSE else difference.abs().mean()
    )
    gradient = 2 * difference if loss_class is MSE else difference.sign()
    if regularize:
        expected = expected + 0.005 * prediction.detach().double().square().mean()
        gradient = gradient + 0.01 * prediction.detach().double()
    gradient = (gradient / prediction.numel()).to(dtype)
    result = loss_class(apply_sd=regularize)(prediction, target)
    assert result.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert result.item() == pytest.approx(expected.item(), rel=2e-7)
    result.backward()
    assert torch.isfinite(prediction.grad).all()
    torch.testing.assert_close(prediction.grad, gradient)


@pytest.mark.parametrize("loss_class", [MSE, MAE])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_floating_bhw_targets_are_shared_across_channels_without_losing_fractions(
    loss_class, dtype
):
    prediction = (
        torch.tensor([0.25, 1.5, -0.75, 2.0], dtype=dtype)
        .reshape(1, 2, 1, 2)
        .requires_grad_()
    )
    target = torch.tensor([[[0.5, -0.25]]], dtype=dtype, requires_grad=True)
    difference = prediction.detach().double() - target.detach().double().unsqueeze(1)
    expected = (
        difference.square().mean() if loss_class is MSE else difference.abs().mean()
    )
    gradient = (
        2 * difference if loss_class is MSE else difference.sign()
    ) / prediction.numel()
    result = loss_class()(prediction, target)
    assert result.item() == pytest.approx(expected.item())
    result.backward()
    torch.testing.assert_close(prediction.grad, gradient.to(dtype))
    torch.testing.assert_close(target.grad, -gradient.sum(1).to(dtype))


@pytest.mark.parametrize("name", ["mse", "mae"])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_shared_half_target_gradients_sum_before_narrowing(name, autocast, device):
    values = [-65504.0, 65504.0] if name == "mse" else [-1.0, 1.0]
    prediction = (
        torch.tensor(values, device=device, dtype=torch.float32)
        .reshape(1, 2, 1, 1)
        .requires_grad_()
    )
    target_value = 16376.0 if name == "mse" else 0.0
    target = torch.full(
        (1, 1, 1), target_value, device=device, dtype=torch.float16, requires_grad=True
    )
    options, inputs = {}, {}
    if name == "mae":
        options = {"edge_weight": 2}
        inputs = {"target_weight": torch.full((1, 1, 1), 18.0, device=device)}

    # Channel gradients can overflow half before their representable sum is formed.
    difference = prediction.detach().double() - target.detach().double().unsqueeze(1)
    if name == "mse":
        expected = difference.square().mean()
        prediction_gradient = 2 * difference / prediction.numel()
        target_gradient = 32752.0
    else:
        expected = difference.abs().mean() * 2**18
        prediction_gradient = difference.sign() * 2**18 / prediction.numel()
        target_gradient = 0.0

    autocast_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with torch.autocast(device, dtype=autocast_dtype, enabled=autocast):
        result = Loss(name, **options)(prediction, target, **inputs)
    assert result.dtype == torch.float32
    assert result.item() == pytest.approx(expected.item(), rel=1e-7)
    result.backward()
    torch.testing.assert_close(prediction.grad, prediction_gradient.float())
    torch.testing.assert_close(
        target.grad, torch.full_like(target, target_gradient), atol=0, rtol=0
    )


@pytest.mark.parametrize("loss_class", [MSE, MAE])
@pytest.mark.parametrize(
    "dtype,exponent",
    [(torch.float32, 0.5), (torch.float64, 0.5), (torch.float64, 1.0 + 1e-9)],
)
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_fractional_edge_exponents_preserve_values_and_gradients(
    loss_class, dtype, exponent, device
):
    prediction = torch.full(
        (1, 2, 1, 1), 0.5, device=device, dtype=dtype, requires_grad=True
    )
    target = torch.zeros_like(prediction)
    edges = torch.full((1, 1, 1), exponent, device=device, dtype=dtype)
    weight = 4**exponent
    expected = (0.25 if loss_class is MSE else 0.5) * weight
    result = loss_class(edge_weight=4)(prediction, target, target_weight=edges)
    assert result.dtype == dtype
    tolerance = 1e-12 if dtype == torch.float64 else 1e-6
    assert result.item() == pytest.approx(expected, rel=tolerance, abs=0)
    result.backward()
    torch.testing.assert_close(
        prediction.grad, torch.full_like(prediction, weight / 2), rtol=tolerance, atol=0
    )


@pytest.mark.parametrize("loss_class", [MSE, MAE])
def test_float64_targets_retain_precision_with_float32_predictions(loss_class):
    prediction = torch.ones(1, 1, 1, 1, requires_grad=True)
    target = torch.tensor([[[[1.0 + 1e-9]]]], dtype=torch.float64, requires_grad=True)
    result = loss_class()(prediction, target)
    assert result.dtype == torch.float64
    assert result.item() == pytest.approx(
        1e-18 if loss_class is MSE else 1e-9, rel=1e-6, abs=1e-24
    )
    result.backward()
    assert prediction.grad.item() < 0
    assert target.grad.item() > 0


@pytest.mark.parametrize("name", ["mse", "dice"])
@pytest.mark.parametrize("autocast", [False, True])
def test_combined_smoothing_matches_independent_spatial_stencil_and_gradients(
    name, autocast
):
    target = torch.tensor([[[0, 0, 1], [1, 0, 1], [1, 1, 0]]])
    soft_target = 0.8 * _spatial_targets(target, 2) + 0.1
    prediction = (
        torch.linspace(-1, 1, 18, dtype=torch.float64)
        .reshape(1, 2, 3, 3)
        .requires_grad_()
    )
    reference = prediction.detach().clone().requires_grad_()
    if name == "mse":
        expected = (reference - soft_target).square().mean()
    else:
        probabilities = reference.softmax(1)
        expected = (
            1
            - 2
            * (probabilities * soft_target).sum((2, 3))
            / (probabilities + soft_target).sum((2, 3))
        ).mean()
    expected.backward()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        result = Loss(name, apply_ls=True, apply_svls=True)(
            prediction,
            target,
            kernel_size=3,
            sigma=1 / math.sqrt(2 * math.log(2)),
            label_smoothing=0.2,
        )
    result.backward()
    # The shared spatial smoother constructs its kernel and targets in float32.
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(prediction.grad, reference.grad, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("name", ["mse", "mae"])
@pytest.mark.parametrize(
    "case", ["prediction", "target", "mask", "edge_missing", "edge_shape"]
)
def test_incompatible_shapes_and_missing_weight_maps_are_rejected(name, case):
    prediction = torch.ones(1, 2, 2, 2)
    target = torch.zeros_like(prediction)
    options, inputs = {}, {}
    if case == "prediction":
        prediction = prediction[:, 0]
    elif case == "target":
        target = target[..., :1]
    elif case == "mask":
        options = {"apply_mask": True}
        inputs = {"mask": torch.ones(1, 2, 2, 2)}
    else:
        options = {"edge_weight": 2}
        if case == "edge_shape":
            inputs = {"target_weight": torch.ones(1, 2, 2, 2)}
    with pytest.raises(ValueError, match="BCHW|BHW"):
        Loss(name, **options)(prediction, target, **inputs)


@pytest.mark.parametrize("case", ["weights_shape", "floating_labels", "bchw_target"])
def test_mse_class_weights_require_bhw_integer_labels_and_c_weights(case):
    prediction = torch.ones(1, 2, 2, 2)
    target = torch.zeros(1, 2, 2, dtype=torch.long)
    weights = torch.ones(3 if case == "weights_shape" else 2)
    if case == "floating_labels":
        target = target.float()
    elif case == "bchw_target":
        target = torch.zeros_like(prediction)
    with pytest.raises(ValueError, match="class_weights"):
        MSE(class_weights=weights)(prediction, target)


@pytest.mark.parametrize("name", ["mse", "mae"])
def test_zero_edge_weights_produce_zero_loss_and_gradients(name):
    prediction = torch.full(
        (1, 2, 1, 2), 300.0, dtype=torch.float16, requires_grad=True
    )
    result = Loss(name, apply_sd=True, edge_weight=0)(
        prediction, torch.zeros_like(prediction), target_weight=torch.ones(1, 1, 2)
    )
    assert result.item() == 0
    result.backward()
    assert torch.count_nonzero(prediction.grad) == 0


def test_zero_mse_class_weights_produce_zero_loss_and_gradients():
    prediction = torch.ones(1, 2, 1, 2, requires_grad=True)
    result = MSE(class_weights=torch.zeros(2), apply_sd=True)(
        prediction, torch.zeros(1, 1, 2, dtype=torch.long)
    )
    assert result.item() == 0
    result.backward()
    assert torch.count_nonzero(prediction.grad) == 0


@pytest.mark.parametrize("loss_class", [MSE, MAE])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_half_edge_exponents_compute_weights_before_multiplying_in_float32(
    loss_class, device
):
    prediction = torch.full(
        (1, 2, 2, 2), 0.5, device=device, dtype=torch.float16, requires_grad=True
    )
    result = loss_class(edge_weight=2)(
        prediction,
        torch.zeros_like(prediction),
        target_weight=torch.full((1, 2, 2), 18.0, device=device, dtype=torch.float16),
    )
    assert result.dtype == torch.float32
    expected = (0.25 if loss_class is MSE else 0.5) * 2**18
    assert result.item() == expected
    result.backward()
    torch.testing.assert_close(prediction.grad, torch.full_like(prediction, 2**18 / 8))


@pytest.mark.parametrize("loss_class", [MSE, MAE])
def test_disabled_or_missing_mask_does_not_enable_background_penalty(loss_class):
    prediction = torch.tensor([[[[2.0, -3.0]]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    expected = loss_class()(prediction, target)
    torch.testing.assert_close(
        loss_class(apply_mask=False)(prediction, target, mask=torch.zeros(1, 1, 2)),
        expected,
    )
    torch.testing.assert_close(
        loss_class(apply_mask=True)(prediction, target), expected
    )


def test_fully_masked_mae_without_penalties_has_zero_loss_and_gradients():
    prediction = torch.ones(1, 2, 1, 2, requires_grad=True)
    result = MAE(alpha=0, apply_mask=True)(
        prediction, torch.zeros_like(prediction), mask=torch.zeros(1, 1, 2)
    )
    assert result.item() == 0
    result.backward()
    assert torch.count_nonzero(prediction.grad) == 0


@pytest.mark.cuda
@pytest.mark.parametrize("name", ["mse", "mae"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
def test_cuda_masked_weighted_precision_matches_analytic_values_and_gradients(
    name, dtype, autocast
):
    prediction = (
        torch.tensor([-8192.0, 4096.0, 8192.0, -2048.0] * 3, device="cuda", dtype=dtype)
        .reshape(2, 2, 1, 3)
        .requires_grad_()
    )
    target = torch.tensor([[[0, 1, 0]], [[1, 0, 1]]], device="cuda")
    mask = torch.tensor(
        [[[1.0, 0.25, 0.0]], [[0.5, 0.0, 1.0]]], device="cuda", dtype=dtype
    )
    edges = torch.tensor(
        [[[0.0, 1.0, 2.0]], [[2.0, 0.0, 1.0]]], device="cuda", dtype=dtype
    )
    expected, gradient = _weighted_reference(
        prediction.detach().cpu().double(),
        target.cpu(),
        name,
        mask.cpu().double(),
        edges.cpu().double(),
        True,
    )
    target_input = (
        target
        if name == "mse"
        else torch.stack([target == 0, target == 1], 1).to(dtype)
    )
    criterion = Loss(
        name,
        alpha=0.2,
        apply_mask=True,
        apply_sd=True,
        edge_weight=2,
        class_weights=torch.tensor([1.5, 0.5], device="cuda", dtype=dtype)
        if name == "mse"
        else None,
    )
    with torch.autocast(
        "cuda",
        dtype=dtype if dtype != torch.float32 else torch.float16,
        enabled=autocast,
    ):
        result = criterion(prediction, target_input, mask=mask, target_weight=edges)
    assert result.dtype == torch.float32
    assert result.device == prediction.device
    assert result.item() == pytest.approx(expected.item(), rel=2e-6)
    result.backward()
    assert torch.isfinite(prediction.grad).all()
    torch.testing.assert_close(
        prediction.grad.cpu(), gradient.to(dtype), rtol=0.01, atol=1e-5
    )


@pytest.mark.cuda
@pytest.mark.parametrize("name", ["mse", "dice"])
def test_cuda_combined_smoothing_with_autocast_has_correct_values_and_gradients(name):
    target = torch.tensor([[[0, 0, 1], [1, 0, 1], [1, 1, 0]]])
    soft_target = 0.8 * _spatial_targets(target, 2) + 0.1
    prediction = (
        torch.linspace(-1, 1, 18, device="cuda", dtype=torch.float16)
        .reshape(1, 2, 3, 3)
        .requires_grad_()
    )
    reference = prediction.detach().cpu().double().requires_grad_()
    if name == "mse":
        expected = (reference - soft_target).square().mean()
    else:
        probabilities = reference.softmax(1)
        expected = (
            1
            - 2
            * (probabilities * soft_target).sum((2, 3))
            / (probabilities + soft_target).sum((2, 3))
        ).mean()
    expected.backward()
    with torch.autocast("cuda", dtype=torch.float16):
        result = Loss(name, apply_ls=True, apply_svls=True)(
            prediction,
            target.cuda(),
            kernel_size=3,
            sigma=1 / math.sqrt(2 * math.log(2)),
            label_smoothing=0.2,
        )
    assert result.dtype == torch.float32
    assert result.item() == pytest.approx(expected.item(), rel=1e-6)
    result.backward()
    torch.testing.assert_close(
        prediction.grad.cpu(), reference.grad.half(), rtol=0.002, atol=1e-6
    )
