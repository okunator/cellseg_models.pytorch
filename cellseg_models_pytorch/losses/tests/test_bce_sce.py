"""Numerical and gradient regressions for binary and symmetric cross-entropy."""

import math

import pytest
import torch
import torch.nn.functional as F

from cellseg_models_pytorch.losses import BCELoss, Loss
from cellseg_models_pytorch.losses.criterions.sce import SCELoss
from cellseg_models_pytorch.losses.tests.test_ce_focal import _spatial_targets


def test_bce_matches_hand_calculated_values_and_gradients():
    logits = torch.full(
        (1, 1, 1, 2), math.log(3), dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor([[[0, 1]]])
    actual = BCELoss()(logits, target)
    assert actual.dtype == torch.float64
    assert actual.item() == pytest.approx((math.log(4) + math.log(4 / 3)) / 2)
    actual.backward()
    torch.testing.assert_close(
        logits.grad, torch.tensor([[[[0.375, -0.125]]]], dtype=torch.float64)
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("correct", [True, False])
def test_bce_extreme_logits_retain_loss_and_gradient(dtype, correct):
    logits = torch.tensor([[[[1000.0, -1000.0]]]], dtype=dtype, requires_grad=True)
    target = torch.tensor([[[1.0, 0.0]]]) if correct else torch.tensor([[[0.0, 1.0]]])
    actual = BCELoss()(logits, target)
    assert actual.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert actual.item() == pytest.approx(0 if correct else 1000)
    actual.backward()
    expected = (
        torch.zeros_like(logits)
        if correct
        else torch.tensor([[[[0.5, -0.5]]]], dtype=dtype)
    )
    torch.testing.assert_close(logits.grad, expected)


@pytest.mark.parametrize("regularize", [False, True])
def test_bce_half_limit_regularization_uses_unclipped_float32_logits(regularize):
    magnitude = torch.finfo(torch.float16).max
    logits = torch.tensor(
        [[[[magnitude, -magnitude]]]], dtype=torch.float16, requires_grad=True
    )
    actual = BCELoss(apply_sd=regularize)(logits, torch.tensor([[[0.0, 1.0]]]))
    expected = magnitude + (0.005 * magnitude**2 if regularize else 0)
    assert actual.item() == pytest.approx(expected)
    actual.backward()
    gradient = 0.5 + (0.005 * magnitude if regularize else 0)
    torch.testing.assert_close(
        logits.grad, torch.tensor([[[[gradient, -gradient]]]], dtype=torch.float16)
    )


@pytest.mark.parametrize("regularize", [False, True])
@pytest.mark.parametrize("shared_target", [False, True])
@pytest.mark.parametrize("fully_masked", [False, True])
def test_bce_channel_weights_masks_and_regularization_have_correct_gradients(
    regularize, shared_target, fully_masked
):
    logits = (
        torch.linspace(-3, 3, 18, dtype=torch.float64)
        .reshape(2, 3, 1, 3)
        .requires_grad_()
    )
    target = torch.tensor([[[0.0, 1.0, 0.0]], [[1.0, 0.0, 1.0]]], dtype=torch.float64)
    dense_target = target[:, None].expand_as(logits).clone()
    if not shared_target:
        dense_target[:, 1] = 0.25
        dense_target[:, 2] = 0.75
    mask = torch.tensor([[[1.0, 0.5, 0.0]], [[0.0, 1.0, 0.5]]], dtype=torch.float64)
    if fully_masked:
        mask.zero_()
    edges = torch.tensor([[[0.0, 1.0, 2.0]], [[2.0, 0.0, 1.0]]], dtype=torch.float64)
    weights = torch.tensor([1.5, 0.0, 2.0], dtype=torch.float64).view(1, 3, 1, 1)
    scale = weights * 2 ** edges[:, None]
    reference = (F.softplus(logits) - dense_target * logits) * mask[:, None]
    if regularize:
        reference = reference + 0.005 * logits.square()
    expected = (reference * scale).mean()
    expected_gradient = (logits.sigmoid() - dense_target) * mask[:, None]
    if regularize:
        expected_gradient = expected_gradient + 0.01 * logits
    expected_gradient = expected_gradient * scale / logits.numel()

    actual = BCELoss(
        apply_mask=True,
        apply_sd=regularize,
        class_weights=weights.flatten(),
        edge_weight=2,
    )(logits, target if shared_target else dense_target, mask=mask, target_weight=edges)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(logits.grad, expected_gradient)
    assert torch.count_nonzero(logits.grad[:, 1]) == 0
    if fully_masked and not regularize:
        assert actual.item() == 0
        assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize("channels", [1, 3])
def test_bce_label_smoothing_is_binary_and_symmetric(channels):
    logits = torch.full(
        (1, channels, 1, 2), math.log(3), dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor([[[0, 1]]])
    actual = BCELoss(apply_ls=True)(logits, target, label_smoothing=0.2)
    expected = math.log(4) - 0.5 * math.log(3)
    assert actual.item() == pytest.approx(expected)
    actual.backward()
    gradient = (
        torch.tensor([[[[0.75 - 0.1, 0.75 - 0.9]]]], dtype=torch.float64)
        / logits.numel()
    )
    torch.testing.assert_close(logits.grad, gradient.expand_as(logits))


@pytest.mark.parametrize("alpha,beta", [(0.5, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)])
def test_sce_hard_labels_match_hand_calculated_values_and_gradients(alpha, beta):
    probabilities = torch.tensor([[[[0.7]], [[0.2]], [[0.1]]]], dtype=torch.float64)
    logits = probabilities.log().requires_grad_()
    actual = SCELoss(alpha=alpha, beta=beta)(
        logits, torch.zeros(1, 1, 1, dtype=torch.long)
    )
    reverse_scale = -math.log(1e-4)
    assert actual.item() == pytest.approx(
        -alpha * math.log(0.7) + beta * 0.3 * reverse_scale
    )
    actual.backward()
    expected_gradient = (alpha + beta * reverse_scale * 0.7) * (
        probabilities - torch.tensor([[[[1]], [[0]], [[0]]]])
    )
    torch.testing.assert_close(logits.grad, expected_gradient)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("correct", [True, False])
def test_sce_extreme_logits_have_finite_values_and_gradients(dtype, correct):
    logits = torch.tensor([[[[1000.0]], [[-1000.0]]]], dtype=dtype, requires_grad=True)
    target = torch.tensor([[[0 if correct else 1]]])
    actual = SCELoss(alpha=1.5, beta=0.7)(logits, target)
    assert actual.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert actual.item() == pytest.approx(0 if correct else 3000 - 0.7 * math.log(1e-4))
    actual.backward()
    expected_gradient = (
        torch.zeros_like(logits)
        if correct
        else torch.tensor([[[[1.5]], [[-1.5]]]], dtype=dtype)
    )
    torch.testing.assert_close(logits.grad, expected_gradient)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("correct", [True, False])
def test_sce_zero_target_terms_do_not_produce_nan_at_dtype_limit(
    device, dtype, correct
):
    magnitude = torch.finfo(dtype).max
    logits = torch.tensor(
        [[[[magnitude]], [[-magnitude]]]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    target = torch.tensor([[[0 if correct else 1]]], device=device)
    actual = SCELoss()(logits, target)
    if correct:
        assert actual.item() == 0
    else:
        # The positive-target CE exceeds the computation dtype's range; keep infinity.
        assert torch.isposinf(actual)
    actual.backward()
    expected = torch.zeros_like(logits)
    if not correct:
        expected[:, 0] = 0.5
        expected[:, 1] = -0.5
    torch.testing.assert_close(logits.grad, expected)


@pytest.mark.parametrize("regularize", [False, True])
@pytest.mark.parametrize("fully_masked", [False, True])
def test_sce_mask_and_weights_scale_both_cross_entropies_and_regularization(
    regularize, fully_masked
):
    logits = (
        torch.linspace(-2, 2, 18, dtype=torch.float64)
        .reshape(2, 3, 1, 3)
        .requires_grad_()
    )
    reference = logits.detach().clone().requires_grad_()
    target = torch.tensor([[[0, 2, 1]], [[2, 0, 1]]])
    mask = torch.tensor([[[1.0, 0.5, 0.0]], [[0.0, 1.0, 0.5]]], dtype=torch.float64)
    if fully_masked:
        mask.zero_()
    edges = torch.tensor([[[0.0, 1.0, 2.0]], [[2.0, 0.0, 1.0]]], dtype=torch.float64)
    weights = torch.tensor([2.0, 0.0, 0.5], dtype=torch.float64)
    p_target = reference.softmax(1).gather(1, target[:, None]).squeeze(1)
    pixels = 0.3 * F.cross_entropy(
        reference, target, reduction="none"
    ) - 0.8 * math.log(1e-4) * (1 - p_target)
    pixels = pixels * mask
    if regularize:
        pixels = pixels + 0.005 * reference.square().mean(1)
    expected = (pixels * weights[target] * 2**edges).mean()
    expected.backward()

    actual = SCELoss(
        alpha=0.3,
        beta=0.8,
        apply_mask=True,
        apply_sd=regularize,
        class_weights=weights,
        edge_weight=2,
    )(logits, target, mask=mask, target_weight=edges)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(logits.grad, reference.grad)
    if fully_masked and not regularize:
        assert actual.item() == 0
        assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize("smoothing", [0.0, 1e-6, 0.2])
def test_sce_target_log_floor_does_not_smooth_forward_cross_entropy(smoothing):
    logits = (
        torch.tensor([[[[0.8]], [[0.2]]]], dtype=torch.float64).log().requires_grad_()
    )
    reference = logits.detach().clone().requires_grad_()
    soft_target = torch.tensor(
        [[[[1 - smoothing / 2]], [[smoothing / 2]]]], dtype=torch.float64
    )
    reverse_logs = torch.tensor(
        [math.log(max(1 - smoothing / 2, 1e-4)), math.log(max(smoothing / 2, 1e-4))],
        dtype=torch.float64,
    ).reshape(1, 2, 1, 1)
    expected = (
        0.5 * F.cross_entropy(reference, soft_target)
        - (reference.softmax(1) * reverse_logs).sum()
    )
    expected.backward()
    actual = SCELoss(apply_ls=True)(
        logits, torch.zeros(1, 1, 1, dtype=torch.long), label_smoothing=smoothing
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
    actual.backward()
    torch.testing.assert_close(logits.grad, reference.grad, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("loss_name", ["bce", "sce"])
def test_spatial_and_uniform_smoothing_match_an_independent_stencil(loss_name):
    target = torch.tensor([[[0, 0, 1], [1, 0, 1], [1, 1, 0]]])
    probabilities = _spatial_targets(target, 2)
    logits = (
        torch.linspace(-1, 1, 18, dtype=torch.float64)
        .reshape(1, 2, 3, 3)
        .requires_grad_()
    )
    reference = logits.detach().clone().requires_grad_()
    soft_target = 0.8 * probabilities + 0.1
    if loss_name == "bce":
        target_input = F.one_hot(target, 2).permute(0, 3, 1, 2)
        expected = (F.softplus(reference) - soft_target * reference).mean()
    else:
        target_input = target
        expected = (
            0.5 * F.cross_entropy(reference, soft_target)
            - (reference.softmax(1) * soft_target.log()).sum(1).mean()
        )
    expected.backward()
    actual = Loss(loss_name, apply_ls=True, apply_svls=True)(
        logits,
        target_input,
        label_smoothing=0.2,
        kernel_size=3,
        sigma=1 / math.sqrt(2 * math.log(2)),
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
    actual.backward()
    torch.testing.assert_close(logits.grad, reference.grad, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("name", ["bce", "sce"])
def test_default_losses_pass_gradient_check(name):
    logits = (
        torch.linspace(-2, 2, 12, dtype=torch.float64)
        .reshape(1, 2, 2, 3)
        .requires_grad_()
    )
    target = torch.tensor([[[0, 1, 0], [1, 0, 1]]])
    assert torch.autograd.gradcheck(
        lambda values: Loss(name)(values, target), (logits,)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target": torch.zeros(2, 2)},
        {"mask": torch.ones(2, 2), "apply_mask": True},
        {"class_weights": torch.ones(2)},
        {"edge_weight": 2},
        {"edge_weight": 2, "target_weight": torch.ones(2, 2)},
    ],
)
def test_bce_rejects_invalid_weight_and_target_shapes(kwargs):
    options = dict(kwargs)
    target = options.pop("target", torch.zeros(1, 2, 2))
    mask = options.pop("mask", None)
    target_weight = options.pop("target_weight", None)
    with pytest.raises(ValueError):
        BCELoss(**options)(
            torch.zeros(1, 1, 2, 2), target, mask=mask, target_weight=target_weight
        )


def test_bce_rejects_logits_without_channel_axis():
    with pytest.raises(ValueError, match="BCHW"):
        BCELoss()(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2))


@pytest.mark.cuda
@pytest.mark.parametrize("name", ["bce", "sce"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("regularize", [False, True])
def test_cuda_masked_weighted_half_limit_logits_match_float64(
    name, dtype, autocast, regularize
):
    magnitude = torch.finfo(torch.float16).max
    logits = torch.tensor(
        [[[[magnitude, -magnitude, 0.75]], [[-magnitude, magnitude, -0.25]]]],
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    reference = logits.detach().cpu().double().requires_grad_()
    target = torch.tensor([[[0, 1, 0]]])
    mask = torch.tensor([[[0.0, 1.0, 0.5]]], dtype=torch.float64)
    edges = torch.tensor([[[2.0, 0.0, 1.0]]], dtype=torch.float64)
    weights = torch.tensor([1.5, 0.5], dtype=torch.float64)
    if name == "bce":
        soft_target = target[:, None].expand_as(reference) * 0.8 + 0.1
        pixels = (F.softplus(reference) - soft_target * reference) * mask[:, None]
        if regularize:
            pixels = pixels + 0.005 * reference.square()
        expected = (pixels * weights.view(1, 2, 1, 1) * 2 ** edges[:, None]).mean()
    else:
        soft_target = F.one_hot(target, 2).permute(0, 3, 1, 2).double() * 0.8 + 0.1
        pixels = 0.5 * F.cross_entropy(reference, soft_target, reduction="none") - (
            reference.softmax(1) * soft_target.log()
        ).sum(1)
        pixels = pixels * mask
        if regularize:
            pixels = pixels + 0.005 * reference.square().mean(1)
        expected = (pixels * weights[target] * 2**edges).mean()
    expected.backward()
    criterion = Loss(
        name,
        apply_ls=True,
        apply_mask=True,
        apply_sd=regularize,
        class_weights=weights.cuda().float(),
        edge_weight=2,
    )
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        actual = criterion(
            logits,
            target.cuda(),
            mask=mask.cuda().float(),
            target_weight=edges.cuda().float(),
            label_smoothing=0.2,
        )
    actual.backward()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(logits.grad.cpu(), reference.grad.to(dtype))
