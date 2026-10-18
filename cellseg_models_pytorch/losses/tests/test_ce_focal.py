"""Check logit losses against independent values and derivatives."""

import math

import pytest
import torch
import torch.nn.functional as F

from cellseg_models_pytorch.losses import CELoss, FocalLoss, Loss


@pytest.fixture(params=[(CELoss, 1.0, 0.0), (FocalLoss, 0.5, 2.0)], ids=["ce", "focal"])
def loss_config(request):
    return request.param


def _spatial_targets(target: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Return CPU float64 BCHW probabilities from BHW class IDs and a 3x3 stencil."""
    batch, height, width = target.shape
    smoothed = torch.zeros(batch, n_classes, height, width, dtype=torch.float64)
    # With Gaussian neighbor/center ratio 1/2, SVLS has these exact weights / 24.
    weights = ((1, 2, 1), (2, 12, 2), (1, 2, 1))
    for b in range(batch):
        for y in range(height):
            for x in range(width):
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        yy = min(max(y + dy, 0), height - 1)
                        xx = min(max(x + dx, 0), width - 1)
                        label = int(target[b, yy, xx])
                        smoothed[b, label, y, x] += weights[dy + 1][dx + 1] / 24
    return smoothed


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("correct", [True, False])
def test_extreme_logits_have_correct_values_and_gradients(loss_config, dtype, correct):
    loss_class, alpha, _ = loss_config
    logits = torch.tensor([[[[1000.0]], [[-1000.0]]]], dtype=dtype, requires_grad=True)
    target = torch.tensor([[[0 if correct else 1]]])

    loss = loss_class()(logits, target)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert loss.dtype == expected_dtype
    assert loss.item() == pytest.approx(0.0 if correct else alpha * 2000)
    loss.backward()
    expected_gradient = torch.zeros_like(logits)
    if not correct:
        expected_gradient[:, 0] = alpha
        expected_gradient[:, 1] = -alpha
    torch.testing.assert_close(logits.grad, expected_gradient)


def test_confident_wrong_ce_matches_pytorch_without_capping_loss_or_gradients():
    logits = torch.tensor([[[[40.0]], [[-40.0]]]], requires_grad=True)
    target = torch.tensor([[[1]]])
    loss = CELoss()(logits, target)
    assert loss.item() == pytest.approx(80.0)
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[[[1.0]], [[-1.0]]]]))


@pytest.mark.parametrize("regularize", [False, True])
def test_float16_limit_does_not_overflow_loss_or_regularization(
    loss_config, regularize
):
    loss_class, alpha, _ = loss_config
    magnitude = torch.finfo(torch.float16).max
    logits = torch.tensor(
        [[[[magnitude]], [[-magnitude]]]], dtype=torch.float16, requires_grad=True
    )
    loss = loss_class(apply_sd=regularize)(logits, torch.tensor([[[1]]]))
    expected = alpha * 2 * magnitude
    if regularize:
        expected += 0.005 * magnitude**2
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    gradient = alpha + (0.005 * magnitude if regularize else 0)
    expected_gradient = torch.tensor(
        [[[[gradient]], [[-gradient]]]], dtype=torch.float16
    )
    torch.testing.assert_close(logits.grad, expected_gradient)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 2.0])
def test_focal_value_and_gradient_match_hand_calculation(gamma):
    alpha = 0.7
    logits = (
        torch.tensor([[[[0.8]], [[0.2]]]], dtype=torch.float64).log().requires_grad_()
    )
    loss = FocalLoss(alpha=alpha, gamma=gamma)(logits, torch.tensor([[[1]]]))
    assert loss.item() == pytest.approx(-alpha * 0.8**gamma * math.log(0.2))
    loss.backward()
    # Derivative with respect to the target logit, using its probability p=0.2.
    target_gradient = alpha * 0.8**gamma * (gamma * 0.2 * math.log(0.2) - 0.8)
    expected = torch.tensor(
        [[[[-target_gradient]], [[target_gradient]]]], dtype=torch.float64
    )
    torch.testing.assert_close(logits.grad, expected)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_fractional_gamma_has_finite_gradient_at_saturated_probability(dtype):
    logits = torch.tensor([[[[1000.0]], [[-1000.0]]]], dtype=dtype, requires_grad=True)
    loss = FocalLoss(gamma=0.5)(logits, torch.tensor([[[0]]]))
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize("regularize", [False, True])
def test_weight_mask_and_regularization_order_is_preserved(loss_config, regularize):
    loss_class, alpha, gamma = loss_config
    logits = torch.tensor(
        [[[[1.5, -0.7, 2.0]], [[-0.3, 0.6, -1.0]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    reference_logits = logits.detach().clone().requires_grad_()
    target = torch.tensor([[[0, 1, 0]]])
    mask = torch.tensor([[[1.0, 0.5, 0.0]]], dtype=torch.float64)
    edges = torch.tensor([[[0.0, 1.0, 2.0]]], dtype=torch.float64)
    class_weights = torch.tensor([2.0, 3.0], dtype=torch.float64)

    nll = F.cross_entropy(reference_logits, target, reduction="none")
    pixel_losses = alpha * (1 - (-nll).exp()) ** gamma * nll
    pixel_losses = pixel_losses * mask
    if regularize:
        pixel_losses = pixel_losses + 0.005 * reference_logits.square().mean(1)
    expected = (pixel_losses * class_weights[target] * 2**edges).mean()

    loss = loss_class(
        apply_mask=True, apply_sd=regularize, class_weights=class_weights, edge_weight=2
    )(logits, target, mask=mask, target_weight=edges)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)
    assert bool(torch.count_nonzero(logits.grad[..., 2:])) == regularize


def test_label_smoothing_retains_per_class_focal_weights(loss_config):
    loss_class, alpha, gamma = loss_config
    logits = torch.tensor([[[[0.8, 0.3]], [[0.2, 0.7]]]], dtype=torch.float64).log()
    target = torch.tensor([[[0, 1]]])
    first = -alpha * (
        0.9 * 0.2**gamma * math.log(0.8) + 0.1 * 0.8**gamma * math.log(0.2)
    )
    second = -alpha * (
        0.1 * 0.7**gamma * math.log(0.3) + 0.9 * 0.3**gamma * math.log(0.7)
    )

    loss = loss_class(apply_ls=True)(logits, target, label_smoothing=0.2)
    assert loss.item() == pytest.approx((first + second) / 2)


def test_smoothed_float16_extreme_logits_have_correct_gradient(loss_config):
    loss_class, alpha, _ = loss_config
    logits = torch.tensor(
        [[[[1000.0]], [[-1000.0]]]], dtype=torch.float16, requires_grad=True
    )
    loss = loss_class(apply_ls=True)(logits, torch.tensor([[[0]]]), label_smoothing=0.2)
    assert loss.item() == pytest.approx(alpha * 200, rel=1e-6)
    loss.backward()
    expected = torch.tensor([[[[alpha * 0.1]], [[-alpha * 0.1]]]], dtype=torch.float16)
    torch.testing.assert_close(logits.grad, expected)


@pytest.mark.parametrize("smoothing", ["apply_ls", "apply_svls"])
def test_smoothed_gradients_match_finite_differences(loss_config, smoothing):
    loss_class, _, _ = loss_config
    logits = (
        torch.linspace(-1, 1, 18, dtype=torch.float64)
        .reshape(1, 2, 3, 3)
        .requires_grad_()
    )
    target = torch.tensor([[[0, 0, 1], [0, 1, 1], [1, 1, 0]]])
    criterion = loss_class(**{smoothing: True})
    assert torch.autograd.gradcheck(lambda values: criterion(values, target), (logits,))


@pytest.mark.parametrize("label_smoothing", [False, True])
def test_spatial_smoothing_matches_independent_values_and_gradients(
    loss_config, label_smoothing
):
    loss_class, alpha, gamma = loss_config
    target = torch.tensor([[[0, 0, 1], [0, 2, 1], [2, 2, 1]]])
    soft_target = _spatial_targets(target, 3)
    torch.testing.assert_close(
        soft_target[0, :, 0, 0], torch.tensor([23, 0, 1]) / 24, check_dtype=False
    )
    torch.testing.assert_close(
        soft_target[0, :, 1, 1], torch.tensor([5, 4, 15]) / 24, check_dtype=False
    )
    if label_smoothing:
        soft_target = 0.8 * soft_target + 0.2 / 3

    logits = torch.tensor(
        [
            [
                [[1.5, -0.5, 0.75], [0.25, 2, -1], [1, -0.25, 0.5]],
                [[-0.5, 1, -0.25], [0.5, -1, 1.25], [0, 1.5, 0.75]],
                [[0.25, 0.5, 1.25], [-1, 0.5, 0.25], [-0.5, 0.25, 1]],
            ]
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    reference = logits.detach().clone().requires_grad_()
    probabilities = reference.softmax(1)
    expected = (
        (-alpha * soft_target * (1 - probabilities) ** gamma * probabilities.log())
        .sum(1)
        .mean()
    )
    expected.backward()

    actual = loss_class(apply_svls=True, apply_ls=label_smoothing)(
        logits,
        target,
        kernel_size=3,
        sigma=1 / math.sqrt(2 * math.log(2)),
        label_smoothing=0.2,
    )
    actual.backward()
    # The smoothing helper constructs its kernel/targets in float32.
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(logits.grad, reference.grad, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("name,expected", [("ce", 2000.0), ("focal", 1000.0)])
def test_loss_factory_uses_stable_losses(name, expected):
    logits = torch.tensor(
        [[[[1000.0]], [[-1000.0]]]], dtype=torch.float16, requires_grad=True
    )
    loss = Loss(name)(logits, torch.tensor([[[1]]]))
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("regularize", [False, True])
def test_cuda_extreme_values_and_gradients_match_float64_reference(
    loss_config, dtype, autocast, regularize
):
    loss_class, alpha, gamma = loss_config
    logits = torch.tensor(
        [[[[1000.0, -1000.0, 1000.0, 0.75]], [[-1000.0, 1000.0, -1000.0, -0.25]]]],
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    target = torch.tensor([[[1, 1, 0, 1]]], device="cuda")
    reference = logits.detach().cpu().double().requires_grad_()
    nll = F.cross_entropy(reference, target.cpu(), reduction="none")
    pixel_losses = alpha * (1 - (-nll).exp()) ** gamma * nll
    if regularize:
        pixel_losses = pixel_losses + 0.005 * reference.square().mean(1)
    expected = pixel_losses.mean()
    expected.backward()
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        actual = loss_class(apply_sd=regularize)(logits, target)
    actual.backward()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu().double(), expected)
    torch.testing.assert_close(logits.grad.cpu(), reference.grad.to(dtype))


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("regularize", [False, True])
@pytest.mark.parametrize("label_smoothing", [False, True])
def test_cuda_smoothed_weighted_loss_matches_independent_reference(
    loss_config, dtype, autocast, regularize, label_smoothing
):
    loss_class, alpha, gamma = loss_config
    magnitude = torch.finfo(torch.float16).max
    logits = torch.tensor(
        [
            [
                [
                    [magnitude, -magnitude, 0.75],
                    [0, 2, -magnitude],
                    [magnitude, 1, -0.5],
                ],
                [
                    [-magnitude, magnitude, -0.25],
                    [magnitude, -3, 1],
                    [0, -magnitude, 0.5],
                ],
                [[0, 0, 1.25], [-magnitude, 1, magnitude], [-magnitude, -0.5, 0]],
            ]
        ],
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    target = torch.tensor([[[0, 0, 1], [0, 2, 1], [2, 2, 1]]])
    mask = torch.tensor([[[0, 1, 0.5], [1, 0, 1], [0.5, 1, 1]]], dtype=torch.float64)
    edges = torch.tensor([[[2, 0, 1], [1, 2, 0], [0, 1, 2]]], dtype=torch.float64)
    class_weights = torch.tensor([1.25, 0.5, 2.0], dtype=torch.float64)
    soft_target = _spatial_targets(target, 3)
    if label_smoothing:
        soft_target = 0.8 * soft_target + 0.2 / 3

    reference = logits.detach().cpu().double().requires_grad_()
    log_probabilities = reference.log_softmax(1)
    pixel_losses = (
        -alpha
        * soft_target
        * (1 - log_probabilities.exp()) ** gamma
        * log_probabilities
    ).sum(1)
    pixel_losses = pixel_losses * mask
    if regularize:
        pixel_losses = pixel_losses + 0.005 * reference.square().mean(1)
    expected = (pixel_losses * class_weights[target] * 2**edges).mean()
    expected.backward()

    criterion = loss_class(
        apply_svls=True,
        apply_ls=label_smoothing,
        apply_mask=True,
        apply_sd=regularize,
        class_weights=class_weights.cuda().float(),
        edge_weight=2,
    )
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        actual = criterion(
            logits,
            target.cuda(),
            mask=mask.cuda().float(),
            target_weight=edges.cuda().float(),
            kernel_size=3,
            sigma=1 / math.sqrt(2 * math.log(2)),
            label_smoothing=0.2,
        )
    actual.backward()
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual)
    assert torch.isfinite(logits.grad).all()
    # Autocast also rounds the target convolution to float16; other arithmetic is float32.
    tolerance = torch.finfo(torch.float16).eps if autocast else 1e-6
    torch.testing.assert_close(
        actual.cpu().double(), expected, rtol=tolerance, atol=1e-6
    )
    torch.testing.assert_close(logits.grad.cpu(), reference.grad.to(dtype))
