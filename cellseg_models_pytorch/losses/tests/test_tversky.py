"""Check spatial Tversky values and derivatives against independent calculations."""

import math

import pytest
import torch
import torch.nn.functional as F

from cellseg_models_pytorch.losses import DiceLoss, IoULoss, Loss, TverskyLoss
from cellseg_models_pytorch.losses.tests.test_ce_focal import _spatial_targets


def _reference_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    pixel_weights: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Average weighted class losses using explicit target and non-target regions."""
    probabilities = logits.softmax(1)
    losses = []
    for b in range(logits.shape[0]):
        for c in range(logits.shape[1]):
            selected = target[b] == c
            if pixel_weights[b][selected].sum() == 0:
                continue
            weights = pixel_weights[b]
            predicted = probabilities[b, c]
            tp = (predicted[selected] * weights[selected]).sum()
            fp = (predicted[~selected] * weights[~selected]).sum()
            fn = ((1 - predicted[selected]) * weights[selected]).sum()
            losses.append(class_weights[c] * (1 - tp / (tp + alpha * fp + beta * fn)))
    return torch.stack(losses).mean()


@pytest.mark.parametrize(
    "alpha,beta",
    [
        (0.7, 0.3),
        (0.3, 0.7),
        (1.0, 1.0),
        (0.5, 0.5),
        (0.0, 1.0),
        (1.0, 0.0),
        (2.0, 1.0),
    ],
)
def test_spatial_overlap_and_alpha_beta_match_values_and_analytic_gradients(
    alpha, beta
):
    probabilities = torch.tensor(
        [[[[0.1, 0.2, 0.9]], [[0.9, 0.8, 0.1]]]], dtype=torch.float64
    )
    logits = probabilities.log().requires_grad_()
    target = torch.tensor([[[1, 0, 0]]])
    # Spatial TP, FP, FN are (1.1, 0.1, 0.9) for background and (0.9, 0.9, 0.1) for nuclei.
    tp = torch.tensor([1.1, 0.9], dtype=torch.float64).view(1, 2, 1, 1)
    fp = torch.tensor([0.1, 0.9], dtype=torch.float64).view(1, 2, 1, 1)
    fn = torch.tensor([0.9, 0.1], dtype=torch.float64).view(1, 2, 1, 1)
    denominator = tp + alpha * fp + beta * fn
    expected = (1 - tp / denominator).mean()
    labels = F.one_hot(target, 2).permute(0, 3, 1, 2)
    denominator_derivative = (1 - beta) * labels + alpha * (1 - labels)
    probability_gradient = (
        (tp * denominator_derivative - labels * denominator) / denominator.square() / 2
    )
    expected_gradient = probabilities * (
        probability_gradient
        - (probabilities * probability_gradient).sum(1, keepdim=True)
    )

    actual = TverskyLoss(alpha=alpha, beta=beta)(logits, target)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(logits.grad, expected_gradient)


def test_swapping_alpha_beta_changes_false_positive_false_negative_tradeoff():
    logits = torch.tensor(
        [[[[0.1, 0.2, 0.9]], [[0.9, 0.8, 0.1]]]], dtype=torch.float64
    ).log()
    target = torch.tensor([[[1, 0, 0]]])
    fp_heavy = TverskyLoss(alpha=0.7, beta=0.3)(logits, target)
    fn_heavy = TverskyLoss(alpha=0.3, beta=0.7)(logits, target)
    assert fp_heavy.item() == pytest.approx((1 - 1.1 / 1.44 + 1 - 0.9 / 1.56) / 2)
    assert fn_heavy.item() == pytest.approx((1 - 1.1 / 1.76 + 1 - 0.9 / 1.24) / 2)
    assert fp_heavy > fn_heavy


@pytest.mark.parametrize("correct,expected", [(True, 0.0), (False, 1.0)])
def test_perfect_and_disjoint_predictions(correct, expected):
    target = torch.tensor([[[0, 1]]])
    predicted = target if correct else 1 - target
    logits = F.one_hot(predicted, 2).permute(0, 3, 1, 2).float() * 100
    assert TverskyLoss()(logits, target).item() == pytest.approx(expected, abs=1e-7)


def test_mean_uses_target_present_image_class_pairs_and_absent_classes_still_compete():
    logits = (
        torch.tensor(
            [
                [[[0.8, 0.8]], [[0.1, 0.1]], [[0.1, 0.1]]],
                [[[0.6, 0.2]], [[0.3, 0.7]], [[0.1, 0.1]]],
            ],
            dtype=torch.float64,
        )
        .log()
        .requires_grad_()
    )
    target = torch.tensor([[[0, 0]], [[0, 1]]])
    expected = (3 / 43 + 13 / 43 + 0.3) / 3
    actual = TverskyLoss()(logits, target)
    assert actual.item() == pytest.approx(expected)
    actual.backward()
    assert (logits.grad[:, 2] > 0).all()


@pytest.mark.parametrize("regularize", [False, True])
@pytest.mark.parametrize("weighted_classes", [False, True])
def test_masks_edge_weights_and_class_weights_match_independent_value_and_gradients(
    regularize, weighted_classes
):
    logits = (
        torch.linspace(-2, 2, 18, dtype=torch.float64)
        .reshape(2, 3, 1, 3)
        .requires_grad_()
    )
    reference = logits.detach().clone().requires_grad_()
    target = torch.tensor([[[0, 2, 1]], [[2, 0, 2]]])
    mask = torch.tensor([[[1.0, 0.5, 0.0]], [[0.0, 1.0, 0.5]]], dtype=torch.float64)
    edges = torch.tensor([[[0.0, 1.0, 2.0]], [[2.0, 0.0, 1.0]]], dtype=torch.float64)
    class_weights = (
        torch.tensor([1.5, 2.0, 0.0], dtype=torch.float64)
        if weighted_classes
        else torch.ones(3, dtype=torch.float64)
    )
    expected = _reference_loss(
        reference, target, 0.7, 0.3, mask * 4**edges, class_weights
    )
    if regularize:
        expected = (
            expected + 0.005 * (reference.square().mean(1) * mask).sum() / mask.sum()
        )
    expected.backward()
    actual = Loss(
        "tversky",
        apply_mask=True,
        apply_sd=regularize,
        edge_weight=4,
        class_weights=class_weights if weighted_classes else None,
    )(logits, target, mask=mask, target_weight=edges)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(logits.grad, reference.grad)
    assert torch.count_nonzero(logits.grad.permute(0, 2, 3, 1)[mask == 0]) == 0


@pytest.mark.parametrize(
    "alpha,beta,other", [(0.5, 0.5, DiceLoss), (1.0, 1.0, IoULoss)]
)
def test_special_cases_match_dice_and_iou_values_and_gradients(alpha, beta, other):
    logits = (
        torch.linspace(-1, 1, 12, dtype=torch.float64)
        .reshape(1, 3, 2, 2)
        .requires_grad_()
    )
    reference = logits.detach().clone().requires_grad_()
    target = torch.tensor([[[0, 1], [1, 0]]])
    expected = other()(reference, target)
    actual = TverskyLoss(alpha=alpha, beta=beta)(logits, target)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference.grad)


@pytest.mark.parametrize("regularize", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fully_masked_batch_has_zero_loss_and_gradients(dtype, regularize):
    logits = torch.full((2, 3, 2, 2), 2.0, dtype=dtype, requires_grad=True)
    target = torch.zeros(2, 2, 2, dtype=torch.long)
    actual = TverskyLoss(apply_mask=True, apply_sd=regularize)(
        logits, target, mask=torch.zeros_like(target)
    )
    assert actual.item() == 0
    actual.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_disabled_mask_is_ignored_and_zero_edge_weights_exclude_pixels():
    logits = torch.zeros(1, 2, 1, 2, requires_grad=True)
    target = torch.tensor([[[0, 1]]])
    ignored = TverskyLoss()(logits, target, mask=torch.zeros_like(target))
    assert ignored.item() == pytest.approx(0.5)
    excluded = TverskyLoss(edge_weight=0)(
        logits, target, target_weight=torch.ones_like(target)
    )
    assert excluded.item() == 0
    excluded.backward()
    assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("spatial", [False, True])
def test_smoothing_matches_independent_targets_and_finite_difference_gradients(
    uniform, spatial
):
    target = torch.tensor([[[0, 0, 1], [0, 2, 1], [2, 2, 1]]])
    logits = (
        torch.linspace(-1, 1, 27, dtype=torch.float64)
        .reshape(1, 3, 3, 3)
        .requires_grad_()
    )
    soft_target = (
        _spatial_targets(target, 3)
        if spatial
        else F.one_hot(target, 3).permute(0, 3, 1, 2).double()
    )
    if uniform:
        soft_target = soft_target * 0.8 + 0.2 / 3
    probabilities = logits.softmax(1)
    intersection = (probabilities * soft_target).sum((2, 3))
    fp = (probabilities * (1 - soft_target)).sum((2, 3))
    fn = ((1 - probabilities) * soft_target).sum((2, 3))
    expected = (1 - intersection / (intersection + 0.7 * fp + 0.3 * fn)).mean()
    criterion = TverskyLoss(apply_svls=spatial, apply_ls=uniform)
    options = dict(
        label_smoothing=0.2, kernel_size=3, sigma=1 / math.sqrt(2 * math.log(2))
    )
    actual = criterion(logits, target, **options)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
    assert torch.autograd.gradcheck(
        lambda values: criterion(values, target, **options), (logits,)
    )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_spatial_sums_do_not_overflow(device, dtype):
    # Each overlap sum is 73,728, exceeding float16's maximum of 65,504.
    logits = torch.zeros(1, 2, 384, 384, device=device, dtype=dtype, requires_grad=True)
    target = torch.zeros(1, 384, 384, device=device, dtype=torch.long)
    actual = TverskyLoss()(logits, target)
    assert actual.dtype == torch.float32
    assert actual.item() == pytest.approx(3 / 13)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[:, 0] < 0).all()
    assert (logits.grad[:, 1] > 0).all()


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_nonuniform_bfloat16_logits_use_float32_overlap_and_gradients(device):
    logits = torch.tensor([0.0, 1.0], device=device, dtype=torch.bfloat16)
    logits = logits.view(1, 2, 1, 1).expand(1, 2, 384, 384).clone().requires_grad_()
    target = torch.zeros(1, 384, 384, device=device, dtype=torch.long)
    probability = 1 / (1 + math.e)
    denominator = probability + 0.3 * (1 - probability)
    actual = TverskyLoss()(logits, target)
    assert actual.dtype == torch.float32
    assert actual.item() == pytest.approx(1 - probability / denominator)
    actual.backward()
    gradient = -0.3 * probability * (1 - probability) / denominator**2 / target.numel()
    expected_gradient = torch.empty_like(logits)
    expected_gradient[:, 0] = gradient
    expected_gradient[:, 1] = -gradient
    torch.testing.assert_close(logits.grad, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("regularize", [False, True])
def test_float16_logit_limit_preserves_loss_and_regularization(regularize):
    magnitude = torch.finfo(torch.float16).max
    logits = torch.tensor(
        [[[[magnitude]], [[-magnitude]]]], dtype=torch.float16, requires_grad=True
    )
    target = torch.ones(1, 1, 1, dtype=torch.long)
    actual = TverskyLoss(apply_sd=regularize)(logits, target)
    assert actual.item() == pytest.approx(
        1 + (0.005 * magnitude**2 if regularize else 0)
    )
    actual.backward()
    gradient = 0.005 * magnitude if regularize else 0
    torch.testing.assert_close(
        logits.grad, torch.tensor([[[[gradient]], [[-gradient]]]], dtype=torch.float16)
    )


@pytest.mark.parametrize(
    "options,arguments,match",
    [
        ({"apply_mask": True}, {"mask": torch.ones(1, 1, 1)}, "mask"),
        ({"edge_weight": 2}, {}, "target_weight"),
        ({"edge_weight": 2}, {"target_weight": torch.ones(1, 1, 1)}, "target_weight"),
        ({"class_weights": torch.ones(3)}, {}, "class_weights"),
        ({}, {"target": torch.zeros(1, 2, 3, dtype=torch.long)}, "BCHW"),
    ],
)
def test_invalid_shapes_raise_contextual_errors(options, arguments, match):
    forward_args = {"target": torch.zeros(1, 2, 2, dtype=torch.long), **arguments}
    with pytest.raises(ValueError, match=match):
        TverskyLoss(**options)(torch.zeros(1, 2, 2, 2), **forward_args)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("regularize", [False, True])
@pytest.mark.parametrize("alpha,beta", [(0.7, 0.3), (0.3, 0.7)])
def test_cuda_weighted_masked_loss_and_gradients_match_independent_reference(
    dtype, autocast, regularize, alpha, beta
):
    logits = (
        torch.linspace(-2, 2, 18, device="cuda", dtype=dtype)
        .reshape(2, 3, 1, 3)
        .requires_grad_()
    )
    reference = logits.detach().cpu().double().requires_grad_()
    target = torch.tensor([[[0, 2, 1]], [[2, 0, 2]]])
    mask = torch.tensor([[[1.0, 0.5, 0.0]], [[0.0, 1.0, 0.5]]], dtype=torch.float64)
    edges = torch.tensor([[[0.0, 1.0, 2.0]], [[2.0, 0.0, 1.0]]], dtype=torch.float64)
    class_weights = torch.tensor([1.5, 2.0, 0.0], dtype=torch.float64)
    expected = _reference_loss(
        reference, target, alpha, beta, mask * 4**edges, class_weights
    )
    if regularize:
        expected = (
            expected + 0.005 * (reference.square().mean(1) * mask).sum() / mask.sum()
        )
    expected.backward()
    criterion = TverskyLoss(
        alpha=alpha,
        beta=beta,
        apply_mask=True,
        apply_sd=regularize,
        edge_weight=4,
        class_weights=class_weights.cuda().float(),
    )
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        actual = criterion(
            logits,
            target.cuda(),
            mask=mask.cuda().float(),
            target_weight=edges.cuda().float(),
        )
    actual.backward()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(logits.grad.cpu(), reference.grad.to(dtype))
