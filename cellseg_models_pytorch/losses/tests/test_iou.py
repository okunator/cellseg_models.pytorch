"""Check spatial IoU loss against hand-calculated values and derivatives."""

import pytest
import torch

from cellseg_models_pytorch.losses import IoULoss, Loss


@pytest.mark.parametrize("correct,expected", [(True, 0.0), (False, 1.0)])
def test_perfect_and_disjoint_predictions(correct, expected):
    target = torch.tensor([[[0, 1]]])
    predicted = target if correct else 1 - target
    logits = torch.nn.functional.one_hot(predicted, 2).permute(0, 3, 1, 2).float() * 100

    assert IoULoss()(logits, target).item() == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("class_weights", [None, [2.0, 3.0]])
def test_spatial_iou_and_class_weights(class_weights):
    probabilities = torch.tensor(
        [[[[0.8, 0.6, 0.9]], [[0.2, 0.4, 0.1]]]], dtype=torch.float64
    )
    target = torch.tensor([[[0, 0, 1]]])
    weights = None if class_weights is None else torch.tensor(class_weights)
    # Class intersections are 1.4 and 0.1; unions are 2.9 and 1.6.
    expected = (15 / 29 + 15 / 16) / 2
    if class_weights is not None:
        expected = (2 * 15 / 29 + 3 * 15 / 16) / 2

    loss = IoULoss(class_weights=weights)(probabilities.log(), target)
    assert loss.item() == pytest.approx(expected)


def test_mean_includes_only_target_present_image_class_pairs():
    probabilities = torch.tensor(
        [
            [[[0.8, 0.8]], [[0.1, 0.1]], [[0.1, 0.1]]],
            [[[0.6, 0.2]], [[0.3, 0.7]], [[0.1, 0.1]]],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor([[[0, 0]], [[0, 1]]])
    # Three present pairs have losses 1/5, 1/2 and 6/13; class 2 is always absent.
    expected = (1 / 5 + 1 / 2 + 6 / 13) / 3
    assert IoULoss()(probabilities.log(), target).item() == pytest.approx(expected)


def test_absent_class_probabilities_still_contribute_to_softmax_gradients():
    logits = torch.tensor([[[[0.7]], [[0.2]], [[0.1]]]], dtype=torch.float64).log()
    logits.requires_grad_()

    loss = IoULoss()(logits, torch.tensor([[[0]]]))
    assert loss.item() == pytest.approx(0.3)
    loss.backward()
    expected = torch.tensor([[[[-0.21]], [[0.14]], [[0.07]]]], dtype=torch.float64)
    torch.testing.assert_close(logits.grad, expected)


@pytest.mark.parametrize("regularize", [False, True])
def test_fractional_mask_excludes_pixels_classes_and_gradients(regularize):
    logits = (
        torch.tensor([[[[0.8, 0.6, 0.9]], [[0.2, 0.4, 0.1]]]], dtype=torch.float64)
        .log()
        .requires_grad_()
    )
    target = torch.tensor([[[0, 0, 1]]])
    mask = torch.tensor([[[1.0, 0.5, 0.0]]], dtype=torch.float64)
    expected = 4 / 15
    if regularize:
        penalty = (logits.detach().square().mean(1) * mask).sum() / mask.sum()
        expected += 0.005 * penalty.item()

    loss = IoULoss(apply_mask=True, apply_sd=regularize)(logits, target, mask=mask)
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[..., :2]) > 0
    assert torch.count_nonzero(logits.grad[..., 2:]) == 0


def test_mask_is_ignored_when_masking_is_disabled():
    logits = torch.tensor([[[[0.8, 0.6, 0.9]], [[0.2, 0.4, 0.1]]]]).log()
    target = torch.tensor([[[0, 0, 1]]])
    loss = IoULoss()(logits, target, mask=torch.zeros_like(target))
    assert loss.item() == pytest.approx((15 / 29 + 15 / 16) / 2)


def test_edge_weights_apply_once_to_both_intersection_and_union():
    probabilities = torch.tensor([[[[0.8, 0.6]], [[0.2, 0.4]]]])
    target = torch.tensor([[[0, 1]]])
    # Weights 1 and 2 give intersection 0.8 and union 2.2 for both classes.
    loss = IoULoss(edge_weight=2)(
        probabilities.log(), target, target_weight=torch.tensor([[[0.0, 1.0]]])
    )
    assert loss.item() == pytest.approx(7 / 11)


def test_loss_factory_combines_mask_edge_and_class_weights():
    logits = torch.tensor([[[[0.8, 0.6, 0.9]], [[0.2, 0.4, 0.1]]]]).log()
    target = torch.tensor([[[0, 1, 1]]])
    # Mask times 4**edge gives weights 1, 2 and 0; class weights scale the losses.
    loss = Loss(
        "iou", apply_mask=True, edge_weight=4, class_weights=torch.tensor([2.0, 3.0])
    )(
        logits,
        target,
        mask=torch.tensor([[[1.0, 0.5, 0.0]]]),
        target_weight=torch.tensor([[[0.0, 1.0, 5.0]]]),
    )
    assert loss.item() == pytest.approx(35 / 22)


def test_label_smoothing_uses_soft_target_intersection_and_union():
    probabilities = torch.tensor([[[[0.8, 0.4]], [[0.2, 0.6]]]])
    target = torch.tensor([[[0, 1]]])
    # Smoothed targets are (0.9, 0.1) and (0.1, 0.9).
    loss = IoULoss(apply_ls=True)(probabilities.log(), target, label_smoothing=0.2)
    assert loss.item() == pytest.approx((17 / 36 + 17 / 31) / 2)


@pytest.mark.parametrize("smoothing", [{}, {"apply_ls": True}, {"apply_svls": True}])
def test_weighted_regularized_gradients_match_finite_differences(smoothing):
    logits = torch.linspace(-1, 1, 18, dtype=torch.float64).reshape(1, 2, 3, 3)
    logits.requires_grad_()
    target = torch.tensor([[[0, 0, 1], [0, 1, 1], [1, 1, 0]]])
    mask = torch.tensor([[[1.0, 0.5, 0], [1, 1, 1], [0, 1, 1]]], dtype=torch.float64)
    edges = torch.tensor([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]])
    criterion = IoULoss(
        apply_mask=True,
        apply_sd=True,
        edge_weight=2,
        class_weights=torch.tensor([0.7, 1.3], dtype=torch.float64),
        **smoothing,
    )
    assert torch.autograd.gradcheck(
        lambda values: criterion(values, target, mask=mask, target_weight=edges),
        (logits,),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_large_low_precision_spatial_sums_do_not_overflow(dtype):
    logits = torch.zeros((1, 2, 256, 256), dtype=dtype, requires_grad=True)
    target = torch.zeros((1, 256, 256), dtype=torch.long)
    loss = IoULoss()(logits, target)
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[:, 0].sum() < 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fully_masked_input_has_zero_loss_and_gradients_including_regularization(dtype):
    logits = torch.full((1, 2, 2, 2), 2.0, dtype=dtype, requires_grad=True)
    target = torch.zeros((1, 2, 2), dtype=torch.long)
    loss = IoULoss(apply_mask=True, apply_sd=True)(
        logits, target, mask=torch.zeros_like(target)
    )
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_zero_edge_weights_exclude_all_pixels_without_nan():
    logits = torch.zeros((1, 2, 2, 2), requires_grad=True)
    target = torch.zeros((1, 2, 2), dtype=torch.long)
    loss = IoULoss(edge_weight=0)(logits, target, target_weight=torch.ones_like(target))
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


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
        IoULoss(**options)(torch.zeros(1, 2, 2, 2), **forward_args)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_weighted_masked_iou_on_cuda_matches_cpu_value_and_gradients(dtype):
    cpu_logits = torch.tensor(
        [[[[2.0, -1.0, 3.0]], [[-2.0, 1.0, -3.0]]]], dtype=dtype, requires_grad=True
    )
    cuda_logits = cpu_logits.detach().cuda().requires_grad_()
    target = torch.tensor([[[0, 1, 0]]])
    mask = torch.tensor([[[1.0, 0.5, 0.0]]])
    edges = torch.tensor([[[0.0, 1.0, 0.0]]])
    criterion = IoULoss(
        apply_mask=True,
        apply_sd=True,
        edge_weight=2,
        class_weights=torch.tensor([1.0, 2.0]),
    )

    expected = criterion(cpu_logits, target, mask=mask, target_weight=edges)
    expected.backward()
    actual = criterion.cuda()(
        cuda_logits, target.cuda(), mask=mask.cuda(), target_weight=edges.cuda()
    )
    actual.backward()
    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(cuda_logits.grad.cpu(), cpu_logits.grad)
