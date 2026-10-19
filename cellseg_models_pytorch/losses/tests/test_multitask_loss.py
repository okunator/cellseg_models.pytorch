import pytest
import torch

from cellseg_models_pytorch.losses import (
    MAE,
    MSE,
    SSIM,
    CELoss,
    IoULoss,
    JointLoss,
    MultiTaskLoss,
    TverskyLoss,
)
from cellseg_models_pytorch.losses.tests.test_losses import _get_dummy_pair


@torch.no_grad()
@pytest.mark.parametrize("n_classes", [1, 3])
@pytest.mark.parametrize(
    "losses",
    [
        {"inst": JointLoss([TverskyLoss(), IoULoss()])},
        {"inst": CELoss(), "types": SSIM()},
    ],
)
def test_multitask_loss(n_classes, losses):
    yhats = {}
    targets = {}
    for i, br in enumerate(losses.keys(), 1):
        yhats[br], targets[br] = _get_dummy_pair(n_classes + i)

    mtl = MultiTaskLoss(losses)
    mtl(yhats, targets)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("mask_kind", ["none", "partial", "empty"])
@pytest.mark.parametrize("edge_mode", ["none", "target_only", "legacy_dummy"])
def test_weighted_heads_masks_and_edges_match_values_and_gradients(
    device, mask_kind, edge_mode
):
    flow = (
        torch.tensor(
            [1.0, -2.0, 3.0, -1.0, -1.0, 4.0, 2.0, -3.0],
            device=device,
            dtype=torch.float64,
        )
        .reshape(2, 2, 1, 2)
        .requires_grad_()
    )
    logits = (
        torch.tensor(
            [0.2, -0.7, -0.3, 0.9, 1.1, -0.4, -0.2, 0.6],
            device=device,
            dtype=torch.float64,
        )
        .reshape(2, 2, 1, 2)
        .requires_grad_()
    )
    flow_target = torch.zeros_like(flow, requires_grad=True)
    classes = torch.tensor([[[0, 1]], [[1, 0]]], device=device)
    mask = None
    if mask_kind != "none":
        mask = flow.new_tensor([1.0, 0.0, 0.5, 1.0]).reshape(2, 1, 2)
        if mask_kind == "empty":
            mask.zero_()
    edges = flow.new_tensor([0.0, 1.0, 2.0, 0.0]).reshape(2, 1, 2)
    edge_base = None if edge_mode == "none" else 3
    criterion = MultiTaskLoss(
        {
            "flow": JointLoss(
                [
                    MSE(apply_mask=True, edge_weight=edge_base),
                    MAE(alpha=0, apply_mask=True, edge_weight=edge_base),
                ],
                [0.2, 0.3],
            ),
            "type_mask": CELoss(apply_mask=True, edge_weight=edge_base),
        },
        loss_weights={"type_mask": 2.0, "flow": 0.5},
    )
    predictions = {"type_mask": logits, "flow": flow}
    targets = {"flow": flow_target, "type_mask": classes}
    if edge_mode != "none":
        targets["edgeweight"] = edges
    if edge_mode == "legacy_dummy":
        # Only target metadata supplies weights; a legacy prediction is ignored.
        predictions["edgeweight"] = torch.full_like(edges, 10)

    # Derive the two different masked reductions before applying either head weight.
    inclusion = torch.ones_like(edges) if mask is None else mask
    spatial_weight = torch.ones_like(edges) if edge_base is None else 3**edges
    errors = flow - flow_target
    squared = errors.square().mean(1) * inclusion
    absolute = errors.abs().mean(1) * inclusion
    if mask is not None:
        absolute = absolute / (mask.mean() + 1e-7)
    flow_loss = ((0.2 * squared + 0.3 * absolute) * spatial_weight).mean()
    true_logits = logits.gather(1, classes[:, None]).squeeze(1)
    type_loss = (
        (logits.logsumexp(1) - true_logits) * inclusion * spatial_weight
    ).mean()
    expected = 0.5 * flow_loss + 2 * type_loss
    actual = criterion(predictions, targets, mask=mask)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    inputs = (flow, flow_target, logits)
    actual_grads = torch.autograd.grad(actual, inputs, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, inputs)
    for result, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(result, reference, rtol=1e-12, atol=1e-12)
        assert torch.isfinite(result).all()
    if mask is not None:
        assert torch.count_nonzero(actual_grads[0][0, :, :, 1]) == 0
        assert torch.count_nonzero(actual_grads[2][0, :, :, 1]) == 0
    assert targets["flow"] is flow_target
    assert predictions["flow"] is flow


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_head_dtypes_preserve_sum_precision_and_gradients(device, reverse):
    first = torch.tensor([[[[1e4]]]], device=device, requires_grad=True)
    second = torch.tensor(
        [[[[1.0]]]], device=device, dtype=torch.float64, requires_grad=True
    )
    heads = {"first": MSE(), "second": MSE()}
    if reverse:
        heads = dict(reversed(list(heads.items())))
    actual = MultiTaskLoss(heads)(
        {"first": first, "second": second},
        {"first": torch.zeros_like(first), "second": torch.zeros_like(second)},
    )
    assert actual.dtype == torch.float64
    assert actual.item() == 100_000_001
    actual.backward()
    assert first.grad.item() == 20_000
    assert second.grad.item() == 2


@pytest.mark.parametrize(
    "weights",
    [
        None,
        {"first": 0.25, "second": 0.75},
        {"first": 0.0, "second": 1.0},
        {"first": 0.0, "second": 0.0},
    ],
)
def test_default_and_zero_head_weights(weights):
    first = torch.tensor([[[[2.0]]]], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([[[[3.0]]]], dtype=torch.float64, requires_grad=True)
    actual = MultiTaskLoss({"first": MSE(), "second": MSE()}, weights)(
        {"first": first, "second": second},
        {"first": torch.zeros_like(first), "second": torch.zeros_like(second)},
    )
    a, b = (1.0, 1.0) if weights is None else (weights["first"], weights["second"])
    assert actual.item() == pytest.approx(4 * a + 9 * b)
    actual.backward()
    assert first.grad.item() == pytest.approx(4 * a)
    assert second.grad.item() == pytest.approx(6 * b)


@pytest.mark.parametrize("weights", [{}, {"wrong": 1.0}, {"head": 1.0, "extra": 1.0}])
def test_constructor_rejects_mismatched_weight_keys(weights):
    with pytest.raises(ValueError):
        MultiTaskLoss({"head": MSE()}, weights)


@pytest.mark.parametrize("weights", [{}, {"wrong": 1.0}, {"head": 1.0, "extra": 1.0}])
def test_mutated_weight_keys_are_revalidated_before_losses_run(weights):
    loss = MSE()
    calls = []
    loss.register_forward_pre_hook(lambda module, args: calls.append(True))
    criterion = MultiTaskLoss({"head": loss})
    criterion.weights.clear()
    criterion.weights.update(weights)
    with pytest.raises(ValueError, match="weight"):
        criterion({"head": torch.ones(1, 1, 1, 1)}, {"head": torch.zeros(1, 1, 1, 1)})
    assert not calls


@pytest.mark.parametrize(
    "prediction_keys,target_keys,error",
    [
        (["head"], ["wrong"], ValueError),
        (["head"], ["head", "extra"], ValueError),
        (["head", "extra"], ["head"], ValueError),
        (["other"], ["other"], KeyError),
        (["head", "edgeweight"], ["head"], ValueError),
    ],
)
def test_invalid_input_keys_are_rejected(prediction_keys, target_keys, error):
    criterion = MultiTaskLoss({"head": MSE()})
    with pytest.raises(error):
        criterion(
            {key: torch.ones(1, 1, 1, 1) for key in prediction_keys},
            {key: torch.zeros(1, 1, 1, 1) for key in target_keys},
        )


def test_extra_matching_heads_remain_unused():
    prediction = torch.ones(1, 1, 1, 1, requires_grad=True)
    unused = torch.ones(1, 1, 1, 1, requires_grad=True)
    actual = MultiTaskLoss({"head": MSE()})(
        {"head": prediction, "unused": unused},
        {"head": torch.zeros_like(prediction), "unused": torch.zeros_like(unused)},
    )
    assert actual.item() == 1
    actual.backward()
    assert prediction.grad.item() == 2
    assert unused.grad is None


def test_edgeweight_is_reserved_for_target_metadata():
    with pytest.raises(ValueError, match="edgeweight"):
        MultiTaskLoss({"edgeweight": MSE()})


def test_missing_later_head_is_rejected_before_evaluating_earlier_heads():
    loss = MSE()
    calls = []
    loss.register_forward_pre_hook(lambda module, args: calls.append(True))
    criterion = MultiTaskLoss({"first": loss, "missing": MSE()})
    with pytest.raises(KeyError, match="missing"):
        criterion(
            {"first": torch.ones(1, 1, 1, 1)},
            {"first": torch.zeros(1, 1, 1, 1)},
        )
    assert not calls


def test_masking_is_controlled_by_each_child_loss():
    first = torch.ones(1, 1, 1, 2, requires_grad=True)
    second = torch.full((1, 1, 1, 2), 2.0, requires_grad=True)
    criterion = MultiTaskLoss({"masked": MSE(apply_mask=True), "unmasked": MSE()})
    result = criterion(
        {"masked": first, "unmasked": second},
        {"masked": torch.zeros_like(first), "unmasked": torch.zeros_like(second)},
        mask=torch.zeros(1, 1, 2),
    )
    assert result.item() == 4
    result.backward()
    assert torch.count_nonzero(first.grad) == 0
    torch.testing.assert_close(second.grad, torch.full_like(second, 2.0))
