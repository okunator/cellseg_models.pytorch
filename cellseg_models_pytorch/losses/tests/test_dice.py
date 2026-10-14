import pytest
import torch

from cellseg_models_pytorch.losses import DiceLoss


def test_missing_rare_foreground_is_not_high_pixel_accuracy():
    logits = torch.full((1, 2, 10, 10), -30.0, requires_grad=True)
    with torch.no_grad():
        logits[:, 0] = 30
    target = torch.zeros((1, 10, 10), dtype=torch.long)
    target[:, 0, 0] = 1
    loss = DiceLoss()(logits, target)
    assert loss.item() == pytest.approx(1 - (198 / 199) / 2)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("class_weights", [None, [2.0, 3.0]])
def test_spatial_dice_and_class_weights(class_weights):
    probabilities = torch.tensor(
        [[[[0.8, 0.6, 0.9]], [[0.2, 0.4, 0.1]]]], dtype=torch.float64
    )
    target = torch.tensor([[[0, 0, 1]]])
    # Volumes are 4.3 and 1.7, so Dice scores are 28/43 and 2/17.
    losses = torch.tensor([1 - 28 / 43, 1 - 2 / 17], dtype=torch.float64)
    weights = None if class_weights is None else torch.tensor(class_weights)
    criterion = DiceLoss(class_weights=weights)
    expected_loss = losses.mean() if weights is None else (losses * weights).mean()
    torch.testing.assert_close(criterion(probabilities.log(), target), expected_loss)
    assert "class_weights=" in repr(criterion)


def test_absent_classes_and_batch_reduction():
    probabilities = torch.tensor(
        [[[[0.8]], [[0.1]], [[0.1]]], [[[0.2]], [[0.7]], [[0.1]]]]
    )
    target = torch.tensor([[[0]], [[1]]])
    expected = ((1 - 1.6 / 1.8) + (1 - 1.4 / 1.7)) / 2
    assert DiceLoss()(probabilities.log(), target).item() == pytest.approx(expected)


def test_mask_excludes_pixels_and_gradients_even_with_regularization():
    logits = torch.tensor([[[[2.0, -1.0]], [[-2.0, 1.0]]]], requires_grad=True)
    target = torch.tensor([[[0, 0]]])
    mask = torch.tensor([[[1.0, 0.0]]])
    criterion = DiceLoss(apply_mask=True, apply_sd=True)
    loss = criterion(logits, target, mask=mask)
    expected = DiceLoss(apply_sd=True)(logits[..., :1], target[..., :1])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[..., 1:]) == 0
    assert torch.count_nonzero(logits.grad[..., :1]) > 0
    logits.grad = None
    empty_loss = criterion(logits, target, mask=torch.zeros_like(mask))
    assert empty_loss.item() == 0
    empty_loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_edges_weight_both_overlap_and_volume_once():
    probabilities = torch.tensor([[[[0.8, 0.6]], [[0.2, 0.4]]]])
    target = torch.tensor([[[0, 1]]])
    loss = DiceLoss(edge_weight=2)(
        probabilities.log(), target, target_weight=torch.tensor([[[0.0, 1.0]]])
    )
    expected = 1 - ((1.6 / 3) + (1.6 / 3)) / 2
    assert loss.item() == pytest.approx(expected)


def test_spectral_decoupling_adds_positive_logit_penalty():
    logits = torch.tensor([[[[2.0, -1.0]], [[-2.0, 1.0]]]])
    target = torch.tensor([[[0, 1]]])
    difference = DiceLoss(apply_sd=True)(logits, target) - DiceLoss()(logits, target)
    assert difference.item() == pytest.approx(0.005 * logits.square().mean().item())


@pytest.mark.parametrize("smoothing", ["apply_ls", "apply_svls"])
def test_smoothed_dice_has_finite_correct_gradients(smoothing):
    torch.manual_seed(42)
    logits = torch.randn((1, 2, 5, 5), dtype=torch.float64, requires_grad=True)
    target = torch.zeros((1, 5, 5), dtype=torch.long)
    target[:, 2:] = 1
    criterion = DiceLoss(**{smoothing: True})
    assert torch.autograd.gradcheck(
        lambda values: criterion(values, target), (logits,), eps=1e-5, atol=1e-4
    )


def test_weights_follow_module_dtype_without_changing_state_dict():
    criterion = DiceLoss(class_weights=torch.tensor([1.0, 2.0])).double()
    assert criterion.class_weights.dtype == torch.float64
    assert criterion.state_dict() == {}


@pytest.mark.cuda
def test_weighted_dice_on_cuda_matches_cpu():
    logits = torch.tensor([[[[2.0, -1.0]], [[-2.0, 1.0]]]], requires_grad=True)
    target = torch.tensor([[[0, 1]]])
    criterion = DiceLoss(class_weights=torch.tensor([1.0, 2.0]))
    expected = criterion(logits, target)
    actual = criterion.cuda()(logits.cuda(), target.cuda())
    torch.testing.assert_close(actual.cpu(), expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_large_low_precision_masks_do_not_overflow(dtype):
    logits = torch.zeros((1, 2, 256, 256), dtype=dtype, requires_grad=True)
    target = torch.zeros((1, 256, 256), dtype=torch.long)
    loss = DiceLoss()(logits, target)
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(1 / 3)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[:, 0].sum() < 0


@pytest.mark.parametrize(
    "options,arguments,match",
    [
        ({"apply_mask": True}, {"mask": torch.ones(1, 1, 1)}, "mask"),
        ({"edge_weight": 2}, {}, "target_weight"),
        ({"class_weights": torch.ones(3)}, {}, "class_weights"),
    ],
)
def test_invalid_weight_shapes_raise(options, arguments, match):
    with pytest.raises(ValueError, match=match):
        DiceLoss(**options)(
            torch.zeros(1, 2, 2, 2), torch.zeros(1, 2, 2, dtype=torch.long), **arguments
        )


@pytest.mark.parametrize("correct,expected", [(True, 0.0), (False, 1.0)])
def test_perfect_and_disjoint_predictions(correct, expected):
    target = torch.tensor([[[0, 1]]])
    predicted = target if correct else 1 - target
    logits = torch.nn.functional.one_hot(predicted, 2).permute(0, 3, 1, 2).float() * 100
    assert DiceLoss()(logits, target).item() == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_empty_low_precision_mask_with_regularization(dtype):
    logits = torch.zeros((1, 2, 2, 2), dtype=dtype, requires_grad=True)
    target = torch.zeros((1, 2, 2), dtype=torch.long)
    loss = DiceLoss(apply_mask=True, apply_sd=True)(
        logits, target, mask=torch.zeros_like(target, dtype=dtype)
    )
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_zero_edge_weights_exclude_overlap_without_nan():
    logits = torch.zeros((1, 2, 2, 2), requires_grad=True)
    target = torch.zeros((1, 2, 2), dtype=torch.long)
    loss = DiceLoss(edge_weight=0)(
        logits, target, target_weight=torch.ones_like(target)
    )
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_mean_weights_present_image_class_pairs_equally():
    probabilities = torch.tensor(
        [
            [[[0.8, 0.8]], [[0.2, 0.2]]],
            [[[0.6, 0.4]], [[0.4, 0.6]]],
        ]
    )
    target = torch.tensor([[[0, 0]], [[0, 1]]])
    # Three present image/class pairs have losses 1/9, 2/5 and 2/5.
    assert DiceLoss()(probabilities.log(), target).item() == pytest.approx(41 / 135)
