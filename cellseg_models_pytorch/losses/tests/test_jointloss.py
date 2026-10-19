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
        [TverskyLoss, CELoss],
        [IoULoss],
        [SSIM, TverskyLoss, CELoss],
    ],
)
def test_joint_loss(n_classes, losses) -> None:
    yhat, target = _get_dummy_pair(n_classes)
    loss = JointLoss([loss_class() for loss_class in losses])
    res = loss(yhat=yhat, target=target)

    assert res.dtype == torch.float32
    assert res.size() == torch.Size([])


@pytest.mark.parametrize("weights", [[], [0.5], [0.2, 0.3, 0.5]])
def test_weight_count_must_match_loss_count(weights):
    with pytest.raises(ValueError, match="one weight per loss"):
        JointLoss([MSE(), MAE()], weights)


@pytest.mark.parametrize("losses", [[], (), None])
def test_losses_must_be_a_nonempty_list(losses):
    with pytest.raises(ValueError, match="nonempty list"):
        JointLoss(losses)


@pytest.mark.parametrize("weights", [[-0.1, 1.0], [1.1, 1.0]])
def test_weights_must_stay_in_the_existing_range(weights):
    with pytest.raises(ValueError, match="0 <= weight <= 1"):
        JointLoss([MSE(), MAE()], weights)


@pytest.mark.parametrize(
    "weights", [None, [0.25, 0.75], [0.2, 0.3], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
)
def test_weighted_values_and_both_input_gradients(weights):
    prediction = torch.tensor(
        [[[[1.0, -2.0, 0.0]]]], dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor(
        [[[[0.0, 1.0, -1.0]]]], dtype=torch.float64, requires_grad=True
    )
    result = JointLoss([MSE(), MAE()], weights)(yhat=prediction, target=target)
    squared_weight, absolute_weight = (1.0, 1.0) if weights is None else weights

    # Pixel errors are [1, -3, 1]; derive the sum and gradients without the loss code.
    expected = (11 * squared_weight + 5 * absolute_weight) / 3
    gradient = (
        torch.tensor(
            [
                2 * squared_weight + absolute_weight,
                -6 * squared_weight - absolute_weight,
                2 * squared_weight + absolute_weight,
            ],
            dtype=torch.float64,
        ).reshape_as(prediction)
        / 3
    )
    assert result.dtype == torch.float64
    assert result.shape == torch.Size([])
    assert result.item() == pytest.approx(expected, abs=1e-12)
    result.backward()
    torch.testing.assert_close(prediction.grad, gradient, atol=1e-12, rtol=0)
    torch.testing.assert_close(target.grad, -gradient, atol=1e-12, rtol=0)


@pytest.mark.parametrize("weights", [[0.5], [0.2, 0.3, 0.5]])
def test_weight_count_changes_cannot_silently_drop_losses(weights):
    criterion = JointLoss([MSE(), MAE()], [0.5, 0.5])
    # The public weight list can be edited after constructor validation.
    criterion.weights[:] = weights
    with pytest.raises(ValueError):
        criterion(yhat=torch.ones(1, 1, 1, 1), target=torch.zeros(1, 1, 1, 1))


def test_weighted_joint_loss_composes_with_multitask_loss():
    prediction = torch.tensor([[[[2.0]]]], dtype=torch.float64, requires_grad=True)
    criterion = MultiTaskLoss(
        {"cellpose": JointLoss([MSE(), MAE()], [0.2, 0.3])},
        loss_weights={"cellpose": 0.5},
    )
    result = criterion(
        {"cellpose": prediction}, {"cellpose": torch.zeros_like(prediction)}
    )
    # Both the inner component weights and the outer head weight scale the result.
    assert result.item() == pytest.approx(0.5 * (0.2 * 4 + 0.3 * 2), abs=1e-12)
    result.backward()
    assert prediction.grad.item() == pytest.approx(0.5 * (0.2 * 4 + 0.3), abs=1e-12)
