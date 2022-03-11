import pytest
import torch

from cellseg_models_pytorch.losses import SSIM, CELoss, IoULoss, JointLoss, TverskyLoss
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
    loss = JointLoss([l() for l in losses])
    res = loss(yhat=yhat, target=target)

    assert res.dtype == torch.float32
    assert res.size() == torch.Size([])
