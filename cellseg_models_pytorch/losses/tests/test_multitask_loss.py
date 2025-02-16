import pytest
import torch

from cellseg_models_pytorch.losses import (
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
