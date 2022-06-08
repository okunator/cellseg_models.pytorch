from typing import Callable, Tuple

import pytest
import scipy.ndimage as ndi
import torch

from cellseg_models_pytorch.losses import Loss
from cellseg_models_pytorch.transforms.functional import gen_weight_maps


def _get_dummy_pair(n_classes: int) -> Tuple[torch.Tensor]:
    dummy_yhat = torch.zeros([1, n_classes, 64, 64]).float()
    dummy_target = torch.zeros([1, 64, 64]).long()

    for i, c in enumerate(range(dummy_yhat.shape[1]), 1):
        dummy_yhat[
            ..., c, (i * c) + 10 : (i * c) + 20, (i * c) + 30 : (i * c) + 40
        ] += 1
        dummy_yhat[
            ..., c, (i * c) + 10 : (i * c) + 20, (i * c) + 30 : (i * c) + 40
        ] += 1
        dummy_target[
            ...,
            (i * c * 2) + 10 : (i * c * 2) + 20,
            (i * c * 2) + 30 : (i * c * 2) + 40,
        ] += 1
        dummy_target[
            ...,
            (i * c * 2) + 15 : (i * c * 2) + 25,
            (i * c * 2) + 15 : (i * c * 2) + 25,
        ] += 1
        dummy_target[dummy_target > c] = c

    return dummy_yhat, dummy_target


def _get_dummy_weight_map(target: torch.Tensor) -> torch.Tensor:
    wmap = gen_weight_maps(ndi.label(target)[0][0])
    wmap = torch.tensor(wmap).unsqueeze(0)
    return wmap


def _get_dummy_class_weights(n_classes: int) -> torch.Tensor:
    w = torch.zeros(1, n_classes)
    w[0][0] += 0.1

    for i in range(1, n_classes):
        w[0][i] += 0.9 + i * 0.01

    return w[0, ...]


@torch.no_grad()
@pytest.mark.parametrize("n_classes", [1, 3])
@pytest.mark.parametrize("class_weights", [True, False])
@pytest.mark.parametrize("edge_weight", [1.1, None])
@pytest.mark.parametrize(
    "loss",
    ["mse", "ssim", "msssim", "dice", "iou", "focal", "tversky", "ce", "sce"],
)
def test_loss(
    n_classes: int, loss: Callable, edge_weight: float, class_weights: bool
) -> None:
    yhat, target = _get_dummy_pair(n_classes)

    wmap = None
    if edge_weight is not None:
        wmap = _get_dummy_weight_map(target)

    cw = None
    if class_weights:
        cw = _get_dummy_class_weights(n_classes)

    criterion = Loss(name=loss, edge_weight=edge_weight, class_weights=cw)
    criterion(yhat=yhat, target=target, target_weight=wmap)
