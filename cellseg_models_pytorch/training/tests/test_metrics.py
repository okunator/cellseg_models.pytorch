import pytest
import torch

from cellseg_models_pytorch.training.functional.train_metrics import accuracy, iou


@pytest.mark.parametrize("metric", [accuracy, iou])
@pytest.mark.parametrize("map_type", ["type", "sem"])
@pytest.mark.parametrize("activation", ["sigmoid", "softmax"])
def test_metrics(type_map_tensor, sem_map_tensor, metric, map_type, activation):
    if map_type == "type":
        target = type_map_tensor
        n_channels = 7
    else:
        target = sem_map_tensor
        n_channels = 5

    yhat = torch.rand(8, n_channels, 320, 320)
    metric(yhat, target, activation)
