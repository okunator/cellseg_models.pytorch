import pytest
import torch

from cellseg_models_pytorch.models import (
    cellpose_base,
    cellpose_plus,
    hovernet_base,
    hovernet_plus,
    hovernet_small,
    hovernet_small_plus,
    omnipose_base,
    omnipose_plus,
    stardist_base,
    stardist_base_multiclass,
    stardist_plus,
)


@pytest.mark.parametrize(
    "model", [hovernet_base, hovernet_plus, hovernet_small_plus, hovernet_small]
)
@pytest.mark.parametrize("style_channels", [None, 32])
def test_hovernet_fwdbwd(model, style_channels):
    x = torch.rand([1, 3, 64, 64])
    m = model(
        type_classes=3,
        sem_classes=3,
        style_channels=style_channels,
    )
    y = m(x)
    y["hovernet"].mean().backward()

    assert y["type"].shape == x.shape


@pytest.mark.parametrize(
    "model", [stardist_base, stardist_plus, stardist_base_multiclass]
)
@pytest.mark.parametrize("style_channels", [None, 32])
def test_stardist_fwdbwd(model, style_channels):
    n_rays = 3
    x = torch.rand([1, 3, 64, 64])
    m = model(
        n_rays=n_rays,
        type_classes=3,
        sem_classes=3,
        style_channels=style_channels,
    )
    y = m(x)
    y["stardist"].mean().backward()

    assert y["stardist"].shape == torch.Size([1, n_rays, 64, 64])


@pytest.mark.parametrize("model", [cellpose_base, cellpose_plus])
def test_cellpose_fwdbwd(model):
    x = torch.rand([1, 3, 64, 64])
    m = model(type_classes=3, sem_classes=3)
    y = m(x)
    y["cellpose"].mean().backward()

    if "sem" in y.keys():
        assert y["sem"].shape == torch.Size([1, 3, 64, 64])

    assert y["cellpose"].shape == torch.Size([1, 2, 64, 64])


@pytest.mark.parametrize("model", [omnipose_base, omnipose_plus])
def test_omnipose_fwdbwd(model):
    x = torch.rand([1, 3, 64, 64])
    m = model(type_classes=3, sem_classes=3)
    y = m(x)
    y["omnipose"].mean().backward()

    if "sem" in y.keys():
        assert y["sem"].shape == torch.Size([1, 3, 64, 64])

    assert y["omnipose"].shape == torch.Size([1, 2, 64, 64])
