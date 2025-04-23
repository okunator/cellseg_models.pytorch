import pytest
import torch

from cellseg_models_pytorch.models.cellpose.cellpose_unet import (
    cellpose_panoptic, cellpose_nuclei, omnipose_nuclei, omnipose_panoptic
)
from cellseg_models_pytorch.models.stardist.stardist_unet import stardist_panoptic, stardist_nuclei
from cellseg_models_pytorch.models.hovernet.hovernet_unet import hovernet_panoptic, hovernet_nuclei
from cellseg_models_pytorch.models.cellvit.cellvit_unet import cellvit_panoptic, cellvit_nuclei
from cellseg_models_pytorch.models.cppnet.cppnet_unet import cppnet_panoptic, cppnet_nuclei

@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
@pytest.mark.parametrize("model_type", ["nuc", "panop"])
def test_cppnet_fwdbwd(enc_name, model_type):
    n_rays = 3
    x = torch.rand([1, 3, 64, 64])
    if model_type == "nuc":
        model = cppnet_nuclei(n_rays, 3, enc_name=enc_name)
    elif model_type == "panop":
        model = cppnet_panoptic(n_rays, 3, 3, enc_name=enc_name)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 3, 64, 64])

    if model_type == "panop":
        assert y["tissue"].type_map.shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize(
    "enc_name",
    [
        "samvit_base_patch16",
        "samvit_base_patch16_224",
        "samvit_huge_patch16",
        "samvit_large_patch16",
    ],
)
@pytest.mark.parametrize("model_type", ["nuc", "panop"])
def test_cellvit_fwdbwd(enc_name, model_type):
    x = torch.rand([1, 3, 64, 64])
    if model_type == "nuc":
        model = cellvit_nuclei(enc_name, 3, enc_pretrain=False)
    elif model_type == "panop":
        model = cellvit_panoptic(enc_name, 3, 3, enc_pretrain=False)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 2, 64, 64])

    if model_type == "panop":
        assert y["tissue"].type_map.shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
@pytest.mark.parametrize("model_type", ["nuc", "panop"])
def test_hovernet_fwdbwd(enc_name, model_type):
    x = torch.rand([1, 3, 64, 64])
    if model_type == "nuc":
        model = hovernet_nuclei(3, enc_name=enc_name)
    elif model_type == "panop":
        model = hovernet_panoptic(3, 3, enc_name=enc_name)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 2, 64, 64])

    if model_type == "panop":
        assert y["tissue"].type_map.shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
@pytest.mark.parametrize("model_type", ["nuc", "panop"])
def test_stardist_fwdbwd(enc_name, model_type):
    n_rays = 3
    x = torch.rand([1, 3, 64, 64])
    if model_type == "nuc":
        model = stardist_nuclei(n_rays, 3, enc_name=enc_name)
    elif model_type == "panop":
        model = stardist_panoptic(n_rays, 3, 3, enc_name=enc_name)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 3, 64, 64])

    if model_type == "panop":
        assert y["tissue"].type_map.shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
@pytest.mark.parametrize("model_type", ["nuc", "panop"])
def test_cellpose_fwdbwd(enc_name, model_type):
    x = torch.rand([1, 3, 64, 64])
    if model_type == "nuc":
        model = cellpose_nuclei(3, enc_name=enc_name)
    elif model_type == "panop":
        model = cellpose_panoptic(3, 3, enc_name=enc_name)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 2, 64, 64])

    if model_type == "panop":
        assert y["tissue"].type_map.shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
@pytest.mark.parametrize("model_type", ["nuc", "panop"])
def test_omnipose_fwdbwd(enc_name, model_type):
    x = torch.rand([1, 3, 64, 64])

    if model_type == "nuc":
        model = omnipose_nuclei(3, enc_name=enc_name)
    elif model_type == "panop":
        model = omnipose_panoptic(3, 3, enc_name=enc_name)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 2, 64, 64])

    if model_type == "panop":
        assert y["tissue"].type_map.shape == torch.Size([1, 3, 64, 64])
