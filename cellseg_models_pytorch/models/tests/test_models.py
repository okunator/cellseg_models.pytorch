import pytest
import torch

from cellseg_models_pytorch.models.cellpose.cellpose_unet import (
    cellpose_nuclei, omnipose_nuclei
)
from cellseg_models_pytorch.models.stardist.stardist_unet import stardist_nuclei
from cellseg_models_pytorch.models.hovernet.hovernet_unet import hovernet_nuclei
from cellseg_models_pytorch.models.cellvit.cellvit_unet import cellvit_nuclei
from cellseg_models_pytorch.models.cppnet.cppnet_unet import cppnet_nuclei

@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
def test_cppnet_fwdbwd(enc_name):
    n_rays = 3
    x = torch.rand([1, 3, 64, 64])
    model = cppnet_nuclei(n_rays, 3, enc_name=enc_name)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize(
    "enc_name",
    [
        "samvit_base_patch16",
        "samvit_base_patch16_224",
        "samvit_huge_patch16",
        "samvit_large_patch16",
    ],
)
def test_cellvit_fwdbwd(enc_name):
    x = torch.rand([1, 3, 64, 64])
    model = cellvit_nuclei(enc_name, 3, enc_pretrain=False)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 2, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
def test_hovernet_fwdbwd(enc_name):
    x = torch.rand([1, 3, 64, 64])
    model = hovernet_nuclei(3, enc_name=enc_name)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 2, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
def test_stardist_fwdbwd(enc_name):
    n_rays = 3
    x = torch.rand([1, 3, 64, 64])
    model = stardist_nuclei(n_rays, 3, enc_name=enc_name)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
def test_cellpose_fwdbwd(enc_name):
    x = torch.rand([1, 3, 64, 64])
    model = cellpose_nuclei(3, enc_name=enc_name)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 2, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
def test_omnipose_fwdbwd(enc_name):
    x = torch.rand([1, 3, 64, 64])

    model = omnipose_nuclei(3, enc_name=enc_name)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 2, 64, 64])
