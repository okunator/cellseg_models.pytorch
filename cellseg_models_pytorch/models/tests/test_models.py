import pytest
import torch

from cellseg_models_pytorch.models import get_model


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
@pytest.mark.parametrize("model_type", ["base", "plus"])
def test_cppnet_fwdbwd(enc_name, model_type):
    n_rays = 3
    x = torch.rand([1, 3, 64, 64])
    model = get_model(
        name="cppnet",
        type=model_type,
        enc_name=enc_name,
        n_rays=n_rays,
        n_type_classes=3,
        n_sem_classes=3,
        enc_pretrain=False,
    )

    y = model(x)
    y["stardist-stardist"].mean().backward()

    assert y["type-type"].shape == x.shape
    assert y["stardist-stardist"].shape == torch.Size([1, n_rays, 64, 64])

    if "sem-sem" in y.keys():
        assert y["sem-sem"].shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize(
    "enc_name",
    [
        "samvit_base_patch16",
        "samvit_base_patch16_224",
        "samvit_huge_patch16",
        "samvit_large_patch16",
    ],
)
@pytest.mark.parametrize("model_type", ["base", "plus", "small_plus", "small"])
def test_cellvit_fwdbwd(enc_name, model_type):
    x = torch.rand([1, 3, 32, 32])
    model = get_model(
        name="cellvit",
        type=model_type,
        enc_name=enc_name,
        n_type_classes=3,
        n_sem_classes=3,
        enc_pretrain=False,
        enc_freeze=True
    )

    y = model(x)
    y["hovernet-hovernet"].mean().backward()

    assert y["type-type"].shape == x.shape

    if "sem-sem" in y.keys():
        assert y["sem-sem"].shape == torch.Size([1, 3, 32, 32])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
@pytest.mark.parametrize("model_type", ["base", "plus", "small_plus", "small"])
@pytest.mark.parametrize("stem_skip_kws", [None, {"short_skip": "residual"}])
@pytest.mark.parametrize("style_channels", [None, 256])
def test_hovernet_fwdbwd(enc_name, model_type, stem_skip_kws, style_channels):
    x = torch.rand([1, 3, 64, 64])
    model = get_model(
        name="hovernet",
        type=model_type,
        enc_name=enc_name,
        n_type_classes=3,
        n_sem_classes=3,
        enc_pretrain=False,
        style_channels=style_channels,
        stem_skip_kws=stem_skip_kws,
    )

    y = model(x)
    y["hovernet-hovernet"].mean().backward()

    assert y["type-type"].shape == x.shape

    if "sem-sem" in y.keys():
        assert y["sem-sem"].shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
@pytest.mark.parametrize("model_type", ["base", "plus"])
def test_stardist_fwdbwd(enc_name, model_type):
    n_rays = 3
    x = torch.rand([1, 3, 64, 64])
    model = get_model(
        name="stardist",
        type=model_type,
        n_rays=n_rays,
        enc_name=enc_name,
        n_type_classes=3,
        n_sem_classes=3,
        enc_pretrain=False,
    )

    y = model(x)
    y["stardist-stardist"].mean().backward()

    assert y["stardist-type"].shape == x.shape
    assert y["stardist-stardist"].shape == torch.Size([1, n_rays, 64, 64])

    if "sem-sem" in y.keys():
        assert y["sem-sem"].shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
@pytest.mark.parametrize("model_type", ["base", "plus"])
def test_cellpose_fwdbwd(enc_name, model_type):
    x = torch.rand([1, 3, 64, 64])
    model = get_model(
        name="cellpose",
        type=model_type,
        enc_name=enc_name,
        n_type_classes=3,
        n_sem_classes=3,
        enc_pretrain=False,
    )

    y = model(x)
    y["cellpose-cellpose"].mean().backward()

    assert y["cellpose-type"].shape == x.shape
    assert y["cellpose-cellpose"].shape == torch.Size([1, 2, 64, 64])

    if "sem-sem" in y.keys():
        assert y["sem-sem"].shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
@pytest.mark.parametrize("model_type", ["base", "plus"])
def test_cellpose_fwdbwd(enc_name, model_type):
    x = torch.rand([1, 3, 64, 64])
    model = get_model(
        name="omnipose",
        type=model_type,
        enc_name=enc_name,
        n_type_classes=3,
        n_sem_classes=3,
        enc_pretrain=False,
    )

    y = model(x)
    y["omnipose-omnipose"].mean().backward()

    assert y["omnipose-type"].shape == x.shape
    assert y["omnipose-omnipose"].shape == torch.Size([1, 2, 64, 64])

    if "sem-sem" in y.keys():
        assert y["sem-sem"].shape == torch.Size([1, 3, 64, 64])
