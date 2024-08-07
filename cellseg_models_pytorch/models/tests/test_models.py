import pytest
import torch

from cellseg_models_pytorch.models import MultiTaskUnet, get_model


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
@pytest.mark.parametrize("model_type", ["base", "plus"])
@pytest.mark.parametrize("style_channels", [None, 32])
@pytest.mark.parametrize("add_stem_skip", [False, True])
def test_cppnet_fwdbwd(enc_name, model_type, style_channels, add_stem_skip):
    n_rays = 3
    x = torch.rand([1, 3, 64, 64])
    model = get_model(
        name="cppnet",
        type=model_type,
        enc_name=enc_name,
        n_rays=n_rays,
        ntypes=3,
        ntissues=3,
        style_channels=style_channels,
        add_stem_skip=add_stem_skip,
        enc_pretrain=False,
    )

    y = model(x)
    y["stardist_refined"].mean().backward()

    assert y["type"].shape == x.shape
    assert y["stardist_refined"].shape == torch.Size([1, n_rays, 64, 64])

    if "sem" in y.keys():
        assert y["sem"].shape == torch.Size([1, 3, 64, 64])


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
@pytest.mark.parametrize("style_channels", [None, 32])
def test_cellvit_fwdbwd(enc_name, model_type, style_channels):
    x = torch.rand([1, 3, 32, 32])
    model = get_model(
        name="cellvit",
        type=model_type,
        enc_name=enc_name,
        ntypes=3,
        ntissues=3,
        style_channels=style_channels,
        enc_pretrain=False,
    )
    model.freeze_encoder()

    y = model(x)
    y["hovernet"].mean().backward()

    assert y["type"].shape == x.shape

    if "sem" in y.keys():
        assert y["sem"].shape == torch.Size([1, 3, 32, 32])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
@pytest.mark.parametrize("model_type", ["base", "plus", "small_plus", "small"])
@pytest.mark.parametrize("style_channels", [None, 32])
@pytest.mark.parametrize("add_stem_skip", [False, True])
def test_hovernet_fwdbwd(enc_name, model_type, style_channels, add_stem_skip):
    x = torch.rand([1, 3, 64, 64])
    model = get_model(
        name="hovernet",
        type=model_type,
        enc_name=enc_name,
        ntypes=3,
        ntissues=3,
        style_channels=style_channels,
        add_stem_skip=add_stem_skip,
        enc_pretrain=False,
    )

    y = model(x)
    y["hovernet"].mean().backward()

    assert y["type"].shape == x.shape

    if "sem" in y.keys():
        assert y["sem"].shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
@pytest.mark.parametrize("model_type", ["base", "plus"])
@pytest.mark.parametrize("style_channels", [None, 32])
@pytest.mark.parametrize("add_stem_skip", [False, True])
def test_stardist_fwdbwd(enc_name, model_type, style_channels, add_stem_skip):
    n_rays = 3
    x = torch.rand([1, 3, 64, 64])
    model = get_model(
        name="stardist",
        type=model_type,
        n_rays=n_rays,
        enc_name=enc_name,
        ntypes=3,
        ntissues=3,
        style_channels=style_channels,
        add_stem_skip=add_stem_skip,
        enc_pretrain=False,
    )

    y = model(x)
    y["stardist"].mean().backward()

    assert y["type"].shape == x.shape
    assert y["stardist"].shape == torch.Size([1, n_rays, 64, 64])

    if "sem" in y.keys():
        assert y["sem"].shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
@pytest.mark.parametrize("model_type", ["base", "plus"])
@pytest.mark.parametrize("add_stem_skip", [False, True])
def test_cellpose_fwdbwd(enc_name, model_type, add_stem_skip):
    x = torch.rand([1, 3, 64, 64])
    model = get_model(
        name="cellpose",
        type=model_type,
        enc_name=enc_name,
        ntypes=3,
        ntissues=3,
        add_stem_skip=add_stem_skip,
        enc_pretrain=False,
    )

    y = model(x)
    y["cellpose"].mean().backward()

    assert y["type"].shape == x.shape
    assert y["cellpose"].shape == torch.Size([1, 2, 64, 64])

    if "sem" in y.keys():
        assert y["sem"].shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
@pytest.mark.parametrize("model_type", ["base", "plus"])
@pytest.mark.parametrize("add_stem_skip", [False, True])
def test_cellpose_fwdbwd(enc_name, model_type, add_stem_skip):
    x = torch.rand([1, 3, 64, 64])
    model = get_model(
        name="omnipose",
        type=model_type,
        enc_name=enc_name,
        ntypes=3,
        ntissues=3,
        add_stem_skip=add_stem_skip,
        enc_pretrain=False,
    )

    y = model(x)
    y["omnipose"].mean().backward()

    assert y["type"].shape == x.shape
    assert y["omnipose"].shape == torch.Size([1, 2, 64, 64])

    if "sem" in y.keys():
        assert y["sem"].shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
@pytest.mark.parametrize("add_stem_skip", [False, True])
def test_multitaskunet_fwdbwd(enc_name, add_stem_skip):
    x = torch.rand([1, 3, 64, 64])
    m = MultiTaskUnet(
        decoders=("sem",),
        heads={"sem": {"sem": 3}},
        n_conv_layers={"sem": (1, 1, 1, 1)},
        n_conv_blocks={"sem": ((2,), (2,), (2,), (2,))},
        out_channels={"sem": (128, 64, 32, 16)},
        long_skips={"sem": "unet"},
        dec_params={"sem": None},
        add_stem_skip=add_stem_skip,
        enc_name=enc_name,
        enc_pretrain=False,
    )
    y = m(x)
    y["sem"].mean().backward()

    assert y["sem"].shape == torch.Size([1, 3, 64, 64])
