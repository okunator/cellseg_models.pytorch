import pytest
import torch

from cellseg_models_pytorch.decoders import Decoder


@pytest.mark.parametrize("long_skip", ["unet", "unetpp", "unet3p"])
@pytest.mark.parametrize("merge_policy", ["cat", "sum"])
def test_decoder_forward(long_skip, merge_policy):
    enc_channels = (128, 64, 32, 16, 8)
    out_dims = [256 // 2**i for i in range(6)][::-1]

    decoder1_kwargs = {"merge_policy": merge_policy}
    decoder2_kwargs = {"merge_policy": merge_policy}
    decoder3_kwargs = {"merge_policy": merge_policy}
    decoder4_kwargs = {"merge_policy": merge_policy}
    decoder5_kwargs = {"merge_policy": merge_policy}
    stage_params = (
        decoder1_kwargs,
        decoder2_kwargs,
        decoder3_kwargs,
        decoder4_kwargs,
        decoder5_kwargs,
    )

    decoder = Decoder(
        enc_channels=enc_channels,
        model_input_size=256,
        out_channels=(128, 64, 32, 16, 8),
        n_layers=(1, 1, 1, 1, 1),
        n_blocks=((2,), (2,), (2,), (2,), (2,)),
        long_skip=long_skip,
        stage_params=stage_params,
    )

    x = [torch.rand([1, enc_channels[i], out_dims[i], out_dims[i]]) for i in range(5)]
    out = decoder(*x)

    assert out.shape[1] == decoder.out_channels


@pytest.mark.parametrize("long_skip", ["unet", "unetpp", "unet3p"])
@pytest.mark.parametrize("merge_policy", ["cat", "sum"])
def test_decoder_backward(long_skip, merge_policy):
    enc_channels = (128, 64, 32, 16, 8)
    out_dims = [256 // 2**i for i in range(6)][::-1]

    decoder1_kwargs = {"merge_policy": merge_policy}
    decoder2_kwargs = {"merge_policy": merge_policy}
    decoder3_kwargs = {"merge_policy": merge_policy}
    decoder4_kwargs = {"merge_policy": merge_policy}
    decoder5_kwargs = {"merge_policy": merge_policy}
    stage_params = (
        decoder1_kwargs,
        decoder2_kwargs,
        decoder3_kwargs,
        decoder4_kwargs,
        decoder5_kwargs,
    )

    decoder = Decoder(
        enc_channels=enc_channels,
        model_input_size=256,
        out_channels=(128, 64, 32, 16, 8),
        n_layers=(1, 1, 1, 1, 1),
        n_blocks=((2,), (2,), (2,), (2,), (2,)),
        long_skip=long_skip,
        stage_params=stage_params,
    )

    x = [torch.rand([1, enc_channels[i], out_dims[i], out_dims[i]]) for i in range(5)]
    out = decoder(*x)

    out.mean().backward()

    assert out.shape[1] == decoder.out_channels
