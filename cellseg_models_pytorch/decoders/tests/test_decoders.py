import pytest
import torch

from cellseg_models_pytorch.decoders import Decoder


@pytest.mark.parametrize("long_skip", ["unet", "unetpp", "unet3p", "unet3p-lite"])
@pytest.mark.parametrize("merge_policy", ["cat", "sum"])
def test_decoder_fwdbwd(long_skip, merge_policy):
    enc_channels = (64, 32, 16, 8, 8)
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
        out_channels=(64, 32, 16, 8, 8),
        n_layers=(1, 1, 1, 1, 1),
        n_blocks=((2,), (2,), (2,), (2,), (2,)),
        long_skip=long_skip,
        stage_params=stage_params,
    )

    x = [torch.rand([1, enc_channels[i], out_dims[i], out_dims[i]]) for i in range(5)]
    out = decoder(*x)

    out[-1].mean().backward()

    assert out[-1].shape[1] == decoder.out_channels


@pytest.mark.slow
@pytest.mark.parametrize("long_skip", ["unet", "unetpp", "unet3p", "unet3p-lite"])
@pytest.mark.parametrize("merge_policy", ["cat", "sum"])
@pytest.mark.parametrize("short_skip", ["residual", "dense", "basic"])
@pytest.mark.parametrize(
    "block_types", ["mbconv", "fmbconv", "basic", "bottleneck", "dws", "hover_dense"]
)
@pytest.mark.parametrize("normalizations", ["bn", "bcn", "gn"])
@pytest.mark.parametrize("convolutions", ["conv", "scaled_wsconv", "wsconv"])
@pytest.mark.parametrize("attentions", ["se", "scse", "eca", "gc"])
def test_decoder_fwdbwd_all(
    long_skip,
    merge_policy,
    short_skip,
    block_types,
    normalizations,
    convolutions,
    attentions,
):
    enc_channels = (64, 32, 16, 8, 8)
    out_dims = [256 // 2**i for i in range(6)][::-1]

    decoder1_kwargs = {
        "short_skips": (short_skip,),
        "block_types": ((block_types, block_types),),
        "normalizations": ((normalizations, normalizations),),
        "activations": (("leaky-relu", "leaky-relu"),),
        "convolutions": ((convolutions, convolutions),),
        "attentions": ((attentions, attentions),),
        "merge_policy": merge_policy,
        "skip_params": {
            "convolutions": ((convolutions,),),
            "normalizations": ((normalizations,),),
            "attentions": ((attentions,),),
            "block_types": ((block_types,),),
        },
    }

    decoder2_kwargs = {
        "short_skips": (short_skip,),
        "block_types": ((block_types, block_types),),
        "normalizations": ((normalizations, normalizations),),
        "activations": (("leaky-relu", "leaky-relu"),),
        "convolutions": ((convolutions, convolutions),),
        "attentions": ((attentions, attentions),),
        "merge_policy": merge_policy,
        "skip_params": {
            "convolutions": ((convolutions,),),
            "normalizations": ((normalizations,),),
            "attentions": ((attentions,),),
            "block_types": ((block_types,),),
        },
    }

    decoder3_kwargs = {
        "short_skips": (short_skip,),
        "block_types": ((block_types, block_types),),
        "normalizations": ((normalizations, normalizations),),
        "activations": (("leaky-relu", "leaky-relu"),),
        "convolutions": ((convolutions, convolutions),),
        "attentions": ((attentions, attentions),),
        "merge_policy": merge_policy,
        "skip_params": {
            "convolutions": ((convolutions,),),
            "normalizations": ((normalizations,),),
            "attentions": ((attentions,),),
            "block_types": ((block_types,),),
        },
    }

    decoder4_kwargs = {
        "short_skips": (short_skip,),
        "block_types": ((block_types, block_types),),
        "normalizations": ((normalizations, normalizations),),
        "activations": (("leaky-relu", "leaky-relu"),),
        "convolutions": ((convolutions, convolutions),),
        "attentions": ((attentions, attentions),),
        "merge_policy": merge_policy,
        "skip_params": {
            "convolutions": ((convolutions,),),
            "normalizations": ((normalizations,),),
            "attentions": ((attentions,),),
            "block_types": ((block_types,),),
        },
    }

    decoder5_kwargs = {
        "short_skips": (short_skip,),
        "block_types": ((block_types, block_types),),
        "normalizations": ((normalizations, normalizations),),
        "activations": (("leaky-relu", "leaky-relu"),),
        "convolutions": ((convolutions, convolutions),),
        "attentions": ((attentions, attentions),),
        "merge_policy": merge_policy,
        "skip_params": {
            "convolutions": ((convolutions,),),
            "normalizations": ((normalizations,),),
            "attentions": ((attentions,),),
            "block_types": ((block_types,),),
        },
    }
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
        out_channels=(64, 32, 16, 8, 8),
        n_layers=(1, 1, 1, 1, 1),
        n_blocks=((2,), (2,), (2,), (2,), (2,)),
        long_skip=long_skip,
        stage_params=stage_params,
    )

    x = [torch.rand([1, enc_channels[i], out_dims[i], out_dims[i]]) for i in range(5)]
    out = decoder(*x)

    out[-1].mean().backward()

    assert out[-1].shape[1] == decoder.out_channels
