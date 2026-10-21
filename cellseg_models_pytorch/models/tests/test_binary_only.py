"""Exercise networks with foreground heads and no cell-type predictions."""

import pytest
import torch

from cellseg_models_pytorch.models.cellpose.cellpose_unet import CellPoseUnet
from cellseg_models_pytorch.models.stardist.stardist_unet import StarDistUnet


@pytest.fixture(params=["cellpose", "stardist"])
def network_factory(request):
    def create(typed_heads=()):
        family = request.param
        heads = {}
        for name in ("nuc", "cyto"):
            heads[f"{name}_{family}"] = 2 if family == "cellpose" else 8
            heads[f"{name}_binary"] = 1
            if name in typed_heads:
                heads[f"{name}_type"] = 3
        model_class = CellPoseUnet if family == "cellpose" else StarDistUnet
        kwargs = {"layer_depths": (1, 1, 1)} if family == "cellpose" else {}
        return model_class(
            decoders=("objects",),
            heads={"objects": heads},
            depth=3,
            out_channels=(16, 8, 4),
            style_channels=None,
            enc_name="resnet18",
            enc_pretrain=False,
            **kwargs,
        ).cpu()

    return create


@pytest.mark.parametrize("typed_heads", [(), ("nuc",), ("cyto",), ("nuc", "cyto")])
def test_binary_only_forward_and_gradients(network_factory, typed_heads):
    torch.manual_seed(0)
    network = network_factory(typed_heads).eval()
    image = torch.rand(1, 3, 32, 32)

    outputs = network(image)
    loss = torch.zeros(())
    for name in ("nuc", "cyto"):
        maps = outputs[name]
        assert maps.aux_map.shape[-2:] == (32, 32)
        assert maps.binary_map.shape == (1, 1, 32, 32)
        assert maps.parents["binary_map"] == ["objects", f"{name}_binary"]
        if name in typed_heads:
            assert maps.type_map.shape == (1, 3, 32, 32)
            assert maps.parents["type_map"] == ["objects", f"{name}_type"]
            loss = loss + maps.type_map.square().mean()
        else:
            assert maps.type_map is None and maps.parents["type_map"] is None
        loss = loss + maps.aux_map.square().mean() + maps.binary_map.square().mean()

    loss.backward()

    # Every configured output head must still train, including foreground heads.
    for name, parameter in network.decoder.named_parameters():
        if "_head." in name:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert torch.count_nonzero(parameter.grad), name
    assert torch.isfinite(next(network.parameters()).grad).all()
    assert outputs["tissue"] is None


@pytest.mark.parametrize("typed_heads", [(), ("nuc", "cyto")])
def test_binary_only_checkpoint_roundtrip(network_factory, typed_heads, tmp_path):
    torch.manual_seed(1)
    original = network_factory(typed_heads).eval()
    image = torch.rand(1, 3, 32, 32)
    checkpoint = tmp_path / "weights.pth"
    torch.save(original.state_dict(), checkpoint)
    restored = network_factory(typed_heads).eval()

    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)

    assert restored.state_dict().keys() == original.state_dict().keys()
    with torch.inference_mode():
        expected, actual = original(image), restored(image)
    for name in ("nuc", "cyto"):
        for field in ("type_map", "aux_map", "binary_map"):
            reference, result = (
                getattr(expected[name], field),
                getattr(actual[name], field),
            )
            if reference is None:
                assert result is None
            else:
                torch.testing.assert_close(result, reference, rtol=0, atol=0)
        has_type_weights = (
            f"decoder.objects-{name}_type_head.head.weight" in restored.state_dict()
        )
        assert has_type_weights == (name in typed_heads)
