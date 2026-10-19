"""Check style-vector normalization and its derivatives independently."""

import pytest
import torch

from cellseg_models_pytorch.models.cellpose.cellpose_unet import cellpose_nuclei
from cellseg_models_pytorch.modules.misc_modules import StyleReshape

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("case", ["zero", "cancelled", "mixed", "small"])
def test_style_values_and_input_gradients_match_analytic_normalization(
    device, dtype, autocast, case
):
    vectors = torch.tensor(
        [[0, 0, 0], [3, 4, 0], [-2, 1, 2]], device=device, dtype=dtype
    )
    if case in ("zero", "cancelled"):
        vectors.zero_()
    elif case == "small":
        vectors = vectors / 65536
    features = vectors[:, :, None, None].expand(-1, -1, 2, 3).clone()
    if case == "cancelled":
        # Nonzero features can still have a zero spatial average.
        features = features + features.new_tensor([-1, -0.5, 0, 0, 0.5, 1]).reshape(
            1, 1, 2, 3
        )
    features = features.transpose(2, 3).requires_grad_()
    before = features.detach().clone()
    layer = StyleReshape(3, 3).to(device)
    mean = features.detach().double().mean((2, 3))
    lengths = mean.square().sum(1, keepdim=True).sqrt()
    denominator = torch.where(lengths > 0, lengths, torch.ones_like(lengths))
    expected = mean / denominator
    weights = mean.new_tensor([[0.25, -0.5, 1], [-1, 0.5, 0.25], [1, -0.25, 0.5]])
    # For a nonzero vector, remove the gradient component parallel to its direction.
    vector_gradient = (
        weights - expected * (expected * weights).sum(1, keepdim=True)
    ) / denominator
    expected_gradient = vector_gradient[:, :, None, None].expand_as(features) / 6
    amp_dtype = (
        torch.bfloat16 if device == "cpu" or dtype == torch.bfloat16 else torch.float16
    )
    with torch.autocast(device, dtype=amp_dtype, enabled=autocast):
        actual = layer(features)
    expected_dtype = (
        torch.float32
        if device == "cuda" and autocast and dtype in (torch.float16, torch.bfloat16)
        else dtype
    )
    assert actual.dtype == expected_dtype
    assert actual.device == features.device
    assert actual.shape == (3, 3)
    assert torch.isfinite(actual).all()
    tolerance = {
        torch.float64: 1e-12,
        torch.float32: 1e-6,
        torch.float16: 0.003,
        torch.bfloat16: 0.02,
    }[dtype]
    torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=1e-7)
    (actual * weights.to(actual.dtype)).sum().backward()
    assert torch.isfinite(features.grad).all()
    torch.testing.assert_close(
        features.grad.double(), expected_gradient, rtol=tolerance, atol=1e-7
    )
    torch.testing.assert_close(features, before, rtol=0, atol=0)
    if case in ("zero", "cancelled"):
        assert torch.count_nonzero(actual) == 0
        # With no direction to normalize, the zero-vector branch leaves gradients unscaled.
        torch.testing.assert_close(
            features.grad,
            (weights[:, :, None, None].expand_as(features) / 6).to(dtype),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("normalization", [None, "bn"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_projected_zero_features_have_finite_values_and_gradients(
    device, normalization, training, autocast
):
    torch.manual_seed(9)
    layer = StyleReshape(4, 2, normalization=normalization).to(device).train(training)
    keys = set(layer.state_dict())
    features = torch.zeros(2, 4, 2, 3, device=device, requires_grad=True)
    amp_dtype = torch.bfloat16 if device == "cpu" else torch.float16
    with torch.autocast(device, dtype=amp_dtype, enabled=autocast):
        actual = layer(features)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
    actual.float().sum().backward()
    assert torch.isfinite(features.grad).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters()
    )
    assert set(layer.state_dict()) == keys
    restored = (
        StyleReshape(4, 2, normalization=normalization).to(device).train(training)
    )
    restored.load_state_dict(layer.state_dict(), strict=True)
    with torch.autocast(device, dtype=amp_dtype, enabled=autocast):
        torch.testing.assert_close(restored(features.detach()), actual, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_nonzero_style_gradients_match_finite_differences(device):
    features = (
        torch.linspace(-0.7, 1.6, 36, device=device, dtype=torch.float64)
        .reshape(2, 3, 2, 3)
        .requires_grad_()
    )
    assert torch.autograd.gradcheck(StyleReshape(3, 3).to(device), (features,))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_cellpose_zero_image_has_finite_maps_and_training_gradients(
    device, training, autocast
):
    torch.manual_seed(11)
    network = (
        cellpose_nuclei(
            3,
            enc_name="resnet18",
            enc_pretrain=False,
            style_channels=8,
            out_channels=(16, 8, 4, 2),
            layer_depths=(1, 1, 1, 1),
        )
        .to(device)
        .train(training)
    )
    keys = set(network.state_dict())
    image = torch.zeros(2, 3, 32, 32, device=device, requires_grad=True)
    amp_dtype = torch.bfloat16 if device == "cpu" else torch.float16
    with torch.autocast(device, dtype=amp_dtype, enabled=autocast):
        output = network(image)["nuc"]
        loss = output.aux_map.float().mean() + output.type_map.float().mean()
    assert torch.isfinite(output.aux_map).all()
    assert torch.isfinite(output.type_map).all()
    loss.backward()
    assert torch.isfinite(image.grad).all()
    gradients = [p.grad for p in network.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
    assert set(network.state_dict()) == keys
