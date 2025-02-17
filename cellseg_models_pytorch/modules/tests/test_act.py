import pytest
import torch
from torch.autograd import gradcheck
from cellseg_models_pytorch.modules.act.swish import Swish
from cellseg_models_pytorch.modules.act.mish import Mish
from cellseg_models_pytorch.modules.act.gated_gelu import GEGLU

@pytest.mark.parametrize("dim_in, dim_out", [
    (64, 128),
    (128, 256),
])
def test_geglu(dim_in, dim_out):
    # Create the GEGLU layer
    geglu = GEGLU(dim_in=dim_in, dim_out=dim_out)

    # Create a random input tensor with the specified shape
    x = torch.rand((1, 32, dim_in))

    # Forward pass
    output = geglu(x)

    # Check the output shape
    assert output.shape == (1, 32, dim_out)

    # Check the output type
    assert isinstance(output, torch.Tensor)


@pytest.mark.parametrize("batch_size, num_features", [
    (1, 10),
    (2, 20),
])
def test_mish_fwdbwd(batch_size, num_features):
    # Create the Mish layer
    mish_layer = Mish()

    # Create a random input tensor with the specified shape
    x = torch.randn(batch_size, num_features, requires_grad=True)

    # Forward pass
    output = mish_layer(x)

    # Check the output shape
    assert output.shape == x.shape

    # Check the output type
    assert isinstance(output, torch.Tensor)

    # Backward pass
    output.sum().backward()
    assert x.grad is not None

    # Gradient check
    # assert gradcheck(mish, (x,), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("batch_size, num_features", [
    (1, 10),
    (2, 20),
])
def test_swish_fwdbwd(batch_size, num_features):
    # Create the Swish layer
    swish_layer = Swish()

    # Create a random input tensor with the specified shape
    x = torch.randn(batch_size, num_features, requires_grad=True)

    # Forward pass
    output = swish_layer(x)

    # Check the output shape
    assert output.shape == x.shape

    # Check the output type
    assert isinstance(output, torch.Tensor)

    # Backward pass
    output.sum().backward()
    assert x.grad is not None

    # Gradient check
    # assert gradcheck(swish, (x,))
