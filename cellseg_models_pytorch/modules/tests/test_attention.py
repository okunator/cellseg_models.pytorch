import pytest
import torch

from cellseg_models_pytorch.modules.attention_modules import Attention


@pytest.mark.parametrize("attention", ["se", "scse", "eca", "gc"])
def test_attention_forward(attention):
    att = Attention(attention, in_channels=8)
    input = torch.rand([1, 8, 16, 16])
    output = att(input)

    assert output.shape == input.shape
    assert output.dtype == input.dtype


@pytest.mark.parametrize("attention", ["se", "scse", "eca", "gc", None])
def test_attention_backward(attention):
    att = Attention(attention, in_channels=8)
    input = torch.rand([1, 8, 16, 16])
    output = att(input)

    assert output.shape == input.shape
    assert output.dtype == input.dtype
