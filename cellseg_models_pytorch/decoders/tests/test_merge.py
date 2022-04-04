import pytest
import torch

from cellseg_models_pytorch.decoders.long_skips.merging import Merge


@pytest.mark.parametrize("name", ["sum", "cat", None])
def test_merge(name):
    t1 = torch.rand([1, 1, 16, 16])
    t2 = torch.rand([1, 2, 16, 16])
    t3 = torch.rand([1, 3, 16, 16])

    merge = Merge(name, in_channels=1, skip_channels=(2, 3))
    out = merge(t1, (t2, t3))

    assert out.shape[1] == merge.out_channels
