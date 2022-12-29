import torch
import torch.nn as nn

__all__ = ["FixedUnpool"]


# Adapted from
# https://github.com/vqdang/hover_net/blob/master/models/hovernet/net_utils.py
class FixedUnpool(nn.Module):
    def __init__(self, scale_factor: int = 2) -> None:
        """Upsample input by a scale factor.

        TensorPack (library) fixed unpooling in pytorch.

        Parameters
        ----------
            scale_factor : int, default=2
                Upsampling scale factor. scale_factor*(H, W)
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.register_buffer(
            "unpool_mat", torch.ones([scale_factor, scale_factor], dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fixed-unpooling forward pass."""
        in_shape = list(x.shape)
        x = x.unsqueeze(-1)  # (B, C, H, W)
        mat = self.unpool_mat.unsqueeze(0)  # (B, C, H, W, 1)
        ret = torch.tensordot(x, mat, dims=1)  # (B, C, H, W, SH, SW)
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        ret = ret.reshape(
            (
                -1,
                in_shape[1],
                in_shape[2] * self.scale_factor,
                in_shape[3] * self.scale_factor,
            )
        )

        return ret

    def extra_repr(self):
        """Print output."""
        return f"scale_factor={self.scale_factor}"  # .format(**self.__dict__)
