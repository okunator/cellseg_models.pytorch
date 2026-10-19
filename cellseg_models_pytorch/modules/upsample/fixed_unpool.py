import torch
import torch.nn as nn

__all__ = ["FixedUnpool"]


# Adapted from
# https://github.com/vqdang/hover_net/blob/master/models/hovernet/net_utils.py
class FixedUnpool(nn.Module):
    def __init__(self, scale_factor: int = 2) -> None:
        """Upsample input by a scale factor.

        TensorPack (library) fixed unpooling in pytorch.

        Args:
            scale_factor: Upsampling scale factor. scale_factor*(H, W)
                Defaults to 2.
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.register_buffer(
            "unpool_mat", torch.ones([scale_factor, scale_factor], dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand each feature pixel into a fixed spatial block.

        Args:
            x: Floating BCHW features on the same device as this layer.

        Returns:
            Features shaped (B, C, H * scale_factor, W * scale_factor) on the
            input device. Preserves input dtype outside autocast; autocast can
            reduce the output precision.
        """
        in_shape = list(x.shape)
        x = x.unsqueeze(-1)  # (B, C, H, W, 1)
        # Autocast can change feature dtype without changing the saved buffer.
        mat = self.unpool_mat.to(dtype=x.dtype).unsqueeze(0)
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
