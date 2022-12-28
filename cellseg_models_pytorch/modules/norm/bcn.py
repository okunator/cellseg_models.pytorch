import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


# Adapted from
# https://github.com/joe-siyuan-qiao/Batch-Channel-Normalization
class BCNorm(nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-7, estimate: bool = False
    ) -> None:
        """Batch channel normalization.

        https://arxiv.org/abs/1911.09738

        Infers the num_groups from the num_features to avoid
        errors. By default: uses 16 channels per group.
        If channels <= 16, squashes to batch layer norm

        magic number 16 comes from the paper:
        https://arxiv.org/abs/1803.08494

        Parameters
        ----------
            num_features : int
                Number of input channels/features.
            num_groups : int, default=None
                Number of groups to group the channels.
            eps : float, default=1e-7
                Small constant for numerical stability.
            estimate : bool, default=False
                If True, Uses `EstBN` instead of BN.
        """
        super().__init__()

        # Infer number of groups
        self.num_groups, remainder = divmod(num_features, 16)
        if remainder:
            self.num_groups = num_features // remainder

        self.num_features = num_features
        self.eps = eps
        self.weight = Parameter(torch.ones(1, self.num_groups, 1))
        self.bias = Parameter(torch.zeros(1, self.num_groups, 1))
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Batch-channel norm forward pass."""
        out = self.bn(x)
        out = out.reshape(1, x.size(0) * self.num_groups, -1)
        out = torch.batch_norm(out, None, None, None, None, True, 0, self.eps, True)
        out = out.reshape(x.size(0), self.num_groups, -1)
        out = self.weight * out + self.bias
        out = out.reshape_as(x)

        return out
