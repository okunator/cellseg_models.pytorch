import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

# https://github.com/joe-siyuan-qiao/Batch-Channel-Normalization
# class EstBN(nn.Module):
#     def __init__(self, num_features: int, eps: float = 1e-7) -> None:
#         """Estimate of the batch statistics.

#         https://arxiv.org/abs/1911.09738

#         Parameters
#         ----------
#             num_features : int
#                 Number of input channels/features.
#             eps : float, default=1e-7
#                 Small constant for numerical stability
#         """
#         super().__init__()
#         self.num_features = num_features
#         self.weight = Parameter(torch.ones(num_features))
#         self.bias = Parameter(torch.zeros(num_features))
#         self.register_buffer("running_mean", torch.zeros(num_features))
#         self.register_buffer("running_var", torch.ones(num_features))
#         self.register_buffer("estbn_moving_speed", torch.zeros(1))
#         self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
#         self.eps = eps

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Batch stat estimation forward pass."""
#         ms = self.estbn_moving_speed.item()
#         if self.training:
#             with torch.no_grad():
#                 xt = x.transpose(0, 1).contiguous().view(self.num_features, -1)
#                 running_mean = xt.mean(dim=1)
#                 xt -= self.running_mean.view(-1, 1)
#                 running_var = torch.mean(xt * xt, dim=1)
#                 self.running_mean.data.mul_(1 - ms).add_(ms * running_mean.data)
#                 self.running_var.data.mul_(1 - ms).add_(ms * running_var.data)

#         out = x - self.running_mean.view(1, -1, 1, 1)
#         out = out / torch.sqrt(self.running_var + self.eps).view(1, -1, 1, 1)
#         weight = self.weight.view(1, -1, 1, 1)
#         bias = self.bias.view(1, -1, 1, 1)
#         out = weight * out + bias

#         return out


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
        out = out.view(1, x.size(0) * self.num_groups, -1)
        out = torch.batch_norm(out, None, None, None, None, True, 0, self.eps, True)
        out = out.view(x.size(0), self.num_groups, -1)
        out = self.weight * out + self.bias
        out = out.view_as(x)

        return out
