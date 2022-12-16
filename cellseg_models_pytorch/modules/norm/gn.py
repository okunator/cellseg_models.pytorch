import torch.nn as nn


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_features: int, num_groups: int = None, **kwargs) -> None:
        """Wrap nn.Groupnorm class.

        To make kwargs compatible with nn.BatchNorm.

        Also, infers the `num_groups` from the num_features to avoid
        errors. By default: uses 16 channels per group. If channels <= 16,
        squashes to layer norm.

        Magic number 16 comes from the paper:
        https://arxiv.org/abs/1803.08494

        Parameters
        ----------
            num_features : int
                Number of input channels/features.
            num_groups : int, default=None
                Number of groups to group the channels.
        """
        if num_groups is None:
            num_groups, remainder = divmod(num_features, 16)
            if remainder:
                num_groups = num_features // remainder

        super().__init__(num_groups=num_groups, num_channels=num_features)
