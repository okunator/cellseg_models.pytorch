from typing import List

import torch
import torch.nn as nn


class JointLoss(nn.ModuleDict):
    """Sum several losses for the same prediction and target."""

    def __init__(
        self,
        losses: List[nn.Module],
        weights: List[float] = None,
    ) -> None:
        """Choose losses and their weights.

        Args:
            losses: Nonempty list of configured PyTorch loss modules. They must accept
                the same keyword inputs used when calling this object.
            weights: One number in [0, 1] per loss. None uses 1 for every loss.
                Supply exactly one per loss; mismatched lengths are not validated
                and unmatched entries would be ignored. Weights need not sum to one.

        Raises:
            ValueError: losses is not a list or a supplied weight is outside [0, 1].
        """
        super().__init__()

        if not isinstance(losses, list):
            raise ValueError("`losses` arg must be a list.")

        self.weights = [1.0] * len(losses)
        if weights is not None:
            if not all(0 <= val <= 1.0 for val in weights):
                raise ValueError(f"Weights need to be 0 <= weight <= 1. Got: {weights}")
            self.weights = weights

        for i in range(len(losses)):
            self.add_module(f"loss{i + 1}", losses[i])

    def forward(self, **kwargs) -> torch.Tensor:
        """Add the weighted losses.

        Args:
            **kwargs: Passed to every loss, typically yhat, target and optional
                target_weight/mask tensors. Each loss must accept these inputs.

        Returns:
            Sum of the weighted losses. For ordinary scalar losses, the result stays
            on the prediction device and keeps gradients for training.
        """
        lw = zip(self.values(), self.weights)
        losses = torch.stack([loss(**kwargs) * weight for loss, weight in lw])
        return torch.sum(losses)

    def extra_repr(self) -> str:
        """Add info to print."""
        s = "loss_weights={weights}"
        return s.format(**self.__dict__)
