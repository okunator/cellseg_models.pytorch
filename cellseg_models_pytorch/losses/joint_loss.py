from typing import List

import torch
import torch.nn as nn


class JointLoss(nn.ModuleDict):
    def __init__(
        self,
        losses: List[nn.Module],
        weights: List[float] = None,
    ) -> None:
        """Joint loss function.

        Takes in a list of nn.Module losses and computes the loss for
        each loss in the list and at the end sums the outputs together
        as one joint loss.

        Parameters:
            losses (List[nn.Module]):
                List of initialized nn.Module losses.
            weights (List[float], default=None):
                List of weights for each loss.

        Raises:
            ValueError:
                If more than 4 losses are given as input.
                If given weights are not between [0, 1].
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
        """Compute the joint-loss.

        Returns:
            torch.Tensor:
                The computed joint loss.
        """
        lw = zip(self.values(), self.weights)
        losses = torch.stack([loss(**kwargs) * weight for loss, weight in lw])
        return torch.sum(losses)

    def extra_repr(self) -> str:
        """Add info to print."""
        s = "loss_weights={weights}"
        return s.format(**self.__dict__)
