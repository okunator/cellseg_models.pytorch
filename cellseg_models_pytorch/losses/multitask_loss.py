from typing import Dict

import torch
import torch.nn as nn

from .joint_loss import JointLoss


class MultiTaskLoss(nn.ModuleDict):
    def __init__(
        self,
        branch_losses: Dict[str, JointLoss],
        loss_weights: Dict[str, float] = None,
        **kwargs,
    ) -> None:
        """Multi-task loss wrapper.

        Combines losses from different branches to one loss function.

        Parameters
        ----------
            branch_losses : Dict[str, nn.Module]
                Dictionary of branch names mapped to a loss module.
                e.g. {"inst": JointLoss(MSE(), Dice()), "type": Dice()}.
            loss_weights : Dict[str, float], default=None
                Dictionary of branch names mapped to the weight used for
                that branch loss.

        Raises
        ------
            ValueError:
                If the input arguments have different lengths.
                If the input arguments have mismatching keys.
        """
        super().__init__()

        self.weights = {k: 1.0 for k in branch_losses.keys()}
        if loss_weights is not None:
            if len(loss_weights) != len(branch_losses):
                raise ValueError(
                    f"""
                    Got {len(loss_weights)} loss weights and {len(branch_losses)}
                    branches. Need to have the same length."""
                )
            if not all(k in branch_losses.keys() for k in loss_weights.keys()):
                raise ValueError(
                    """Mismatching keys in `loss_weights` and `branch_losses`"""
                )
            else:
                self.weights = loss_weights

        for branch, loss in branch_losses.items():
            self.add_module(f"{branch}_loss", loss)

    def forward(
        self, yhats: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """Compute the joint loss of the multi-task network.

        Parameters
        ----------
            yhats : Dict[str, torch.Tensor]
                Dictionary of branch names mapped to the predicted masks.
            targets : Dict[str, torch.Tensor]
                Dictionary of branch names mapped to the GT masks.

        Returns
        -------
            torch.Tensor:
                Computed multi-task loss (Scalar).
        """
        weight_map = None
        if "edgeweight" in targets.keys():
            weight_map = targets["edgeweight"]

        multitask_loss = 0.0
        for branch, loss in self.items():
            branch = branch.split("_")[0]
            branch_loss = loss(
                yhat=yhats[branch],
                target=targets[branch],
                target_weight=weight_map,
            )
            multitask_loss += branch_loss * self.weights[branch]

        return multitask_loss

    def extra_repr(self) -> str:
        """Add info to print."""
        s = "branch_loss_weights={weights}"
        return s.format(**self.__dict__)
