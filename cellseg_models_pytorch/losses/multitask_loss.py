from typing import Dict

import torch
import torch.nn as nn

from .joint_loss import JointLoss


class MultiTaskLoss(nn.ModuleDict):
    def __init__(
        self,
        head_losses: Dict[str, JointLoss],
        loss_weights: Dict[str, float] = None,
        **kwargs,
    ) -> None:
        """Multi-task loss wrapper.

        Combines losses from different heades to one loss function.

        Parameters:
            head_losses (Dict[str, nn.Module]):
                Dictionary of head names mapped to a loss module.
                e.g. {"inst": JointLoss(MSE(), Dice()), "type": Dice()}.
            loss_weights (Dict[str, float], default=None):
                Dictionary of head names mapped to the weight used for
                that head loss.

        Raises:
            ValueError:
                If the input arguments have different lengths.
                If the input arguments have mismatching keys.
        """
        super().__init__()

        self.weights = {k: 1.0 for k in head_losses.keys()}
        if loss_weights is not None:
            if len(loss_weights) != len(head_losses):
                raise ValueError(
                    f"""
                    Got {len(loss_weights)} loss weights and {len(head_losses)}
                    heads. Need to have the same length."""
                )
            if not all(k in head_losses.keys() for k in loss_weights.keys()):
                raise ValueError(
                    """Mismatching keys in `loss_weights` and `head_losses`"""
                )
            else:
                self.weights = loss_weights

        for head, loss in head_losses.items():
            self.add_module(head, loss)

    def forward(
        self,
        yhats: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the joint loss of the multi-task network.

        Parameters:
            yhats (Dict[str, torch.Tensor]):
                Dictionary of head names mapped to the predicted masks.
                e.g. {"inst": (B, C, H, W), "type": (B, C, H, W)}.
            targets (Dict[str, torch.Tensor]):
                Dictionary of head names mapped to the GT masks.
                e.g. {"inst": (B, C, H, W), "type": (B, C, H, W)}.
            mask (torch.Tensor, default=None):
                The mask for masked losses. Shape (B, H, W).

        Returns:
            torch.Tensor: Computed multi-task loss (Scalar).
        """
        if yhats.keys() != targets.keys():
            raise ValueError(
                "Mismatching keys in `yhats` and `targets`"
                f"Got {list(yhats.keys())} and {list(targets.keys())}"
            )

        weight_map = None
        if "edgeweight" in targets.keys():
            weight_map = targets["edgeweight"]

        multitask_loss = 0.0
        for head, loss in self.items():
            head_loss = loss(
                yhat=yhats[head],
                target=targets[head],
                target_weight=weight_map,
                mask=mask,
            )
            multitask_loss += head_loss * self.weights[head]

        return multitask_loss

    def extra_repr(self) -> str:
        """Add info to print."""
        s = "head_loss_weights={weights}"
        return s.format(**self.__dict__)
