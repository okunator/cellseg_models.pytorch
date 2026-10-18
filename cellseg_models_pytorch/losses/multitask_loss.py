from typing import Dict

import torch
import torch.nn as nn

from .joint_loss import JointLoss


class MultiTaskLoss(nn.ModuleDict):
    """Sum losses for several named model outputs."""

    def __init__(
        self,
        head_losses: Dict[str, JointLoss],
        loss_weights: Dict[str, float] = None,
        **kwargs,
    ) -> None:
        """Assign a loss to each model output.

        Args:
            head_losses: Dictionary mapping output names to callable loss modules,
                usually JointLoss objects. Names are the training dictionaries' keys,
                for example cellpose and type_mask; extract tensors from network
                output containers before calling this loss.
            loss_weights: Optional weights with exactly the same keys as head_losses.
                None uses 1 for each output. Weights need not sum to one.
            **kwargs: Additional constructor arguments are accepted but unused.

        Raises:
            ValueError: Weight keys or counts differ from head_losses.
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
    ) -> torch.Tensor | float:
        """Compute and add the weighted losses for the configured outputs.

        Args:
            yhats: Dictionary of prediction tensors. Keys must exactly match targets
                and include every configured output; extra matching keys are not evaluated.
            targets: Dictionary of matching class-label or regression tensors. An
                edgeweight entry supplies spatial weights to all losses that enable
                edge weighting; it must also be present in yhats for the key check.
            mask: Optional BHW inclusion mask passed to each output's loss.
            **kwargs: Additional call arguments are accepted but unused.

        Returns:
            Weighted sum, normally a scalar tensor on the prediction device with
            gradients. With no configured outputs, returns the number 0.0.

        Raises:
            ValueError: Prediction and target dictionary keys differ. A missing
                configured output raises KeyError.
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
