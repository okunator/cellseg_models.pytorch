import torch
import torch.nn as nn


class MultiTaskLoss(nn.ModuleDict):
    """Sum losses for several named model outputs."""

    def __init__(
        self,
        head_losses: dict[str, nn.Module],
        loss_weights: dict[str, float] | None = None,
        **kwargs: object,
    ) -> None:
        """Assign a loss to each model output.

        Args:
            head_losses: Dictionary mapping output names to callable loss modules,
                usually JointLoss objects. Names are the training dictionaries' keys,
                for example cellpose and type_mask; extract tensors from network
                output containers before calling this loss. edgeweight is reserved
                for target metadata rather than a prediction head.
            loss_weights: Optional weights with exactly the same keys as head_losses.
                None uses 1 for each output. Weights need not sum to one.
            **kwargs: Additional constructor arguments are accepted but unused.

        Raises:
            ValueError: Weight keys or counts differ from head_losses, or a head
                uses the reserved edgeweight name.
        """
        super().__init__()

        if "edgeweight" in head_losses:
            raise ValueError("edgeweight is reserved for target metadata")

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
        yhats: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | float:
        """Compute and add the weighted losses for the configured outputs.

        Args:
            yhats: Dictionary of prediction tensors. Head keys must match targets
                and include every configured output; extra matching heads are not
                evaluated. A legacy dummy edgeweight entry is accepted when the
                real map is supplied in targets.
            targets: Dictionary of matching class-label or regression tensors. An
                edgeweight entry supplies spatial weights to the child losses;
                no dummy prediction is needed. Each child must accept target_weight.
            mask: Optional BHW inclusion mask passed to each output's loss.
            **kwargs: Additional call arguments are accepted but unused.

        Returns:
            Weighted sum, normally a scalar tensor on the prediction device with
            gradients. With no configured outputs, returns the number 0.0.

        Raises:
            ValueError: Prediction/target head keys or configured weight keys
                disagree. A missing configured output raises KeyError before
                any child loss runs.
        """
        if self.weights.keys() != self.keys():
            raise ValueError("loss weight keys must match the configured heads")
        if "edgeweight" in yhats and "edgeweight" not in targets:
            raise ValueError("edgeweight must be supplied in targets")
        # Edge weights describe the targets, not an additional model output.
        prediction_heads = yhats.keys() - {"edgeweight"}
        target_heads = targets.keys() - {"edgeweight"}
        if prediction_heads != target_heads:
            raise ValueError(
                "Mismatching keys in `yhats` and `targets`"
                f"Got {list(yhats.keys())} and {list(targets.keys())}"
            )

        missing = self.keys() - prediction_heads
        if missing:
            raise KeyError(f"Missing configured loss heads: {sorted(missing)}")

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
            # Out-of-place addition preserves precision when head dtypes differ.
            multitask_loss = multitask_loss + head_loss * self.weights[head]

        return multitask_loss

    def extra_repr(self) -> str:
        """Add info to print."""
        s = "head_loss_weights={weights}"
        return s.format(**self.__dict__)
