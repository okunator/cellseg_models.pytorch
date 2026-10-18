from typing import Any

import torch
import torch.nn.functional as F

from ..weighted_base_loss import WeightedBaseLoss

__all__ = ["TverskyLoss"]


class TverskyLoss(WeightedBaseLoss):
    """Compare class regions with separate false-positive and false-negative weights."""

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        edge_weight: float | None = None,
        class_weights: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure spatial Tversky loss for class masks.

        https://arxiv.org/abs/1706.05721

        Args:
            alpha: False-positive coefficient in the overlap denominator.
            beta: False-negative coefficient in the overlap denominator.
            apply_sd: Add 0.005 times the mask-weighted mean squared logit,
                independently of class and edge weights (spectral decoupling).
            apply_ls: Soften targets by sharing weight across classes.
            apply_svls: Smooth targets using neighboring pixels.
            apply_mask: Exclude pixels with zero mask weight.
            edge_weight: Base raised to each target_weight exponent; weights
                true positives, false positives and false negatives at object borders.
            class_weights: C-element weights scaling class losses without
                renormalizing the mean.
            **kwargs: Reserved for compatibility with the loss factory.
        """
        super().__init__(
            apply_sd, apply_ls, apply_svls, apply_mask, class_weights, edge_weight
        )
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-8

    def forward(
        self,
        yhat: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute spatial Tversky loss from multiclass logits.

        For each image and class, use 1 - TP / (TP + alpha * FP + beta * FN),
        summing soft true/false positives and false negatives across height and width.
        Average target-present image/class pairs, including background; absent
        classes still compete through softmax. Presence is determined before
        smoothing. Mask and edge weights apply once inside all three sums.
        A fully masked batch has zero loss and gradients, including regularization.

        Args:
            yhat: Floating BCHW logits on the target's device, before softmax.
            target: BHW integer class IDs in [0, C).
            target_weight: BHW edge exponents when edge weighting is enabled.
            mask: BHW nonnegative inclusion weights when masking is enabled.
            **kwargs: Options for smoothing, such as label_smoothing,
                kernel_size and sigma.

        Returns:
            Scalar differentiable loss on the logits' device. Float16 and bfloat16
            logits use float32 overlap and regularization arithmetic.

        Raises:
            ValueError: Shapes disagree or enabled edge weighting has no map.

        Examples:
            >>> logits = torch.zeros(1, 2, 1, 2)
            >>> target = torch.tensor([[[0, 1]]])
            >>> round(TverskyLoss()(logits, target).item(), 6)
            0.5
        """
        dtype = (
            torch.float32
            if yhat.dtype in (torch.float16, torch.bfloat16)
            else yhat.dtype
        )
        probabilities = F.softmax(yhat, dim=1, dtype=dtype)
        n_classes = yhat.shape[1]
        target_one_hot = F.one_hot(target.long(), n_classes).permute(0, 3, 1, 2)
        if target_one_hot.shape != yhat.shape:
            raise ValueError(
                "Tversky logits and one-hot targets must have matching BCHW shapes."
            )

        pixel_weights = torch.ones_like(target, dtype=dtype)
        if self.apply_mask and mask is not None:
            if mask.shape != target.shape:
                raise ValueError("Tversky mask must have the target's BHW shape.")
            pixel_weights = pixel_weights * mask
        if self.edge_weight is not None:
            if target_weight is None or target_weight.shape != target.shape:
                raise ValueError(
                    "Tversky edge weighting requires a BHW target_weight map."
                )
            pixel_weights = self.apply_edge_weights(pixel_weights, target_weight)
        present = (target_one_hot * pixel_weights.unsqueeze(1)).sum((2, 3)) > 0

        if self.apply_svls:
            target_one_hot = self.apply_svls_to_target(
                target_one_hot, n_classes, **kwargs
            )

        if self.apply_ls:
            target_one_hot = self.apply_ls_to_target(
                target_one_hot,
                n_classes,
                label_smoothing=kwargs.get("label_smoothing", 0.1),
            )

        target_one_hot = target_one_hot.to(dtype=dtype)
        weights = pixel_weights.unsqueeze(1)
        intersection = (probabilities * target_one_hot * weights).sum((2, 3))
        fps = (probabilities * (1 - target_one_hot) * weights).sum((2, 3))
        fns = ((1 - probabilities) * target_one_hot * weights).sum((2, 3))
        denom = intersection + self.alpha * fps + self.beta * fns
        losses = 1 - intersection / denom.clamp_min(self.eps)
        if self.class_weights is not None:
            if self.class_weights.shape != (n_classes,):
                raise ValueError("Tversky class_weights must have shape (C,).")
            losses = losses * self.class_weights
        loss = (losses * present).sum() / present.sum().clamp_min(1)

        if self.apply_sd:
            regularization = self.apply_spectral_decouple(
                torch.zeros_like(target), yhat.to(dtype=dtype)
            )
            inclusion = (
                mask.to(dtype=dtype)
                if self.apply_mask and mask is not None
                else torch.ones_like(target)
            )
            loss = loss + (
                regularization * inclusion
            ).sum() / inclusion.sum().clamp_min(self.eps)

        return loss
