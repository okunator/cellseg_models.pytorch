from typing import Any

import torch
import torch.nn.functional as F

from ..weighted_base_loss import WeightedBaseLoss

__all__ = ["IoULoss"]


class IoULoss(WeightedBaseLoss):
    """Compare predicted and target class regions using spatial soft IoU."""

    def __init__(
        self,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        edge_weight: float | None = None,
        class_weights: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure IoU loss for class masks.

        Args:
            apply_sd: Add 0.005 times the mask-weighted mean squared logit
                (spectral decoupling), independently of class and edge weights.
            apply_ls: Soften target labels by sharing a small weight across all classes.
            apply_svls: Smooth targets using neighboring pixels.
            apply_mask: Exclude pixels with zero mask weight.
            edge_weight: Base raised to each target_weight exponent; weights
                spatial intersection and union at object borders.
            class_weights: Nonnegative C-element weights scaling class losses.
            **kwargs: Reserved for compatibility with the loss factory.
        """
        super().__init__(
            apply_sd, apply_ls, apply_svls, apply_mask, class_weights, edge_weight
        )
        self.eps = 1e-8

    def forward(
        self,
        yhat: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute spatial soft IoU loss from multiclass logits.

        Calculate soft intersection over union separately for each image and class,
        then average target-present image/class pairs, including background.
        Absent target classes are excluded; their probabilities still compete
        with present classes through softmax. Class weights scale losses without
        renormalizing their mean. Mask and edge weights apply once to both
        intersection and union. A fully masked batch contributes zero, including
        regularization.

        Args:
            yhat: Floating BCHW logits on the same device as the target.
            target: BHW integer class labels in [0, C).
            target_weight: BHW edge exponents when edge weighting is enabled.
            mask: BHW nonnegative inclusion weights when masking is enabled.
            **kwargs: Arguments for the selected target smoothing method.

        Returns:
            Scalar differentiable loss on the logits' device. Half-precision logits
            use float32 overlap accumulation.

        Raises:
            ValueError: Shapes disagree or enabled edge weighting has no map.

        Examples:
            >>> logits = torch.zeros(1, 2, 1, 2)
            >>> target = torch.tensor([[[0, 1]]])
            >>> round(IoULoss()(logits, target).item(), 6)
            0.666667
        """
        # Pixel counts can overflow float16 on large images; sum them in float32.
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
                "IoU logits and one-hot targets must have matching BCHW shapes."
            )

        pixel_weights = torch.ones_like(target, dtype=probabilities.dtype)
        if self.apply_mask and mask is not None:
            if mask.shape != target.shape:
                raise ValueError("IoU mask must have the target's BHW shape.")
            pixel_weights = pixel_weights * mask

        if self.edge_weight is not None:
            if target_weight is None or target_weight.shape != target.shape:
                raise ValueError("IoU edge weighting requires a BHW target_weight map.")
            pixel_weights = self.apply_edge_weights(pixel_weights, target_weight)
        # Smoothing must not make an absent class count as present in the target.
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

        # Weight overlap and total area equally, then compare each image/class pair.
        weights = pixel_weights.unsqueeze(1)
        intersection = (probabilities * target_one_hot * weights).sum((2, 3))
        union = ((probabilities + target_one_hot) * weights).sum((2, 3)) - intersection
        losses = 1 - intersection / union.clamp_min(self.eps)
        if self.class_weights is not None:
            if self.class_weights.shape != (n_classes,):
                raise ValueError("IoU class_weights must have shape (C,).")
            losses = losses * self.class_weights
        # Average only included classes; a fully masked batch contributes zero.
        loss = (losses * present).sum() / present.sum().clamp_min(1)

        if self.apply_sd:
            # Mask this penalty too, while keeping it separate from class/edge weights.
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
