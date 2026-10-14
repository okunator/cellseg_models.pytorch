import torch
import torch.nn.functional as F

from ..weighted_base_loss import WeightedBaseLoss

__all__ = ["DiceLoss"]


class DiceLoss(WeightedBaseLoss):
    def __init__(
        self,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        edge_weight: float | None = None,
        class_weights: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """Configure multiclass spatial Sørensen-Dice loss.

        Args:
            apply_sd: Add positive spectral-decoupling regularization.
            apply_ls: Smooth one-hot targets uniformly.
            apply_svls: Smooth targets using neighboring pixels.
            apply_mask: Exclude pixels with zero mask weight.
            edge_weight: Base raised to each target_weight exponent; weights
                spatial overlap and volume at object borders.
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
        **kwargs,
    ) -> torch.Tensor:
        """Compute spatial soft Dice loss from multiclass logits.

        Reduce over H and W separately for each image and class, then average
        losses over image/class pairs present in the target (including background).
        Absent target classes are excluded; their probabilities still compete
        with present classes through softmax. Class weights scale losses without
        renormalizing their mean. Mask and edge weights weight overlap and volume
        equally. A fully masked batch contributes zero, including regularization.

        Args:
            yhat: Floating BCHW logits on the same device as the target.
            target: BHW integer class labels in [0, C).
            target_weight: BHW edge exponents when edge weighting is enabled.
            mask: BHW nonnegative inclusion weights when masking is enabled.
            **kwargs: Arguments for the selected target smoothing method.

        Returns:
            Scalar differentiable loss on the logits' device; float32 for half-precision
            inputs, otherwise the logits' dtype.

        Raises:
            ValueError: Shapes disagree or enabled edge weighting has no map.
        """
        # Accumulate low-precision overlaps in float32 to avoid spatial-sum overflow.
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
                "Dice logits and one-hot targets must have matching BCHW shapes."
            )

        pixel_weights = torch.ones_like(target, dtype=probabilities.dtype)
        if self.apply_mask and mask is not None:
            if mask.shape != target.shape:
                raise ValueError("Dice mask must have the target's BHW shape.")
            pixel_weights = pixel_weights * mask
        present = (target_one_hot * pixel_weights.unsqueeze(1)).sum((2, 3)) > 0

        if self.edge_weight is not None:
            if target_weight is None or target_weight.shape != target.shape:
                raise ValueError(
                    "Dice edge weighting requires a BHW target_weight map."
                )
            pixel_weights = self.apply_edge_weights(pixel_weights, target_weight)

        if self.apply_svls:
            target_one_hot = self.apply_svls_to_target(
                target_one_hot, n_classes, **kwargs
            )
        if self.apply_ls:
            target_one_hot = self.apply_ls_to_target(
                target_one_hot, n_classes, **kwargs
            )

        weights = pixel_weights.unsqueeze(1)
        intersection = (probabilities * target_one_hot * weights).sum((2, 3))
        volume = ((probabilities + target_one_hot) * weights).sum((2, 3))
        losses = 1 - 2 * intersection / volume.clamp_min(self.eps)
        if self.class_weights is not None:
            if self.class_weights.shape != (n_classes,):
                raise ValueError("Dice class_weights must have shape (C,).")
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
