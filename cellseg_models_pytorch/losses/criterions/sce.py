from typing import Any

import torch
import torch.nn.functional as F

from ..weighted_base_loss import WeightedBaseLoss

__all__ = ["SCELoss"]


class SCELoss(WeightedBaseLoss):
    """Combine forward and reverse multiclass cross-entropy.

    Inputs must share a device. B is batch size, C channels, H height and W width.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 1.0,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        edge_weight: float | None = None,
        class_weights: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        """Combine forward and reverse multiclass cross-entropy.

        https://arxiv.org/abs/1908.06112

        Args:
            alpha: Scale on the forward cross-entropy term.
            beta: Scale on the reverse cross-entropy term.
            apply_sd: Add 0.005 times the mean squared prediction across channels to
                the per-pixel loss.
            apply_ls: Mix the target with a uniform class distribution.
            apply_svls: Smooth targets using neighboring pixels.
            apply_mask: Multiply pixel losses by the supplied mask. The average still
                includes all pixels, including those with zero mask weight.
            edge_weight: Multiply pixel losses by edge_weight ** target_weight.
                Supply that map when enabled; None disables this weighting.
            class_weights: Per-class multiplier tensor of shape (C,), on the prediction device.
                Use with integer class labels.
            **kwargs: Additional constructor arguments are accepted but unused.
        """
        super().__init__(
            apply_sd, apply_ls, apply_svls, apply_mask, class_weights, edge_weight
        )
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-4

    def forward(
        self,
        yhat: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the loss for one batch.

        Masking multiplies per-pixel losses before the full-image mean; it does
        not divide by the number of included pixels. Spectral regularization is
        added after masking; class and edge weights scale the resulting pixel losses.
        Reverse cross-entropy floors targets at 1e-4 inside the logarithm, following
        the authors' reference implementation, so hard-label reverse terms stay finite.

        Args:
            yhat: Floating BCHW class logits.
            target: Integer BHW class IDs in [0, C).
            target_weight: Optional BHW edge exponents when edge weighting is enabled.
            mask: Optional BHW inclusion weights, used only with apply_mask=True.
            **kwargs: Forward options for enabled smoothing methods, such as
                label_smoothing, kernel_size and sigma; unused otherwise.

        Returns:
            A single loss value on the prediction device, with gradients for training.
            Combine channel errors first, then average across images and pixels.
            Float16 and bfloat16 logits use float32 loss and regularization arithmetic.

        Examples:
            >>> logits = torch.zeros(1, 2, 1, 1)
            >>> target = torch.zeros(1, 1, 1, dtype=torch.long)
            >>> round(SCELoss()(logits, target).item(), 6)
            4.951744
        """
        # Keep large scores and their squared penalty from overflowing float16.
        dtype = (
            torch.float32
            if yhat.dtype in (torch.float16, torch.bfloat16)
            else yhat.dtype
        )
        logits = yhat.to(dtype=dtype)
        n_classes = yhat.shape[1]
        target_one_hot = F.one_hot(target.long(), n_classes).permute(0, 3, 1, 2)
        log_probabilities = F.log_softmax(logits, dim=1)
        assert target_one_hot.shape == yhat.shape

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
        # Ignore zero target entries before multiplying, since 0 * -inf would be NaN.
        forward = target_one_hot * log_probabilities.masked_fill(target_one_hot == 0, 0)
        # The reverse term swaps prediction/target roles; the floor avoids log(0).
        reverse = log_probabilities.exp() * target_one_hot.clamp_min(self.eps).log()

        cross_entropy = -torch.sum(forward, dim=1)  # to (B, H, W)
        reverse_cross_entropy = -torch.sum(reverse, dim=1)  # to (B, H, W)
        loss = self.alpha * cross_entropy + self.beta * reverse_cross_entropy

        if self.apply_mask and mask is not None:
            loss = self.apply_mask_weight(loss, mask, norm=False)  # (B, H, W)

        if self.apply_sd:
            # This penalty also applies outside the mask, before class/edge weighting.
            loss = self.apply_spectral_decouple(loss, logits)

        if self.class_weights is not None:
            loss = self.apply_class_weights(loss, target)

        if self.edge_weight is not None:
            loss = self.apply_edge_weights(loss, target_weight)

        return loss.mean()
