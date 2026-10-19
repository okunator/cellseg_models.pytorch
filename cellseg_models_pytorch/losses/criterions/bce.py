from typing import Any

import torch
import torch.nn.functional as F

from cellseg_models_pytorch.losses.weighted_base_loss import WeightedBaseLoss

__all__ = ["BCELoss"]


class BCELoss(WeightedBaseLoss):
    """Compute binary cross-entropy independently for each output channel.

    Inputs must share a device. B is batch size, C channels, H height and W width.
    """

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
        """Configure binary cross-entropy from raw logits.

        Args:
            apply_sd: Add 0.005 times the mean squared prediction across channels to
                the per-pixel loss.
            apply_ls: Smooth binary targets toward 0.5.
            apply_svls: Smooth targets using neighboring pixels.
            apply_mask: Multiply pixel losses by the supplied mask. The average still
                includes all pixels, including those with zero mask weight.
            edge_weight: Multiply pixel losses by edge_weight ** target_weight.
                Supply that map when enabled; None disables this weighting.
            class_weights: C-element output-channel multipliers on the prediction
                device, applied to both classification and regularization terms.
            **kwargs: Additional constructor arguments are accepted but unused.
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
        """Compute the loss for one batch.

        Masking multiplies per-pixel losses before the full-image mean; it does
        not divide by the number of included pixels. Spectral regularization is
        added after masking; channel and edge weights scale the resulting losses.

        Args:
            yhat: Floating BCHW logits, without sigmoid or clipping.
            target: BCHW binary or soft targets in [0, 1], or a BHW binary map
                shared across channels. Channels are independent binary tasks,
                not integer multiclass labels.
            target_weight: Optional BHW edge exponents when edge weighting is enabled.
            mask: Optional BHW inclusion weights, used only with apply_mask=True.
            **kwargs: Forward options for enabled smoothing methods, such as
                label_smoothing, kernel_size and sigma; unused otherwise.

        Returns:
            A single loss value on the prediction device, with gradients for training.
            Average over batch, channels and pixels, without weight normalization.
            Float16 and bfloat16 logits use float32 loss and regularization arithmetic.

        Raises:
            ValueError: Tensor shapes disagree or enabled edge weighting has no map.

        Examples:
            >>> logits = torch.zeros(1, 1, 1, 2)
            >>> target = torch.tensor([[[0, 1]]])
            >>> round(BCELoss()(logits, target).item(), 6)
            0.693147
        """
        if yhat.ndim != 4:
            raise ValueError("BCE logits must have BCHW shape.")
        num_classes = yhat.shape[1]
        pixel_shape = (yhat.shape[0], *yhat.shape[2:])
        # Large scores and their squared penalty need more range than float16 offers.
        dtype = (
            torch.float32
            if yhat.dtype in (torch.float16, torch.bfloat16)
            else yhat.dtype
        )
        logits = yhat.to(dtype=dtype)

        if target.shape == pixel_shape:
            target = target.unsqueeze(1).expand_as(logits)
        elif target.shape != yhat.shape:
            raise ValueError("BCE targets must have matching BCHW or BHW shape.")
        target = target.to(dtype=dtype)

        if self.apply_svls:
            target = self.apply_svls_to_target(target, num_classes, **kwargs)

        if self.apply_ls:
            # Each channel is a separate yes/no task, so smooth toward 0.5.
            target = self.apply_ls_to_target(
                target, 2, label_smoothing=kwargs.get("label_smoothing", 0.1)
            )

        # Combining sigmoid and cross-entropy avoids logs of rounded 0/1 probabilities.
        bce = F.binary_cross_entropy_with_logits(
            logits, target.to(dtype=dtype), reduction="none"
        )  # (B, C, H, W)

        if self.apply_mask and mask is not None:
            if mask.shape != pixel_shape:
                raise ValueError("BCE mask must have BHW shape.")
            bce = self.apply_mask_weight(bce, mask.unsqueeze(1), norm=False)

        if self.apply_sd:
            # Keep the penalty per channel so each channel weight scales its own term.
            bce = bce + 0.005 * logits.square()

        if self.class_weights is not None:
            if self.class_weights.shape != (num_classes,):
                raise ValueError("BCE class_weights must have shape (C,).")
            bce = bce * self.class_weights.view(1, -1, 1, 1)

        if self.edge_weight is not None:
            if target_weight is None or target_weight.shape != pixel_shape:
                raise ValueError("BCE edge weighting requires a BHW target_weight map.")
            bce = self.apply_edge_weights(bce, target_weight.unsqueeze(1))

        return torch.mean(bce)
