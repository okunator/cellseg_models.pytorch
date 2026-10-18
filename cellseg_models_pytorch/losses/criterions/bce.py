import torch
import torch.nn.functional as F

from cellseg_models_pytorch.losses.weighted_base_loss import WeightedBaseLoss

__all__ = ["BCELoss"]


class BCELoss(WeightedBaseLoss):
    """Compute the package's binary cross-entropy variant.

    Inputs must share a device. B is batch size, C channels, H height and W width.
    """

    def __init__(
        self,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs,
    ) -> None:
        """Compute the package's binary cross-entropy variant.

        Args:
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
        self.eps = 1e-8

    def forward(
        self,
        yhat: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the loss for one batch.

        Masking multiplies per-pixel losses before the full-image mean; it does
        not divide by the number of included pixels.

        Args:
            yhat: Floating BCHW predictions; currently clipped to [1e-8, 1-1e-8]
                before binary_cross_entropy_with_logits is called. This differs from
                standard BCE-with-logits and limits its response to large logits.
            target: Binary targets matching BCHW, or BHW replicated across channels.
            target_weight: Optional BHW edge exponents when edge weighting is enabled.
            mask: Optional BHW inclusion weights, used only with apply_mask=True.
            **kwargs: Forward options for enabled smoothing methods, such as
                label_smoothing, kernel_size and sigma; unused otherwise.

        Returns:
            A single loss value on the prediction device, with gradients for training.
            Combine channel errors first, then average across images and pixels.
        """
        num_classes = yhat.shape[1]
        yhat = torch.clip(yhat, self.eps, 1.0 - self.eps)

        if target.size() != yhat.size():
            target = target.unsqueeze(1).repeat_interleave(num_classes, dim=1)

        if self.apply_svls:
            target = self.apply_svls_to_target(target, num_classes, **kwargs)

        if self.apply_ls:
            target = self.apply_ls_to_target(target, num_classes, **kwargs)

        bce = F.binary_cross_entropy_with_logits(
            yhat.float(), target.float(), reduction="none"
        )  # (B, C, H, W)
        bce = torch.mean(bce, dim=1)  # (B, H, W)

        if self.apply_mask and mask is not None:
            bce = self.apply_mask_weight(bce, mask, norm=False)  # (B, H, W)

        if self.apply_sd:
            bce = self.apply_spectral_decouple(bce, yhat)

        if self.class_weights is not None:
            bce = self.apply_class_weights(bce, target)

        if self.edge_weight is not None:
            bce = self.apply_edge_weights(bce, target_weight)

        return torch.mean(bce)
