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
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs,
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
            yhat: Floating BCHW class logits.
            target: Integer BHW class IDs in [0, C). Default unsmoothed one-hot targets
                contain zeros, so reverse log terms can produce infinity. Uniform
                label smoothing can make targets positive; do not assume finite
                default results.
            target_weight: Optional BHW edge exponents when edge weighting is enabled.
            mask: Optional BHW inclusion weights, used only with apply_mask=True.
            **kwargs: Forward options for enabled smoothing methods, such as
                label_smoothing, kernel_size and sigma; unused otherwise.

        Returns:
            A single loss value on the prediction device, with gradients for training.
            Combine channel errors first, then average across images and pixels.
        """
        n_classes = yhat.shape[1]
        target_one_hot = F.one_hot(target.long(), n_classes).permute(0, 3, 1, 2)
        yhat_soft = F.softmax(yhat, dim=1) + self.eps
        assert target_one_hot.shape == yhat.shape

        if self.apply_svls:
            target_one_hot = self.apply_svls_to_target(
                target_one_hot, n_classes, **kwargs
            )

        if self.apply_ls:
            target_one_hot = self.apply_ls_to_target(
                target_one_hot, n_classes, **kwargs
            )

        forward = target_one_hot * torch.log(yhat_soft)
        reverse = yhat_soft * torch.log(target_one_hot)

        cross_entropy = -torch.sum(forward, dim=1)  # to (B, H, W)
        reverse_cross_entropy = -torch.sum(reverse, dim=1)  # to (B, H, W)
        loss = self.alpha * cross_entropy + self.beta * reverse_cross_entropy

        if self.apply_mask and mask is not None:
            loss = self.apply_mask_weight(loss, mask, norm=False)  # (B, H, W)

        if self.apply_sd:
            loss = self.apply_spectral_decouple(loss, yhat)

        if self.class_weights is not None:
            loss = self.apply_class_weights(loss, target)

        if self.edge_weight is not None:
            loss = self.apply_edge_weights(loss, target_weight)

        return loss.mean()
