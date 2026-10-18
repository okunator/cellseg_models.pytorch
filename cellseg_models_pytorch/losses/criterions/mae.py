import torch

from cellseg_models_pytorch.losses.weighted_base_loss import WeightedBaseLoss

__all__ = ["MAE"]


class MAE(WeightedBaseLoss):
    """Compare raw maps using mean absolute error.

    Inputs must share a device. B is batch size, C channels, H height and W width.
    """

    def __init__(
        self,
        alpha: float = 1e-4,
        apply_sd: bool = False,
        apply_mask: bool = False,
        edge_weight: float = None,
        **kwargs,
    ) -> None:
        """Compare raw maps using mean absolute error.

            https://arxiv.org/pdf/1806.03535.pdf

        Args:
            alpha: Penalty on absolute predictions outside the mask when masking is enabled.
            apply_sd: Add 0.005 times the mean squared prediction across channels to
                the per-pixel loss.
            apply_mask: Multiply pixel losses by the supplied mask and divide by
                its mean weight. alpha also controls a penalty outside the mask.
            edge_weight: Multiply pixel losses by edge_weight ** target_weight.
                Supply that map when enabled; None disables this weighting.
            **kwargs: Additional constructor arguments are accepted but unused.
        """
        super().__init__(apply_sd, False, False, apply_mask, None, edge_weight)
        self.alpha = alpha
        self.eps = 1e-7

    def forward(
        self,
        yhat: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the loss for one batch.

        The masked sum is normalized by mean mask weight; alpha can also
        penalize predictions outside the mask.

        Args:
            yhat: Floating BCHW maps; no activation is applied.
            target: Floating targets matching BCHW, or BHW replicated across channels.
            target_weight: Optional BHW edge exponents when edge weighting is enabled.
            mask: Optional BHW inclusion weights, used only with apply_mask=True.
            **kwargs: Forward options for enabled smoothing methods, such as
                label_smoothing, kernel_size and sigma; unused otherwise.

        Returns:
            A single loss value on the prediction device, with gradients for training.
            Combine channel errors first, then average across images and pixels.
        """
        n_classes = yhat.shape[1]
        if target.size() != yhat.size():
            target = target.unsqueeze(1).repeat_interleave(n_classes, dim=1)

        if not yhat.shape == target.shape:
            raise ValueError(
                f"Pred and target shapes must match. Got: {yhat.shape}, {target.shape}"
            )

        # compute the MAE loss with alpha as weight
        mae_loss = torch.mean(torch.abs(target - yhat), axis=1)  # (B, H, W)

        if self.apply_mask and mask is not None:
            mae_loss = self.apply_mask_weight(mae_loss, mask, norm=True)  # (B, H, W)

            # add the background regularization
            if self.alpha > 0:
                reg = torch.mean(((1 - mask).unsqueeze(1)) * torch.abs(yhat), axis=1)
                mae_loss += self.alpha * reg

        if self.apply_sd:
            mae_loss = self.apply_spectral_decouple(mae_loss, yhat)

        if self.edge_weight is not None:
            mae_loss = self.apply_edge_weights(mae_loss, target_weight)

        return mae_loss.mean()
