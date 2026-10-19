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
        edge_weight: float | None = None,
        **kwargs: object,
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
        target_weight: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Compute the loss for one batch.

        The masked sum is normalized by mean mask weight; alpha can also
        penalize predictions outside the mask.

        Args:
            yhat: Floating BCHW maps; no activation is applied.
            target: Floating targets matching BCHW, or BHW replicated across channels.
            target_weight: Optional BHW edge exponents when edge weighting is enabled.
            mask: Optional BHW inclusion weights in [0, 1], used with apply_mask=True.
            **kwargs: Accepted but unused; smoothing and class weights are unsupported.

        Returns:
            A single loss value on the prediction device, with gradients for training.
            Combine channel errors first, then average across images and pixels.
            Float16/bfloat16 inputs compute in float32; float64 is preserved.
            Divide masked errors by mean(mask) + 1e-7. Outside-mask and spectral
            penalties are then added, including for fully masked inputs; edge
            weights scale all terms.

        Raises:
            ValueError: Shapes disagree or enabled edge weighting has no map.

        Examples:
            >>> prediction = torch.tensor([[[[1.0, 3.0]]]])
            >>> MAE()(prediction, torch.zeros_like(prediction)).item()
            2.0
        """
        if yhat.ndim != 4:
            raise ValueError("MAE predictions must have BCHW shape.")
        pixel_shape = (yhat.shape[0], *yhat.shape[2:])
        if target.shape not in (yhat.shape, pixel_shape):
            raise ValueError("MAE targets must have matching BCHW or BHW shape.")
        # Preserve the more precise input and keep differences within float32 range.
        dtype = torch.promote_types(yhat.dtype, target.dtype)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        prediction, target = yhat.to(dtype=dtype), target.to(dtype=dtype)
        if target.shape == pixel_shape:
            # Convert before sharing so channel gradients add up before rounding.
            target = target.unsqueeze(1).expand_as(prediction)

        # Opposite large values can overflow if subtracted while still in float16.
        mae_loss = (target - prediction).abs().mean(1)

        if self.apply_mask and mask is not None:
            if mask.shape != pixel_shape:
                raise ValueError("MAE mask must have BHW shape.")
            mask = mask.to(dtype=dtype)
            # Normalize by the mask so excluded pixels do not dilute the average error.
            mae_loss = self.apply_mask_weight(mae_loss, mask, norm=True)  # (B, H, W)

            if self.alpha > 0:
                # Outside the mask, discourage nonzero predictions instead of matching targets.
                reg = ((1 - mask).unsqueeze(1) * prediction.abs()).mean(1)
                mae_loss += self.alpha * reg

        if self.apply_sd:
            mae_loss = self.apply_spectral_decouple(mae_loss, prediction)

        if self.edge_weight is not None:
            if target_weight is None or target_weight.shape != pixel_shape:
                raise ValueError("MAE edge weighting requires a BHW target_weight map.")
            mae_loss = self.apply_edge_weights(mae_loss, target_weight)

        return mae_loss.mean()
