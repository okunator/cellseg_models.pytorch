import torch
import torch.nn.functional as F

from ..weighted_base_loss import WeightedBaseLoss

__all__ = ["MSE"]


class MSE(WeightedBaseLoss):
    """Compare raw maps using mean squared error.

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
        **kwargs: object,
    ) -> None:
        """Compare raw maps using mean squared error.

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

    def forward(
        self,
        yhat: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs: float | int,
    ) -> torch.Tensor:
        """Compute the loss for one batch.

        Masking multiplies per-pixel losses before the full-image mean; it does
        not divide by the number of included pixels.

        Args:
            yhat: Floating BCHW regression maps; no activation is applied.
            target: Matching BCHW maps, floating BHW maps shared across channels,
                or integer BHW class IDs converted to one-hot targets.
            target_weight: Optional BHW edge exponents when edge weighting is enabled.
            mask: Optional BHW inclusion weights, used only with apply_mask=True.
            **kwargs: Forward options for enabled smoothing methods, such as
                label_smoothing, kernel_size and sigma; unused otherwise.

        Returns:
            A single loss value on the prediction device, with gradients for training.
            Combine channel errors first, then average across images and pixels.
            Float16/bfloat16 inputs compute in float32; float64 is preserved.
            Regularization is added after masking and scaled by class/edge weights.

        Raises:
            ValueError: Shapes disagree, class weights lack integer BHW labels,
                or enabled edge weighting has no map.

        Examples:
            >>> prediction = torch.tensor([[[[1.0, 3.0]]]])
            >>> MSE()(prediction, torch.zeros_like(prediction)).item()
            5.0
        """
        if yhat.ndim != 4:
            raise ValueError("MSE predictions must have BCHW shape.")
        pixel_shape = (yhat.shape[0], *yhat.shape[2:])
        if target.shape not in (yhat.shape, pixel_shape):
            raise ValueError("MSE targets must have matching BCHW or BHW shape.")
        # Preserve the more precise input; float16 errors can overflow when squared.
        dtype = torch.promote_types(yhat.dtype, target.dtype)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        prediction = yhat.to(dtype=dtype)
        target_one_hot = target
        n_classes = yhat.shape[1]

        if target.shape == pixel_shape:
            if target.is_floating_point():
                # Convert before sharing so channel gradients add up before rounding.
                target_one_hot = (
                    target.to(dtype=dtype).unsqueeze(1).expand_as(prediction)
                )
            else:
                # Integer targets identify classes rather than shared pixel values.
                target_one_hot = F.one_hot(target.long(), n_classes).permute(0, 3, 1, 2)
        target_one_hot = target_one_hot.to(dtype=dtype)

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

        mse = F.mse_loss(prediction, target_one_hot.to(dtype=dtype), reduction="none")
        mse = torch.mean(mse, dim=1)  # to (B, H, W)

        if self.apply_mask and mask is not None:
            if mask.shape != pixel_shape:
                raise ValueError("MSE mask must have BHW shape.")
            # Masked-out pixels still count toward the final full-image average.
            mse = self.apply_mask_weight(mse, mask.to(dtype=dtype), norm=False)

        if self.apply_sd:
            # This separate penalty still discourages large values outside the mask.
            mse = self.apply_spectral_decouple(mse, prediction)

        if self.class_weights is not None:
            if (
                self.class_weights.shape != (n_classes,)
                or target.shape != pixel_shape
                or target.is_floating_point()
            ):
                raise ValueError(
                    "MSE class_weights require shape (C,) and BHW integer labels."
                )
            mse = self.apply_class_weights(mse, target)

        if self.edge_weight is not None:
            if target_weight is None or target_weight.shape != pixel_shape:
                raise ValueError("MSE edge weighting requires a BHW target_weight map.")
            mse = self.apply_edge_weights(mse, target_weight)

        return torch.mean(mse)
