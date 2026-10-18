import torch
import torch.nn as nn

from .criterions import SEG_LOSS_LOOKUP


class Loss(nn.Module):
    """Select a loss by its registered name.

    Use a loss class directly when you need its specific options or target format.
    """

    def __init__(
        self,
        name: str,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs,
    ) -> None:
        """Choose a loss and its settings.

        Args:
            name: ce, bce, dice, iou, focal, mse, mae, ssim, msssim or sce.
                tversky remains accepted, but its calculation across classes
                needs correction before use. The registered instanseg route
                currently rejects these forwarded options; construct InstanSegCriterion
                directly and call it with pred and target.
            apply_sd: Add a squared-prediction penalty for losses supporting it.
            apply_ls: Uniformly smooth targets for losses supporting it.
            apply_svls: Smooth targets using neighboring pixels where supported.
            edge_weight: Use edge_weight ** target_weight as per-pixel weights
                where supported. None disables this weighting.
            class_weights: Per-class weights for losses that support class weighting.
            **kwargs: Other constructor settings for the selected loss. Supported settings
                vary by loss; see its own entry.

        Raises:
            ValueError: The name is unknown. Errors from the selected loss are passed through.
        """
        super().__init__()

        allowed = list(SEG_LOSS_LOOKUP.keys())
        if name not in allowed:
            raise ValueError(
                f"Illegal loss function given. Allowed: {allowed}. Got: '{name}'"
            )

        self.loss = SEG_LOSS_LOOKUP[name](
            apply_sd=apply_sd,
            apply_ls=apply_ls,
            apply_svls=apply_svls,
            edge_weight=edge_weight,
            class_weights=class_weights,
            **kwargs,
        )

    def forward(
        self,
        yhat: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the selected loss.

        Args:
            yhat: Prediction tensor with the shape and values required by the selected loss.
            target: Matching class labels or regression targets.
            target_weight: Optional per-pixel exponents for edge weighting.
            **kwargs: Additional forward arguments accepted by the selected loss.

        Returns:
            The loss on the prediction device; normally a single tensor value.
            SSIM can return a pair when return_cs is enabled.
        """
        return self.loss(
            yhat=yhat, target=target, target_weight=target_weight, **kwargs
        )
