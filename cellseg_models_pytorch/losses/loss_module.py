import torch
import torch.nn as nn

from .criterions import SEG_LOSS_LOOKUP


class Loss(nn.Module):
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
        """Loss wrapper class.

        Args:
            name: Name of the loss function.
            apply_sd:
                If True, Spectral decoupling regularization will be applied  to the
                loss matrix.
                Defaults to False.
            apply_ls: If True, Label smoothing will be applied to the target.
                Defaults to False.
            apply_svls:
                If True, spatially varying label smoothing will be applied to the target
                Defaults to False.
            edge_weight: Weight that is added to object borders.
                Defaults to none.
            class_weights: Class weights. A tensor of shape (n_classes,).
                Defaults to None.

        Raises:
            ValueError: if the loss function name is illegal.
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
        """Forward pass for the loss function."""
        return self.loss(
            yhat=yhat, target=target, target_weight=target_weight, **kwargs
        )
