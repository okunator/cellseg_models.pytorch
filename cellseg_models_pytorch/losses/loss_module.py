import torch
import torch.nn as nn

from .criterions import SEG_LOSS_LOOKUP


class Loss(nn.Module):
    def __init__(
        self,
        name: str,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs,
    ) -> None:
        """Loss wrapper class.

        Parameters:
        -----------
            name : str
                Name of the loss function.
            edge_weight : float, default=none
                Weight that is added to object borders.
            class_weights : torch.Tensor, default=None
                Class weights. A tensor of shape (n_classes,).

        Raises
        ------
            ValueError: if the loss function name is illegal.
        """
        super().__init__()

        allowed = list(SEG_LOSS_LOOKUP.keys())
        if name not in allowed:
            raise ValueError(
                f"Illegal loss function given. Allowed: {allowed}. Got: '{name}'"
            )

        self.loss = SEG_LOSS_LOOKUP[name](
            edge_weight=edge_weight, class_weights=class_weights, **kwargs
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
