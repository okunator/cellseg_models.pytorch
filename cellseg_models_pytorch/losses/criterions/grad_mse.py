import torch
import torch.nn as nn
import torch.nn.functional as F

from cellseg_models_pytorch.utils import filter2D, sobel_hv


def grad_mse(yhat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the gradient mse loss.

    Parameters
    ----------
        yhat : torch.Tensor
            Input tensor of size (B, 2, H, W). Regressed HoVer map.
        target : torch.Tensor
            Target tensor of shape (B, 2, H, W). GT HoVer-map.

    Returns
    -------
        torch.Tensor:
            Computed gradient mse loss matrix. Shape (B, H, W).

    """
    kernel = sobel_hv(window_size=5, device=yhat.device)
    grad_yhat = filter2D(yhat, kernel)
    grad_target = filter2D(target, kernel)
    msge = F.mse_loss(grad_yhat, grad_target, reduction="none")

    return msge


class GradMSE(nn.Module):
    def __init__(self, **kwargs) -> None:
        """Gradient MSE loss.

        Used in the horizontal and vertical gradient branch of HoVer-Net.
        https://arxiv.org/abs/1812.06499
        """
        super().__init__()
        self.eps = 1e-6

    def forward(
        self,
        yhat: torch.Tensor,
        target: torch.Tensor,
        target_inst: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute the msge-loss.

        Parameters
        ----------
            yhat : torch.Tensor
                Input tensor (B, 2, H, W). Regressed HoVer map.
            target : torch.Tensor
                Target tensor (B, 2, H, W). GT HoVer map.
            target_inst : torch.Tensor
                instance map target that is used to focus loss to the
                correct objects. Shape (B, H, W).

        Returns
        -------
            torch.Tensor:
                Computed gradient mse loss (scalar).
        """
        if target_inst is not None:
            focus = torch.stack([target_inst, target_inst], dim=1)
        else:
            focus = torch.ones_like(target)

        loss = focus * grad_mse(yhat, target)
        loss = loss.sum() / focus.clamp_min(self.eps).sum()

        return loss
