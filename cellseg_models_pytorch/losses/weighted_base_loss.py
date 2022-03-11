import torch
import torch.nn as nn


class WeightedBaseLoss(nn.Module):
    def __init__(
        self, class_weights: torch.Tensor = None, edge_weight: float = None
    ) -> None:
        """Init a base class for weighted cross entropy based losses.

        Enables weighting for object instance edges and classes.

        Parameters
        ----------
            class_weights : torch.Tensor, default=None
                Class weights. A tensor of shape (C, )
            edge_weight : float, default=None
                Weight for the object instance border pixels
        """
        super().__init__()
        self.class_weights = class_weights
        self.edge_weight = edge_weight

    def apply_class_weights(
        self, loss_matrix: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Multiply pixelwise loss matrix by the class weights.

        Note: No normalization

        Parameters
        ----------
            loss_matrix : torch.Tensor
                Pixelwise losses. A tensor of shape (B, H, W).
            target : torch.Tensor
                The target mask. Shape (B, H, W).

        Returns
        -------
            torch.Tensor:
                The loss matrix scaled with the weight matrix. Shape (B, H, W).
        """
        weight_mat = self.class_weights[target].to(target.device)  # to (B, H, W)
        loss = loss_matrix * weight_mat

        return loss

    def apply_edge_weights(
        self, loss_matrix: torch.Tensor, weight_map: torch.Tensor
    ) -> torch.Tensor:
        """Apply weights to the object boundaries.

        Basically just computes `edge_weight`**`weight_map`.

        Parameters
        ----------
            loss_matrix : torch.Tensor
                Pixelwise losses. A tensor of shape (B, H, W).
            weight_map : torch.Tensor
                Map that points to the pixels that will be weighted.
                Shape (B, H, W).

        Returns
        -------
            torch.Tensor:
                The loss matrix scaled with the nuclear boundary weights.
                Shape (B, H, W).
        """
        return loss_matrix * self.edge_weight**weight_map

    def extra_repr(self) -> str:
        """Add info to print."""
        s = "class_weights={class_weights}, edge_weight={edge_weight}"
        return s.format(**self.__dict__)


# def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
#     logsoftmax = nn.LogSoftmax()
#     n_classes = pred.size(1)
#     # convert to one-hot
#     target = torch.unsqueeze(target, 1)
#     soft_target = torch.zeros_like(pred)
#     soft_target.scatter_(1, target, 1)
#     # label smoothing
#     soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
#     return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))
