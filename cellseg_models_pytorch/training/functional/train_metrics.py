import torch
import torch.nn.functional as F


def confusion_mat(
    yhat: torch.Tensor, target: torch.Tensor, activation: str = None
) -> torch.Tensor:
    """Compute the confusion matrix from the soft mask and target tensor.

    Parameters
    ----------
        yhat : torch.Tensor
            The soft mask from the network of shape (B, C, H, W).
        target : torch.Tensor
            The target matrix of shape (B, H, W)
        activation : str, optional:
            Apply sigmoid or softmax activation before taking argmax.

    Raises
    ------
        ValueError if an illegal activation is given.

    Returns
    -------
        torch.Tensor:
            A confusion matrix tensor of shape (B, num_classes, num_classes).

    """
    yhat_soft = yhat

    allowed = ("sigmoid", "softmax")
    if activation is not None:
        if activation not in allowed:
            raise ValueError(
                f"Illegal activation given. Got '{activation}'. Allowed: {allowed}."
            )

        if activation == "sigmoid":
            yhat_soft = torch.sigmoid(yhat)
        elif activation == "softmax":
            yhat_soft = F.softmax(yhat, dim=1)

    n_classes = yhat_soft.shape[1]
    batch_size = yhat_soft.shape[0]
    bins = target + torch.argmax(yhat_soft, dim=1) * n_classes
    bins_vec = bins.view(batch_size, -1)

    confusion_list = []
    for i in range(batch_size):
        pb = bins_vec[i]
        bin_count = torch.bincount(pb, minlength=n_classes**2)
        confusion_list.append(bin_count)

    confusion_vec = torch.stack(confusion_list)
    confusion_mat = confusion_vec.view(batch_size, n_classes, n_classes).to(
        torch.float32
    )

    return confusion_mat


def iou(
    yhat: torch.Tensor,
    target: torch.Tensor,
    activation: str = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute the per class intersection over union for dense predictions.

    Parameters
    ----------
        yhat : torch.Tensor
            The soft mask from the network of shape (B, C, H, W).
        target : torch.Tensor
            The target matrix of shape (B, H, W).
        activation : str, optional
            Apply sigmoid or softmax activation before taking argmax.
        eps : float, default=1e-7
            Small constant to avoid zero div error.

    Returns
    -------
        torch.Tensor:
            An iou matrix of shape (B, num_classes, num_classes).
    """
    conf_mat = confusion_mat(yhat, target, activation)
    rowsum = torch.sum(conf_mat, dim=1)  # [(TP + FP), (FN + TN)]
    colsum = torch.sum(conf_mat, dim=2)  # [(TP + FN), (FP + TN)]
    diag = torch.diagonal(conf_mat, dim1=-2, dim2=-1)  # [TP, TN]
    denom = rowsum + colsum - diag  # [(TP + FN + FP), (TN + FN + FP)]
    ious = (diag + eps) / (denom + eps)

    return ious  # [(TP/(TP + FN + FP)), (TN/(TN + FN + FP))]


def accuracy(
    yhat: torch.Tensor,
    target: torch.Tensor,
    activation: str = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute the per class accuracy for dense predictions.

    Parameters
    ----------
        yhat : torch.Tensor
            The soft mask from the network of shape (B, C, H, W).
        target : torch.Tensor
            The target matrix of shape (B, H, W).
        activation : str, optional
            apply sigmoid or softmax activation before taking argmax
        eps : float, default=1e-7
            Small constant to avoid zero div error.

    Returns
    -------
        torch.Tensor:
            The accuracy. A tensor of shape (1).
    """
    conf_mat = confusion_mat(yhat, target, activation)
    diag = torch.diagonal(conf_mat, dim1=-2, dim2=-1)  # batch diagonal
    denom = conf_mat.sum()
    accuracies = (diag + eps) / (denom + eps)

    return accuracies.sum()
