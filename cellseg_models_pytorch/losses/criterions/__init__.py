from .bce import BCELoss
from .ce import CELoss
from .dice import DiceLoss
from .focal import FocalLoss
from .iou import IoULoss
from .mae import MAE
from .mse import MSE
from .sce import SCELoss
from .ssim import MSSSIM, SSIM
from .tversky import TverskyLoss

SEG_LOSS_LOOKUP = {
    "iou": IoULoss,
    "dice": DiceLoss,
    "tversky": TverskyLoss,
    "ce": CELoss,
    "sce": SCELoss,
    "focal": FocalLoss,
    "mse": MSE,
    "ssim": SSIM,
    "msssim": MSSSIM,
    "mae": MAE,
    "bce": BCELoss,
}


__all__ = [
    "SEG_LOSS_LOOKUP",
    "REG_LOSS_LOOKUP",
    "MSE",
    "SSIM",
    "MSSSIM",
    "DiceLoss",
    "IoULoss",
    "TverskyLoss",
    "FocalLoss",
    "CELoss",
    "SCELoss",
    "MAE",
    "BCELoss",
]
