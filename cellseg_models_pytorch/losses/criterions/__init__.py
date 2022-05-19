from .ce import CELoss
from .dice import DiceLoss
from .focal import FocalLoss
from .grad_mse import GradMSE
from .iou import IoULoss
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
    "gmse": GradMSE,
    "ssim": SSIM,
    "msssim": MSSSIM,
}


__all__ = [
    "SEG_LOSS_LOOKUP",
    "REG_LOSS_LOOKUP",
    "MSE",
    "GradMSE",
    "SSIM",
    "MSSSIM",
    "DiceLoss",
    "IoULoss",
    "TverskyLoss",
    "FocalLoss",
    "CELoss",
    "SCELoss",
]
