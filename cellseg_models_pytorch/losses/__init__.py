from itertools import permutations

from .criterions import (
    MSE,
    MSSSIM,
    SEG_LOSS_LOOKUP,
    SSIM,
    CELoss,
    DiceLoss,
    FocalLoss,
    GradMSE,
    IoULoss,
    TverskyLoss,
)
from .joint_loss import JointLoss
from .loss_module import Loss
from .multitask_loss import MultiTaskLoss

JOINT_SEG_LOSSES = []
for i in range(1, 5):
    JOINT_SEG_LOSSES.extend(
        ["_".join(t) for t in permutations(SEG_LOSS_LOOKUP.keys(), i)]
    )

__all__ = [
    "SEG_LOSS_LOOKUP",
    "JOINT_SEG_LOSSES",
    "Loss",
    "JointLoss",
    "MultiTaskLoss",
    "MSE",
    "MSSSIM",
    "CELoss",
    "SSIM",
    "GradMSE",
    "TverskyLoss",
    "FocalLoss",
    "DiceLoss",
    "IoULoss",
]
