from torch.nn import ConvTranspose2d, Upsample

from .fixed_unpool import FixedUnpool

UP_LOOKUP = {
    "fixed-unpool": FixedUnpool,
    "bilinear": Upsample,
    "bicubic": Upsample,
    "nearest": Upsample,
    "conv_transpose": ConvTranspose2d,
}

__all__ = ["UP_LOOKUP", "FixedUnpool", "ConvTranspose2d", "Upsample"]
