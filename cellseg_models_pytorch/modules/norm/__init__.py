from torch.nn import BatchNorm2d, InstanceNorm2d, SyncBatchNorm

from .bcn import BCNorm
from .gn import GroupNorm
from .ln import LayerNorm2d

NORM_LOOKUP = {
    "bn": BatchNorm2d,
    "ln": LayerNorm2d,
    "bcn": BCNorm,
    "gn": GroupNorm,
    "in": InstanceNorm2d,
}

__all__ = [
    "NORM_LOOKUP",
    "BCNorm",
    "GroupNorm",
    "BatchNorm2d",
    "InstanceNorm2d",
    "SyncBatchNorm",
    "LayerNorm2d",
]
