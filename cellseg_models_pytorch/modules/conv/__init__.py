from torch.nn import Conv2d

from .scaled_ws_conv import ScaledWSConv2d
from .ws_conv import WSConv2d

CONV_LOOKUP = {"conv": Conv2d, "wsconv": WSConv2d, "scaled_wsconv": ScaledWSConv2d}


__all__ = ["CONV_LOOKUP", "WSConv2d", "ScaledWSConv2d"]
