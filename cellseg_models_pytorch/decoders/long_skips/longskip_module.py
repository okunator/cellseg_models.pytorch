from typing import Tuple

import torch
import torch.nn as nn

from ...modules import Identity
from .cross_attn_skip import CrossAttentionSkip
from .unet import UnetSkip
from .unet3p import Unet3pSkip
from .unetpp import UnetppSkip

__all__ = ["LongSkip"]


LONGSKIP_LOOKUP = {
    "unet": UnetSkip,
    "unet3p": Unet3pSkip,
    "unet3p-lite": Unet3pSkip,
    "unetpp": UnetppSkip,
    "cross-attn": CrossAttentionSkip,
}


class LongSkip(nn.Module):
    def __init__(self, name: str, **kwargs) -> None:
        """Long skip wrapper class.

        Parameters
        ----------
            name : str
                The name of the long skip method.
                One of "unet", "unet3p", "unet3p-lite", "unetpp".

        Raises
        ------
            ValueError: if the long skip method name is illegal.
        """
        super().__init__()

        allowed = list(LONGSKIP_LOOKUP.keys()) + [None]
        if name not in allowed:
            raise ValueError(
                f"Illegal long skip method given. Allowed: {allowed}. Got: '{name}'"
            )

        if name is not None:
            if name == "unet3p-lite":
                kwargs["lite_version"] = True
            try:
                self.skip = LONGSKIP_LOOKUP[name](**kwargs)
            except Exception as e:
                raise Exception(
                    "Encountered an error when trying to init long-skip module: "
                    f"LongSkip(name='{name}'): {e.__class__.__name__}: {e}"
                )
        else:
            self.skip = Identity()

    @property
    def out_channels(self) -> int:
        """Out channels."""
        return self.skip.out_channels

    def forward(
        self, x: torch.Tensor, skips: Tuple[torch.Tensor, ...], **kwargs
    ) -> torch.Tensor:
        """Forward pass of the long skip function."""
        return self.skip(x, skips, **kwargs)
