from typing import Tuple

import torch
import torch.nn as nn

from ....modules import Identity
from .cat import CatBlock
from .sum import SumBlock

__all__ = ["Merge"]


MERGE_LOOKUP = {
    "cat": CatBlock,
    "sum": SumBlock,
}


class Merge(nn.Module):
    def __init__(self, name: str, **kwargs) -> None:
        """Merge wrapper class.

        Parameters
        ----------
            name : str
                The name of the merging method. One of "sum", "cat".

        Raises
        ------
            ValueError: if the merging method name is illegal.
        """
        super().__init__()

        allowed = list(MERGE_LOOKUP.keys()) + [None]
        if name not in allowed:
            raise ValueError(
                f"Illegal merging method given. Allowed: {allowed}. Got: '{name}'"
            )

        if name is not None:
            self.merge = MERGE_LOOKUP[name](**kwargs)
        else:
            self.merge = Identity(**kwargs)

    @property
    def out_channels(self) -> int:
        """Out channels."""
        out_channels = self.merge.in_channels
        if not isinstance(self.merge, Identity):
            out_channels = self.merge.out_channels

        return out_channels

    def forward(
        self, x: torch.Tensor, skips: Tuple[torch.Tensor, ...], **kwargs
    ) -> torch.Tensor:
        """Forward pass of the merge module."""
        return self.merge(x, skips)
