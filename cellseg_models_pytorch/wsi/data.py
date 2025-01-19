"""Ported from:
https://github.com/jopo666/HistoPrep/tree/master/histoprep/_data.py

MIT License

Copyright (c) 2023 jopo666

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__all__ = ["TileCoordinates", "SpotCoordinates"]

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=False)
class TileCoordinates:
    """Data class representing a collection of tile coordinates.

    Parameters:
        coordinates (list[tuple[int, int, int, int]]):
            List of `xywh`-coordinates.
        width (int):
            Tile width.
        height (int):
            Tile height.
        overlap (float):
            Overlap between neighbouring tiles.
        max_background (float):
            Maximum amount of background in each tile.
        tissue_mask (np.ndarray):
            Tissue mask used for filtering tiles based on `max_background`.
    """

    coordinates: list[tuple[int, int, int, int]]
    width: int
    height: int
    overlap: float
    max_background: Optional[float] = field(default=None)
    tissue_mask: Optional[np.ndarray] = field(default=None)

    def get_properties(self, level: int, level_downsample: tuple[float, float]) -> dict:
        """Generate dictonary of properties."""
        return {
            "num_tiles": len(self),
            "level": level,
            "level_downsample": level_downsample,
            "width": self.width,
            "height": self.height,
            "overlap": self.overlap,
            "max_background": self.max_background,
        }

    def __len__(self) -> int:
        return len(self.coordinates)

    def __iter__(self) -> Iterator[tuple[int, int, int, int]]:
        return iter(self.coordinates)

    def __getitem__(self, index: int) -> tuple[int, int, int, int]:
        return self.coordinates[index]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_tiles={len(self)}, "
            f"shape={self.height, self.width})"
        )


@dataclass(frozen=False)
class SpotCoordinates:
    """Data class representing a collection of spot coordinates.

    Args:
        coordinates (list[tuple[int, int, int, int]]):
            List of XYWH-coordinates.
        spot_names (list[str]):
            Spot numbers.
        tissue_mask (np.ndarray):
            Tissue mask used to detect spots.
    """

    coordinates: tuple[int, int, int, int] = field(repr=False)
    spot_names: list[str] = field(repr=False)
    tissue_mask: np.ndarray = field(repr=False)

    def __len__(self) -> int:
        return len(self.coordinates)

    def __iter__(self) -> Iterator[str, tuple[int, int, int, int]]:
        return iter(self.coordinates)

    def __getitem__(self, index: int) -> tuple[int, int, int, int]:
        return self.coordinates[index]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_spots={len(self)})"
