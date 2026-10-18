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
    """Tile bounds and the settings used to generate them.

    Iterate or index this collection to get (x, y, width, height) tuples in
    level-zero slide pixels. Names and filtering settings are stored separately.

    Args:
        coordinates: Tile bounds in reading order.
        width: Tile width in level-zero pixels.
        height: Tile height in level-zero pixels.
        overlap: Fraction shared by neighboring tiles, from 0 up to but excluding 1.
        max_background: Largest allowed background fraction, or None if not filtered.
        tissue_mask: Optional 2D mask used for filtering: 1 is tissue, 0 background.
    """

    coordinates: list[tuple[int, int, int, int]]
    width: int
    height: int
    overlap: float
    max_background: Optional[float] = field(default=None)
    tissue_mask: Optional[np.ndarray] = field(default=None)

    def get_properties(
        self, level: int, level_downsample: tuple[float, float]
    ) -> dict[str, int | float | tuple[float, float] | None]:
        """Return the tile settings as a dictionary.

        Args:
            level: Read level to record.
            level_downsample: Height/width downsample factors to record for that level.

        Returns:
            Keys num_tiles, level, level_downsample, width, height, overlap and
            max_background. Width and height keep their level-zero values; this
            method records the supplied level metadata without changing coordinates.
        """
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
        """Return the number of tile bounds."""
        return len(self.coordinates)

    def __iter__(self) -> Iterator[tuple[int, int, int, int]]:
        """Iterate over tile bounds in their stored order."""
        return iter(self.coordinates)

    def __getitem__(self, index: int) -> tuple[int, int, int, int]:
        """Return the tile bounds at an index; negative indices count from the end."""
        return self.coordinates[index]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_tiles={len(self)}, "
            f"shape={self.height, self.width})"
        )


@dataclass(frozen=False)
class SpotCoordinates:
    """Detected tissue microarray spots and their names.

    Iterate or index this collection to get (x, y, width, height) bounds in
    level-zero pixels. Iteration returns bounds only; pair them with spot_names
    when you also need names.

    Args:
        coordinates: One bounding box per spot.
        spot_names: Names in the same order as coordinates, such as "spot_1".
        tissue_mask: Cleaned 2D uint8 mask at the detection image size; 1 is tissue.
    """

    coordinates: list[tuple[int, int, int, int]] = field(repr=False)
    spot_names: list[str] = field(repr=False)
    tissue_mask: np.ndarray = field(repr=False)

    def __len__(self) -> int:
        """Return the number of spots."""
        return len(self.coordinates)

    def __iter__(self) -> Iterator[tuple[int, int, int, int]]:
        """Iterate over spot bounds; names remain in spot_names."""
        return iter(self.coordinates)

    def __getitem__(self, index: int) -> tuple[int, int, int, int]:
        """Return the spot bounds at an index; negative indices count from the end."""
        return self.coordinates[index]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_spots={len(self)})"
