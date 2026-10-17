"""Ported from:
https://github.com/jopo666/HistoPrep/tree/master

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

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np


if TYPE_CHECKING:
    import torch


def _pixel_size_um(
    values: Sequence[float | str | None],
    units: Sequence[str] = ("µm", "µm"),
) -> tuple[float, float]:
    if len(values) != 2 or len(units) != 2:
        raise ValueError("Pixel-size metadata must contain both height and width")
    factors = {
        "µm": 1.0,
        "μm": 1.0,
        "um": 1.0,
        "micrometer": 1.0,
        "nm": 1e-3,
        "nanometer": 1e-3,
        "mm": 1e3,
        "millimeter": 1e3,
    }
    try:
        y, x = (float(value) * factors[unit] for value, unit in zip(values, units))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Pixel-size metadata is missing, invalid or has unknown units"
        ) from error
    if not all(np.isfinite(value) and value > 0 for value in (y, x)):
        raise ValueError("Pixel sizes must be positive finite micrometers per pixel")
    return y, x


class SlideReaderBackend(ABC):
    """Base class for all backends."""

    def __init__(self, path: Union[str, Path]) -> None:
        if not isinstance(path, Path):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path.resolve()))
        self.__path = path if isinstance(path, Path) else Path(path)
        self.__name = self.__path.stem

    @property
    def path(self) -> str:
        """Full slide filepath."""
        return str(self.__path.resolve())

    @property
    def name(self) -> str:
        """Slide filename without an extension."""
        return self.__name

    @property
    def suffix(self) -> str:
        """Slide file-extension."""
        return self.__path.suffix

    @property
    def pixel_size(self) -> tuple[float, float]:
        """Level-zero pixel size (height, width) in micrometers.

        Raises:
            ValueError: Physical calibration is unavailable.
        """
        raise ValueError("This backend does not provide pixel-size metadata")

    @property
    @abstractmethod
    def reader(self):  # noqa
        pass

    @property
    @abstractmethod
    def data_bounds(self) -> tuple[int, int, int, int]:
        """Data bounds defined by `xywh`-coordinates at `level=0`."""

    @property
    @abstractmethod
    def dimensions(self) -> tuple[int, int]:
        """Image dimensions (height, width) at `level=0`."""

    @property
    @abstractmethod
    def level_count(self) -> int:
        """Number of slide pyramid levels."""

    @property
    @abstractmethod
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        """Image dimensions (height, width) for each pyramid level."""

    @property
    @abstractmethod
    def level_downsamples(self) -> dict[int, tuple[float, float]]:
        """Image downsample factors (height, width) for each pyramid level."""

    @abstractmethod
    def read_level(self, level: int) -> np.ndarray:
        """Read full pyramid level data.

        Args:
            level: Slide pyramid level.

        Raises:
            ValueError: Invalid `level` argument.

        Returns:
            Array containing image data from `level`.
        """

    @abstractmethod
    def read_region(
        self,
        xywh: tuple[int, int, int, int],
        level: int,
        *,
        device: str | torch.device | None = None,
    ) -> np.ndarray | torch.Tensor:
        """Read region based on `xywh`-coordinates.

        Args:
            xywh: Coordinates for the region.
            level: Slide pyramid level to read from. Defaults to 0.
            device: None returns NumPy pixels; supported backends can return
                a tensor on the explicitly requested CUDA device.

        Raises:
            ValueError: Invalid `level` argument.

        Returns:
            Array containing image data from `xywh`-region.
        """
