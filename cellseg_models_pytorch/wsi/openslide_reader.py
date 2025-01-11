"""Ported from (with tiny modifications):
https://github.com/jopo666/HistoPrep/tree/master/histoprep/_backend.py

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

from pathlib import Path

import numpy as np

from ._base_reader import SlideReaderBackend
from .tiles import _divide_xywh, _get_allowed_dimensions, _pad_tile, format_level

try:
    import openslide

    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False


OPENSLIDE_READABLE_FORMATS = (
    ".svs",
    ".vms",
    ".vmu",
    ".ndpi",
    ".scn",
    ".mrxs",
    ".tiff",
    ".svslide",
    ".tif",
    ".bif",
)


class OpenSlideReader(SlideReaderBackend):
    """Slide reader using `OpenSlide` as a backend."""

    BACKEND_NAME = "OPENSLIDE"

    def __init__(self, path: str) -> None:
        """Initialize OpenSlideBackend class instance.

        Parameters:
            path (str):
                Path to the slide image.

        Raises:
            ImportError: OpenSlide could not be imported.
        """
        if not HAS_OPENSLIDE:
            raise ImportError(
                "OpenSlide not installed. `pip install openslide-python`",
                "Install binaries with linux: `apt-get install openslide-tools`",
                "Or download from https://openslide.org/download/",
            )

        path = Path(path)
        if not path.name.endswith(OPENSLIDE_READABLE_FORMATS):
            raise ValueError(
                f"File format {path.suffix} not supported by OpenSlide. "
                f"Supported formats: {OPENSLIDE_READABLE_FORMATS}"
            )

        super().__init__(path)
        self.__reader = openslide.OpenSlide(path)

        # Openslide has (width, height) dimensions.
        self.__level_dimensions = {
            lvl: (h, w) for lvl, (w, h) in enumerate(self.__reader.level_dimensions)
        }
        # Calculate actual downsamples.
        slide_h, slide_w = self.dimensions
        self.__level_downsamples = {}
        for lvl, (level_h, level_w) in self.__level_dimensions.items():
            self.__level_downsamples[lvl] = (slide_h / level_h, slide_w / level_w)

    @property
    def reader(self):
        """OpenSlide instance."""
        return self.__reader

    @property
    def data_bounds(self) -> tuple[int, int, int, int]:
        properties = dict(self.__reader.properties)
        x_bound = int(properties.get("openslide.bounds-x", 0))
        y_bound = int(properties.get("openslide.bounds-y", 0))
        w_bound = int(properties.get("openslide.bounds-width", self.dimensions[1]))
        h_bound = int(properties.get("openslide.bounds-heigh", self.dimensions[0]))
        return (x_bound, y_bound, w_bound, h_bound)

    @property
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        """dim order (H, W)"""
        return self.__level_dimensions

    @property
    def level_downsamples(self) -> dict[int, tuple[int, int]]:
        """dim order (H, W)"""
        return self.__level_downsamples

    @property
    def dimensions(self) -> tuple[int, int]:
        """dim order (H, W)"""
        return self.level_dimensions[0]

    @property
    def level_count(self) -> int:
        return len(self.level_dimensions)

    def read_level(self, level: int) -> np.ndarray:
        level = format_level(level, available=list(self.level_dimensions))
        level_h, level_w = self.level_dimensions[level]
        return np.array(self.__reader.get_thumbnail(size=(level_w, level_h)))

    def read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        level = format_level(level, available=list(self.level_dimensions))
        # Only width and height have to be adjusted for the level.
        x, y, *__ = xywh
        *__, w, h = _divide_xywh(xywh, self.level_downsamples[level])
        # Read allowed region.
        allowed_h, allowed_w = _get_allowed_dimensions((x, y, w, h), self.dimensions)
        tile = self.__reader.read_region(
            location=(x, y), level=level, size=(allowed_w, allowed_h)
        )
        tile = np.array(tile)[..., :3]  # only rgb channels

        # Pad tile.
        return _pad_tile(tile, shape=(h, w))
