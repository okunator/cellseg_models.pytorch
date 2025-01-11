from pathlib import Path

import numpy as np

from ._base_reader import SlideReaderBackend
from .tiles import _divide_xywh, _get_allowed_dimensions, _pad_tile, format_level

try:
    from cucim import CuImage

    HAS_CUCIM = True
except ImportError:
    HAS_CUCIM = False


CUCIM_READABLE_FORMATS = (
    ".svs",
    ".tiff",
)


class CucimReader(SlideReaderBackend):
    """Slide reader using `CUCIM` as a backend."""

    BACKEND_NAME = "CUCIM"

    def __init__(self, path: str) -> None:
        """Initialize CUCIMBackend class instance.

        Parameters:
            path (str): Path to the slide image.

        Raises:
            ImportError: CUCIM could not be imported.
        """
        if not HAS_CUCIM:
            raise ImportError(
                "CUCIM not installed. `pip install cucim-cu12`",
            )

        path = Path(path)
        if not path.name.endswith(CUCIM_READABLE_FORMATS):
            raise ValueError(
                f"File format {path.suffix} not supported by CUCIM. "
                f"Supported formats: {CUCIM_READABLE_FORMATS}"
            )

        super().__init__(path)
        self.__reader = CuImage(path.as_posix())

        # CUCIM has (width, height) dimensions.
        self.__level_dimensions = {
            lvl: (h, w)
            for lvl, (w, h) in enumerate(self.__reader.resolutions["level_dimensions"])
        }
        # Calculate actual downsamples.
        slide_h, slide_w = self.dimensions
        self.__level_downsamples = {}
        for lvl, (level_h, level_w) in self.__level_dimensions.items():
            self.__level_downsamples[lvl] = (slide_h / level_h, slide_w / level_w)

    @property
    def reader(self):
        """CUCIM instance."""
        return self.__reader

    @property
    def data_bounds(self) -> tuple[int, int, int, int]:
        properties = self.reader.metadata
        x_bound = int(properties.get("bounds-x", 0))
        y_bound = int(properties.get("bounds-y", 0))
        w_bound = int(properties.get("bounds-width", self.dimensions[1]))
        h_bound = int(properties.get("bounds-heigh", self.dimensions[0]))
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
        slide_h, slide_w = self.dimensions
        return self.read_region((0, 0, slide_w, slide_h), level)

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
