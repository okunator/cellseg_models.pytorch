from pathlib import Path

import numpy as np

from ._base_reader import SlideReaderBackend
from .tiles import _divide_xywh, _get_allowed_dimensions, _pad_tile, format_level

try:
    from bioio import BioImage

    HAS_BIOIO = True
except ImportError:
    HAS_BIOIO = False


BIOIO_READABLE_FORMATS = (
    ".ome.tiff",
    ".tiff",
    ".zarr",
    ".svs",
)


__all__ = ["BioIOReader"]


class BioIOReader(SlideReaderBackend):
    """Slide reader using `BioIO` as a backend."""

    BACKEND_NAME = "BIOIO"

    def __init__(self, path: str) -> None:
        """Initialize BIOIOBackend class instance.

        Parameters:
            path (str): Path to the slide image.

        Raises:
            ImportError: BIOIO could not be imported.
        """
        if not HAS_BIOIO:
            raise ImportError(
                "bioio not installed. `pip install bioio`",
                "check-out additional installation instructions at "
                "https://bioio-devs.github.io/bioio/OVERVIEW.html",
            )

        path = Path(path)
        if not path.name.endswith(BIOIO_READABLE_FORMATS):
            raise ValueError(
                f"File format {path.suffix} not supported by BIOIO. "
                f"Supported formats: {BIOIO_READABLE_FORMATS}"
            )

        super().__init__(path)
        self.__reader = BioImage(path)

        # BIOIO has (width, height) dimensions.
        self.__level_dimensions = {
            lvl: shp[:2] for lvl, shp in self.__reader.resolution_level_dims.items()
        }

        # Calculate actual downsamples.
        slide_h, slide_w = self.dimensions
        self.__level_downsamples = {}
        for lvl, (level_h, level_w) in self.__level_dimensions.items():
            self.__level_downsamples[lvl] = (slide_h / level_h, slide_w / level_w)

    @property
    def reader(self):
        """BIOIO instance."""
        return self.__reader

    @property
    def data_bounds(self) -> tuple[int, int, int, int]:
        h, w = self.dimensions
        return (0, 0, w, h)

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

    def get_downsample_img(self, level: int, factor: int) -> np.ndarray:
        # Get the downsample factors for the specified level
        slide_h, slide_w = self.dimensions
        wsi = self._read_region((0, 0, slide_w, slide_h))
        wsi.rechunk((1, 1000, 1000))

        # Calculate the downsampled dimensions    # Downsample using map_blocks
        def downsample_block(block):
            return block[::factor, ::factor]

        thumbnail = wsi.map_blocks(downsample_block, dtype=wsi.dtype)

        return thumbnail.compute()

    def read_level(self, level: int, factor: int = 100) -> np.ndarray:
        # if only one level is available, return a downsampled image
        if len(self.level_dimensions) == 1 or level == 0:
            return self.get_downsample_img(level, factor=factor)

        slide_h, slide_w = self.dimensions
        return self._read_region((0, 0, slide_w, slide_h), level)

    def _read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        level = format_level(level, available=list(self.level_dimensions))
        # Only width and height have to be adjusted for the level.
        x, y, *__ = xywh
        *__, w, h = _divide_xywh(xywh, self.level_downsamples[level])
        # Read allowed region.
        allowed_h, allowed_w = _get_allowed_dimensions((x, y, w, h), self.dimensions)
        return self.__reader.xarray_data[
            0, 0, 0, range(y, y + allowed_h), range(x, x + allowed_w)
        ]

    def read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        *__, w, h = _divide_xywh(xywh, self.level_downsamples[level])
        tile = self._read_region(xywh, level)
        tile = np.array(tile)[..., :3]  # only rgb channels

        # Pad tile.
        return _pad_tile(tile, shape=(h, w))
