"""Ported from (with tiny modifications):
https://github.com/jopo666/HistoPrep/tree/master/histoprep/_reader.py

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

from collections.abc import Iterator
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from .bioio_reader import BioIOReader
from .cucim_reader import CucimReader
from .data import SpotCoordinates, TileCoordinates
from .dearray import get_spot_coordinates
from .image import get_annotated_image

# from .ometiff_reader import OMETIFFReader
from .openslide_reader import OpenSlideReader
from .tiles import (
    _multiply_xywh,
    format_level,
    get_background_percentages,
    get_downsample,
    get_tile_coordinates,
)
from .tissue import clean_tissue_mask, get_tissue_mask

AVAILABLE_BACKENDS = ("OPENSLIDE", "CUCIM", "BIOIO")


class SlideReader:
    """Reader class for histological slide images."""

    def __init__(
        self,
        path: Union[str, Path],
        backend: str = "OPENSLIDE",
    ) -> None:
        """Initialize `SlideReader` instance.

        Parameters:
            path (str, Path):
                Path to slide image.
            backend (str, default="OPENSLIDE"):
                Backend to use for reading slide images.

        Raises:
            FileNotFoundError: Path does not exist.
            ValueError: Backend name not recognised.
        """
        super().__init__()
        if backend not in AVAILABLE_BACKENDS:
            raise ValueError(f"Backend {backend} not recognised or not supported.")

        if backend == "OPENSLIDE":
            self._reader = OpenSlideReader(path=path)
        elif backend == "CUCIM":
            self._reader = CucimReader(path=path)
        elif backend == "BIOIO":
            self._reader = BioIOReader(path=path)

    @property
    def path(self) -> str:
        """Full slide filepath."""
        return self._reader.path

    @property
    def name(self) -> str:
        """Slide filename without an extension."""
        return self._reader.name

    @property
    def suffix(self) -> str:
        """Slide file-extension."""
        return self._reader.suffix

    @property
    def backend_name(self) -> str:
        """Name of the slide reader backend."""
        return self._reader.BACKEND_NAME

    @property
    def data_bounds(self) -> tuple[int, int, int, int]:
        """Data bounds defined by `xywh`-coordinates at `level=0`.

        Some image formats (eg. `.mrxs`) define a bounding box where image data resides,
        which may differ from the actual image dimensions. `HistoPrep` always uses the
        full image dimensions, but other software (such as `QuPath`) uses the image
        dimensions defined by this data bound.
        """
        return self._reader.data_bounds

    @property
    def dimensions(self) -> tuple[int, int]:
        """Image dimensions (height, width) at `level=0`."""
        return self._reader.dimensions

    @property
    def level_count(self) -> int:
        """Number of slide pyramid levels."""
        return self._reader.level_count

    @property
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        """Image dimensions (height, width) for each pyramid level."""
        return self._reader.level_dimensions

    @property
    def level_downsamples(self) -> dict[int, tuple[float, float]]:
        """Image downsample factors (height, width) for each pyramid level."""
        return self._reader.level_downsamples

    def read_level(self, level: int) -> np.ndarray:
        """Read full pyramid level data.

        Parameters:
            level (int):
                Slide pyramid level to read.

        Raises:
            ValueError: Invalid level argument.

        Returns:
            np.ndarray:
                Array containing image data from `level`.
        """
        return self._reader.read_level(level=level)

    def read_region(
        self, xywh: tuple[int, int, int, int], level: int = 0
    ) -> np.ndarray:
        """Read region based on `xywh`-coordinates.

        Parameters:
            xywh (tuple[int, int, int, int]):
                Coordinates for the region.
            level (int, default=0):
                Slide pyramid level to read from.

        Raises:
            ValueError: Invalid `level` argument.

        Returns:
            np.ndarray:
                Array containing image data from `xywh`-region.
        """
        return self._reader.read_region(xywh=xywh, level=level)

    def level_from_max_dimension(self, max_dimension: int = 4096) -> int:
        """Find pyramid level with *both* dimensions less or equal to `max_dimension`.
        If one isn't found, return the last pyramid level.

        Parameters:
            max_dimension (int, default=4096):
                Maximum dimension for the level.

        Returns:
            int:
                Slide pyramid level.
        """
        for level, (level_h, level_w) in self.level_dimensions.items():
            if level_h <= max_dimension and level_w <= max_dimension:
                return level
        return list(self.level_dimensions.keys())[-1]

    def level_from_dimensions(self, dimensions: tuple[int, int]) -> int:
        """Find pyramid level which is closest to `dimensions`.

        Parameters:
            dimensions (tuple[int, int]):
                Height and width.

        Returns:
            int:
                Slide pyramid level.
        """
        height, width = dimensions
        available = []
        distances = []
        for level, (level_h, level_w) in self.level_dimensions.items():
            available.append(level)
            distances.append(abs(level_h - height) + abs(level_w - width))
        return available[distances.index(min(distances))]

    def get_tissue_mask(
        self,
        *,
        level: Optional[int] = None,
        threshold: Optional[int] = None,
        multiplier: float = 1.05,
        sigma: float = 0.0,
    ) -> tuple[int, np.ndarray]:
        """Detect tissue from slide pyramid level image.

        Parameters:
            level (int, default=None):
                Slide pyramid level to use for tissue detection. If None, uses the
                `level_from_max_dimension` method.
            threshold (int, default=None):
                Threshold for tissue detection. If set, will detect tissue by global
                thresholding. Otherwise Otsu's method is used to find a threshold.
            multiplier (float, default=1.05):
                Otsu's method finds an optimal threshold by minimizing the weighted
                within-class variance. This threshold is then multiplied with
                `multiplier`. Ignored if `threshold` is not None.
            sigma (float, default=0.0):
                Sigma for gaussian blurring.

        Raises:
            ValueError: Threshold not between 0 and 255.

        Returns:
            tuple[int, np.ndarray]:
                Threshold and tissue mask.
        """
        level = (
            self.level_from_max_dimension()
            if level is None
            else format_level(level, available=list(self.level_dimensions))
        )
        return get_tissue_mask(
            image=self.read_level(level),
            threshold=threshold,
            multiplier=multiplier,
            sigma=sigma,
        )

    def get_tile_coordinates(
        self,
        width: int,
        *,
        tissue_mask: Optional[np.ndarray],
        height: Optional[int] = None,
        overlap: float = 0.0,
        max_background: float = 0.95,
        out_of_bounds: bool = True,
    ) -> TileCoordinates:
        """Generate tile coordinates.

        Parameters:
            width (int):
                Width of a tile.
            tissue_mask (np.ndarray, default=None):
                Tissue mask for filtering tiles with too much background. If None,
                the filtering is disabled.
            height (int, default=None):
                Height of a tile. If None, will be set to `width`.
            overlap (float, default=0.0):
                Overlap between neighbouring tiles.
            max_background (float, default=0.95):
                Maximum proportion of background in tiles. Ignored if `tissue_mask`
                is None.
            out_of_bounds (bool, default=True):
                Keep tiles which contain regions outside of the image.

        Raises:
            ValueError: Height and/or width are smaller than 1.
            ValueError: Height and/or width is larger than dimensions.
            ValueError: Overlap is not in range [0, 1).

        Returns:
            TileCoordinates:
                `TileCoordinates` dataclass.
        """
        tile_coordinates = get_tile_coordinates(
            dimensions=self.dimensions,
            width=width,
            height=height,
            overlap=overlap,
            out_of_bounds=out_of_bounds,
        )
        if tissue_mask is not None:
            all_backgrounds = get_background_percentages(
                tile_coordinates=tile_coordinates,
                tissue_mask=tissue_mask,
                downsample=get_downsample(tissue_mask, self.dimensions),
            )
            filtered_coordinates = []
            for xywh, background in zip(tile_coordinates, all_backgrounds):
                if background <= max_background:
                    filtered_coordinates.append(xywh)
            tile_coordinates = filtered_coordinates

        return TileCoordinates(
            coordinates=tile_coordinates,
            width=width,
            height=width if height is None else height,
            overlap=overlap,
            max_background=None if tissue_mask is None else max_background,
            tissue_mask=tissue_mask,
        )

    def get_spot_coordinates(
        self,
        tissue_mask: np.ndarray,
        *,
        min_area_pixel: int = 10,
        max_area_pixel: Optional[int] = None,
        min_area_relative: float = 0.2,
        max_area_relative: Optional[float] = 2.0,
    ) -> SpotCoordinates:
        """Generate tissue microarray spot coordinates.

        Parameters:
            tissue_mask:
                Tissue mask of the slide. It's recommended to increase `sigma` value when
                detecting tissue to remove non-TMA spots from the mask. Rest of the areas
                can be handled with the following arguments.
            min_area_pixel (int, default=10):
                Minimum pixel area for contours.
            max_area_pixel (int, default=None):
                Maximum pixel area for contours.
            min_area_relative (float, default=0.2):
                Relative minimum contour area, calculated from the median contour area
                after filtering contours with `[min,max]_pixel` arguments
                (`min_area_relative * median(contour_areas)`).
            max_area_relative (float, default=2.0):
                Relative maximum contour area, calculated from the median contour area
                after filtering contours with `[min,max]_pixel` arguments
                (`max_area_relative * median(contour_areas)`).

        Returns:
            SpotCoordinates:
                `SpotCoordinates` instance.
        """
        spot_mask = clean_tissue_mask(
            tissue_mask=tissue_mask,
            min_area_pixel=min_area_pixel,
            max_area_pixel=max_area_pixel,
            min_area_relative=min_area_relative,
            max_area_relative=max_area_relative,
        )
        # Dearray spots.
        spot_info = get_spot_coordinates(spot_mask)
        spot_coordinates = [  # upsample to level zero.
            _multiply_xywh(x, get_downsample(tissue_mask, self.dimensions))
            for x in spot_info.values()
        ]

        return SpotCoordinates(
            coordinates=spot_coordinates,
            spot_names=list(spot_info.keys()),
            tissue_mask=spot_mask,
        )

    def get_annotated_thumbnail(
        self,
        image: np.ndarray,
        coordinates: Iterator[tuple[int, int, int, int]],
        linewidth: int = 1,
    ) -> Image.Image:
        """Generate annotated thumbnail from coordinates.

        Parameters:
            image (np.ndarray):
                Input image.
            coordinates (Iterator[tuple[int, int, int, int]]):
                Coordinates to annotate.
            linewidth (int, default=1):
                Width of rectangle lines.

        Returns:
            PIL.Image.Image:
                Annotated thumbnail.
        """
        kwargs = {
            "image": image,
            "downsample": get_downsample(image, self.dimensions),
            "rectangle_width": linewidth,
        }
        if isinstance(coordinates, SpotCoordinates):
            text_items = [x.lstrip("spot_") for x in coordinates.spot_names]
            kwargs.update(
                {"coordinates": coordinates.coordinates, "text_items": text_items}
            )
        elif isinstance(coordinates, TileCoordinates):
            kwargs.update(
                {"coordinates": coordinates.coordinates, "highlight_first": True}
            )
        else:
            kwargs.update({"coordinates": coordinates})
        return get_annotated_image(**kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(path={self.path}, "
            f"backend={self._reader.BACKEND_NAME})"
        )
