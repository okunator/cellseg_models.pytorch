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

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

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


if TYPE_CHECKING:
    import torch


class SlideReader:
    """Read slide images, find tissue and choose tiles for prediction."""

    def __init__(
        self,
        path: Union[str, Path],
        backend: str = "OPENSLIDE",
    ) -> None:
        """Open a slide with the selected reader.

        Args:
            path: Path to a slide supported by the backend.
            backend: "OPENSLIDE", "CUCIM" or "BIOIO". Install the chosen reader and
                its native libraries or format plugins before opening the slide.

        Raises:
            FileNotFoundError: The slide path does not exist.
            ValueError: The backend name or file suffix is unsupported.
            ImportError: The selected reader is not installed.
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
        """Absolute slide file path."""
        return self._reader.path

    @property
    def name(self) -> str:
        """Slide filename without its final extension."""
        return self._reader.name

    @property
    def suffix(self) -> str:
        """Final file extension, including the dot."""
        return self._reader.suffix

    @property
    def backend_name(self) -> str:
        """Active reader name: OPENSLIDE, CUCIM or BIOIO."""
        return self._reader.BACKEND_NAME

    @property
    def data_bounds(self) -> tuple[int, int, int, int]:
        """Image-data bounds as (x, y, width, height) in level-zero pixels.

        Some formats store tissue in a smaller area than the full image.
        Region coordinates still refer to the full image origin.
        """
        return self._reader.data_bounds

    @property
    def dimensions(self) -> tuple[int, int]:
        """Full-resolution slide size as (height, width) in pixels."""
        return self._reader.dimensions

    @property
    def pixel_size(self) -> tuple[float, float]:
        """Level-zero pixel size as (height, width) in micrometers.

        The two values can differ. They describe physical size, not level downsampling.

        Raises:
            ValueError: Calibration is missing, invalid or uses unsupported units.
        """
        return self._reader.pixel_size

    @property
    def level_count(self) -> int:
        """Number of image levels, from full resolution to smaller images."""
        return self._reader.level_count

    @property
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        """Dictionary mapping each level to its (height, width) in pixels."""
        return self._reader.level_dimensions

    @property
    def level_downsamples(self) -> dict[int, tuple[float, float]]:
        """Dictionary mapping each level to its height/width reduction factors.

        For example, (2.0, 2.0) means half as many pixels along each axis.
        """
        return self._reader.level_downsamples

    def read_level(self, level: int) -> np.ndarray:
        """Read a whole image level into CPU memory.

        Args:
            level: Image level; 0 is full resolution and -1 is the smallest level.

        Returns:
            OpenSlide/CUCIM return an RGB uint8 NumPy image with height, width,
            channels axes. OpenSlide uses
            its thumbnail reader, which can return a slightly different image size
            from the level metadata. A full-resolution read can be large.
            BIOIO whole-level reading currently needs a separate correction.

        Raises:
            ValueError: The requested level is unavailable.
        """
        return self._reader.read_level(level=level)

    def read_region(
        self,
        xywh: tuple[int, int, int, int],
        level: int = 0,
        *,
        device: str | torch.device | None = None,
    ) -> np.ndarray | torch.Tensor:
        """Read one rectangular region of the slide.

        Args:
            xywh: (x, y, width, height) bounds in level-zero pixels, even when reading
                another level. Out-of-image areas are padded with white pixels.
            level: Image level to read; 0 gives full resolution.
            device: None returns a CPU NumPy array. A CUDA device is supported only
                by CUCIM and returns a tensor on that device.

        Returns:
            RGB uint8 pixels with height, width, channels axes. Width and height
            are reduced by the chosen level's downsample factors.

        Raises:
            ValueError: The level or device request is unsupported.
        """
        if device is not None:
            return self._reader.read_region(xywh=xywh, level=level, device=device)
        return self._reader.read_region(xywh=xywh, level=level)

    def level_from_max_dimension(self, max_dimension: int = 4096) -> int:
        """Choose a level small enough to fit within the requested size.

        Args:
            max_dimension: Maximum height and width, in pixels.

        Returns:
            First level with both dimensions within the limit, or the smallest
            available level if none fits.
        """
        for level, (level_h, level_w) in self.level_dimensions.items():
            if level_h <= max_dimension and level_w <= max_dimension:
                return level
        return list(self.level_dimensions.keys())[-1]

    def level_from_dimensions(self, dimensions: tuple[int, int]) -> int:
        """Choose the level closest to an image size.

        Args:
            dimensions: Desired (height, width) in pixels.

        Returns:
            Level with the smallest total difference in height and width.
            Ties use the first matching level.
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
        """Find tissue on a slide image level.

        Args:
            level: Level for detection. None chooses a level with
                level_from_max_dimension().
            threshold: Grayscale cutoff from 0 to 255. None chooses it automatically
                with Otsu's method. Darker pixels are treated as tissue.
            multiplier: Scale the automatic cutoff; ignored with a fixed threshold.
            sigma: Blur strength in detection-image pixels; 0 disables blurring.

        Returns:
            (threshold, mask), with a 2D uint8 mask at the detection image size:
            1 is tissue and 0 is background. The first value is the cutoff, not
            the image level.

        Raises:
            ValueError: The threshold or level is invalid.
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
        """Choose slide tiles, optionally excluding mostly empty regions.

        Args:
            width: Tile width in level-zero pixels; no larger than the slide width.
            tissue_mask: 2D tissue mask spanning the full slide, scaled equally
                along both axes. Pass None explicitly to disable background filtering.
            height: Tile height in level-zero pixels. None uses width.
            overlap: Fraction shared by neighboring tiles, in [0, 1).
            max_background: Largest allowed background fraction. Ignored without
                a tissue mask. Outside-slide pixels count as background.
            out_of_bounds: Keep tiles extending beyond the slide; reads pad those areas.

        Returns:
            TileCoordinates with level-zero (x, y, width, height) bounds in row order.
            An all-background mask can leave the collection empty.

        Raises:
            TypeError: Tile dimensions are not integers.
            ValueError: Tile dimensions are invalid or overlap is outside [0, 1).
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
        """Find tissue microarray spots from a tissue mask.

        Args:
            tissue_mask: 2D uint8 full-slide mask, scaled equally along both axes.
                Use 1 for tissue and 0 for background. Remove artifacts first.
            min_area_pixel: Smallest contour area in mask pixels.
            max_area_pixel: Largest contour area in mask pixels, or None for no limit.
            min_area_relative: Smallest area as a fraction of the median area after
                pixel-area filtering.
            max_area_relative: Largest area as a multiple of that median, or None.

        Returns:
            SpotCoordinates with level-zero bounds, corresponding spot_names and
            the cleaned mask at its original resolution. Bounds and names have
            matching order; names describe positions in the detected spot grid.
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
        coordinates: Sequence[tuple[int, int, int, int]]
        | TileCoordinates
        | SpotCoordinates,
        linewidth: int = 1,
    ) -> Image.Image:
        """Draw tile or spot rectangles on a slide overview.

        Args:
            image: RGB uint8 NumPy overview with the slide's aspect ratio.
            coordinates: Level-zero bounds, as a TileCoordinates, SpotCoordinates
                or a list of (x, y, width, height) tuples.
            linewidth: Rectangle line width in overview pixels.

        Returns:
            A new PIL image. Spot collections include names; tile collections
            highlight the first tile. Coordinates are scaled to the overview size.
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
