from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._base_reader import SlideReaderBackend, _pixel_size_um
from .tiles import _pad_tile, format_level

if TYPE_CHECKING:
    import torch

try:
    from cucim import CuImage

    HAS_CUCIM = True
except ImportError:
    HAS_CUCIM = False


CUCIM_READABLE_FORMATS = (".svs", ".tiff", ".tif")


class CucimReader(SlideReaderBackend):
    """Slide reader using `CUCIM` as a backend."""

    BACKEND_NAME = "CUCIM"

    def __init__(self, path: str) -> None:
        """Initialize CUCIMBackend class instance.

        Args:
            path: Path to the slide image.

        Raises:
            ImportError: CUCIM could not be imported.
        """
        if not HAS_CUCIM:
            raise ImportError(
                "cuCIM not installed. Install cucim-cu12 or cucim-cu13 "
                "matching your CUDA version.",
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
    def pixel_size(self) -> tuple[float, float]:
        """Level-zero pixel size (height, width) in micrometers."""
        units = self.__reader.spacing_units("YX")
        # Aperio SVS stores isotropic MPP separately from generic TIFF spacing.
        if not any(units) and "MPP" in self.__reader.metadata.get("aperio", {}):
            mpp = self.__reader.metadata["aperio"]["MPP"]
            return _pixel_size_um((mpp, mpp))
        return _pixel_size_um(self.__reader.spacing("YX"), units)

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

    def read_region(
        self,
        xywh: tuple[int, int, int, int],
        level: int,
        *,
        device: str | torch.device | None = None,
    ) -> np.ndarray | torch.Tensor:
        """Read RGB uint8 HWC pixels on CPU, or a CUDA tensor when requested.

        Args:
            xywh: Region bounds in level-zero pixels.
            level: Pyramid level; output size is downsampled accordingly.
            device: None returns a NumPy array. A CUDA device returns a tensor
                sharing the decoder allocation through DLPack. Padding stays
                on that device; out-of-slide pixels are white.

        Raises:
            ValueError: The device is not CUDA or bounds are invalid.
        """
        if device is not None:
            import torch

            device = torch.device(device)
            if device.type != "cuda":
                raise ValueError("cuCIM tensor reads require a CUDA device")
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
        level = format_level(level, available=list(self.level_dimensions))
        x, y, width, height = xywh
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("Region needs nonnegative x/y and positive width/height")
        down_y, down_x = self.level_downsamples[level]
        w, h = round(width / down_x), round(height / down_y)
        # Locations stay at level zero; only the decoded sizes are downsampled.
        allowed_w = min(w, max(0, int(np.ceil((self.dimensions[1] - x) / down_x))))
        allowed_h = min(h, max(0, int(np.ceil((self.dimensions[0] - y) / down_y))))
        if allowed_w == 0 or allowed_h == 0:
            if device is None:
                return np.full((h, w, 3), 255, dtype=np.uint8)
            return torch.full((h, w, 3), 255, dtype=torch.uint8, device=device)
        tile = self.__reader.read_region(
            location=(x, y),
            level=level,
            size=(allowed_w, allowed_h),
            device="cpu" if device is None else str(device),
        )
        if device is None:
            return _pad_tile(np.array(tile)[..., :3], shape=(h, w))

        import cupy as cp
        from torch.nn.functional import pad

        # CuPy keeps the decoder owner alive; DLPack shares its CUDA storage.
        tensor = torch.from_dlpack(cp.asarray(tile))[..., :3]
        if tensor.device != device:
            raise ValueError("cuCIM returned pixels on a different CUDA device")
        if tensor.shape[:2] != (h, w):
            tensor = pad(
                tensor,
                (0, 0, 0, w - tensor.shape[1], 0, h - tensor.shape[0]),
                value=255,
            )
        return tensor
