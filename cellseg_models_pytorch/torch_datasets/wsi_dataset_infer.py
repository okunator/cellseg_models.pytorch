"""Ported from https://github.com/jopo666/HistoPrep/blob/master/histoprep/utils/_torch.py
with very minor modifications.

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

from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from cellseg_models_pytorch.wsi.reader import SlideReader

__all__ = ["WSIDatasetInfer"]


class WSIDatasetInfer(Dataset):
    """Read slide regions as HWC images for tiled inference.

    Unlike the HDF5 inference reader, this dataset keeps channels last. Use the
    whole-slide guide for the reader and tile-coordinate workflow.
    """

    def __init__(
        self,
        reader: SlideReader,
        coordinates: Sequence[tuple[int, int, int, int]],
        level: int = 0,
        transforms: Callable[..., dict[str, np.ndarray | torch.Tensor]] | None = None,
        *,
        read_device: str | torch.device | None = None,
    ) -> None:
        """Choose slide regions, read level and optional normalization.

        Args:
            reader: SlideReader used to read regions and supply the slide name.
            coordinates: Indexable sequence of (x, y, width, height) tuples following
                SlideReader.read_region's coordinate contract. Pass a TileCoordinates
                object's coordinates field for a plain sequence.
            level: Slide pyramid level passed to read_region.
            transforms: Callable receiving image=tile and returning a dictionary
                with an HWC image. Albumentations Compose follows this contract.
            read_device: None reads NumPy tiles on CPU. A CUDA device requests GPU
                tensors through cuCIM; transforms must then accept and return HWC tensors.
        """
        super().__init__()
        self.reader = reader
        self.coordinates = coordinates
        self.level = level
        self.transforms = transforms
        self.read_device = read_device

    def __len__(self) -> int:
        """Return the number of requested regions."""

        return len(self.coordinates)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | torch.Tensor | str]:
        """Read and optionally transform one slide region.

        Args:
            index: Position in the supplied coordinate sequence.

        Returns:
            Dictionary with image (HWC NumPy array or tensor), name (slide name),
            and coords (NumPy array holding x, y, width, height). Pixel values and
            image dtype follow the reader and transform; this dataset does not
            automatically normalize or convert to channels-first tensors.
        """

        xywh = self.coordinates[index]
        if self.read_device is None:
            tile = self.reader.read_region(xywh, level=self.level)
        else:
            tile = self.reader.read_region(
                xywh, level=self.level, device=self.read_device
            )
        if self.transforms is not None:
            tile = self.transforms(image=tile)["image"]

        return {"image": tile, "name": self.reader.name, "coords": np.array(xywh)}
