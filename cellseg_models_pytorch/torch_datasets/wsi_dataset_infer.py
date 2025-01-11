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

from typing import Any, Callable, Dict, Iterator, Optional

import numpy as np
from torch.utils.data import Dataset

from cellseg_models_pytorch.wsi.reader import SlideReader

__all__ = ["WSIDatasetInfer"]


class WSIDatasetInfer(Dataset):
    """Torch dataset yielding tile images and `xywh`-coordinates from reader (requires
    `PyTorch`)."""

    def __init__(
        self,
        reader: SlideReader,
        coordinates: Iterator[tuple[int, int, int, int]],
        level: int = 0,
        transform: Optional[Callable[[np.ndarray], Any]] = None,
    ) -> None:
        """Initialize WSIReaderDataset.

        Parameters:
            reader (SlideReader):
                `SlideReader` instance.
            coordinates (Iterator[tuple[int, int, int, int]]):
                Iterator of xywh-coordinates.
            level (int):
                Slide level for reading tile image. Defaults to 0.
            transform (Optional[Callable[[np.ndarray], Any]]):
                Transform function for tile images. Defaults to None.

        Raises:
            ImportError: Could not import `PyTorch`.
        """
        super().__init__()
        self.reader = reader
        self.coordinates = coordinates
        self.level = level
        self.transform = transform

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        xywh = self.coordinates[index]
        tile = self.reader.read_region(xywh, level=self.level)

        if self.transform is not None:
            tile = self.transform(tile)

        return {"image": tile, "name": self.reader.name, "coords": np.array(xywh)}
