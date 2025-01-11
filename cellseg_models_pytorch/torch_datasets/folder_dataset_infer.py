from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from cellseg_models_pytorch.utils import FileHandler

SUFFIXES = (".jpeg", ".jpg", ".png")


__all__ = ["FolderDatasetInfer"]


class FolderDatasetInfer(Dataset):
    def __init__(
        self,
        path: Union[str, Path],
        pattern: str = "*",
        transform: Optional[Callable[[np.ndarray], Any]] = None,
        n_images: int = None,
    ) -> None:
        """Folder dataset that can be used during inference for loading images.

        NOTE: loads only images.

        Parameters
        ----------
            path : str | Path
                Path to the folder containing image files.
            pattern: str, default="*"
                File pattern for filtering only the files that contain the pattern.
            transform : Callable, optional
                Transform to be applied to the images.
            n_images : int, optional
                First n-number of images used from the folder.

        Raises
        ------
            ValueError if `path` does not exist.
            ValueError if `path` is not a folder.
        """
        super().__init__()

        folder_path = Path(path)
        if not folder_path.exists():
            raise ValueError(f"folder: {folder_path} does not exist")

        if not folder_path.is_dir():
            raise ValueError(f"path: {folder_path} is not a folder")

        self.fnames = sorted(folder_path.glob(pattern))
        if n_images is not None:
            self.fnames = self.fnames[:n_images]

        illegal_suffix_files = [fn for fn in self.fnames if fn.suffix not in SUFFIXES]
        if illegal_suffix_files:
            raise ValueError(
                f"Following files have illegal suffixes: {illegal_suffix_files}"
                "Allowed suffixes are: '.jpeg', '.jpg', '.png'"
            )

        self.transform = transform

    def __len__(self) -> int:
        """Length of folder."""
        return len(self.fnames)

    def __getitem__(self, ix: int) -> torch.Tensor:
        """Read image."""
        fn = self.fnames[ix]
        im = FileHandler.read_img(fn.as_posix())
        if self.transform is not None:
            im = self.transform(im)

        return {"image": im, "name": fn.stem}
