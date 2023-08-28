from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset

from ..utils import FileHandler

SUFFIXES = (".jpeg", ".jpg", ".tif", ".tiff", ".png")


__all__ = ["FolderDatasetInfer"]


class FolderDatasetInfer(Dataset, FileHandler):
    def __init__(
        self, path: Union[str, Path], pattern: str = "*", n_images: int = None
    ) -> None:
        """Folder dataset that can be used during inference for loading images.

        NOTE: loads only images.

        Parameters
        ----------
            path : str | Path
                Path to the folder containing image files.
            pattern: str, default="*"
                File pattern for filtering only the files that contain the pattern.
            n_images : int, optional
                First n-number of images used from the folder.

        Raises
        ------
            ValueError if `path` does not exist.
            ValueError if `path` is not a folder.
            ValueError if `path` contains images with illegal suffices.
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

    def __len__(self) -> int:
        """Length of folder."""
        return len(self.fnames)

    def __getitem__(self, ix: int) -> torch.Tensor:
        """Read image."""
        fn = self.fnames[ix]
        if fn.suffix in SUFFIXES:
            im = FileHandler.read_img(fn.as_posix())
            im = torch.from_numpy(im.transpose(2, 0, 1))
        else:
            raise ValueError(
                f"Received an illegal file format in: {fn}."
                f"file format {fn.suffix} not supported."
                f"Supported formats: {SUFFIXES}."
            )

        return {"im": im, "file": fn.with_suffix("").name}
