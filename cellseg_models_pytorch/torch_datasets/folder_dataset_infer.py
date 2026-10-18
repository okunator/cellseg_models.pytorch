from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
from torch.utils.data import Dataset

from cellseg_models_pytorch.utils import FileHandler

SUFFIXES = (".jpeg", ".jpg", ".png")


__all__ = ["FolderDatasetInfer"]


class FolderDatasetInfer(Dataset):
    """Load RGB images from a folder for inference.

    Samples contain ``image`` and ``name``. Without a transform, images are
    uint8 NumPy arrays with height, width, channels axes and no normalization.
    """

    def __init__(
        self,
        path: Union[str, Path],
        pattern: str = "*",
        transform: Optional[Callable[[np.ndarray], Any]] = None,
        n_images: int = None,
    ) -> None:
        """Select images and an optional image transform.

        Args:
            path: Existing folder containing images.
            pattern: Glob pattern evaluated inside the folder. Every selected entry
                must have a lowercase .jpeg, .jpg or .png suffix.
            transform: Callable receiving an image positionally. Its return value
                becomes the sample's image. For Albumentations Compose, adapt the call
                to supply image by keyword and extract its image result.
            n_images: Optional limit applied after sorting paths, before suffix checks.

        Raises:
            ValueError: The path is missing/not a directory, or a selected suffix
                is unsupported.
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
        """Return the number of selected image files."""
        return len(self.fnames)

    def __getitem__(self, ix: int) -> dict[str, Any]:
        """Read one selected image.

        Args:
            ix: Index in the sorted, optionally limited file list.

        Returns:
            Dictionary with ``image`` (RGB HWC NumPy array, or the transform's
            return value) and ``name`` (filename stem without extension).
        """
        fn = self.fnames[ix]
        im = FileHandler.read_img(fn.as_posix())
        if self.transform is not None:
            im = self.transform(im)

        return {"image": im, "name": fn.stem}
