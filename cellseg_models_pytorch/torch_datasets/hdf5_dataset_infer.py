from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset

from cellseg_models_pytorch.utils import FileHandler

try:
    import tables as tb

    HAS_TABLES = True
except Exception:
    HAS_TABLES = False


__all__ = ["HDF5DatasetInfer"]


class HDF5DatasetInfer(Dataset, FileHandler):
    def __init__(self, path: Union[str, Path], n_images: int = None, **kwargs) -> None:
        """Folder dataset that can be used during inference for loading images.

        NOTE: loads only images.

        Parameters
        ----------
            path : str | Path
                Path to the folder containing image files.
            n_images : int, optional
                First n-number of images used from the folder.

        Raises
        ------
            ValueError if the input path has incorrect suffix.
        """
        if not HAS_TABLES:
            raise ImportError(
                "`pytables` needed for this class. Install with: `pip install tables`"
            )

        self.path = Path(path)

        if self.path.suffix not in (".h5", ".hdf5"):
            raise ValueError(
                f"The input path has to be a hdf5 db. Got suffix: {self.path.suffix} "
                "Allowed suffices: {('.h5', '.hdf5')}"
            )

        super().__init__()

        with tb.open_file(self.path) as h5:
            if n_images is not None:
                self.fnames = h5.root.fnames[:n_images]
            else:
                self.fnames = h5.root.fnames[:]

    def __len__(self) -> int:
        """Return the number of items in the db."""
        return len(self.fnames)

    def __getitem__(self, ix: int) -> torch.Tensor:
        """Read image."""
        fn = self.fnames[ix]

        with tb.open_file(self.path.as_posix(), "r") as h5:
            im = h5.root.imgs[ix, ...]

        im = torch.from_numpy(im.transpose(2, 0, 1))
        return {"im": im, "file": Path(fn.decode("UTF-8")).name}
