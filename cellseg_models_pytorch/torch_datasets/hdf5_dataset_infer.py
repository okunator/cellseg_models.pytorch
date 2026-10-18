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
    """Load image tensors and filenames from an existing HDF5 file.

    This reader expects root nodes imgs and fnames. It does not use the configurable
    training schema or apply image normalization.
    """

    def __init__(self, path: Union[str, Path], n_images: int = None, **kwargs) -> None:
        """Choose an HDF5 image file and optional sample limit.

        Args:
            path: .h5 or .hdf5 file with imgs shaped (N, H, W, C) and UTF-8 byte
                filenames in fnames, in matching order.
            n_images: Optional limit on the stored filename list and dataset length.
            **kwargs: Additional keyword arguments are accepted but unused.

        Raises:
            ImportError: PyTables is unavailable.
            ValueError: The file suffix is unsupported. Missing files/nodes raise
                the underlying PyTables error.
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
        """Return the number of selected stored filenames."""
        return len(self.fnames)

    def __getitem__(self, ix: int) -> dict[str, torch.Tensor | str]:
        """Read one HDF5 image and its basename.

        Args:
            ix: Sample index.

        Returns:
            Dictionary with ``im`` as a CHW CPU tensor retaining the stored dtype
            and pixel values, and ``file`` as a decoded filename including extension.
            These keys differ from FolderDatasetInfer's image/name keys.
        """
        fn = self.fnames[ix]

        with tb.open_file(self.path.as_posix(), "r") as h5:
            im = h5.root.imgs[ix, ...]

        im = torch.from_numpy(im.transpose(2, 0, 1))
        return {"im": im, "file": Path(fn.decode("UTF-8")).name}
