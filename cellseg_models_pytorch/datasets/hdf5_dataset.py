from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np

from ._base_dataset import TrainDatasetBase

try:
    import tables as tb
except Exception:
    raise ImportError(
        "`pytables` needed for this class. Install with: `pip install tables`"
    )


__all__ = ["SegmentationHDF5Dataset"]


class SegmentationHDF5Dataset(TrainDatasetBase):
    def __init__(
        self,
        path: str,
        img_transforms: List[str],
        inst_transforms: List[str],
        normalization: str = None,
        return_inst: bool = False,
        return_binary: bool = True,
        return_type: bool = True,
        return_sem: bool = False,
        return_weight: bool = False,
        **kwargs,
    ) -> None:
        """Create a dataset class that reads images/patches from a hdf5 database.

        NOTE: The h5-db needs to be written with `HDF5Writer` (/utils/h5_writer.py).

        Parameters
        ----------
            path : str
                Path to the hdf5 db.
            img_transforms : List[str]
                A list containing all the transformations that are applied to the input
                images and corresponding masks. Allowed ones: "blur", "non_spatial",
                "non_rigid", "rigid", "hue_sat", "random_crop", "center_crop", "resize"
            inst_transforms : List[str]
                A list containg all the transformations that are applied to only the
                instance labelled masks. Allowed ones: "cellpose", "contour", "dist",
                "edgeweight", "hovernet", "omnipose", "smooth_dist", "binarize"
            normalization : str, optional
                Apply img normalization after all the transformations. One of "minmax",
                "norm", "percentile", None.
            return_inst : bool, default=False
                If True, returns the instance labelled mask. (If the db contains these.)
            return_binary : bool, default=True
                If True, returns a binarized instance mask. (If the db contains these.)
            return_type : bool, default=True
                If True, returns a type mask. (If the db contains these.)
            return_sem : bool, default=False
                If True, returns a semantic mask, (If the db contains these.)
            return_weight : bool, default=False
                Include a nuclear border weight map in the output.
        """
        super().__init__(
            img_transforms=img_transforms,
            inst_transforms=inst_transforms,
            normalization=normalization,
            return_inst=return_inst,
            return_type=return_type,
            return_sem=return_sem,
            return_weight=return_weight,
            return_binary=return_binary,
        )

        self.path = Path(path)

        if self.path.suffix not in (".h5", ".hdf5"):
            raise ValueError(
                f"""The input path has to be a hdf5 db. Got suffix: {self.path.suffix}
                Allowed suffices: {(".h5", ".hdf5")}"""
            )

        # n imgs in the db.
        with tb.open_file(self.path) as h5:
            self.n_items = h5.root._v_attrs["n_items"]

    def read_h5_patch(
        self,
        ix: int,
        return_type: bool = True,
        return_sem: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Read img & mask patches at index `ix`.

        Parameters
        ----------
            ix : int
                Index for the hdf5 db-arrays.
            return_type : bool, default=True
                If True, returns a type mask. (If the db contains these.)
            return_sem : bool, default=False
                If True, returns a semantic mask, (If the db contains these.)

        Returns
        -------
            OrderedDict[str, np.ndarray]:
                A Dict of numpy matrices. Img shape: (H, W, 3), mask shapes: (H, W).
                keys of the dict are: "im", "inst", "type", "sem"

        Raises
        ------
            IOError: If a mask that does not exist in the db is being read.
        """
        with tb.open_file(self.path.as_posix(), "r") as h5:
            out = OrderedDict()
            out["image"] = h5.root.imgs[ix, ...]

            try:
                out["inst"] = h5.root.insts[ix, ...]
            except Exception:
                raise IOError(
                    "The HDF5 database does not contain instance labelled masks."
                )

            if return_type:
                try:
                    out["type"] = h5.root.types[ix, ...]
                except Exception:
                    raise IOError("The HDF5 database does not contain type masks.")

            if return_sem:
                try:
                    out["sem"] = h5.root.areas[ix, ...]
                except Exception:
                    raise IOError("The HDF5 database does not contain semantic masks.")

            return out

    def __len__(self) -> int:
        """Return the number of items in the db."""
        return self.n_items

    def __getitem__(self, ix: int) -> Dict[str, np.ndarray]:
        """Get item.

        Parameters
        ----------
            ix : int
                An index for the iterable dataset.

        Returns
        -------
            Dict[str, np.ndarray]:
                A dictionary containing all the augmented data patches.
                Keys are: "im", "inst", "type", "sem". Image shape: (B, 3, H, W).
                Mask shapes: (B, C_mask, H, W).

        """
        return self._getitem(ix, self.read_h5_patch)
