from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..datasets.folder_dataset_train import SegmentationFolderDataset
from ..datasets.hdf5_dataset import SegmentationHDF5Dataset

__all__ = ["BaseDataModule"]


DATASET_LOOKUP = {
    "hdf5": SegmentationHDF5Dataset,
    "folder": SegmentationFolderDataset,
}


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_path: str,
        valid_data_path: str,
        test_data_path: str,
        img_transforms: List[str],
        inst_transforms: List[str],
        train_mask_path: str = None,
        valid_mask_path: str = None,
        test_mask_path: str = None,
        normalization: str = None,
        return_weight: bool = False,
        return_inst: bool = True,
        return_type: bool = True,
        return_sem: bool = False,
        batch_size: int = 8,
        num_workers: int = 8,
    ) -> None:
        """Set up a base datamodule for any of the inbuilt dataset types.

        Parameters
        ----------
            train_data_path : str, optional
                Path to train data folder or database.
            valid_data_path : str, optional
                Path to validation data folder or database.
            test_data_path : str, optional
                Path to the test data folder or database.
            img_transforms : List[str]
                A list containing all the transformations that are applied to the input
                images and corresponding masks. Allowed ones: "blur", "non_spatial",
                "non_rigid", "rigid", "hue_sat", "random_crop", "center_crop", "resize"
            inst_transforms : List[str]
                A list containg all the transformations that are applied to only the
                instance labelled masks. Allowed ones: "cellpose", "contour", "dist",
                "edgeweight", "hovernet", "omnipose", "smooth_dist", "binarize"
            train_mask_path : str, optional
                Path to the masks of the training dataset. Needed for a folder dataset.
            valid_mask_path : str, optional
                Path to the masks of the valid dataset. Needed for a folder dataset.
            test_mask_path : str, optional
                Path to the masks of the test dataset. Needed for a folder dataset.
            normalization : str, optional
                Apply img normalization after all the transformations. One of "minmax",
                "norm", "percentile", None.
            return_inst : bool, default=True
                If True, Dataloader returns an instance labelled mask.
            return_type : bool, default=True
                If True, Dataloader returns a type mask.
            return_sem : bool, default=False
                If True, Dataloeader returns a semantic mask.
            return_weight : bool, default=False
                Include a nuclear border weight map in the output.
            batch_size : int, default=8
                Batch size for the dataloader
            num_workers : int, default=8
                number of cpu cores/threads used in the dataloading process.

        Raises
        ------
            ValueError: If a wrong dataset type is given.
        """
        super().__init__()
        self.train_data_path = Path(train_data_path)
        self.valid_data_path = Path(valid_data_path)
        self.test_data_path = Path(test_data_path)
        self.train_mask_path = Path(train_mask_path) if train_mask_path else None
        self.valid_mask_path = Path(valid_mask_path) if valid_mask_path else None
        self.test_mask_path = Path(test_mask_path) if test_mask_path else None
        self.train_ds_type = self.get_ds_type(self.train_data_path)
        self.valid_ds_type = self.get_ds_type(self.valid_data_path)
        self.test_ds_type = self.get_ds_type(self.test_data_path)

        self.img_transforms = img_transforms
        self.inst_transforms = inst_transforms
        self.normalization = normalization
        self.return_weight = return_weight
        self.return_inst = return_inst
        self.return_type = return_type
        self.return_sem = return_sem
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_ds_type(self, path: Path) -> str:
        """Infer the dataset type from file suffix."""
        h5_suffices = (".h5", ".hdf5", "he5")
        if path.is_file():
            if path.suffix in h5_suffices:
                suf = "hdf5"
            else:
                raise ValueError(
                    f"Illegal file suffix. Got: {path.suffix}. Allowed: {h5_suffices}."
                )
        elif path.is_dir():
            suf = "folder"
        else:
            raise ValueError(f"{path} not a folder nor .h5 db.")

        return suf

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the train, valid, and test datasets."""
        self.trainset = DATASET_LOOKUP[self.train_ds_type](
            path=self.train_data_path,
            mask_path=self.train_mask_path,
            img_transforms=self.img_transforms,
            inst_transforms=self.inst_transforms,
            normalization=self.normalization,
            return_inst=self.return_inst,
            return_type=self.return_type,
            return_sem=self.return_sem,
            return_weight=self.return_weight,
        )
        self.validset = DATASET_LOOKUP[self.valid_ds_type](
            path=self.valid_data_path,
            mask_path=self.valid_mask_path,
            img_transforms=self.img_transforms,
            inst_transforms=self.inst_transforms,
            normalization=self.normalization,
            return_inst=self.return_inst,
            return_type=self.return_type,
            return_sem=self.return_sem,
            return_weight=self.return_weight,
        )
        self.testset = DATASET_LOOKUP[self.test_ds_type](
            path=self.test_data_path,
            mask_path=self.test_mask_path,
            img_transforms=self.img_transforms,
            inst_transforms=self.inst_transforms,
            normalization=self.normalization,
            return_inst=self.return_inst,
            return_type=self.return_type,
            return_sem=self.return_sem,
            return_weight=self.return_weight,
        )

    def train_dataloader(self) -> DataLoader:
        """Initialize train dataloader."""
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Initialize val dataloader."""
        return DataLoader(
            self.validset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Initialize test dataloader."""
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )
