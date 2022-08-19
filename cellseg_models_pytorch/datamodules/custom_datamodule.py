from typing import List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        custom_datasets: List[Dataset],
        batch_size: int = 8,
        num_workers: int = 8,
    ) -> None:
        """Set up a custom datamodule with custom datasets.

        Parameters
        ----------
            custom_datasets : List[torch.utils.Dataset]
                A list of initialized torch Datasets. Order: train. valid, test.
            batch_size : int, default=8
                Batch size for the dataloader.
            num_workers : int, default=8
                number of cpu cores/threads used in the dataloading process.
        """
        super().__init__()
        self.custom_datasets = custom_datasets
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the train, valid, and test datasets."""
        self.trainset = self.custom_datasets[0]
        self.validset = self.custom_datasets[1]
        self.testset = self.custom_datasets[2]

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
