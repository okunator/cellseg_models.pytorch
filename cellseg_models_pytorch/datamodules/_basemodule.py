import zipfile
from pathlib import Path
from typing import Union

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "To use the `csmp.datamodules` module, the pytorch-lightning lib, is needed. "
        "Install with `pip install pytorch-lightning`"
    )
from torch.utils.data import DataLoader

__all__ = ["BaseDataModule"]


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 8,
    ) -> None:
        """Set up pannuke datamodule..

        Parameters
        ----------
            batch_size : int, default=8
                Batch size for the dataloader.
            num_workers : int, default=8
                number of cpu cores/threads used in the dataloading process.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

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

    @staticmethod
    def extract_zips(path: Union[str, Path], rm: bool = False) -> None:
        """Extract files from all the .zip files inside a folder.

        Parameters
        ----------
            path : str or Path
                Path to a folder containing .zip files.
            rm :bool, default=False
                remove the .zip files after extraction.
        """
        for f in Path(path).iterdir():
            if f.is_file() and f.suffix == ".zip":
                with zipfile.ZipFile(f, "r") as z:
                    z.extractall(path)
                if rm:
                    f.unlink()
