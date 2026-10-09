from typing import TYPE_CHECKING

from .folder_dataset_infer import FolderDatasetInfer
from .wsi_dataset_infer import WSIDatasetInfer

if TYPE_CHECKING:
    from .folder_dataset_train import TrainDatasetFolder
    from .hdf5_dataset_train import TrainDatasetH5

__all__ = [
    "FolderDatasetInfer",
    "WSIDatasetInfer",
    "TrainDatasetH5",
    "TrainDatasetFolder",
]


def __getattr__(name: str) -> type:
    """Load optional training datasets only when requested."""
    if name == "TrainDatasetFolder":
        from .folder_dataset_train import TrainDatasetFolder

        return TrainDatasetFolder
    if name == "TrainDatasetH5":
        from .hdf5_dataset_train import TrainDatasetH5

        return TrainDatasetH5
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
