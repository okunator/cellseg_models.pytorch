from .folder_dataset import FolderDataset

# from .hdf5_dataset import SegmentationHDF5Dataset

DATASET_LOOKUP = {
    # "hdf5": SegmentationHDF5Dataset,
    "folder": FolderDataset
}


# __all__ = ["DATASET_LOOKUP", "SegmentationHDF5Dataset", "FolderDataset"]
__all__ = ["DATASET_LOOKUP", "FolderDataset"]
