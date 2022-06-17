# from .hdf5_dataset import SegmentationHDF5Dataset

DATASET_LOOKUP = {
    "dd": "dd"
    # "hdf5": SegmentationHDF5Dataset,
    # "folder": FolderDataset
    # "inference": InferenceDataset
}


# __all__ = ["DATASET_LOOKUP", "SegmentationHDF5Dataset"]
__all__ = ["DATASET_LOOKUP"]
