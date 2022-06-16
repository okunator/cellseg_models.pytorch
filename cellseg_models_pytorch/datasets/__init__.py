from .hdf5_dataset import SegmentationHDF5Dataset

DATASET_LOOKUP = {
    "hdf5": SegmentationHDF5Dataset,
    # "folder": FolderDataset
    # "inference": InferenceDataset
}


__all__ = ["DATASET_LOOKUP", "SegmentationHDF5Dataset"]
