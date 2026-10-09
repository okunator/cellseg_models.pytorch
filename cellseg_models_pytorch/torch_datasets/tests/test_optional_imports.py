import subprocess
import sys


def test_inference_datasets_without_training_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
for name in ("albumentations", "tables", "matplotlib"):
    sys.modules[name] = None

import numpy as np
import pytest
from types import SimpleNamespace

import cellseg_models_pytorch.torch_datasets as datasets
from cellseg_models_pytorch.torch_datasets import FolderDatasetInfer, WSIDatasetInfer
from cellseg_models_pytorch.inference.wsi_segmenter import WsiSegmenter

assert callable(FolderDatasetInfer)
assert "cellseg_models_pytorch.torch_datasets.folder_dataset_train" not in sys.modules
assert "cellseg_models_pytorch.torch_datasets.hdf5_dataset_train" not in sys.modules
image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
coordinates = [(5, 7, 3, 2)]
calls = []
def read_region(xywh, level):
    calls.append((xywh, level))
    return image.copy()
reader = SimpleNamespace(name="slide", read_region=read_region)
dataset = WSIDatasetInfer(reader, coordinates, level=2)
assert len(dataset) == 1
sample = dataset[0]
np.testing.assert_array_equal(sample["image"], image)
np.testing.assert_array_equal(sample["coords"], coordinates[0])
assert sample["name"] == "slide"
assert calls == [(coordinates[0], 2)]

def normalize(*, image):
    return {"image": image.astype(np.float32) / 255}
segmenter = WsiSegmenter(reader, object(), 2, coordinates, normalization=normalize)
np.testing.assert_array_equal(segmenter.dataset[0]["image"], normalize(image=image)["image"])

with pytest.raises(AttributeError, match="UnknownDataset"):
    datasets.UnknownDataset
for name in ("TrainDatasetFolder", "TrainDatasetH5"):
    with pytest.raises(ModuleNotFoundError, match="albumentations"):
        getattr(datasets, name)
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
