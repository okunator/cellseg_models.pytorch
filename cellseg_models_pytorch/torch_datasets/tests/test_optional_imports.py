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
from cellseg_models_pytorch.torch_datasets import (
    FolderDatasetInfer, WSIDatasetInfer, TrainDatasetFolder, TrainDatasetH5
)
from cellseg_models_pytorch.inference.wsi_segmenter import WsiSegmenter

assert callable(FolderDatasetInfer)
assert datasets.TrainDatasetFolder is TrainDatasetFolder
assert datasets.TrainDatasetH5 is TrainDatasetH5
assert "cellseg_models_pytorch.transforms.albu_transforms" not in sys.modules
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
with pytest.raises(ModuleNotFoundError, match="albumentations.*TrainDatasetFolder"):
    TrainDatasetFolder("unused", "unused", ("inst",), None, None)
with pytest.raises(ModuleNotFoundError, match="albumentations.*TrainDatasetH5"):
    TrainDatasetH5("unused", "image", ("inst",), ("inst",), None, None)

from types import ModuleType
sys.modules["albumentations"] = ModuleType("albumentations")
with pytest.raises(ModuleNotFoundError, match="tables.*TrainDatasetH5"):
    TrainDatasetH5("unused", "image", ("inst",), ("inst",), None, None)
with pytest.raises(ValueError, match="Invalid keys"):
    TrainDatasetFolder("unused", "unused", ("invalid",), None, None)
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_training_dependency_error_is_not_hidden(monkeypatch) -> None:
    import pytest

    from cellseg_models_pytorch.torch_datasets import folder_dataset_train

    failure = ModuleNotFoundError("broken Albumentations dependency", name="qudida")

    def fail_import(name):
        raise failure

    monkeypatch.setattr(folder_dataset_train, "import_module", fail_import)
    with pytest.raises(ModuleNotFoundError) as error:
        folder_dataset_train.TrainDatasetFolder(
            "unused", "unused", ("inst",), None, None
        )
    assert error.value is failure
