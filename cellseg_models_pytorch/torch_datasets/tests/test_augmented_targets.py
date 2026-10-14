"""Auxiliary targets must describe the augmented instance geometry."""

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.io import savemat


@pytest.mark.parametrize("operation", ["horizontal", "vertical", "crop"])
def test_folder_targets_follow_augmented_instances(tmp_path, operation):
    A = pytest.importorskip("albumentations")
    from cellseg_models_pytorch.torch_datasets import TrainDatasetFolder
    from cellseg_models_pytorch.transforms.albu_transforms import (
        ApplyEach,
        CellposeTransform,
        HoverNetTransform,
    )

    instances = np.zeros((32, 40), dtype=np.int32)
    instances[4:19, 7:24] = 11
    instances[20:29, 27:36] = 22
    classes = np.where(instances > 0, 3, 0).astype(np.int32)
    image = np.repeat(classes[..., None], 3, axis=2).astype(np.uint8)
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    Image.fromarray(image).save(tmp_path / "images/sample.png")
    savemat(tmp_path / "labels/sample.mat", {"inst": instances, "type": classes})
    operations = {
        "horizontal": A.HorizontalFlip(p=1),
        "vertical": A.VerticalFlip(p=1),
        "crop": A.Crop(x_min=10, y_min=8, x_max=34, y_max=28),
    }
    transforms = [CellposeTransform(), HoverNetTransform()]
    dataset = TrainDatasetFolder(
        str(tmp_path / "images"),
        str(tmp_path / "labels"),
        ("type", "inst"),
        A.Compose([operations[operation]]),
        ApplyEach(transforms, as_list=True),
    )
    sample = dataset[0]
    augmented = sample["inst"].numpy().astype(np.int32)
    torch.testing.assert_close(sample["image"][0].long(), sample["type"].long())
    for transform in transforms:
        expected = torch.from_numpy(transform(augmented))
        torch.testing.assert_close(sample[f"{transform.name}_inst"], expected)
