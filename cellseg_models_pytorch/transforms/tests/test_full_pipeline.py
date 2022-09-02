import numpy as np
import pytest
import torch

from cellseg_models_pytorch.transforms.albu_transforms import (
    apply_each,
    blur_transforms,
    cellpose_transform,
    compose,
    edgeweight_transform,
    non_rigid_transforms,
    smooth_dist_transform,
    to_tensorv3,
)


@pytest.mark.parametrize("zero_input", [None, np.zeros((11, 10))])
def test_full_transform_pipeline(img_sample, inst_map, sem_map, zero_input):
    basic_transforms = compose([blur_transforms(), non_rigid_transforms()])
    inst_transforms = apply_each(
        [edgeweight_transform(), cellpose_transform(), smooth_dist_transform()]
    )
    to_tensor = to_tensorv3()

    if zero_input is not None:
        inst_map = zero_input

    aug = basic_transforms(image=img_sample, masks=[inst_map, sem_map])
    aux = inst_transforms(image=aug["image"], inst_map=aug["masks"][0])
    all_data = to_tensor(image=aug["image"], masks=aug["masks"], aux=aux)

    assert all_data["image"].shape == torch.Size([3, 500, 500])
    assert all_data["image"].dtype == torch.float32
    assert all_data["aux"]["edgeweight"].shape == torch.Size([11, 10])
    assert all_data["aux"]["cellpose"].shape == torch.Size([2, 11, 10])
    assert all_data["aux"]["smoothdist"].shape == torch.Size([11, 10])
    assert all(mask.shape == torch.Size([11, 10]) for mask in all_data["masks"])
