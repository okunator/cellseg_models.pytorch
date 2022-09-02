import pytest
import torch

from cellseg_models_pytorch.transforms.albu_transforms import (
    blur_transforms,
    center_crop,
    compose,
    hue_saturation_transforms,
    non_rigid_transforms,
    non_spatial_transforms,
    random_crop,
    resize,
    rigid_transforms,
    to_tensor,
)


@pytest.mark.parametrize(
    "transform",
    [
        rigid_transforms,
        non_rigid_transforms,
        hue_saturation_transforms,
        blur_transforms,
        non_spatial_transforms,
    ],
)
def test_img_transforms(img_sample, transform):
    im = img_sample
    transforms = compose(transform())
    transformed = transforms(image=im)

    assert transformed["image"].shape == im.shape
    assert transformed["image"].dtype == im.dtype


@pytest.mark.parametrize(
    "transform",
    [center_crop, random_crop, resize],
)
def test_shape_transforms(img_sample, transform):
    expected_shape = (400, 420, 3)

    im = img_sample
    transforms = compose(transform(expected_shape[0], expected_shape[1]))
    transformed = transforms(image=im)

    assert transformed["image"].shape == expected_shape


def test_to_tensor(img_sample):
    transform = to_tensor()
    transformed = transform[0](image=img_sample)

    assert transformed["image"].dtype == torch.uint8


def test_tranform_pipeline(img_sample):
    trans_list = compose(
        [rigid_transforms(), non_rigid_transforms(), center_crop(450, 450), to_tensor()]
    )
    transformed = trans_list(image=img_sample)

    assert transformed["image"].shape == torch.Size([3, 450, 450])
    assert transformed["image"].dtype == torch.uint8
