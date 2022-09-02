import numpy as np
import pytest

from cellseg_models_pytorch.transforms.albu_transforms import (
    compose,
    imgnorm_transform,
    minmaxnorm_transform,
    percentilenorm_transform,
)


@pytest.mark.parametrize(
    "transform",
    [imgnorm_transform, percentilenorm_transform, minmaxnorm_transform],
)
@pytest.mark.parametrize("zero_input", [None, np.zeros((14, 14, 3), dtype=np.float32)])
def test_norm_transforms(img_sample, transform, zero_input):
    trans = compose([transform()])

    if zero_input is not None:
        img_sample = zero_input

    transformed = trans(image=img_sample)

    assert transformed["image"].dtype == np.float32
    assert transformed["image"].shape == img_sample.shape
