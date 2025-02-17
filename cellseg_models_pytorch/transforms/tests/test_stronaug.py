import numpy as np
import pytest

from cellseg_models_pytorch.transforms.functional.generic_transforms import (
    AUGMENT_SPACE,
    _apply_operation,
    _magnitude_kwargs,
)
from cellseg_models_pytorch.utils import FileHandler


@pytest.mark.parametrize("op_name", list(AUGMENT_SPACE.keys()))
def test_stronaug_transforms(img_sample, op_name):
    rng = np.random.RandomState(seed=123)
    kwargs = dict(
        name=op_name,
        **_magnitude_kwargs(op_name, bounds=AUGMENT_SPACE[op_name], rng=rng)
    )

    tr_img = _apply_operation(img_sample, op_name, **kwargs)

    assert tr_img.shape == img_sample.shape
    assert tr_img.dtype == img_sample.dtype
    assert tr_img.max() <= 255
