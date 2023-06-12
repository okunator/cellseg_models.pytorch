import numpy as np
import pytest

from cellseg_models_pytorch.transforms.albu_transforms.strong_augment import (
    StrongAugment,
)
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


def test_strongaug(img_patch_dir, mask_patch_dir):
    imp = sorted(img_patch_dir.glob("*"))[0]
    mp = sorted(mask_patch_dir.glob("*"))[0]

    im = FileHandler.read_img(imp)
    mask = FileHandler.read_mat(mp, key="inst_map")
    mask2 = FileHandler.read_mat(mp, key="type_map")

    sa = StrongAugment()
    tr_data = sa(image=im, masks=[mask, mask2])

    assert tr_data["image"].shape == im.shape
    assert tr_data["image"].dtype == im.dtype
    assert list(tr_data.keys()) == ["image", "masks"]
