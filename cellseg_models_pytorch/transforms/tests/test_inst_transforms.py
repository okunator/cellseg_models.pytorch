import numpy as np
import pytest

from cellseg_models_pytorch.transforms import (
    apply_each,
    binarize_transform,
    cellpose_transform,
    compose,
    contour_transform,
    dist_transform,
    edgeweight_transform,
    hovernet_transform,
    omnipose_transform,
    smooth_dist_transform,
)


@pytest.mark.parametrize(
    "transform",
    [cellpose_transform, hovernet_transform, omnipose_transform],
)
@pytest.mark.parametrize("zero_input", [None, np.zeros((14, 14))])
def test_gradient_transforms(inst_map, transform, zero_input):
    trans = compose([transform()])

    if zero_input is not None:
        inst_map = zero_input

    transformed = trans(image=inst_map, inst_map=inst_map)

    assert transformed["inst_map"].dtype == np.float64
    assert transformed["inst_map"].shape == (2,) + inst_map.shape


@pytest.mark.parametrize(
    "transform",
    [dist_transform, smooth_dist_transform, contour_transform, edgeweight_transform],
)
@pytest.mark.parametrize("zero_input", [None, np.zeros((14, 14))])
def test_inst_transforms(inst_map, transform, zero_input):
    trans = compose([transform()])

    if zero_input is not None:
        inst_map = zero_input

    transformed = trans(image=inst_map, inst_map=inst_map)

    assert transformed["inst_map"].dtype == np.float64
    assert transformed["inst_map"].shape == inst_map.shape


@pytest.mark.parametrize("transform", [binarize_transform])
@pytest.mark.parametrize("zero_input", [None, np.zeros((14, 14))])
def test_binarize_transforms(inst_map, transform, zero_input):
    trans = compose([transform()])

    if zero_input is not None:
        inst_map = zero_input

    transformed = trans(image=inst_map, inst_map=inst_map)

    assert transformed["inst_map"].dtype == np.uint8
    assert transformed["inst_map"].shape == inst_map.shape


@pytest.mark.parametrize("zero_input", [None, np.zeros((14, 14))])
def test_inst_transform_pipeline(inst_map, zero_input):
    pipeline = apply_each(
        [edgeweight_transform(), cellpose_transform(), smooth_dist_transform()]
    )

    if zero_input is not None:
        inst_map = zero_input

    transformed = pipeline(image=inst_map, inst_map=inst_map)

    assert transformed["edgeweight"]["inst_map"].dtype == np.float64
    assert transformed["cellpose"]["inst_map"].dtype == np.float64
    assert transformed["smoothdist"]["inst_map"].dtype == np.float64
