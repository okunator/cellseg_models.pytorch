import numpy as np
import pytest

from cellseg_models_pytorch.transforms.functional import (
    gen_stardist_maps,
    gen_flow_maps,
    gen_hv_maps,
    gen_omni_flow_maps,
    smooth_distance,
    gen_dist_maps,
    gen_contour_maps,
    gen_weight_maps,
)


@pytest.mark.parametrize(
    "transform",
    [gen_hv_maps, gen_flow_maps, gen_omni_flow_maps],
)
@pytest.mark.parametrize("zero_input", [None, np.zeros((14, 14), np.int32)])
def test_gradient_transforms(inst_map, transform, zero_input):
    if zero_input is not None:
        inst_map = zero_input

    transformed = transform(inst_map=inst_map)

    assert transformed.dtype == np.float64
    assert transformed.shape == (2,) + inst_map.shape


@pytest.mark.parametrize(
    "transform",
    [smooth_distance, gen_dist_maps, gen_contour_maps, gen_weight_maps],
)
@pytest.mark.parametrize("zero_input", [None, np.zeros((14, 14))])
def test_inst_transforms(inst_map, transform, zero_input):
    if zero_input is not None:
        inst_map = zero_input

    transformed = transform(inst_map=inst_map)

    assert transformed.dtype == np.float64
    assert transformed.shape == inst_map.shape



@pytest.mark.parametrize("transform", [gen_stardist_maps])
@pytest.mark.parametrize("zero_input", [None, np.zeros((14, 14), np.int32)])
def test_stardist_transforms(inst_map, transform, zero_input):
    if zero_input is not None:
        inst_map = zero_input

    transformed = transform(inst_map=inst_map, n_rays=3)

    assert transformed.dtype == np.float32
    assert transformed.shape == (3,) + inst_map.shape
