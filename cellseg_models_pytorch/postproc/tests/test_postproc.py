import numpy as np
import pytest

from cellseg_models_pytorch.postproc import (
    fill_holes_and_remove_small_masks,
    gen_flows,
    post_proc_cellpose,
    post_proc_cellpose_old,
    post_proc_dcan,
    post_proc_drfns,
    post_proc_hovernet,
    post_proc_omnipose,
    post_proc_stardist,
)
from cellseg_models_pytorch.transforms.functional import (
    gen_contour_maps,
    gen_dist_maps,
    gen_flow_maps,
    gen_hv_maps,
    gen_stardist_maps,
)
from cellseg_models_pytorch.utils import FileHandler, binarize, remap_label


def test_gen_flows(inst_map):
    flows = gen_flow_maps(inst_map)
    flows = gen_flows(flows)

    assert flows.shape == (inst_map.shape[0], inst_map.shape[1], 3)


def test_fill_holes(inst_map):
    out = fill_holes_and_remove_small_masks(inst_map.copy(), min_size=7)

    assert len(np.unique(out)) == 6


@pytest.mark.parametrize("return_flows", [True, False])
@pytest.mark.parametrize("interp", [True, False])
def test_postproc_cellpose(inst_map, return_flows, interp):
    flows = gen_flow_maps(inst_map)

    if not return_flows:
        rebuild = post_proc_cellpose(
            inst_map, flows, min_size=0, interp=interp, use_gpu=False
        )
    else:
        rebuild, _ = post_proc_cellpose(
            inst_map,
            flows,
            min_size=0,
            return_flows=return_flows,
            interp=interp,
            use_gpu=False,
        )

    assert rebuild.shape == inst_map.shape
    assert rebuild.dtype == "int32"

    if interp:
        assert len(np.unique(rebuild)) == 4


@pytest.mark.parametrize("return_flows", [True, False])
def test_postproc_omnipose(inst_map, return_flows):
    flows = gen_flow_maps(inst_map)

    if not return_flows:
        rebuild = post_proc_omnipose(inst_map, flows, min_size=0)
    else:
        rebuild, _ = post_proc_omnipose(
            inst_map, flows, min_size=0, return_flows=return_flows
        )

    assert rebuild.dtype == "int32"
    np.testing.assert_array_equal(
        np.unique(remap_label(inst_map)), np.unique(remap_label(rebuild))
    )


def test_postproc_cellpose_old(inst_map):
    flows = gen_hv_maps(inst_map)
    rebuild = rebuild = post_proc_cellpose_old(inst_map, flows)

    assert rebuild.shape == inst_map.shape
    assert rebuild.dtype == "int32"


@pytest.mark.parametrize("enhance", [True, False])
def test_postproc_hovernet(inst_map, enhance):
    hover = gen_hv_maps(inst_map)
    rebuild = post_proc_hovernet(binarize(inst_map), hover, enhance=enhance)

    assert rebuild.shape == inst_map.shape
    assert rebuild.dtype == "int32"


def test_postproc_drfns(inst_map):
    dist = gen_dist_maps(inst_map)
    rebuild = post_proc_drfns(inst_map, dist)

    assert rebuild.dtype == "int32"
    assert rebuild.shape == inst_map.shape
    np.testing.assert_array_equal(
        np.unique(remap_label(inst_map)), np.unique(remap_label(rebuild))
    )


def test_postproc_dcan(inst_map):
    cont = gen_contour_maps(inst_map)
    rebuild = post_proc_dcan(inst_map, cont)

    assert rebuild.dtype == "int32"
    assert rebuild.shape == inst_map.shape


def test_postproc_stardist(mask_patch_dir):
    mask_path = sorted(mask_patch_dir.glob("*"))[0]
    inst_map = FileHandler.read_mat(mask_path)["inst_map"]
    stardist = gen_stardist_maps(inst_map, 32)
    dist = gen_dist_maps(inst_map)
    rebuild = post_proc_stardist(dist, stardist)

    assert rebuild.dtype == "int32"
    assert rebuild.shape == inst_map.shape
