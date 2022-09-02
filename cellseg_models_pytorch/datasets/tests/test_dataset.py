import pytest
import torch

from cellseg_models_pytorch.datasets.hdf5_dataset import SegmentationHDF5Dataset

img_transforms = ["rigid", "blur"]
inst_transforms = ["smooth_dist"]


@pytest.mark.optional
@pytest.mark.parametrize("return_inst", [True, False])
@pytest.mark.parametrize("return_type", [True, False])
@pytest.mark.parametrize("return_sem", [True, False])
@pytest.mark.parametrize("normalization", [None, "minmax"])
def test_hdf5_dataset(hdf5db, return_inst, return_type, return_sem, normalization):
    ds = SegmentationHDF5Dataset(
        path=hdf5db,
        img_transforms=img_transforms,
        inst_transforms=inst_transforms,
        normalization=normalization,
        return_inst=return_inst,
        return_type=return_type,
        return_sem=return_sem,
    )

    out = next(iter(ds))

    if return_inst:
        assert "inst" in out.keys()
        assert out["inst"].dtype == torch.int64
    else:
        assert "binary" not in out.keys()

    if return_type:
        assert "type" in out.keys()
        assert out["type"].dtype == torch.int64
    else:
        assert "type" not in out.keys()

    if return_sem:
        assert "sem" in out.keys()
        assert out["sem"].dtype == torch.int64
    else:
        assert "sem" not in out.keys()

    assert "smoothdist" in out.keys()
    assert out["image"].dtype == torch.float32
