import numpy as np
import pytest
import torch

from cellseg_models_pytorch.torch_datasets import TrainDatasetH5

pytestmark = pytest.mark.optional


@pytest.fixture(params=["array", "carray", "earray"])
def h5_samples(tmp_path, request):
    tb = pytest.importorskip("tables")
    pytest.importorskip("albumentations")
    from cellseg_models_pytorch.utils import H5Handler

    image_key = "pixels" if request.param == "carray" else "image"
    image = np.arange(2 * 12 * 16 * 3, dtype=np.uint8).reshape(2, 12, 16, 3)
    inst = np.zeros((2, 12, 16), dtype=np.int32)
    cyto = np.zeros_like(inst)
    inst[:, 2:5, 2:6] = 7
    cyto[:, 6:10, 8:13] = 23
    arrays = {image_key: image, "inst": inst, "cyto_inst": cyto}
    path = tmp_path / "patches.h5"
    with tb.open_file(path, "w") as h5:
        h5.create_array("/", "aaa_metadata", np.arange(9))
        if request.param == "earray":
            H5Handler.init_img(h5, (12, 16))
            H5Handler.init_mask(h5, "inst", (12, 16))
            H5Handler.init_mask(h5, "cyto_inst", (12, 16))
            for key, data in arrays.items():
                H5Handler.append_array(h5, data, key)
        else:
            create = h5.create_array if request.param == "array" else h5.create_carray
            for key, data in arrays.items():
                create("/", key, obj=data)
    return path, arrays, image_key


@pytest.mark.parametrize("two_transforms", [False, True])
@pytest.mark.parametrize("inst_keys", [("inst",), ("inst", "cyto_inst")])
def test_hdf5_length_and_auxiliary_targets(h5_samples, two_transforms, inst_keys):
    import albumentations as A

    from cellseg_models_pytorch.transforms.albu_transforms import (
        ApplyEach,
        CellposeTransform,
        HoverNetTransform,
    )

    path, arrays, image_key = h5_samples
    transforms = [CellposeTransform()]
    if two_transforms:
        transforms.append(HoverNetTransform())
    dataset = TrainDatasetH5(
        str(path), image_key, inst_keys, inst_keys,
        A.Compose([]), ApplyEach(transforms, as_list=True),
    )
    assert len(dataset) == 2
    sample = dataset[1]
    np.testing.assert_array_equal(sample[image_key], arrays[image_key][1].transpose(2, 0, 1))
    assert sample[image_key].dtype == torch.float32
    for key in inst_keys:
        np.testing.assert_array_equal(sample[key], arrays[key][1])
        assert sample[key].dtype == torch.int64
        for transform in transforms:
            expected = transform(arrays[key][1]).astype(np.float32)
            target = sample[f"{key}_{transform.name}"]
            np.testing.assert_allclose(target, expected)
            assert target.dtype == torch.float32
    if len(inst_keys) > 1:
        assert not torch.equal(sample["inst_cellpose"], sample["cyto_inst_cellpose"])


def test_hdf5_rejects_unaligned_sample_counts(h5_samples):
    import tables as tb

    path, arrays, image_key = h5_samples
    with tb.open_file(path, "a") as h5:
        h5.remove_node("/", "inst")
        h5.create_array("/", "inst", arrays["inst"][:1])
    with pytest.raises(ValueError, match="inst"):
        TrainDatasetH5(str(path), image_key, ("inst",), ("inst",), None, None)
