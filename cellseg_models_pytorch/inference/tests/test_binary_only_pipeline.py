"""Reconstruct binary-only predictions with the real StarDist post-processor."""

import numpy as np
import pytest
import torch

from cellseg_models_pytorch.decoders.multitask_decoder import SoftInstanceOutput
from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.inference.predictor import Predictor
from cellseg_models_pytorch.models.base._base_model_inst import BaseModelInst
from cellseg_models_pytorch.models.cellpose.onnx import CellPoseONNXWrapper
from cellseg_models_pytorch.models.cellvit.onnx import CellVitONNXWrapper
from cellseg_models_pytorch.models.hovernet.onnx import HoverNetONNXWrapper
from cellseg_models_pytorch.models.instanseg.onnx import InstanSegONNXWrapper
from cellseg_models_pytorch.models.stardist.onnx import StarDistONNXWrapper


class StarDistPixelModel(torch.nn.Module):
    def __init__(self, typed: bool = False) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.typed = typed

    def forward(self, image: torch.Tensor) -> dict[str, SoftInstanceOutput | None]:
        batch, _, height, width = image.shape
        rays = image.new_full((batch, 8, height, width), 4)
        types = (
            torch.cat((1 - image[:, :1], image[:, :1]), dim=1) if self.typed else None
        )
        return {
            "nuc": SoftInstanceOutput(types, rays, image[:, :1] * self.scale),
            "cyto": None,
            "tissue": None,
        }


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("sliding", [False, True])
def test_binary_only_public_pipeline_reconstructs_nuclei(batch_size, sliding):
    model = BaseModelInst()
    model.model = StarDistPixelModel().eval()
    model.device = torch.device("cpu")
    model.predictor = Predictor(model.model, mixed_precision=False)
    model.post_processor = PostProcessor(
        "stardist", {"trim_bboxes": False, "normalize": False}
    )
    model.inference_mode = True
    image = torch.zeros(batch_size, 3, 31, 47)
    centers = [(8, 8), (22, 36)]
    for row, col in centers:
        image[:, 0, row, col] = 0.9

    prediction = model.predict(
        image, use_sliding_win=sliding, window_size=(16, 16), stride=8
    )
    result = model.post_process(prediction, use_async_postproc=False)

    assert prediction["nuc"].type_map is None
    torch.testing.assert_close(prediction["nuc"].binary_map, image[:, :1])
    assert len(result["nuc"]) == batch_size
    for instances, classes in result["nuc"]:
        assert instances.shape == (31, 47) and instances.dtype == np.int32
        assert classes is None
        np.testing.assert_array_equal(np.unique(instances), [0, 1, 2])
        ids = [instances[row, col] for row, col in centers]
        assert all(ids) and len(set(ids)) == 2
        assert instances[0, 0] == instances[-1, -1] == 0
    assert result["cyto"] is result["tissue"] is None


def test_missing_classes_do_not_change_stardist_instance_masks():
    image = torch.zeros(1, 3, 31, 47)
    image[0, 0, 8, 8] = image[0, 0, 22, 36] = 0.9
    binary = Predictor(StarDistPixelModel(), mixed_precision=False).predict(image)
    typed = {
        "nuc": SoftInstanceOutput(
            (image[:, 0] > 0).long() * 3,
            binary["nuc"].aux_map,
            binary["nuc"].binary_map,
        ),
        "cyto": None,
        "tissue": None,
    }
    processor = PostProcessor("stardist", {"trim_bboxes": False, "normalize": False})

    binary_instances, binary_classes = processor.postproc_serial(binary)["nuc"][0]
    typed_instances, typed_classes = processor.postproc_serial(typed)["nuc"][0]
    direct_instances, direct_classes = processor.postproc_inst(
        image[0, 0].numpy(), binary["nuc"].aux_map[0].numpy()
    )

    np.testing.assert_array_equal(binary_instances, typed_instances)
    np.testing.assert_array_equal(direct_instances, binary_instances)
    assert direct_classes is None
    assert binary_classes is None
    np.testing.assert_array_equal(
        typed_classes, (typed_instances > 0).astype(np.int32) * 3
    )


@pytest.mark.parametrize(
    "wrapper",
    [
        CellPoseONNXWrapper,
        HoverNetONNXWrapper,
        CellVitONNXWrapper,
        InstanSegONNXWrapper,
        StarDistONNXWrapper,
    ],
)
@pytest.mark.parametrize("typed", [False, True])
def test_onnx_wrappers_keep_their_existing_typed_output_contract(wrapper, typed):
    model = StarDistPixelModel(typed=typed)
    image = torch.zeros(1, 3, 16, 24)
    if not typed:
        with pytest.raises(RuntimeError, match="cell-type output head"):
            wrapper(model)(image)
        return

    maps = model(image)["nuc"]
    expected = (maps.aux_map, maps.type_map)
    if wrapper is StarDistONNXWrapper:
        expected = (maps.binary_map, *expected)
    actual = wrapper(model)(image)

    assert len(actual) == len(expected)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
