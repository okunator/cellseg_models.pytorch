"""Check image prediction against a model with known pixel outputs."""

import numpy as np
import pytest
import torch

from cellseg_models_pytorch.decoders.multitask_decoder import (
    SoftInstanceOutput,
    SoftSemanticOutput,
)
from cellseg_models_pytorch.inference.predictor import Predictor


class PixelModel(torch.nn.Module):
    def __init__(self, optional_heads: bool = False) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.optional_heads = optional_heads
        self.window_shapes: list[tuple[int, int]] = []

    def forward(
        self, image: torch.Tensor
    ) -> dict[str, SoftInstanceOutput | SoftSemanticOutput | None]:
        self.window_shapes.append(tuple(image.shape[-2:]))
        foreground = image[:, :1] * self.scale
        logits = torch.cat((1 - foreground, foreground), dim=1)
        outputs: dict[str, SoftInstanceOutput | SoftSemanticOutput | None] = {
            "nuc": SoftInstanceOutput(
                type_map=logits,
                aux_map=torch.cat((foreground, foreground + 2), dim=1),
                binary_map=foreground + 4 if self.optional_heads else None,
            ),
            "cyto": None,
            "tissue": None,
        }
        if self.optional_heads:
            outputs["cyto"] = SoftInstanceOutput(
                type_map=logits,
                aux_map=torch.cat((foreground + 6, foreground + 8), dim=1),
                binary_map=foreground + 10,
            )
            outputs["tissue"] = SoftSemanticOutput(
                type_map=torch.cat(
                    (torch.zeros_like(foreground), torch.ones_like(foreground)), dim=1
                ),
                binary_map=foreground + 12,
            )
        return outputs


@pytest.mark.parametrize("mixed_precision", [False, True])
@pytest.mark.parametrize("optional_heads", [False, True])
def test_direct_prediction_applies_boundary_weights(
    mixed_precision: bool, optional_heads: bool
) -> None:
    image = torch.linspace(0, 1, 2 * 3 * 3 * 5).reshape(2, 3, 3, 5)
    original = image.clone()
    model = PixelModel(optional_heads=optional_heads).eval()
    predictor = Predictor(model, mixed_precision=mixed_precision)

    # Hand-calculated center/edge distance ratios for a 3-by-5 image.
    edge = 0.5**0.5
    corner = edge / (5**0.5 + edge)
    top_middle = edge / (1 + edge)
    side_middle = edge / (2 + edge)
    inner = 2.5**0.5 / (1 + 2.5**0.5)
    weights = torch.tensor(
        [
            [corner, 1 / 3, top_middle, 1 / 3, corner],
            [side_middle, inner, 1, inner, side_middle],
            [corner, 1 / 3, top_middle, 1 / 3, corner],
        ]
    )
    weights = weights / weights.mean()
    foreground = image[:, :1]

    weighted = predictor.predict(image, apply_boundary_weight=True)
    # Reusing the predictor must not apply its cached weights when disabled.
    unweighted = predictor.predict(image, apply_boundary_weight=False)
    default = predictor.predict(image)

    for head, offset in (("nuc", 0), ("cyto", 6)):
        if head == "cyto" and not optional_heads:
            assert weighted[head] is unweighted[head] is default[head] is None
            continue
        auxiliary = torch.cat((foreground + offset, foreground + offset + 2), dim=1)
        for result, weight in ((weighted, weights), (unweighted, 1), (default, 1)):
            torch.testing.assert_close(result[head].aux_map, auxiliary * weight)
            torch.testing.assert_close(
                result[head].type_map, (foreground[:, 0] > 0.5).long()
            )
            assert not result[head].aux_map.requires_grad
            if optional_heads:
                torch.testing.assert_close(
                    result[head].binary_map, (foreground + offset + 4) * weight
                )
            else:
                assert result[head].binary_map is None
    for result in (weighted, unweighted, default):
        if optional_heads:
            torch.testing.assert_close(
                result["tissue"].type_map, torch.ones((2, 3, 5), dtype=torch.long)
            )
            torch.testing.assert_close(result["tissue"].binary_map, foreground + 12)
        else:
            assert result["tissue"] is None
    torch.testing.assert_close(image, original)
    assert not model.training


@pytest.mark.parametrize("shape", [(64, 80), (65, 79), (13, 17), (1, 1)])
@pytest.mark.parametrize("window_size", [(32, 32), (32, 48)])
@pytest.mark.parametrize("weighted", [False, True])
def test_sliding_windows_preserve_pixel_outputs(shape, window_size, weighted):
    image = np.linspace(0, 1, np.prod(shape) * 3, dtype=np.float32).reshape(*shape, 3)
    original = image.copy()
    model = PixelModel().eval()
    predictor = Predictor(model, mixed_precision=False)

    actual = predictor.predict_sliding_win(
        image, window_size, stride=16, apply_boundary_weight=weighted
    )

    foreground = torch.from_numpy(image[..., 0].copy())[None, None]
    torch.testing.assert_close(actual["nuc"].type_map, (foreground[:, 0] > 0.5).long())
    expected_aux = torch.cat((foreground, foreground + 2), dim=1)
    torch.testing.assert_close(actual["nuc"].aux_map, expected_aux)
    assert actual["nuc"].binary_map is None
    assert actual["cyto"] is actual["tissue"] is None
    assert all(size == window_size for size in model.window_shapes)
    assert actual["nuc"].aux_map.dtype == torch.float32
    assert not actual["nuc"].aux_map.requires_grad
    np.testing.assert_array_equal(image, original)


def test_sliding_windows_preserve_optional_head_maps():
    image = torch.linspace(0, 1, 2 * 3 * 65 * 79).reshape(2, 3, 65, 79)
    predictor = Predictor(PixelModel(optional_heads=True).eval(), mixed_precision=False)

    actual = predictor.predict_sliding_win(image, (32, 48), stride=16)

    foreground = image[:, :1]
    for head, offset in (("nuc", 0), ("cyto", 6)):
        torch.testing.assert_close(
            actual[head].type_map, (foreground[:, 0] > 0.5).long()
        )
        torch.testing.assert_close(
            actual[head].aux_map,
            torch.cat((foreground + offset, foreground + offset + 2), dim=1),
        )
        torch.testing.assert_close(actual[head].binary_map, foreground + offset + 4)
    torch.testing.assert_close(
        actual["tissue"].type_map, torch.ones((2, 65, 79), dtype=torch.long)
    )
    torch.testing.assert_close(actual["tissue"].binary_map, foreground + 12)


@pytest.mark.parametrize("padding", [0, 1, 20])
def test_sliding_windows_restore_coordinates_with_zero_or_odd_padding(padding):
    image = torch.linspace(0, 1, 3 * 32 * 48).reshape(1, 3, 32, 48)
    predictor = Predictor(PixelModel().eval(), mixed_precision=False)

    actual = predictor.predict_sliding_win(image, (32, 48), stride=16, padding=padding)

    torch.testing.assert_close(actual["nuc"].aux_map[:, :1], image[:, :1])


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_sliding_windows_can_disable_weights_in_both_precision_modes(mixed_precision):
    image = torch.linspace(0, 1, 3 * 32 * 48).reshape(1, 3, 32, 48)
    predictor = Predictor(PixelModel().eval(), mixed_precision=mixed_precision)

    actual = predictor.predict_sliding_win(
        image, (32, 32), stride=16, apply_boundary_weight=False
    )

    torch.testing.assert_close(actual["nuc"].aux_map[:, :1], image[:, :1])


@pytest.mark.parametrize(
    ("window_size", "stride", "padding", "message"),
    [
        ((0, 32), 16, 20, "window_size"),
        ((32,), 16, 20, "window_size"),
        ((32, 32), 0, 20, "stride"),
        ((32, 32), -1, 20, "stride"),
        ((32, 48), 33, 20, "stride"),
        ((32, 32), 16, -1, "padding"),
    ],
)
def test_sliding_windows_reject_invalid_grid_before_prediction(
    window_size, stride, padding, message
):
    model = PixelModel().eval()
    predictor = Predictor(model, mixed_precision=False)

    with pytest.raises(ValueError, match=message):
        predictor.predict_sliding_win(
            torch.zeros(1, 3, 32, 32), window_size, stride, padding=padding
        )
    assert not model.window_shapes


@pytest.mark.parametrize("shape", [(0, 32), (32, 0)])
def test_sliding_windows_reject_empty_images(shape):
    predictor = Predictor(PixelModel().eval(), mixed_precision=False)
    with pytest.raises(ValueError, match="positive height and width"):
        predictor.predict_sliding_win(torch.empty(1, 3, *shape), (32, 32), 16)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_public_sliding_window_pipeline_preserves_nuclei_at_seams_and_borders(
    monkeypatch, batch_size
):
    from scipy.ndimage import label

    from cellseg_models_pytorch.inference.post_processor import PostProcessor
    from cellseg_models_pytorch.inference.tests.test_wsi_segmenter import (
        baseline_instances,
    )
    from cellseg_models_pytorch.models.base._base_model_inst import BaseModelInst
    from cellseg_models_pytorch.postproc import POSTPROC_LOOKUP

    monkeypatch.setitem(POSTPROC_LOOKUP, "baseline", baseline_instances)
    model = BaseModelInst()
    model.model = PixelModel().eval()
    model.device = torch.device("cpu")
    model.predictor = Predictor(model.model, mixed_precision=False)
    model.post_processor = PostProcessor("baseline")
    model.inference_mode = True
    image = np.zeros((65, 79, 3), dtype=np.float32)
    image[12:24, 12:24] = 1
    image[:7, 65:] = 1
    image[56:, :9] = 1
    expected = label(image[..., 0])[0]
    inputs = (
        image
        if batch_size == 1
        else torch.from_numpy(image.transpose(2, 0, 1)).repeat(batch_size, 1, 1, 1)
    )

    prediction = model.predict(
        inputs, use_sliding_win=True, window_size=(32, 32), stride=16
    )
    result = model.post_process(prediction, use_async_postproc=False)

    assert len(result["nuc"]) == batch_size
    for instances, classes in result["nuc"]:
        np.testing.assert_array_equal(instances, expected)
        np.testing.assert_array_equal(classes, image[..., 0].astype(np.int32))


@pytest.mark.cuda
@pytest.mark.parametrize("weighted", [False, True])
def test_cuda_sliding_windows_preserve_device_and_output_maps(weighted):
    image = torch.linspace(0, 1, 3 * 65 * 79, device="cuda").reshape(1, 3, 65, 79)
    predictor = Predictor(
        PixelModel(optional_heads=True).cuda().eval(), mixed_precision=True
    )

    actual = predictor.predict_sliding_win(
        image, (32, 48), stride=16, apply_boundary_weight=weighted
    )

    assert actual["nuc"].aux_map.device == image.device
    torch.testing.assert_close(actual["nuc"].aux_map[:, :1], image[:, :1])
    torch.testing.assert_close(actual["nuc"].binary_map, image[:, :1] + 4)
    torch.testing.assert_close(actual["cyto"].binary_map, image[:, :1] + 10)
    torch.testing.assert_close(actual["tissue"].binary_map, image[:, :1] + 12)
