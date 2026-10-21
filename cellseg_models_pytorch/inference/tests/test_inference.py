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


class BinaryPixelModel(PixelModel):
    def forward(
        self, image: torch.Tensor
    ) -> dict[str, SoftInstanceOutput | SoftSemanticOutput | None]:
        outputs = super().forward(image)
        for name in ("nuc", "cyto"):
            if outputs[name] is not None:
                outputs[name].type_map = None
        return outputs


@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("mixed_precision", [False, True])
@pytest.mark.parametrize("optional_heads", [False, True])
@pytest.mark.parametrize("binary_only", [False, True])
def test_saved_intermediate_preserves_probabilities(
    sliding, weighted, mixed_precision, optional_heads, binary_only
):
    image = torch.linspace(0, 1, 31 * 47).reshape(1, 1, 31, 47).repeat(2, 3, 1, 1)
    model_class = BinaryPixelModel if binary_only else PixelModel
    model = model_class(optional_heads=optional_heads).eval()
    predictor = Predictor(model, mixed_precision=mixed_precision)
    predict = predictor.predict_sliding_win if sliding else predictor.predict
    kwargs = {"window_size": (24, 32), "stride": 16} if sliding else {}

    labels, intermediate = predict(
        image, save_intermediate=True, apply_boundary_weight=weighted, **kwargs
    )
    ordinary = predict(image, apply_boundary_weight=weighted, **kwargs)

    assert isinstance(ordinary, dict)
    assert labels is not intermediate
    # The two-logit softmax reduces to this independent sigmoid calculation.
    foreground_probability = torch.sigmoid(2 * image[:, :1] - 1)
    expected = torch.cat((1 - foreground_probability, foreground_probability), dim=1)
    for name in ("nuc", "cyto", "tissue"):
        output, saved = labels[name], intermediate[name]
        if output is None:
            assert saved is ordinary[name] is None
            continue

        assert output is not saved
        if name != "tissue" and binary_only:
            assert output.type_map is saved.type_map is ordinary[name].type_map is None
        else:
            probabilities = saved.type_map
            reference = expected
            if name == "tissue":
                tissue_probability = torch.full_like(image[:, :1], 1 / (1 + np.exp(-1)))
                reference = torch.cat(
                    (1 - tissue_probability, tissue_probability), dim=1
                )
            # Tissue convolution may run in float16 under autocast.
            torch.testing.assert_close(
                probabilities,
                reference.to(probabilities),
                atol=5e-4 if mixed_precision else 1e-6,
                rtol=1e-3 if mixed_precision else 1e-5,
            )
            assert probabilities.shape == (2, 2, 31, 47)
            assert probabilities.is_floating_point()
            assert probabilities.device == image.device
            assert not probabilities.requires_grad
            torch.testing.assert_close(output.type_map, probabilities.argmax(1))
            torch.testing.assert_close(output.type_map, ordinary[name].type_map)

        # Retaining probabilities must not duplicate the larger reconstruction maps.
        for field in ("aux_map", "binary_map"):
            if not hasattr(output, field):
                continue
            value = getattr(output, field)
            assert value is getattr(saved, field)
            if value is None:
                assert getattr(ordinary[name], field) is None
            else:
                torch.testing.assert_close(value, getattr(ordinary[name], field))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("inplace", [False, True])
def test_argmax_preserves_tensor_storage_and_dtype(dtype, inplace):
    probabilities = torch.tensor([[[[0.2, 0.7]], [[0.8, 0.3]]]], dtype=dtype)
    auxiliary = torch.ones(1, 2, 1, 2, dtype=dtype)
    binary = torch.ones(1, 1, 1, 2, dtype=dtype)
    parents = {"type_map": ["objects", "nuc_type"]}
    nucleus = SoftInstanceOutput(probabilities, auxiliary, binary, parents)
    cytoplasm = SoftInstanceOutput(None, auxiliary, binary)
    tissue = SoftSemanticOutput(probabilities, binary)
    maps = {"nuc": nucleus, "cyto": cytoplasm, "tissue": tissue}
    predictor = Predictor(PixelModel(), mixed_precision=False)

    labels = predictor._argmax(maps, inplace=inplace)

    assert (labels is maps) is inplace
    for name, original in (("nuc", nucleus), ("cyto", cytoplasm), ("tissue", tissue)):
        assert (labels[name] is original) is inplace
        assert labels[name].binary_map is binary
        if name == "cyto":
            assert labels[name].type_map is None
        else:
            torch.testing.assert_close(labels[name].type_map, torch.tensor([[[1, 0]]]))
            if not inplace:
                assert maps[name].type_map is probabilities
                assert maps[name].type_map.dtype == dtype
    assert labels["nuc"].aux_map is auxiliary
    assert labels["cyto"].aux_map is auxiliary
    assert labels["nuc"].parents is parents


@pytest.mark.parametrize(
    "sliding,weighted", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("shape", [(13, 17), (65, 79)])
def test_binary_only_prediction_preserves_foreground_and_auxiliary(
    sliding, weighted, shape
):
    image = torch.linspace(0, 1, 2 * 3 * np.prod(shape)).reshape(2, 3, *shape)
    original = image.clone()
    predictor = Predictor(
        BinaryPixelModel(optional_heads=True).eval(), mixed_precision=False
    )

    if sliding:
        actual = predictor.predict_sliding_win(
            image, (32, 48), stride=16, apply_boundary_weight=weighted
        )
    else:
        actual = predictor.predict(image, apply_boundary_weight=False)
    foreground = image[:, :1]

    for name, offset in (("nuc", 0), ("cyto", 6)):
        assert actual[name].type_map is None
        torch.testing.assert_close(
            actual[name].aux_map,
            torch.cat((foreground + offset, foreground + offset + 2), dim=1),
        )
        torch.testing.assert_close(actual[name].binary_map, foreground + offset + 4)
        assert actual[name].aux_map.dtype == image.dtype
        assert actual[name].binary_map.device == image.device
        assert not actual[name].binary_map.requires_grad
    torch.testing.assert_close(
        actual["tissue"].type_map, torch.ones((2, *shape), dtype=torch.long)
    )
    torch.testing.assert_close(actual["tissue"].binary_map, foreground + 12)
    torch.testing.assert_close(image, original, rtol=0, atol=0)


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
