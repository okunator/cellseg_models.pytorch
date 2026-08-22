import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from PIL import Image

from cellseg_models_pytorch.decoders.multitask_decoder import SoftInstanceOutput
from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.models.cellpose import (
    CellPose,
    CellPoseONNXWrapper,
    export_cellpose_onnx,
)
from cellseg_models_pytorch.models.cellpose.cellpose_unet import cellpose_nuclei
from cellseg_models_pytorch.transforms.functional.normalization import minmax_normalize


def _make_cellpose() -> torch.nn.Module:
    return cellpose_nuclei(
        n_nuc_classes=3,
        enc_name="resnet18",
        enc_pretrain=False,
    ).eval()


def _to_soft_output(outputs: list[np.ndarray]) -> dict:
    flow_map, type_logits = outputs
    return {
        "nuc": SoftInstanceOutput(
            type_map=torch.from_numpy(type_logits).argmax(1),
            aux_map=torch.from_numpy(flow_map),
            binary_map=None,
        ),
        "cyto": None,
        "tissue": None,
    }


def test_cellpose_onnx_wrapper_matches_model_output() -> None:
    model = _make_cellpose()
    wrapper = CellPoseONNXWrapper(model).eval()
    x = torch.rand(1, 3, 64, 64)

    with torch.inference_mode():
        expected = model(x)["nuc"]
        flow_map, type_map = wrapper(x)

    torch.testing.assert_close(flow_map, expected.aux_map)
    torch.testing.assert_close(type_map, expected.type_map)


def test_export_cellpose_onnx_validates_input_shape(tmp_path: Path) -> None:
    model = _make_cellpose()

    with pytest.raises(ValueError, match="four positive BCHW dimensions"):
        export_cellpose_onnx(
            model,
            tmp_path / "cellpose.onnx",
            input_shape=(1, 3, 64, 0),
        )


def test_export_cellpose_onnx_requires_torch_2_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _make_cellpose()
    monkeypatch.setattr(torch, "__version__", "2.4.1")

    with pytest.raises(RuntimeError, match=r"PyTorch >= 2\.5"):
        export_cellpose_onnx(model, tmp_path / "cellpose.onnx")


def test_export_cellpose_onnx_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _make_cellpose()
    model.train()
    output_path = tmp_path / "nested" / "cellpose.onnx"
    export_call = {}

    monkeypatch.setitem(sys.modules, "onnx", ModuleType("onnx"))
    monkeypatch.setitem(sys.modules, "onnxscript", ModuleType("onnxscript"))

    def fake_export(*args, **kwargs) -> None:
        export_call["args"] = args
        export_call["kwargs"] = kwargs

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    result = export_cellpose_onnx(
        model,
        output_path,
        input_shape=(1, 3, 64, 64),
        dynamic_batch=True,
    )

    assert result == output_path
    assert output_path.parent.is_dir()
    assert model.training
    assert export_call["kwargs"]["input_names"] == ["image"]
    assert export_call["kwargs"]["output_names"] == ["flow_map", "type_map"]
    assert export_call["kwargs"]["dynamo"] is True
    assert "dynamic_axes" not in export_call["kwargs"]
    assert set(export_call["kwargs"]["dynamic_shapes"]) == {"x"}
    assert set(export_call["kwargs"]["dynamic_shapes"]["x"]) == {0}
    assert export_call["kwargs"]["opset_version"] == 18


def test_cellpose_onnxruntime_matches_pytorch(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    model = _make_cellpose()
    wrapper = CellPoseONNXWrapper(model).eval()
    output_path = export_cellpose_onnx(
        model,
        tmp_path / "cellpose.onnx",
        input_shape=(1, 3, 64, 64),
        dynamic_batch=True,
    )

    session = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    x = torch.rand(2, 3, 64, 64)

    with torch.inference_mode():
        expected = [tensor.cpu().numpy() for tensor in wrapper(x)]

    actual = session.run(None, {"image": x.numpy()})

    assert [output.name for output in session.get_outputs()] == [
        "flow_map",
        "type_map",
    ]
    for expected_tensor, actual_tensor in zip(expected, actual):
        np.testing.assert_allclose(
            actual_tensor,
            expected_tensor,
            rtol=1e-4,
            atol=1e-5,
        )


def test_pretrained_cellpose_real_image_onnx_parity(tmp_path: Path) -> None:
    """Validate dense and postprocessed parity with a real checkpoint/image.

    Set CELLSEG_CELLPOSE_IMAGE to a real RGB image path to enable this integration
    test. CELLSEG_CELLPOSE_WEIGHTS may be a local checkpoint path or a registered
    checkpoint name; it defaults to the HGSC EfficientNet-B5 CellPose checkpoint.
    """
    image_path = os.environ.get("CELLSEG_CELLPOSE_IMAGE")
    if image_path is None:
        pytest.skip("set CELLSEG_CELLPOSE_IMAGE to enable real-checkpoint validation")

    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    weights = os.environ.get("CELLSEG_CELLPOSE_WEIGHTS", "hgsc_v1_efficientnet_b5")
    tile_size = int(os.environ.get("CELLSEG_CELLPOSE_TILE_SIZE", "1024"))

    image = Image.open(image_path).convert("RGB")
    image = image.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    array = minmax_normalize(np.asarray(image))
    x = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).contiguous()

    model = CellPose.from_pretrained(weights, device=torch.device("cpu"))
    model.set_inference_mode(mixed_precision=False)
    model.post_processor = PostProcessor(
        postproc_method="cellpose",
        postproc_kwargs={"use_gpu": False},
    )
    wrapper = CellPoseONNXWrapper(model.model).eval()

    output_path = model.export_onnx(
        tmp_path / "cellpose_pretrained.onnx",
        input_shape=(1, 3, tile_size, tile_size),
        dynamic_batch=True,
    )
    session = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )

    with torch.inference_mode():
        expected = [tensor.cpu().numpy() for tensor in wrapper(x)]
    actual = session.run(None, {"image": x.numpy()})

    dense_tolerances = {"flow_map": 5e-5, "type_map": 2e-4}
    for name, expected_tensor, actual_tensor in zip(
        ("flow_map", "type_map"), expected, actual
    ):
        np.testing.assert_allclose(
            actual_tensor,
            expected_tensor,
            rtol=1e-4,
            atol=dense_tolerances[name],
        )

    expected_post = model.post_processor.postproc_serial(_to_soft_output(expected))["nuc"][0]
    actual_post = model.post_processor.postproc_serial(_to_soft_output(actual))["nuc"][0]

    assert np.count_nonzero(expected_post[0]) > 0, (
        "pretrained CellPose validation produced an empty instance mask"
    )
    np.testing.assert_array_equal(actual_post[0], expected_post[0])
    np.testing.assert_array_equal(actual_post[1], expected_post[1])
