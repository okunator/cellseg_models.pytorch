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
    torch.manual_seed(0)
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


@pytest.mark.parametrize("version", ["2.4.1", "2.5.1", "2.6.0"])
def test_export_cellpose_onnx_requires_torch_2_7(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    model = _make_cellpose()
    monkeypatch.setattr(torch, "__version__", version)

    with pytest.raises(RuntimeError, match=r"PyTorch >= 2\.7"):
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


@pytest.mark.parametrize("dynamic_batch", [True, False])
def test_cellpose_onnxruntime_matches_pytorch(
    tmp_path: Path, dynamic_batch: bool
) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    model = _make_cellpose()
    wrapper = CellPoseONNXWrapper(model).eval()
    output_path = export_cellpose_onnx(
        model,
        tmp_path / "cellpose.onnx",
        input_shape=(1, 3, 64, 64),
        dynamic_batch=dynamic_batch,
    )

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    assert [output.name for output in session.get_outputs()] == [
        "flow_map",
        "type_map",
    ]
    assert session.get_inputs()[0].shape[2:] == [64, 64]
    batch_dimension = session.get_inputs()[0].shape[0]
    if dynamic_batch:
        assert isinstance(batch_dimension, str)
    else:
        assert batch_dimension == 1
    for batch in (1, 2, 3) if dynamic_batch else (1,):
        x = torch.rand(batch, 3, 64, 64)
        with torch.inference_mode():
            expected = [tensor.cpu().numpy() for tensor in wrapper(x)]
        actual = session.run(None, {"image": x.numpy()})
        for expected_tensor, actual_tensor in zip(expected, actual):
            np.testing.assert_allclose(
                actual_tensor, expected_tensor, rtol=1e-4, atol=1e-5
            )

    with pytest.raises(
        ort.capi.onnxruntime_pybind11_state.InvalidArgument, match="invalid dimensions"
    ):
        session.run(None, {"image": np.zeros((1, 3, 32, 64), dtype=np.float32)})
    if not dynamic_batch:
        with pytest.raises(
            ort.capi.onnxruntime_pybind11_state.InvalidArgument,
            match="invalid dimensions",
        ):
            session.run(None, {"image": np.zeros((2, 3, 64, 64), dtype=np.float32)})


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
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])

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

    expected_post = model.post_processor.postproc_serial(_to_soft_output(expected))[
        "nuc"
    ][0]
    actual_post = model.post_processor.postproc_serial(_to_soft_output(actual))["nuc"][
        0
    ]

    assert (
        np.count_nonzero(expected_post[0]) > 0
    ), "pretrained CellPose validation produced an empty instance mask"
    np.testing.assert_array_equal(actual_post[0], expected_post[0])
    np.testing.assert_array_equal(actual_post[1], expected_post[1])


@pytest.mark.parametrize("fails", [False, True])
def test_export_cellpose_onnx_preserves_child_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.BatchNorm2d(3))
    model.train()
    model[1].eval()
    before = [module.training for module in model.modules()]
    monkeypatch.setitem(sys.modules, "onnx", ModuleType("onnx"))
    monkeypatch.setitem(sys.modules, "onnxscript", ModuleType("onnxscript"))

    def fake_export(*args, **kwargs):
        assert not any(module.training for module in model.modules())
        if fails:
            raise RuntimeError("export failed")

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    if fails:
        with pytest.raises(RuntimeError, match="export failed"):
            export_cellpose_onnx(
                model, tmp_path / "model.onnx", input_shape=(1, 3, 8, 8)
            )
    else:
        export_cellpose_onnx(model, tmp_path / "model.onnx", input_shape=(1, 3, 8, 8))
    assert [module.training for module in model.modules()] == before


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_export_cellpose_onnx_matches_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    model = torch.nn.Conv2d(3, 3, 1).to(dtype=dtype)
    monkeypatch.setitem(sys.modules, "onnx", ModuleType("onnx"))
    monkeypatch.setitem(sys.modules, "onnxscript", ModuleType("onnxscript"))

    def fake_export(wrapper, args, *extra, **kwargs):
        assert args[0].dtype == dtype
        assert args[0].device == model.weight.device

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    export_cellpose_onnx(model, tmp_path / "model.onnx", input_shape=(1, 3, 8, 8))


def test_export_cellpose_onnx_preserves_modes_on_input_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = torch.nn.Conv2d(3, 3, 1).train()
    monkeypatch.setitem(sys.modules, "onnx", ModuleType("onnx"))
    monkeypatch.setitem(sys.modules, "onnxscript", ModuleType("onnxscript"))

    def fail(*args, **kwargs):
        raise RuntimeError("allocation failed")

    monkeypatch.setattr(torch, "zeros", fail)
    with pytest.raises(RuntimeError, match="allocation failed"):
        export_cellpose_onnx(model, tmp_path / "model.onnx", input_shape=(1, 3, 8, 8))
    assert model.training


@pytest.mark.parametrize("shape", [(1, 3, 8, 8.5), (True, 3, 8, 8)])
def test_export_cellpose_onnx_rejects_noninteger_shape(
    tmp_path: Path, shape: tuple
) -> None:
    model = torch.nn.Conv2d(3, 3, 1).train()
    with pytest.raises(ValueError, match="four positive BCHW dimensions"):
        export_cellpose_onnx(model, tmp_path / "model.onnx", input_shape=shape)
    assert model.training


def test_export_cellpose_onnx_requires_optional_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = torch.nn.Conv2d(3, 3, 1).train()
    monkeypatch.setitem(sys.modules, "onnx", None)
    monkeypatch.setitem(sys.modules, "onnxscript", None)
    with pytest.raises(ImportError, match="pip install onnx onnxscript"):
        export_cellpose_onnx(model, tmp_path / "model.onnx")
    assert model.training
