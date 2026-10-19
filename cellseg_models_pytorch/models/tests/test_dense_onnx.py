"""Check HoverNet, InstanSeg and CellVit export contracts and optional CPU runtime parity."""

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from timm.models.vision_transformer_sam import Attention

from cellseg_models_pytorch.models.cellvit import (
    CellVit,
    CellVitONNXWrapper,
    export_cellvit_onnx,
)
from cellseg_models_pytorch.models.hovernet import (
    HoverNet,
    HoverNetONNXWrapper,
    export_hovernet_onnx,
)
from cellseg_models_pytorch.models.instanseg import (
    InstanSeg,
    InstanSegONNXWrapper,
    export_instanseg_onnx,
)

Model = HoverNet | InstanSeg | CellVit
CONFIG = {
    CellVit: (CellVitONNXWrapper, export_cellvit_onnx, "hv_map", 2),
    HoverNet: (HoverNetONNXWrapper, export_hovernet_onnx, "hv_map", 2),
    InstanSeg: (InstanSegONNXWrapper, export_instanseg_onnx, "instanseg_map", 4),
}


def _make_model(model_cls: type[Model]) -> Model:
    torch.manual_seed(0)
    return model_cls(
        3,
        enc_name="samvit_base_patch16_224" if model_cls is CellVit else "resnet18",
        enc_pretrain=False,
        device=torch.device("cpu"),
        model_kwargs={"out_channels": (64, 64, 16, 8)},
    )


@pytest.mark.parametrize("model_cls", CONFIG)
def test_dense_wrapper_matches_network(model_cls: type[Model]) -> None:
    model = _make_model(model_cls)
    model.model.eval()
    wrapper_cls, _, _, channels = CONFIG[model_cls]
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        expected = model.model(image)["nuc"]
        maps, logits = wrapper_cls(model.model)(image)
    assert maps.shape == (1, channels, 64, 64)
    assert logits.shape == (1, 3, 64, 64)
    torch.testing.assert_close(maps, expected.aux_map)
    torch.testing.assert_close(logits, expected.type_map)


@pytest.mark.parametrize("model_cls", CONFIG)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
@pytest.mark.parametrize("dynamic_batch", [False, True])
def test_dense_export_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_cls: type[Model],
    dtype: torch.dtype,
    dynamic_batch: bool,
) -> None:
    model = _make_model(model_cls)
    model.model.to(dtype=dtype).train()
    next(m for m in model.model.modules() if isinstance(m, torch.nn.BatchNorm2d)).eval()
    states = [m.training for m in model.model.modules()]
    attentions = [m for m in model.model.modules() if isinstance(m, Attention)]
    if attentions:
        attentions[0].fused_attn = False
    attention_states = [m.fused_attn for m in attentions]
    _, export, map_name, _ = CONFIG[model_cls]
    monkeypatch.setitem(sys.modules, "onnx", ModuleType("onnx"))
    monkeypatch.setitem(sys.modules, "onnxscript", ModuleType("onnxscript"))
    calls = []

    def fake_export(wrapper, args, path, **kwargs) -> None:
        assert not any(m.training for m in model.model.modules())
        assert not any(m.fused_attn for m in attentions)
        assert args[0].dtype == dtype
        assert args[0].device == next(model.model.parameters()).device
        assert args[0].shape == ((2 if dynamic_batch else 1), 3, 64, 64)
        assert kwargs["input_names"] == ["image"]
        assert kwargs["output_names"] == [map_name, "type_map"]
        assert kwargs["dynamo"] is True
        assert kwargs["opset_version"] == 18
        if dynamic_batch:
            assert set(kwargs["dynamic_shapes"]["x"]) == {0}
        else:
            assert kwargs["dynamic_shapes"] is None
        calls.append(path)

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    path = tmp_path / "nested" / "network.onnx"
    assert (
        export(model.model, path, (1, 3, 64, 64), dynamic_batch=dynamic_batch) == path
    )
    assert calls == [str(path)]
    assert path.parent.is_dir()
    assert [m.training for m in model.model.modules()] == states
    assert [m.fused_attn for m in attentions] == attention_states
    assert all(p.dtype == dtype for p in model.model.parameters())


@pytest.mark.parametrize("model_cls", CONFIG)
@pytest.mark.parametrize("failure", ["allocation", "export"])
def test_dense_export_restores_modes_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_cls: type[Model],
    failure: str,
) -> None:
    model = _make_model(model_cls)
    model.model.train()
    next(m for m in model.model.modules() if isinstance(m, torch.nn.BatchNorm2d)).eval()
    states = [m.training for m in model.model.modules()]
    attentions = [m for m in model.model.modules() if isinstance(m, Attention)]
    if attentions:
        attentions[0].fused_attn = False
    attention_states = [m.fused_attn for m in attentions]
    monkeypatch.setitem(sys.modules, "onnx", ModuleType("onnx"))
    monkeypatch.setitem(sys.modules, "onnxscript", ModuleType("onnxscript"))

    def fail(*args, **kwargs) -> None:
        if failure == "export":
            assert not any(m.training for m in model.model.modules())
            assert not torch.is_autocast_enabled("cpu")
            assert not any(m.fused_attn for m in attentions)
        raise RuntimeError("test failure")

    if failure == "allocation":
        monkeypatch.setattr(torch, "zeros", fail)
    else:
        monkeypatch.setattr(torch.onnx, "export", fail)
    autocast_before = torch.is_autocast_enabled("cpu")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="test failure"):
            model.export_onnx(tmp_path / "network.onnx", (1, 3, 64, 64))
        assert torch.is_autocast_enabled("cpu")
    assert torch.is_autocast_enabled("cpu") == autocast_before
    assert [m.training for m in model.model.modules()] == states
    assert [m.fused_attn for m in attentions] == attention_states


@pytest.mark.parametrize("model_cls", CONFIG)
@pytest.mark.parametrize(
    "shape", [(1, 3, 0, 64), (1, 3, 8.5, 64), (True, 3, 64, 64), (3, 64, 64)]
)
def test_dense_export_rejects_invalid_shape(
    tmp_path: Path, model_cls: type[Model], shape: tuple
) -> None:
    model = _make_model(model_cls)
    with pytest.raises(ValueError, match="four positive BCHW dimensions"):
        model.export_onnx(tmp_path / "network.onnx", shape)


@pytest.mark.parametrize("model_cls", CONFIG)
def test_dense_export_requires_optional_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_cls: type[Model],
) -> None:
    model = _make_model(model_cls)
    monkeypatch.setitem(sys.modules, "onnx", None)
    monkeypatch.setitem(sys.modules, "onnxscript", None)
    with pytest.raises(ImportError, match="pip install onnx onnxscript"):
        model.export_onnx(tmp_path / "network.onnx", (1, 3, 64, 64))


@pytest.mark.optional
@pytest.mark.parametrize("dynamic_batch", [False, True])
def test_hovernet_onnxruntime_matches_pytorch(
    tmp_path: Path, dynamic_batch: bool
) -> None:
    _check_runtime(tmp_path, HoverNet, dynamic_batch)


@pytest.mark.optional
@pytest.mark.parametrize("dynamic_batch", [False, True])
def test_instanseg_onnxruntime_matches_pytorch(
    tmp_path: Path, dynamic_batch: bool
) -> None:
    _check_runtime(tmp_path, InstanSeg, dynamic_batch)


@pytest.mark.optional
@pytest.mark.parametrize("dynamic_batch", [False, True])
def test_cellvit_onnxruntime_matches_pytorch(
    tmp_path: Path, dynamic_batch: bool
) -> None:
    _check_runtime(tmp_path, CellVit, dynamic_batch)


def _check_runtime(tmp_path: Path, model_cls: type[Model], dynamic_batch: bool) -> None:
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    model = _make_model(model_cls)
    size = 224 if model_cls is CellVit else 64
    model.model.train()
    next(m for m in model.model.modules() if isinstance(m, torch.nn.BatchNorm2d)).eval()
    states = [m.training for m in model.model.modules()]
    attentions = [m for m in model.model.modules() if isinstance(m, Attention)]
    if attentions:
        attentions[0].fused_attn = False
    attention_states = [m.fused_attn for m in attentions]
    with torch.no_grad():
        for attention in attentions:
            attention.rel_pos_h.uniform_(-0.1, 0.1)
            attention.rel_pos_w.uniform_(-0.1, 0.1)
    _, _, map_name, channels = CONFIG[model_cls]
    classifier_state = (
        {
            name: value.clone()
            for name, value in model.model.probability_net.state_dict().items()
        }
        if model_cls is InstanSeg
        else {}
    )
    autocast_before = torch.is_autocast_enabled("cpu")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        path = model.export_onnx(
            tmp_path / "network.onnx", (1, 3, size, size), dynamic_batch=dynamic_batch
        )
        assert torch.is_autocast_enabled("cpu")
    assert torch.is_autocast_enabled("cpu") == autocast_before
    assert [m.training for m in model.model.modules()] == states
    assert [m.fused_attn for m in attentions] == attention_states
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    assert all(
        item.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        for item in (*graph.graph.input, *graph.graph.output)
    )
    if model_cls is InstanSeg:
        assert not any(
            "probability_net" in item.name for item in graph.graph.initializer
        )
        for name, expected_value in classifier_state.items():
            torch.testing.assert_close(
                model.model.probability_net.state_dict()[name],
                expected_value,
                rtol=0,
                atol=0,
            )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [o.name for o in session.get_outputs()] == [map_name, "type_map"]
    assert session.get_inputs()[0].shape[2:] == [size, size]
    if dynamic_batch:
        assert isinstance(session.get_inputs()[0].shape[0], str)
    else:
        assert session.get_inputs()[0].shape[0] == 1
    model.model.eval()
    for batch in (1, 2, 3) if dynamic_batch else (1,):
        image = torch.rand(batch, 3, size, size)
        with torch.no_grad():
            expected = model.model(image)["nuc"]
        maps, logits = session.run(None, {"image": image.numpy()})
        assert maps.shape == (batch, channels, size, size)
        np.testing.assert_allclose(maps, expected.aux_map.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            logits, expected.type_map.numpy(), rtol=1e-4, atol=1e-5
        )
    with pytest.raises(
        ort.capi.onnxruntime_pybind11_state.InvalidArgument, match="invalid dimensions"
    ):
        session.run(
            None, {"image": np.zeros((1, 3, size // 2, size), dtype=np.float32)}
        )
    if not dynamic_batch:
        with pytest.raises(
            ort.capi.onnxruntime_pybind11_state.InvalidArgument,
            match="invalid dimensions",
        ):
            session.run(None, {"image": np.zeros((2, 3, size, size), dtype=np.float32)})
