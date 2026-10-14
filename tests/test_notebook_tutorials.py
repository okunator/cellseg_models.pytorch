"""Regression checks for the code rendered in the tutorial notebooks."""

import json
import re
from pathlib import Path
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from skimage.color import label2rgb

TUTORIALS = Path(__file__).resolve().parents[1] / "docs" / "tutorials"


def _code_cells(name: str) -> list[str]:
    notebook = json.loads((TUTORIALS / f"{name}.ipynb").read_text())
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def _check_targets(namespace):
    instances = namespace["instances"]
    assert set(np.unique(instances)) == {0, 1, 2}
    expected_classes = torch.from_numpy((instances > 0).astype(np.int64))
    classes = namespace["class_targets"]
    torch.testing.assert_close(classes, expected_classes.expand(2, -1, -1))
    assert classes.dtype == torch.int64

    flows = namespace["flow_targets"]
    assert flows.shape == (2, 2, 64, 64) and torch.isfinite(flows).all()
    assert flows.dtype == torch.float32
    assert not flows[:, :, instances == 0].any()
    # On the axes of a round object, flows point toward its centre.
    samples = flows[:, :, [20, 28, 24, 24], [24, 24, 20, 28]]
    expected_flows = torch.tensor([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])
    torch.testing.assert_close(samples, expected_flows.expand(2, -1, -1))


@pytest.fixture(autouse=True)
def preserve_torch_state(monkeypatch, request):
    monkeypatch.setattr(plt, "show", Mock())
    run_cuda = request.node.get_closest_marker("cuda") is not None
    if run_cuda and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    previous_threads = torch.get_num_threads()
    previous_rng = torch.get_rng_state()
    previous_cuda_rng = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    if not run_cuda:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)
        torch.set_rng_state(previous_rng)
        if previous_cuda_rng is not None:
            torch.cuda.set_rng_state_all(previous_cuda_rng)


def test_cpu_fixture_restores_cuda_rng(monkeypatch):
    original = torch.tensor([7, 11], dtype=torch.uint8)
    cuda_rng = original.clone()
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", lambda: True)
        patch.setattr(torch.cuda, "get_rng_state_all", lambda: [cuda_rng.clone()])
        patch.setattr(
            torch.cuda, "set_rng_state_all", lambda states: cuda_rng.copy_(states[0])
        )
        patch.setattr(torch.cuda, "manual_seed_all", lambda seed: cuda_rng.fill_(seed))
        request = Mock()
        request.node.get_closest_marker.return_value = None
        fixture = preserve_torch_state.__wrapped__(patch, request)
        next(fixture)
        try:
            assert not torch.cuda.is_available()
            torch.manual_seed(0)
            assert not torch.equal(cuda_rng, original)
        finally:
            with pytest.raises(StopIteration):
                next(fixture)
        torch.testing.assert_close(cuda_rng, original)


@pytest.mark.parametrize(
    "execution_device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_inference_notebook_outputs(execution_device):
    namespace = {}
    for source in _code_cells("inference"):
        exec(source, namespace)

    for outputs in (namespace["prediction"], namespace["result"]):
        assert outputs["cyto"] is None and outputs["tissue"] is None
    assert namespace["device"].type == execution_device
    image = namespace["image"]
    instances, classes = namespace["instances"], namespace["classes"]
    assert image.shape == (64, 64, 3)
    assert image.dtype == np.float32 and np.isfinite(image).all()
    assert instances.shape == classes.shape == image.shape[:2]
    assert instances.dtype == classes.dtype == np.int32
    assert not classes[instances == 0].any()
    assert set(np.unique(classes)) <= {0, 1, 2}
    assert not namespace["class_colors"][instances == 0].any()
    np.testing.assert_array_equal(
        namespace["class_colors"][instances > 0],
        label2rgb(classes, bg_label=-1)[instances > 0],
    )
    _check_plot(namespace, ["Image", "Predicted instances", "Predicted classes"])


def test_inference_plot_distinguishes_class_zero_from_background():
    namespace = {
        "plt": plt,
        "label2rgb": label2rgb,
        "image": np.zeros((2, 2, 3)),
        "instances": np.array([[0, 1], [2, 0]], dtype=np.int32),
        "classes": np.array([[0, 0], [1, 0]], dtype=np.int32),
    }
    exec(_code_cells("inference")[2], namespace)
    colors = namespace["class_colors"]
    assert not colors[[0, 1], [0, 1]].any()
    assert colors[0, 1].any()
    assert not np.array_equal(colors[0, 1], colors[1, 0])
    _check_plot(namespace, ["Image", "Predicted instances", "Predicted classes"])


@pytest.mark.parametrize(
    "execution_device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_training_notebook_updates_parameters(execution_device):
    *setup, training_step, evaluation, visualization = _code_cells("training")
    namespace = {}
    for source in setup:
        exec(source, namespace)
    model = namespace["model"]
    parameter = next(model.parameters())
    before = parameter.detach().clone()
    assert namespace["device"].type == execution_device
    _check_targets(namespace)

    exec(training_step, namespace)
    assert torch.isfinite(namespace["loss"])
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert not torch.equal(before, parameter.detach())

    exec(evaluation, namespace)
    assert not model.training
    assert torch.isfinite(namespace["validation_loss"])
    exec(visualization, namespace)
    np.testing.assert_array_equal(
        namespace["predicted_classes"],
        namespace["output"].type_map.argmax(dim=1)[0].detach().cpu().numpy(),
    )
    np.testing.assert_array_equal(
        namespace["class_colors"], label2rgb(namespace["predicted_classes"], bg_label=0)
    )
    _check_plot(namespace, ["Example image", "Target instances", "Predicted classes"])


@pytest.mark.parametrize(
    "execution_device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_finetuning_notebook_freezing_and_resume(execution_device):
    *setup, frozen_step, unfrozen_step, resume, visualization = _code_cells(
        "finetuning"
    )
    namespace = {}
    for source in setup:
        exec(source, namespace)
    assert namespace["device"].type == execution_device
    _check_targets(namespace)
    encoder = namespace["encoder"]
    encoder_before = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    assert all(not p.requires_grad for p in encoder.parameters())
    head = next(p for p in namespace["model"].parameters() if p.requires_grad)
    head_before = head.detach().clone()

    exec(frozen_step, namespace)
    assert not encoder.training
    for name, value in encoder_before.items():
        torch.testing.assert_close(encoder.state_dict()[name], value, rtol=0, atol=0)
    assert all(p.grad is None for p in encoder.parameters())
    assert not torch.equal(head_before, head.detach())
    assert torch.isfinite(namespace["loss"])

    parameter = next(encoder.parameters())
    before = parameter.detach().clone()
    exec(unfrozen_step, namespace)
    assert all(p.requires_grad for p in encoder.parameters())
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert not torch.equal(before, parameter.detach())

    exec(resume, namespace)
    model, restored = namespace["model"], namespace["restored"]
    assert restored is not model
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    model.eval()
    restored.eval()
    with torch.inference_mode():
        expected = model(namespace["images"].to(namespace["device"]))["nuc"]
        actual = restored(namespace["images"].to(namespace["device"]))["nuc"]
    torch.testing.assert_close(actual.type_map, expected.type_map, rtol=0, atol=0)
    torch.testing.assert_close(actual.aux_map, expected.aux_map, rtol=0, atol=0)

    original = namespace["optimizer"].state_dict()
    resumed = namespace["resumed_optimizer"].state_dict()
    assert original["param_groups"] == resumed["param_groups"]
    assert original["state"].keys() == resumed["state"].keys()
    for parameter_id, state in original["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(resumed["state"][parameter_id][key], value)
    torch.testing.assert_close(torch.get_rng_state(), namespace["state"]["torch_rng"])
    if namespace["device"].type == "cuda":
        torch.testing.assert_close(
            torch.cuda.get_rng_state(namespace["device"]),
            namespace["state"]["cuda_rng"],
        )
    assert namespace["state"]["epoch"] == 1
    exec(visualization, namespace)
    np.testing.assert_array_equal(
        namespace["predicted_classes"], actual.type_map.argmax(dim=1)[0].cpu().numpy()
    )
    np.testing.assert_array_equal(
        namespace["class_colors"], label2rgb(namespace["predicted_classes"], bg_label=0)
    )
    _check_plot(namespace, ["Example image", "Target instances", "Predicted classes"])


def _check_plot(namespace, titles):
    fig = namespace["fig"]
    fig.canvas.draw()
    namespace["plt"].show.assert_called_once()
    assert [axis.get_title() for axis in fig.axes] == titles
    for axis, expected in zip(
        fig.axes,
        (namespace["image"], namespace["instance_colors"], namespace["class_colors"]),
    ):
        np.testing.assert_array_equal(axis.images[0].get_array(), expected)
    np.testing.assert_array_equal(
        namespace["instance_colors"], label2rgb(namespace["instances"], bg_label=0)
    )
    namespace["plt"].close(fig)


def test_pretrained_loading_snippet(monkeypatch):
    from cellseg_models_pytorch.models.cellpose import CellPose

    notebook = json.loads((TUTORIALS / "finetuning.ipynb").read_text())
    introduction = "".join(notebook["cells"][0]["source"])
    match = re.search(r"```python\n(.*?)```", introduction, re.DOTALL)
    assert match is not None
    snippet = match.group(1)
    loader = Mock()
    monkeypatch.setattr(CellPose, "from_pretrained", loader)
    namespace = {}
    exec(snippet, namespace)
    loader.assert_called_once_with(
        Path("checkpoints/cellpose.safetensors"),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    assert namespace["model"] is loader.return_value.model
