"""Regression checks for the code rendered in the tutorial notebooks."""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image
from scipy.io import savemat
from skimage.color import label2rgb

TUTORIALS = Path(__file__).resolve().parents[1] / "docs" / "tutorials"


def _code_cells(name: str) -> list[str]:
    notebook = json.loads((TUTORIALS / f"{name}.ipynb").read_text())
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


@pytest.fixture(autouse=True)
def preserve_torch_state(monkeypatch, request):
    monkeypatch.setenv("NO_ALBUMENTATIONS_UPDATE", "1")
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
def test_inference_notebook_outputs(execution_device, monkeypatch):
    import huggingface_hub

    from cellseg_models_pytorch.models.cellpose import CellPose
    from cellseg_models_pytorch.transforms.functional.normalization import (
        minmax_normalize,
    )

    download = Mock(return_value="example.safetensors")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    loader = Mock(
        side_effect=lambda path, device: CellPose(
            7, enc_name="resnet18", enc_pretrain=False, device=device
        )
    )
    monkeypatch.setattr(CellPose, "from_pretrained", loader)
    monkeypatch.chdir(TUTORIALS)
    namespace = {}
    for source in _code_cells("inference"):
        exec(source, namespace)

    for outputs in (namespace["prediction"], namespace["result"]):
        assert outputs["cyto"] is None and outputs["tissue"] is None
    assert namespace["device"].type == execution_device
    image = namespace["image"]
    instances, classes = namespace["instances"], namespace["classes"]
    assert image.shape == (1024, 1024, 3)
    assert image.dtype == np.float32 and np.isfinite(image).all()
    with Image.open(
        TUTORIALS.parents[1] / "cellseg_models_pytorch/utils/tests/data/HE.png"
    ) as source:
        expected_image = np.asarray(
            source.convert("RGB").resize((1024, 1024), Image.Resampling.BILINEAR)
        )
    np.testing.assert_array_equal(image, minmax_normalize(expected_image, copy=True))
    download.assert_called_once_with(
        repo_id="csmp-hub/cellpose-histo-hgsc-nuc-v1",
        filename="cellpose_hgsc_v1_efficientnet_b5.safetensors",
        revision="88c7ffd3a9106d1673c3dd6e6a9e42476e8320d2",
    )
    loader.assert_called_once_with(
        Path("example.safetensors"), device=namespace["device"]
    )
    assert instances.shape == classes.shape == image.shape[:2]
    assert instances.dtype == classes.dtype == np.int32
    assert not classes[instances == 0].any()
    assert set(np.unique(classes)) <= set(range(7))
    assert not namespace["class_colors"][instances == 0].any()
    np.testing.assert_array_equal(
        namespace["class_colors"][instances > 0],
        label2rgb(classes, bg_label=-1)[instances > 0],
    )
    _check_plot(namespace, ["Image", "Predicted instances", "Predicted classes"])


@pytest.mark.optional
def test_pretrained_inference_notebook_outputs(monkeypatch):
    monkeypatch.chdir(TUTORIALS)
    if cache := os.environ.get("CELLSEG_BASELINE_CACHE"):
        import huggingface_hub

        original_download = huggingface_hub.hf_hub_download
        monkeypatch.setattr(
            huggingface_hub,
            "hf_hub_download",
            lambda **kwargs: original_download(**kwargs, cache_dir=cache),
        )
    namespace = {}
    for source in _code_cells("inference"):
        exec(source, namespace)
    assert hashlib.sha256(namespace["checkpoint_path"].read_bytes()).hexdigest() == (
        "d6e29abb0b1c60668961f1bf074e609091583cae8340f1360e6ccfcb1cf17bb4"
    )
    assert namespace["model"].model.enc_name == "efficientnet_b5"
    assert not namespace["model"].model.training
    instances, classes = namespace["instances"], namespace["classes"]
    assert instances.shape == classes.shape == (1024, 1024)
    assert instances.dtype == classes.dtype == np.int32
    assert len(np.unique(instances)) > 2 and (instances == 0).any()
    assert set(np.unique(classes)) <= set(range(7))
    assert not classes[instances == 0].any()


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


@pytest.fixture
def prepared_panoptils(tmp_path, monkeypatch):
    pytest.importorskip("albumentations")
    rows, cols = np.mgrid[:64, :64]
    instances = np.zeros((64, 64), dtype=np.int32)
    instances[(rows - 24) ** 2 + (cols - 24) ** 2 < 8**2] = 11
    instances[(rows - 44) ** 2 + (cols - 44) ** 2 < 6**2] = 22
    classes = np.zeros_like(instances)
    classes[instances == 11] = 3
    classes[instances == 22] = 3
    image = np.repeat((instances > 0)[..., None], 3, axis=2).astype(np.uint8) * 255
    root = tmp_path / "data/panoptils"
    for split, count in (("train", 3), ("validation", 1)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        split_image = np.roll(image, 7, axis=1) if split == "validation" else image
        split_instances = (
            np.roll(instances, 7, axis=1) if split == "validation" else instances
        )
        split_classes = (
            np.roll(classes, 7, axis=1) if split == "validation" else classes
        )
        for i in range(count):
            Image.fromarray(split_image).save(
                root / split / "images" / f"sample_{i}.png"
            )
            savemat(
                root / split / "labels" / f"sample_{i}.mat",
                {
                    "inst": split_instances,
                    "type": split_classes,
                    "sem": np.full_like(instances, 9),
                },
            )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "offline fixture",
                "samples": [
                    {"split": "train", "patient": "training"},
                    {"split": "validation", "patient": "heldout"},
                ],
            }
        )
    )
    notebook_directory = tmp_path / "docs/tutorials"
    notebook_directory.mkdir(parents=True)
    monkeypatch.chdir(notebook_directory)

    return instances, classes, image


@pytest.mark.parametrize(
    "execution_device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_training_notebook_updates_parameters(execution_device, prepared_panoptils):
    instances, classes, image = prepared_panoptils
    *setup, training_loop, visualization = _code_cells("training")
    namespace = {}
    for source in setup:
        exec(source, namespace)
    model = namespace["model"]
    parameter = next(model.parameters())
    before = parameter.detach().clone()
    assert namespace["device"].type == execution_device
    assert len(namespace["train_dataset"]) == 3
    assert len(namespace["validation_dataset"]) == 1
    example = namespace["train_dataset"][0]
    assert set(example) == {"image", "inst", "type", "cellpose_inst"}
    torch.testing.assert_close(
        example["inst"].long(), torch.from_numpy(instances).long()
    )
    torch.testing.assert_close(example["type"].long(), torch.from_numpy(classes).long())
    from cellseg_models_pytorch.transforms.functional.cellpose import gen_flow_maps

    torch.testing.assert_close(
        example["cellpose_inst"].float(),
        torch.from_numpy(gen_flow_maps(instances).astype(np.float32)),
    )

    exec(training_loop, namespace)
    assert torch.isfinite(namespace["loss"])
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert not torch.equal(before, parameter.detach())
    assert not model.training
    assert np.isfinite(namespace["training_loss"])
    assert np.isfinite(namespace["validation_loss"])
    validation_instances = np.roll(instances, 7, axis=1)
    validation_classes = (
        torch.from_numpy(np.roll(classes, 7, axis=1)).long().to(namespace["device"])
    )
    validation_image = (
        torch.from_numpy(np.roll(image, 7, axis=1).transpose(2, 0, 1))
        .float()
        .unsqueeze(0)
        .to(namespace["device"])
        / 255
    )
    validation_flows = (
        torch.from_numpy(gen_flow_maps(validation_instances).astype(np.float32))
        .unsqueeze(0)
        .to(namespace["device"])
    )
    with torch.inference_mode():
        expected_output = model(validation_image)["nuc"]
        expected_loss = torch.nn.functional.cross_entropy(
            expected_output.type_map, validation_classes.unsqueeze(0)
        ) + torch.nn.functional.mse_loss(expected_output.aux_map, validation_flows)
    assert namespace["validation_loss"] == pytest.approx(float(expected_loss))

    exec(visualization, namespace)
    assert namespace["example"] is not example
    torch.testing.assert_close(
        namespace["example"]["inst"].long(),
        torch.from_numpy(validation_instances).long(),
    )
    assert namespace["output"].type_map.shape == (1, 7, 64, 64)
    np.testing.assert_array_equal(
        namespace["predicted_classes"],
        namespace["output"].type_map.argmax(dim=1)[0].cpu().numpy(),
    )
    np.testing.assert_array_equal(
        namespace["class_colors"], label2rgb(namespace["predicted_classes"], bg_label=0)
    )
    _check_plot(namespace, ["Held-out image", "Target instances", "Predicted classes"])


@pytest.mark.parametrize(
    "execution_device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_finetuning_notebook_freezing_and_resume(
    execution_device, prepared_panoptils, monkeypatch
):
    import huggingface_hub
    from cellseg_models_pytorch.models.cellpose import CellPose

    torch.manual_seed(19)
    pretrained = CellPose(
        7,
        enc_name="resnet18",
        enc_pretrain=False,
        device=torch.device(execution_device),
    )
    pretrained_state = {
        k: v.detach().clone() for k, v in pretrained.model.state_dict().items()
    }
    download = Mock(return_value="example.safetensors")
    loader = Mock(return_value=pretrained)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(CellPose, "from_pretrained", loader)

    *setup, frozen_step, unfrozen_step, resume, visualization = _code_cells(
        "finetuning"
    )
    namespace = {}
    for source in setup:
        exec(source, namespace)
    assert namespace["device"].type == execution_device
    loader.assert_called_once_with(
        Path("example.safetensors"), device=namespace["device"]
    )
    download.assert_called_once_with(
        repo_id="csmp-hub/cellpose-histo-hgsc-nuc-v1",
        filename="cellpose_hgsc_v1_efficientnet_b5.safetensors",
        revision="88c7ffd3a9106d1673c3dd6e6a9e42476e8320d2",
    )
    model = namespace["model"]
    assert model is pretrained.model
    for name, value in pretrained_state.items():
        if name == "decoder.type-nuc_type_head.head.weight":
            assert not torch.equal(model.state_dict()[name], value)
        else:
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    instances, classes, image = prepared_panoptils
    example = namespace["train_dataset"][0]
    torch.testing.assert_close(
        example["inst"].long(), torch.from_numpy(instances).long()
    )
    torch.testing.assert_close(example["type"].long(), torch.from_numpy(classes).long())
    assert set(example) == {"image", "inst", "type", "cellpose_inst"}
    assert len(namespace["train_dataset"]) == 3
    assert len(namespace["validation_dataset"]) == 1

    encoder = namespace["encoder"]
    encoder_before = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    assert all(not p.requires_grad for p in encoder.parameters())
    head = model.decoder["type-nuc_type_head"].head.weight
    head_before = head.detach().clone()

    exec(frozen_step, namespace)
    assert not encoder.training
    for name, value in encoder_before.items():
        torch.testing.assert_close(encoder.state_dict()[name], value, rtol=0, atol=0)
    assert all(p.grad is None for p in encoder.parameters())
    assert not torch.equal(head_before, head.detach())
    assert head.grad is not None and torch.isfinite(head.grad).all()
    assert np.isfinite(namespace["training_loss"])

    parameter = next(encoder.parameters())
    before = parameter.detach().clone()
    exec(unfrozen_step, namespace)
    assert all(p.requires_grad for p in encoder.parameters())
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert not torch.equal(before, parameter.detach())

    exec(resume, namespace)
    restored = namespace["restored"]
    assert restored is not model
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    model.eval()
    restored.eval()
    validation_image = (
        namespace["validation_dataset"][0]["image"].unsqueeze(0).to(namespace["device"])
    )
    with torch.inference_mode():
        expected = model(validation_image)["nuc"]
        actual = restored(validation_image)["nuc"]
    torch.testing.assert_close(actual.type_map, expected.type_map, rtol=0, atol=0)
    torch.testing.assert_close(actual.aux_map, expected.aux_map, rtol=0, atol=0)
    from cellseg_models_pytorch.transforms.functional.cellpose import gen_flow_maps

    validation_classes = (
        torch.from_numpy(np.roll(classes, 7, axis=1)).long().to(namespace["device"])
    )
    validation_flows = (
        torch.from_numpy(
            gen_flow_maps(np.roll(instances, 7, axis=1)).astype(np.float32)
        )
        .unsqueeze(0)
        .to(namespace["device"])
    )
    expected_loss = torch.nn.functional.cross_entropy(
        actual.type_map, validation_classes.unsqueeze(0)
    ) + torch.nn.functional.mse_loss(actual.aux_map, validation_flows)
    assert namespace["validation_loss"] == pytest.approx(float(expected_loss))

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
    assert namespace["state"]["epoch"] == 4
    assert namespace["state"]["phase"] == "unfrozen"
    assert namespace["state"]["class_names"] == [
        "background",
        "neoplastic",
        "stromal",
        "inflammatory",
        "epithelial",
        "other",
        "unknown",
    ]
    assert namespace["state"]["data_manifest"]["dataset"] == "offline fixture"
    checkpoint_bytes = namespace["saved_checkpoint"].read_bytes()
    with pytest.raises(FileExistsError):
        exec(resume, namespace)
    assert namespace["saved_checkpoint"].read_bytes() == checkpoint_bytes

    exec(visualization, namespace)
    np.testing.assert_array_equal(namespace["instances"], np.roll(instances, 7, axis=1))
    np.testing.assert_array_equal(
        namespace["predicted_classes"], actual.type_map.argmax(dim=1)[0].cpu().numpy()
    )
    np.testing.assert_array_equal(
        namespace["class_colors"], label2rgb(namespace["predicted_classes"], bg_label=0)
    )
    _check_plot(namespace, ["Held-out image", "Target instances", "Predicted classes"])


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
