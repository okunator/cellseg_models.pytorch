"""Regression checks for the README quick start and tutorial notebook code."""

import json
import random
import re
from pathlib import Path
from unittest.mock import Mock, patch

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
    previous_numpy = np.random.get_state()
    previous_python = random.getstate()
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
        np.random.set_state(previous_numpy)
        random.setstate(previous_python)
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


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize(
    "image_kind", ["channels", "constant", "low_contrast", "histology"]
)
def test_readme_quickstart_minmax(monkeypatch, img_sample, dtype, image_kind):
    albu = pytest.importorskip("albumentations")
    from cellseg_models_pytorch.models.cellpose import CellPose
    from cellseg_models_pytorch.transforms.functional.normalization import (
        minmax_normalize,
    )
    from cellseg_models_pytorch.utils import FileHandler

    # Separate channel ranges distinguish global from per-channel normalization.
    pixels = np.arange(77).reshape(7, 11)
    image = np.stack([pixels, pixels + 80, pixels + 160], axis=-1).astype(dtype)
    if image_kind == "constant":
        image.fill(37)
    elif image_kind == "low_contrast":
        image = image % 2 + 100
    elif image_kind == "histology":
        image = img_sample.astype(dtype)
    if dtype == np.float32:
        image = image / 255 - 0.5
    original = image.copy()

    model = Mock()
    monkeypatch.setattr(CellPose, "from_pretrained", Mock(return_value=model))
    monkeypatch.setattr(FileHandler, "read_img", Mock(return_value=image))
    readme = (TUTORIALS.parents[1] / "README.md").read_text()
    source = readme.split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {"IMG_PATH": "example.png"}
    exec(source, namespace)

    transform = namespace["normalize"]
    assert isinstance(transform, albu.Normalize)
    assert transform.normalization == "min_max"
    actual = model.predict.call_args.args[0]
    assert actual.dtype == np.float32
    assert actual.shape == (1024, 1024, 3)
    resized = albu.Resize(1024, 1024)(image=original)["image"]
    expected = minmax_normalize(resized, copy=True)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, original)
    if image_kind == "constant":
        np.testing.assert_array_equal(actual, np.zeros_like(actual))
    model.post_process.assert_called_once_with(model.predict.return_value)


@pytest.mark.parametrize(
    "execution_device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_inference_notebook_outputs(execution_device, prepared_panoptils):
    from cellseg_models_pytorch.models.cellpose import CellPose
    from cellseg_models_pytorch.transforms.functional.normalization import (
        minmax_normalize,
    )

    pretrained = CellPose(
        7,
        enc_name="resnet18",
        enc_pretrain=False,
        device=torch.device(execution_device),
    )
    checkpoint_dir = Path("training-checkpoints/epoch_05")
    checkpoint_dir.mkdir(parents=True)
    class_names = [
        "background",
        "neoplastic",
        "stromal",
        "inflammatory",
        "epithelial",
        "other",
        "unknown",
    ]
    from safetensors.torch import save_model

    save_model(
        pretrained.model,
        checkpoint_dir / "model.safetensors",
        metadata={"class_names": json.dumps(class_names)},
    )
    namespace = {}
    with patch.object(
        CellPose, "predict", autospec=True, side_effect=CellPose.predict
    ) as predict:
        for source in _code_cells("inference"):
            exec(source, namespace)
    assert predict.call_count == 2
    assert predict.call_args_list[0].kwargs == {}
    assert predict.call_args_list[1].kwargs == {
        "use_sliding_win": True,
        "window_size": (256, 256),
        "stride": 128,
    }
    assert predict.call_args_list[1].args[1] is namespace["image"]
    assert namespace["device"].type == execution_device
    for outputs in (namespace["prediction"], namespace["result"]):
        assert outputs["cyto"] is None and outputs["tissue"] is None
    _, _, raw_image = prepared_panoptils
    expected_image = minmax_normalize(np.roll(raw_image, 7, axis=1), copy=True)
    np.testing.assert_array_equal(namespace["image"], expected_image)
    assert namespace["image"].dtype == np.float32
    instances, classes = namespace["instances"], namespace["classes"]
    assert instances.shape == classes.shape == expected_image.shape[:2]
    assert instances.dtype == classes.dtype == np.int32
    assert set(np.unique(classes)) <= set(range(7))
    assert not classes[instances == 0].any()
    assert not namespace["class_colors"][instances == 0].any()
    np.testing.assert_array_equal(
        namespace["class_colors"][instances > 0],
        namespace["class_palette"][classes[instances > 0]],
    )
    for name, value in pretrained.model.state_dict().items():
        torch.testing.assert_close(
            namespace["model"].model.state_dict()[name], value, rtol=0, atol=0
        )
    fig = namespace["fig"]
    fig.canvas.draw()
    assert [axis.get_title() for axis in fig.axes] == [
        "Held-out image",
        "Target instances",
        "Target cell types",
        "Held-out image",
        "Predicted instances",
        "Predicted cell types",
    ]
    np.testing.assert_array_equal(
        fig.axes[4].images[0].get_array(), label2rgb(instances, bg_label=0)
    )
    np.testing.assert_array_equal(
        fig.axes[5].images[0].get_array(), namespace["class_colors"]
    )
    sliding_instances = namespace["sliding_instances"]
    sliding_classes = namespace["sliding_classes"]
    assert sliding_instances.shape == sliding_classes.shape == expected_image.shape[:2]
    assert sliding_instances.dtype == sliding_classes.dtype == np.int32
    assert set(np.unique(sliding_classes)) <= set(range(7))
    assert namespace["sliding_prediction"]["nuc"].aux_map.shape == (
        1,
        2,
        *expected_image.shape[:2],
    )
    sliding_fig = namespace["sliding_fig"]
    sliding_fig.canvas.draw()
    assert [axis.get_title() for axis in sliding_fig.axes] == [
        "Held-out image",
        "Sliding-window instances",
        "Sliding-window cell types",
    ]
    np.testing.assert_array_equal(
        sliding_fig.axes[0].images[0].get_array(), namespace["raw_image"]
    )
    np.testing.assert_array_equal(
        sliding_fig.axes[1].images[0].get_array(),
        label2rgb(sliding_instances, bg_label=0),
    )
    expected_colors = namespace["class_palette"][sliding_classes]
    expected_colors[sliding_instances == 0] = 0
    np.testing.assert_array_equal(
        sliding_fig.axes[2].images[0].get_array(), expected_colors
    )
    assert plt.show.call_count == 2
    plt.close(fig)
    plt.close(sliding_fig)


def test_inference_plot_distinguishes_class_zero_from_background():
    from matplotlib.patches import Patch

    instances = np.array([[0, 1], [2, 0]], dtype=np.int32)
    classes = np.array([[0, 0], [1, 0]], dtype=np.int32)
    namespace = {
        "plt": plt,
        "label2rgb": label2rgb,
        "np": np,
        "Patch": Patch,
        "raw_image": np.zeros((2, 2, 3)),
        "instances": instances,
        "classes": classes,
        "labels": {"inst": instances, "type": classes},
        "class_names": [
            "background",
            "neoplastic",
            "stromal",
            "inflammatory",
            "epithelial",
            "other",
            "unknown",
        ],
    }
    exec(
        next(
            source
            for source in _code_cells("inference")
            if source.startswith("# Use the same class colours")
        ),
        namespace,
    )
    colors = namespace["class_colors"]
    assert not colors[[0, 1], [0, 1]].any()
    assert colors[0, 1].any()
    assert not np.array_equal(colors[0, 1], colors[1, 0])
    plt.close(namespace["fig"])
    namespace.update(sliding_instances=instances, sliding_classes=classes)
    exec(
        next(
            source
            for source in _code_cells("inference")
            if source.startswith("# Keep the same class colours")
        ),
        namespace,
    )
    np.testing.assert_array_equal(namespace["sliding_class_colors"], colors)
    plt.close(namespace["sliding_fig"])


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
    root = tmp_path / "data/panoptils-nuclei"
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
    (root / "validation/regions/images").mkdir(parents=True)
    (root / "validation/regions/labels").mkdir()
    Image.fromarray(np.roll(image, 7, axis=1)).save(
        root / "validation/regions/images/region.png"
    )
    savemat(
        root / "validation/regions/labels/region.mat",
        {
            "inst": np.roll(instances, 7, axis=1),
            "type": np.roll(classes, 7, axis=1),
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


@pytest.fixture
def offline_notebook_loaders(monkeypatch):
    from torch.utils.data import DataLoader

    original = DataLoader.__init__
    monkeypatch.setattr(
        DataLoader,
        "__init__",
        lambda self, dataset, **kwargs: original(
            self, dataset, **(kwargs | {"num_workers": 0, "batch_size": 2})
        ),
    )


def _check_validation_loss(namespace, classes, instances):
    from cellseg_models_pytorch.transforms.functional.cellpose import gen_flow_maps

    image = (
        namespace["validation_dataset"][0]["image"].unsqueeze(0).to(namespace["device"])
    )
    target = (
        torch.from_numpy(np.roll(classes, 7, axis=1)).long().to(namespace["device"])
    )
    flows = (
        torch.from_numpy(
            gen_flow_maps(np.roll(instances, 7, axis=1)).astype(np.float32)
        )
        .unsqueeze(0)
        .to(namespace["device"])
    )
    with torch.inference_mode():
        output = namespace["model"](image)["nuc"]
    probabilities = output.type_map.softmax(dim=1)
    one_hot = torch.nn.functional.one_hot(target.unsqueeze(0), 7).permute(0, 3, 1, 2)
    ce = -(one_hot * (probabilities + 1e-8).log()).sum(dim=1).mean()
    regularization = 0.005 * output.type_map.square().mean()
    present = one_hot.sum((2, 3)) > 0
    per_class_dice = 1 - 2 * (probabilities * one_hot).sum((2, 3)) / (
        probabilities + one_hot
    ).sum((2, 3)).clamp_min(1e-8)
    dice = per_class_dice[present].mean()
    mse = torch.nn.functional.mse_loss(output.aux_map, flows)
    assert namespace["validation_loss"] == pytest.approx(
        float(ce + regularization + dice + mse)
    )


def _check_reload_and_prediction(namespace, resume, evaluation, visualization):
    exec(resume, namespace)
    original = namespace["model"]
    restored = namespace["restored"]
    for name, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    before = namespace["optimizer"].state_dict()
    after = namespace["resumed_optimizer"].state_dict()
    assert before["param_groups"] == after["param_groups"]
    assert before["state"].keys() == after["state"].keys()
    for key, state in before["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(after["state"][key][name], value, rtol=0, atol=0)
    assert (
        namespace["scheduler"].state_dict()
        == namespace["resumed_scheduler"].state_dict()
    )
    torch.testing.assert_close(torch.get_rng_state(), namespace["state"]["torch_rng"])
    assert random.getstate() == namespace["state"]["python_rng"]
    np.testing.assert_array_equal(
        np.random.get_state()[1], namespace["state"]["numpy_rng"][1]
    )
    if namespace["device"].type == "cuda":
        torch.testing.assert_close(
            torch.cuda.get_rng_state(namespace["device"]),
            namespace["state"]["cuda_rng"],
        )
    image = (
        namespace["validation_dataset"][0]["image"].unsqueeze(0).to(namespace["device"])
    )
    with torch.inference_mode():
        old = original(image)["nuc"]
        new = restored(image)["nuc"]
    torch.testing.assert_close(old.type_map, new.type_map, rtol=0, atol=0)
    torch.testing.assert_close(old.aux_map, new.aux_map, rtol=0, atol=0)
    saved = namespace["saved_checkpoint"].read_bytes()
    with pytest.raises(FileExistsError):
        namespace["save_checkpoint"](namespace["state"]["epoch"])
    assert namespace["saved_checkpoint"].read_bytes() == saved
    exec(evaluation, namespace)
    assert len(namespace["evaluation_paths"]) == 1
    assert list(namespace["metrics"].index) == [
        "all nuclei",
        *namespace["class_names"][1:],
    ]
    assert np.isfinite(namespace["metrics"]["PQ"].dropna()).all()
    exec(visualization, namespace)
    fig = namespace["fig"]
    fig.canvas.draw()
    assert [axis.get_title() for axis in fig.axes] == [
        "Held-out image",
        "Target instances",
        "Target cell types",
        "Held-out image",
        "Predicted instances",
        "Predicted cell types",
    ]
    np.testing.assert_array_equal(
        fig.axes[4].images[0].get_array(),
        label2rgb(namespace["predicted_instances"], bg_label=0),
    )
    np.testing.assert_array_equal(
        fig.axes[2].images[0].get_array(),
        namespace["class_palette"][namespace["labels"]["type"].astype(int)],
    )
    np.testing.assert_array_equal(
        fig.axes[5].images[0].get_array(),
        namespace["class_palette"][namespace["predicted_classes"]],
    )
    plt.show.assert_called_once()
    plt.close(fig)


@pytest.mark.parametrize(
    "execution_device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("freeze_encoder", [False, True])
def test_training_notebook_updates_parameters(
    execution_device,
    freeze_encoder,
    prepared_panoptils,
    offline_notebook_loaders,
    monkeypatch,
):
    from cellseg_models_pytorch.models.cellpose import CellPose

    factory = Mock(
        side_effect=lambda *args, **kwargs: CellPose(
            7, enc_name="resnet18", enc_pretrain=False, device=kwargs["device"]
        )
    )
    monkeypatch.setattr("cellseg_models_pytorch.models.cellpose.CellPose", factory)
    *setup, loop, resume, evaluation, visualization = _code_cells("training")
    namespace = {}
    for source in setup:
        exec(source, namespace)
    assert namespace["device"].type == execution_device
    assert factory.call_args.kwargs["enc_pretrain"] is True
    assert isinstance(namespace["optimizer"], torch.optim.AdamW)
    assert namespace["criterion"].weights == {"cellpose": 1.0, "type_mask": 1.0}
    assert namespace["criterion"]["type_mask"]["loss1"].apply_sd
    from cellseg_models_pytorch.losses import DiceLoss

    assert isinstance(namespace["criterion"]["type_mask"]["loss2"], DiceLoss)
    model = namespace["model"]
    encoder = model[model.enc_name]
    encoder_before = {
        name: value.detach().clone() for name, value in encoder.state_dict().items()
    }
    if freeze_encoder:
        notebook = json.loads((TUTORIALS / "training.ipynb").read_text())
        section = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if "".join(cell["source"]).startswith("## Optional: freeze the encoder")
        )
        freeze, evaluation_mode = re.findall(
            r"```python\n(.*?)```", section, flags=re.S
        )
        exec(freeze, namespace)
        loop = loop.replace(
            "    model.train()",
            "    model.train()\n"
            + "\n".join("    " + line for line in evaluation_mode.splitlines()),
        )
    head = model.decoder["type-nuc_type_head"].head.weight
    before = head.detach().clone()
    namespace["num_epochs"] = 1
    exec(loop, namespace)
    assert not torch.equal(before, head.detach())
    if freeze_encoder:
        assert all(parameter.grad is None for parameter in encoder.parameters())
        for name, value in encoder_before.items():
            torch.testing.assert_close(
                encoder.state_dict()[name], value, rtol=0, atol=0
            )
    else:
        assert any(
            not torch.equal(encoder.state_dict()[name], value)
            for name, value in encoder_before.items()
        )
    assert namespace["scheduler"].last_epoch == 1
    instances, classes, _ = prepared_panoptils
    _check_validation_loss(namespace, classes, instances)
    weights_path = namespace["saved_checkpoint"].with_name("model.safetensors")
    saved_weights = weights_path.read_bytes()
    _check_reload_and_prediction(namespace, resume, evaluation, visualization)
    assert "model" not in namespace["state"]
    assert weights_path.read_bytes() == saved_weights
    from safetensors import safe_open

    with safe_open(weights_path, framework="pt", device="cpu") as weights:
        assert json.loads(weights.metadata()["class_names"]) == namespace["class_names"]
        assert weights.metadata()["normalization"] == "global min-max, RGB float32"


def test_evaluation_counts_foreground_without_background_and_marks_absent_classes(
    prepared_panoptils,
):
    import pandas as pd
    from cellseg_models_pytorch.metrics import panoptic_quality
    from cellseg_models_pytorch.transforms.functional.normalization import (
        minmax_normalize,
    )
    from cellseg_models_pytorch.utils import FileHandler

    instance_mask = np.ones((64, 64), dtype=np.int32)
    inference = Mock()
    inference.post_process.return_value = {
        "nuc": [(instance_mask, instance_mask.copy())]
    }
    namespace = {
        "Path": Path,
        "np": np,
        "pd": pd,
        "FileHandler": FileHandler,
        "panoptic_quality": panoptic_quality,
        "minmax_normalize": minmax_normalize,
        "inference_model": inference,
        "data_root": Path("../../data/panoptils-nuclei"),
        "class_names": [
            "background",
            "neoplastic",
            "stromal",
            "inflammatory",
            "epithelial",
            "other",
            "unknown",
        ],
    }
    exec(_code_cells("training")[-2], namespace)
    metrics = namespace["metrics"]
    assert metrics.loc["all nuclei", "target objects"] == 2
    assert metrics.loc["all nuclei", "predicted objects"] == 1
    assert metrics.loc["neoplastic", "predicted objects"] == 1
    assert metrics.loc["neoplastic", "PQ"] == 0
    assert metrics.loc["inflammatory", "target objects"] == 2
    assert metrics.loc["inflammatory", "PQ"] == 0
    assert np.isnan(metrics.loc["epithelial", "PQ"])


def test_csmp_hub_loading_example(tmp_path, monkeypatch):
    from safetensors.torch import save_model

    from cellseg_models_pytorch.models.cellpose import CellPose

    original = CellPose(
        7, enc_name="resnet18", enc_pretrain=False, device=torch.device("cpu")
    )
    weights_path = tmp_path / "model.safetensors"
    save_model(original.model, weights_path)
    download = Mock(return_value=str(weights_path))
    monkeypatch.setattr(
        "cellseg_models_pytorch.models.base._base_model_inst.hf_hub_download", download
    )
    notebook = json.loads((TUTORIALS / "inference.ipynb").read_text())
    section = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "".join(cell["source"]).startswith("## Load a csmp-hub model")
    )
    [source] = re.findall(r"```python\n(.*?)```", section, flags=re.S)
    namespace = {"CellPose": CellPose, "device": torch.device("cpu")}
    exec(source, namespace)
    download.assert_called_once_with(
        repo_id="csmp-hub/cellpose-histo-hgsc-nuc-v1",
        filename="cellpose_hgsc_v1_efficientnet_b5.safetensors",
    )
    restored = namespace["pretrained_model"]
    assert restored.inference_mode
    assert not restored.model.training
    for name, value in original.model.state_dict().items():
        torch.testing.assert_close(
            restored.model.state_dict()[name], value, rtol=0, atol=0
        )
