"""Load local checkpoints without downloading encoder or segmentation weights."""

import weakref
from pathlib import Path
from unittest.mock import Mock, create_autospec

import pytest
import safetensors.torch as safetensors_torch
import torch
from safetensors.torch import save_file, save_model

from cellseg_models_pytorch.models.base._base_model_inst import BaseModelInst
from cellseg_models_pytorch.models.cellpose import CellPose
from cellseg_models_pytorch.models.cellvit import CellVit
from cellseg_models_pytorch.models.base import PRETRAINED
from cellseg_models_pytorch.models.stardist import StarDist


@pytest.mark.parametrize(
    ("device_name", "loading_device"),
    [
        ("cpu", "cpu"),
        ("cpu:0", "cpu"),
        ("cpu:1", "cpu"),
        ("cuda", "cuda"),
        ("cuda:0", "cuda:0"),
        ("cuda:1", "cuda:1"),
    ],
    ids=["cpu", "cpu-0", "cpu-1", "default-cuda", "cuda-0", "cuda-1"],
)
def test_safetensors_loading_preserves_device_and_strictness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_name: str,
    loading_device: str,
) -> None:
    path = tmp_path / "weights.safetensors"
    path.touch()
    device = torch.device(device_name)
    state = {
        "resnet18.encoder.weight": torch.ones(2, 2),
        "decoder.nuc_type_head.head.weight": torch.ones(3, 2, 1, 1),
    }
    read = create_autospec(safetensors_torch.load_file, return_value=state)
    load = create_autospec(safetensors_torch.load_model, return_value=([], []))

    def initialize_without_cuda(
        self: CellPose, *, device: torch.device, **kwargs: object
    ) -> None:
        BaseModelInst.__init__(self)
        self.model = torch.nn.Identity()
        self.device = device

    # Check real loader signatures and routing without requiring CUDA hardware.
    monkeypatch.setattr(CellPose, "__init__", initialize_without_cuda)
    monkeypatch.setattr(safetensors_torch, "load_file", read)
    monkeypatch.setattr(safetensors_torch, "load_model", load)

    loaded = CellPose.from_pretrained(path, device=device)

    assert loaded.device == device
    assert not loaded.inference_mode
    read.assert_called_once_with(path, device=loading_device)
    load.assert_called_once_with(loaded.model, path, strict=True, device=loading_device)


@pytest.mark.parametrize("model_class", [CellPose, StarDist])
def test_safetensors_releases_metadata_tensors_before_final_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_class: type[CellPose] | type[StarDist],
) -> None:
    torch.manual_seed(0)
    options = {"out_channels": (16, 8, 4, 2)}
    source = model_class(
        3,
        enc_name="resnet18",
        enc_pretrain=False,
        device=torch.device("cpu"),
        model_kwargs=options,
    )
    path = tmp_path / "weights.safetensors"
    save_model(source.model, path)
    read = safetensors_torch.load_file
    load = safetensors_torch.load_model
    metadata_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def track_metadata_read(
        filename: Path, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        state = read(filename, **kwargs)
        # Weak references observe lifetime without keeping checkpoint tensors alive.
        if not metadata_refs:
            metadata_refs.extend(weakref.ref(value) for value in state.values())
        return state

    def load_after_metadata_release(
        model: torch.nn.Module, filename: Path, **kwargs: object
    ) -> tuple[list[str], list[str]]:
        assert metadata_refs
        assert all(value() is None for value in metadata_refs)
        return load(model, filename, **kwargs)

    monkeypatch.setattr(safetensors_torch, "load_file", track_metadata_read)
    monkeypatch.setattr(safetensors_torch, "load_model", load_after_metadata_release)

    loaded = model_class.from_pretrained(
        path, device=torch.device("cpu"), model_kwargs=options
    )

    for key, expected in source.model.state_dict().items():
        torch.testing.assert_close(
            loaded.model.state_dict()[key], expected, rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "name", ["hgsc_v1_samvit_base_patch16", "hgsc_v1_efficientnet_b5"]
)
def test_cellvit_checkpoint_names_resolve_same_weights(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    download = Mock(side_effect=ConnectionError("offline test"))
    monkeypatch.setattr(
        "cellseg_models_pytorch.models.base._base_model_inst.hf_hub_download", download
    )
    with pytest.raises(ConnectionError, match="offline test"):
        CellVit.from_pretrained(name, device=torch.device("cpu"))
    download.assert_called_once_with(
        repo_id="csmp-hub/cellvit-histo-hgsc-nuc-v1",
        filename="cellvit_hgsc_v1_efficientnet_b5.safetensors",
    )


@pytest.mark.parametrize("n_rays", [8, 32, 48])
@pytest.mark.parametrize(
    ("suffix", "registered"),
    [(".pt", False), (".safetensors", False), (".safetensors", True)],
)
def test_checkpoint_loading_infers_ray_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    n_rays: int,
    suffix: str,
    registered: bool,
) -> None:
    torch.manual_seed(0)
    options = {"out_channels": (16, 8, 4, 2)}
    source = StarDist(
        3,
        n_rays=n_rays,
        enc_name="resnet18",
        enc_pretrain=False,
        device=torch.device("cpu"),
        model_kwargs=options,
    )
    source.model.eval()
    path = tmp_path / f"weights{suffix}"
    if suffix == ".safetensors":
        save_model(source.model, path)
    else:
        torch.save(source.model.state_dict(), path)

    download = Mock(return_value=str(path))
    monkeypatch.setattr(
        "cellseg_models_pytorch.models.base._base_model_inst.hf_hub_download", download
    )
    name = "hgsc_v1_efficientnet_b5"
    loaded = StarDist.from_pretrained(
        name if registered else path, device=torch.device("cpu"), model_kwargs=options
    )
    if registered:
        download.assert_called_once_with(**PRETRAINED[StarDist.model_name][name])
    else:
        download.assert_not_called()
    path.unlink()
    assert loaded.model.enc_name == "resnet18"
    assert loaded.model.decoder.heads["stardist"]["nuc_stardist"] == n_rays
    assert not loaded.inference_mode
    for key, expected in source.model.state_dict().items():
        torch.testing.assert_close(
            loaded.model.state_dict()[key], expected, rtol=0, atol=0
        )
    image = torch.rand(1, 3, 64, 64)
    loaded.set_inference_mode(mixed_precision=False)
    with torch.no_grad():
        expected = source.model(image)["nuc"]
        actual = loaded.model(image)["nuc"]
    assert actual.aux_map.shape == (1, n_rays, 64, 64)
    for field in ("aux_map", "binary_map", "type_map"):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field))
    prediction = loaded.predict(image)["nuc"]
    torch.testing.assert_close(prediction.aux_map, expected.aux_map)
    torch.testing.assert_close(prediction.type_map, expected.type_map.argmax(1))


@pytest.mark.parametrize(
    "ray_weights", [None, torch.tensor(1), torch.empty(0, 2, 1, 1)]
)
def test_checkpoint_loading_rejects_invalid_ray_head(
    tmp_path: Path, ray_weights: torch.Tensor | None
) -> None:
    state = {"decoder.type-nuc_type_head.head.weight": torch.ones(3, 2, 1, 1)}
    if ray_weights is not None:
        state["decoder.stardist-nuc_stardist_head.head.weight"] = ray_weights
    path = tmp_path / "invalid.pt"
    torch.save(state, path)

    with pytest.raises(RuntimeError, match="ray-head weights"):
        StarDist.from_pretrained(path, device=torch.device("cpu"))


@pytest.mark.parametrize(
    ("suffix", "device_name"),
    [
        (".pt", "cpu"),
        (".safetensors", "cpu"),
        (".safetensors", "cpu:0"),
        (".safetensors", "cpu:1"),
    ],
)
def test_non_ray_checkpoint_loading_is_unchanged(
    tmp_path: Path, suffix: str, device_name: str
) -> None:
    torch.manual_seed(0)
    source = CellPose(
        3, enc_name="resnet18", enc_pretrain=False, device=torch.device("cpu")
    )
    path = tmp_path / f"weights{suffix}"
    if suffix == ".safetensors":
        save_model(source.model, path)
    else:
        torch.save(source.model.state_dict(), path)
    loaded = CellPose.from_pretrained(path, device=torch.device(device_name))
    path.unlink()
    assert loaded.device == torch.device(device_name)
    for key, expected in source.model.state_dict().items():
        torch.testing.assert_close(
            loaded.model.state_dict()[key], expected, rtol=0, atol=0
        )
    source.model.eval()
    loaded.model.eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        expected_maps = source.model(image)["nuc"]
        actual_maps = loaded.model(image)["nuc"]
    for field in ("type_map", "aux_map", "binary_map"):
        expected = getattr(expected_maps, field)
        actual = getattr(actual_maps, field)
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("suffix", [".pt", ".safetensors"])
@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("missing", "missing key"),
        ("unexpected", "unexpected key"),
        ("shape", "size mismatch"),
    ],
)
def test_checkpoint_loading_remains_strict(
    tmp_path: Path, suffix: str, corruption: str, message: str
) -> None:
    torch.manual_seed(0)
    options = {"out_channels": (16, 8, 4, 2)}
    source = CellPose(
        3,
        enc_name="resnet18",
        enc_pretrain=False,
        device=torch.device("cpu"),
        model_kwargs=options,
    )
    state = {name: value.clone() for name, value in source.model.state_dict().items()}
    key = "resnet18.encoder.encoder.conv1.weight"
    assert key in state
    if corruption == "missing":
        del state[key]
    elif corruption == "unexpected":
        state["unexpected.weight"] = torch.ones(1)
    else:
        state[key] = state[key][:0]
    path = tmp_path / f"invalid{suffix}"
    if suffix == ".safetensors":
        save_file(state, path)
    else:
        torch.save(state, path)

    with pytest.raises(RuntimeError, match=f"(?i){message}"):
        CellPose.from_pretrained(path, device=torch.device("cpu"), model_kwargs=options)
