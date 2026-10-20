"""Load local checkpoints without downloading encoder or segmentation weights."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_model

from cellseg_models_pytorch.models.cellpose import CellPose
from cellseg_models_pytorch.models.cellvit import CellVit
from cellseg_models_pytorch.models.base import PRETRAINED
from cellseg_models_pytorch.models.stardist import StarDist


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


@pytest.mark.parametrize("suffix", [".pt", ".safetensors"])
def test_non_ray_checkpoint_loading_is_unchanged(tmp_path: Path, suffix: str) -> None:
    torch.manual_seed(0)
    source = CellPose(
        3, enc_name="resnet18", enc_pretrain=False, device=torch.device("cpu")
    )
    path = tmp_path / f"weights{suffix}"
    if suffix == ".safetensors":
        save_model(source.model, path)
    else:
        torch.save(source.model.state_dict(), path)
    loaded = CellPose.from_pretrained(path, device=torch.device("cpu"))
    path.unlink()
    for key, expected in source.model.state_dict().items():
        torch.testing.assert_close(
            loaded.model.state_dict()[key], expected, rtol=0, atol=0
        )
