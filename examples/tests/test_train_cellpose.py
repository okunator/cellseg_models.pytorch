import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.io import savemat


@pytest.mark.optional
def test_training_example_updates_weights_and_preserves_checkpoint(tmp_path):
    pytest.importorskip("albumentations")
    path = Path(__file__).parents[1] / "train_cellpose.py"
    spec = importlib.util.spec_from_file_location("train_cellpose_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    inst = np.zeros((64, 64), dtype=np.int32)
    inst[16:32, 16:32] = 7
    rgb = np.repeat((inst > 0)[..., None], 3, axis=-1).astype(np.uint8) * 255
    Image.fromarray(rgb).save(images / "patch.png")
    savemat(labels / "patch.mat", {"inst": inst})
    checkpoint = tmp_path / "checkpoint.pt"
    example.train(images, labels, checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["epoch"] == 1
    optimizer_states = list(state["optimizer"]["state"].values())
    assert optimizer_states and all(item["step"] == 1 for item in optimizer_states)
    assert any(torch.count_nonzero(item["exp_avg"]) for item in optimizer_states)
    assert all(torch.isfinite(value).all() for value in state["model"].values())
    torch.manual_seed(0)
    model = example.cellpose_nuclei(**state["model_kwargs"])
    assert any(
        not torch.equal(value, state["model"][name])
        for name, value in model.named_parameters()
    )
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255)["nuc"]
    assert output.type_map.shape == (1, 2, 64, 64)
    assert output.aux_map.shape == (1, 2, 64, 64)
    original = checkpoint.read_bytes()
    with pytest.raises(FileExistsError):
        example.train(images, labels, checkpoint)
    assert checkpoint.read_bytes() == original
