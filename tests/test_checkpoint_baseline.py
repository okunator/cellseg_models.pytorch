"""Pinned CPU predictions; real checkpoint runs require --optional."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.models.cellpose import CellPose
from cellseg_models_pytorch.models.stardist import StarDist
from cellseg_models_pytorch.transforms.functional.normalization import minmax_normalize

DATA = Path(__file__).parent / "fixtures" / "checkpoint_baseline"
ROOT = Path(__file__).resolve().parents[1]
METADATA = json.loads((DATA / "metadata.json").read_text())
MODELS = {"cellpose": CellPose, "stardist": StarDist}


def test_checkpoint_baseline_fixture_identity() -> None:
    """Catch edited inputs, corrupt references, and inconsistent output contracts."""
    assert (
        hashlib.sha256((ROOT / METADATA["image"]).read_bytes()).hexdigest()
        == METADATA["image_sha256"]
    )
    for family, record in METADATA["models"].items():
        reference = DATA / f"{family}.npz"
        assert (
            hashlib.sha256(reference.read_bytes()).hexdigest()
            == record["reference_sha256"]
        )
        with np.load(reference, allow_pickle=False) as expected:
            channels = [2, 7] if family == "cellpose" else [1, 32, 7]
            for index, count in enumerate(channels):
                dense = expected[f"dense_{index}"]
                assert dense.shape == (1, count, *METADATA["size"])
                assert dense.dtype == np.float32 and np.isfinite(dense).all()
            instances, classes = expected["instance_mask"], expected["type_mask"]
            assert instances.shape == classes.shape == tuple(METADATA["size"])
            assert instances.dtype == classes.dtype == np.int32
            assert len(np.unique(instances)) - 1 == record["instances"]
            assert set(np.unique(classes)) <= set(METADATA["class_ids"])
            assert not classes[instances == 0].any()


@pytest.mark.optional
@pytest.mark.parametrize("family", ["cellpose", "stardist"])
def test_pretrained_cpu_checkpoint_baseline(
    family: str, subtests: pytest.Subtests
) -> None:
    """Compare pinned public weights against preserved dense arrays and masks."""
    record = METADATA["models"][family]
    checkpoint = record["checkpoint"]
    path = Path(
        hf_hub_download(
            repo_id=checkpoint["repo_id"],
            filename=checkpoint["filename"],
            revision=checkpoint["revision"],
            cache_dir=os.environ.get("CELLSEG_BASELINE_CACHE"),
        )
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint["sha256"]
    with Image.open(ROOT / METADATA["image"]) as source:
        image = source.convert("RGB").resize(
            tuple(METADATA["size"]), Image.Resampling.BILINEAR
        )
    array = minmax_normalize(np.asarray(image), copy=True)
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    assert tensor.dtype == torch.float32 and tensor.device.type == "cpu"
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(METADATA["threads"])
        model = MODELS[family].from_pretrained(path, device=torch.device("cpu"))
        assert model.model.enc_name == METADATA["encoder"]
        model.set_inference_mode(mixed_precision=False)
        if family == "cellpose":
            model.post_processor = PostProcessor("cellpose", {"use_gpu": False})
        with torch.inference_mode():
            output = model.model(tensor)["nuc"]
        dense = (
            (output.aux_map, output.type_map)
            if family == "cellpose"
            else (output.binary_map, output.aux_map, output.type_map)
        )
        instances, classes = model.post_process(model.predict(tensor))["nuc"][0]
        with np.load(DATA / f"{family}.npz", allow_pickle=False) as expected:
            for index, actual in enumerate(dense):
                with subtests.test(output=f"dense_{index}"):
                    np.testing.assert_allclose(
                        actual.numpy(),
                        expected[f"dense_{index}"],
                        atol=METADATA["dense_atol"],
                        rtol=METADATA["dense_rtol"],
                    )
            with subtests.test(output="instance_mask"):
                reference = expected["instance_mask"]
                pairs = np.unique(
                    np.stack((instances.ravel(), reference.ravel()), axis=1), axis=0
                )
                foreground_changes = np.count_nonzero(
                    (instances > 0) != (reference > 0)
                )
                same_partition = foreground_changes == 0 and len(pairs) == len(
                    np.unique(instances)
                ) == len(np.unique(reference))
                np.testing.assert_array_equal(
                    instances,
                    reference,
                    err_msg=f"foreground changes={foreground_changes}, identical object partitions={same_partition}",
                )
            with subtests.test(output="type_mask"):
                np.testing.assert_array_equal(classes, expected["type_mask"])
            with subtests.test(output="instance_count"):
                assert len(np.unique(instances)) - 1 == record["instances"]
    finally:
        torch.set_num_threads(previous_threads)
