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
from scipy.special import softmax

from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.models.cellpose import CellPose
from cellseg_models_pytorch.models.stardist import StarDist
from cellseg_models_pytorch.transforms.functional.normalization import minmax_normalize

DATA = Path(__file__).parent / "fixtures" / "checkpoint_baseline"
ROOT = Path(__file__).resolve().parents[1]
METADATA = json.loads((DATA / "metadata.json").read_text())
MODELS = {"cellpose": CellPose, "stardist": StarDist}


def _type_probabilities(logits: np.ndarray) -> np.ndarray:
    """Evaluate BCHW class probabilities in float64 for numerical comparison."""
    return softmax(logits.astype(np.float64), axis=1)


@pytest.mark.parametrize("offset", [0, 1000])
def test_type_probabilities_preserve_class_axis_and_common_offset(offset: int) -> None:
    logits = np.array([1, 0, 0, 1], dtype=np.float32).reshape(1, 2, 1, 2)
    probability = 1 / (1 + np.exp(-1))
    expected = np.array(
        [probability, 1 - probability, 1 - probability, probability]
    ).reshape(1, 2, 1, 2)
    actual = _type_probabilities(logits + offset)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


def test_type_probabilities_detect_confidence_change_with_same_classes() -> None:
    logits = np.array([1, 0, 0, 1], dtype=np.float32).reshape(1, 2, 1, 2)
    np.testing.assert_array_equal(logits.argmax(1), (2 * logits).argmax(1))
    assert not np.allclose(
        _type_probabilities(logits),
        _type_probabilities(2 * logits),
        atol=METADATA["dense_atol"],
        rtol=METADATA["dense_rtol"],
    )


def _same_instance_partition(actual: np.ndarray, expected: np.ndarray) -> bool:
    """Compare exact object pixels and background, allowing positive IDs to permute."""
    if np.any(actual < 0) or not np.array_equal(actual == 0, expected == 0):
        return False
    pairs = np.unique(np.stack((actual.ravel(), expected.ravel()), axis=1), axis=0)
    return len(pairs) == len(np.unique(actual)) == len(np.unique(expected))


@pytest.mark.parametrize(
    ("actual", "matches"),
    [
        ([[0, 1, 1], [2, 2, 0]], True),
        ([[0, 9, 9], [4, 4, 0]], True),
        ([[0, 1, 3], [2, 2, 0]], False),  # split
        ([[0, 1, 1], [1, 1, 0]], False),  # merge
        ([[0, 1, 2], [2, 2, 0]], False),  # moved boundary
        ([[1, 0, 0], [2, 2, 1]], False),  # background swapped
        ([[0, 1, 1], [2, 0, 0]], False),  # lost foreground
        ([0, 1, 1, 2, 2, 0], False),  # wrong shape
        ([[0, -1, -1], [2, 2, 0]], False),
    ],
)
def test_instance_partition_comparison(
    actual: list[list[int]] | list[int], matches: bool
) -> None:
    expected = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int32)
    assert (
        _same_instance_partition(np.array(actual, dtype=np.int32), expected) is matches
    )


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
                type_head = index == len(dense) - 1
                output = "type_probabilities" if type_head else f"dense_{index}"
                with subtests.test(output=output):
                    current, reference = actual.numpy(), expected[f"dense_{index}"]
                    assert current.shape == reference.shape
                    assert current.dtype == reference.dtype
                    if type_head:
                        current = _type_probabilities(current)
                        reference = _type_probabilities(reference)
                    np.testing.assert_allclose(
                        current,
                        reference,
                        atol=METADATA["dense_atol"],
                        rtol=METADATA["dense_rtol"],
                    )
            with subtests.test(output="instance_mask"):
                reference = expected["instance_mask"]
                assert instances.dtype == reference.dtype
                assert _same_instance_partition(instances, reference), (
                    "Object pixels or background differ"
                )
            with subtests.test(output="type_mask"):
                assert classes.dtype == expected["type_mask"].dtype
                np.testing.assert_array_equal(classes, expected["type_mask"])
            with subtests.test(output="instance_count"):
                assert len(np.unique(instances)) - 1 == record["instances"]
    finally:
        torch.set_num_threads(previous_threads)
