"""Pinned HGSC ONNX checks; real checkpoints require --optional."""

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download

from cellseg_models_pytorch.decoders.multitask_decoder import SoftInstanceOutput
from cellseg_models_pytorch.models.cellvit import CellVit
from cellseg_models_pytorch.models.hovernet import HoverNet
from cellseg_models_pytorch.transforms.functional.normalization import minmax_normalize

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "hovernet": {
        "repo_id": "csmp-hub/hovernet-histo-hgsc-nuc-v1",
        "revision": "bf5d963e61ea9356f648f495d4857645a871027b",
        "filename": "hovernet_hgsc_v1_efficientnet_b5.safetensors",
        "sha256": "4eda0ae59bb88e11e6dee8dd2a6b5f7142f18866af9f266285b8bf4050bc9c87",
        "type_atol": 2e-5,
    },
    "cellvit": {
        "repo_id": "csmp-hub/cellvit-histo-hgsc-nuc-v1",
        "revision": "9173af3d7d8ffbb55844b1f1a0ac465d593bbe61",
        "filename": "cellvit_hgsc_v1_efficientnet_b5.safetensors",
        "sha256": "3462737aee2f833fd564bdcb64623126d96b8e2e91723314db519a8e6c2d461d",
        "type_atol": 4e-5,
    },
}
MODELS = {"hovernet": HoverNet, "cellvit": CellVit}


@pytest.fixture(scope="module", autouse=True)
def cpu_threads() -> Iterator[None]:
    """Keep tiny comparison fixtures from starting a large CPU thread pool."""
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        yield
    finally:
        torch.set_num_threads(previous)


def _file_sha256(path: Path) -> str:
    """Hash checkpoint bytes without copying the full file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_checkpoint_hash_covers_multiple_chunks(tmp_path: Path) -> None:
    data = b"calibration\n" * 100_000
    path = tmp_path / "checkpoint"
    path.write_bytes(data)
    assert _file_sha256(path) == hashlib.sha256(data).hexdigest()


def _assert_pinned_parity(
    family: str,
    checkpoint_sha256: str,
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
) -> None:
    """Compare raw maps and exact masks only for the calibrated checkpoint bytes."""
    record = CHECKPOINTS[family]
    assert checkpoint_sha256 == record["sha256"], "Uncalibrated checkpoint"
    assert (
        actual.keys()
        == expected.keys()
        == {"hv_map", "type_map", "instances", "classes"}
    )
    # Only these pinned class heads get the native float32/float64 calibration.
    for key, atol in (("hv_map", 1e-5), ("type_map", record["type_atol"])):
        assert actual[key].shape == expected[key].shape
        assert actual[key].dtype == expected[key].dtype == np.float32
        assert np.isfinite(actual[key]).all() and np.isfinite(expected[key]).all()
        np.testing.assert_allclose(
            actual[key], expected[key], rtol=1e-4, atol=atol, equal_nan=False
        )
    batch, channels, height, width = expected["hv_map"].shape
    assert channels == 2 and expected["type_map"].shape == (batch, 7, height, width)
    np.testing.assert_array_equal(
        actual["type_map"].argmax(1), expected["type_map"].argmax(1)
    )
    # Predictor uses float32 softmax; very close logits can round to a tie there.
    np.testing.assert_array_equal(
        torch.from_numpy(actual["type_map"]).softmax(1).argmax(1).numpy(),
        torch.from_numpy(expected["type_map"]).softmax(1).argmax(1).numpy(),
    )
    for key in ("instances", "classes"):
        assert actual[key].shape == expected[key].shape == (batch, height, width)
        assert actual[key].dtype == expected[key].dtype == np.int32
    for current, reference in zip(actual["instances"], expected["instances"]):
        assert not np.any(current < 0) and not np.any(reference < 0)
        np.testing.assert_array_equal(current == 0, reference == 0)
        pairs = np.unique(
            np.stack((current.ravel(), reference.ravel()), axis=1), axis=0
        )
        # IDs may differ; an object cannot split, merge, move or disappear.
        assert len(pairs) == len(np.unique(current)) == len(np.unique(reference))
    np.testing.assert_array_equal(actual["classes"], expected["classes"])


@pytest.fixture
def predictions() -> dict[str, np.ndarray]:
    logits = np.zeros((1, 7, 2, 3), dtype=np.float32)
    logits[:, 1] = 1
    instances = np.array([[[0, 1, 1], [2, 2, 0]]], dtype=np.int32)
    return {
        "hv_map": np.zeros((1, 2, 2, 3), dtype=np.float32),
        "type_map": logits,
        "instances": instances,
        "classes": (instances > 0).astype(np.int32),
    }


@pytest.mark.parametrize(
    ("family", "within", "outside"),
    [
        ("hovernet", 1.5e-5, 2.5e-5),
        ("cellvit", 3e-5, 4.5e-5),
        # Zero reference logits isolate each absolute tolerance from rtol.
        ("hovernet", 1.5e-5, 2.01e-5),
        ("cellvit", 3e-5, 4.01e-5),
    ],
)
def test_pinned_logit_budget_has_a_limit(
    family: str, within: float, outside: float, predictions: dict[str, np.ndarray]
) -> None:
    actual = {key: value.copy() for key, value in predictions.items()}
    actual["type_map"][:, 0] += within
    assert not np.allclose(
        actual["type_map"], predictions["type_map"], rtol=1e-4, atol=1e-5
    )
    _assert_pinned_parity(family, CHECKPOINTS[family]["sha256"], actual, predictions)
    actual["type_map"][:, 0] = outside
    with pytest.raises(AssertionError):
        _assert_pinned_parity(
            family, CHECKPOINTS[family]["sha256"], actual, predictions
        )


@pytest.mark.parametrize("family", CHECKPOINTS)
def test_pinned_budget_does_not_relax_reconstruction_maps(
    family: str, predictions: dict[str, np.ndarray]
) -> None:
    actual = {key: value.copy() for key, value in predictions.items()}
    actual["hv_map"] += 1.5e-5
    with pytest.raises(AssertionError):
        _assert_pinned_parity(
            family, CHECKPOINTS[family]["sha256"], actual, predictions
        )


@pytest.mark.parametrize("family", CHECKPOINTS)
def test_pinned_budget_rejects_different_checkpoint(
    family: str, predictions: dict[str, np.ndarray]
) -> None:
    with pytest.raises(AssertionError, match="Uncalibrated checkpoint"):
        _assert_pinned_parity(family, "0" * 64, predictions, predictions)


@pytest.mark.parametrize("family", CHECKPOINTS)
@pytest.mark.parametrize("decision", ["raw", "softmax"])
def test_pinned_budget_rejects_class_changes_inside_raw_budget(
    family: str, decision: str, predictions: dict[str, np.ndarray]
) -> None:
    predictions["type_map"].fill(0)
    predictions["type_map"][:, 0 if decision == "raw" else 1] = 1e-9
    actual = {key: value.copy() for key, value in predictions.items()}
    actual["type_map"].fill(0)
    actual["type_map"][:, 1] = 1e-9 if decision == "raw" else 5e-6
    assert np.allclose(actual["type_map"], predictions["type_map"], atol=1e-5)
    with pytest.raises(AssertionError):
        _assert_pinned_parity(
            family, CHECKPOINTS[family]["sha256"], actual, predictions
        )


@pytest.mark.parametrize(
    "instances",
    [
        [[0, 1, 3], [2, 2, 0]],  # split
        [[0, 1, 1], [1, 1, 0]],  # merge
        [[0, 1, 2], [2, 2, 0]],  # moved boundary, same object count
        [[0, 1, 1], [2, 0, 0]],  # missing foreground
        [[1, 0, 0], [2, 2, 1]],  # changed background
        [[0, -1, -1], [2, 2, 0]],
    ],
)
def test_pinned_budget_rejects_changed_objects(
    instances: list[list[int]], predictions: dict[str, np.ndarray]
) -> None:
    actual = {key: value.copy() for key, value in predictions.items()}
    actual["instances"] = np.array([instances], dtype=np.int32)
    with pytest.raises(AssertionError):
        _assert_pinned_parity(
            "hovernet", CHECKPOINTS["hovernet"]["sha256"], actual, predictions
        )


def test_pinned_budget_allows_only_per_image_id_permutations(
    predictions: dict[str, np.ndarray],
) -> None:
    expected = {key: np.repeat(value, 2, axis=0) for key, value in predictions.items()}
    actual = {key: value.copy() for key, value in expected.items()}
    for index in range(2):
        mask = actual["instances"][index]
        mask[mask > 0] += 10 * (index + 1)
    _assert_pinned_parity(
        "hovernet", CHECKPOINTS["hovernet"]["sha256"], actual, expected
    )


def test_pinned_budget_rejects_changed_object_class(
    predictions: dict[str, np.ndarray],
) -> None:
    actual = {key: value.copy() for key, value in predictions.items()}
    actual["classes"][0, 0, 1] = 2
    with pytest.raises(AssertionError):
        _assert_pinned_parity(
            "hovernet", CHECKPOINTS["hovernet"]["sha256"], actual, predictions
        )


@pytest.mark.parametrize("key", ["hv_map", "type_map", "instances", "classes"])
@pytest.mark.parametrize("change", ["shape", "dtype"])
def test_pinned_budget_rejects_changed_output_contract(
    key: str, change: str, predictions: dict[str, np.ndarray]
) -> None:
    actual = {name: value.copy() for name, value in predictions.items()}
    actual[key] = (
        actual[key].reshape(-1) if change == "shape" else actual[key].astype(np.float64)
    )
    with pytest.raises(AssertionError):
        _assert_pinned_parity(
            "hovernet", CHECKPOINTS["hovernet"]["sha256"], actual, predictions
        )


@pytest.mark.parametrize("key", ["hv_map", "type_map"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_pinned_budget_rejects_matching_nonfinite_maps(
    key: str, value: float, predictions: dict[str, np.ndarray]
) -> None:
    predictions[key].flat[0] = value
    with pytest.raises(AssertionError):
        _assert_pinned_parity(
            "hovernet", CHECKPOINTS["hovernet"]["sha256"], predictions, predictions
        )


def _with_masks(
    model: HoverNet | CellVit, maps: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Use identical public serial reconstruction for both inference backends."""
    pairs = []
    for index in range(maps["hv_map"].shape[0]):
        soft = SoftInstanceOutput(
            type_map=torch.from_numpy(maps["type_map"][index : index + 1])
            .softmax(1)
            .argmax(1),
            aux_map=torch.from_numpy(maps["hv_map"][index : index + 1]),
        )
        pairs.append(
            model.post_process({"nuc": soft, "cyto": None, "tissue": None})["nuc"][0]
        )
    return {
        **maps,
        "instances": np.stack([pair[0] for pair in pairs]),
        "classes": np.stack([pair[1] for pair in pairs]),
    }


@pytest.mark.optional
@pytest.mark.parametrize("family", CHECKPOINTS)
@pytest.mark.parametrize("dynamic_batch", [False, True])
def test_pinned_pretrained_onnx_parity(
    family: str, dynamic_batch: bool, tmp_path: Path
) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    record = CHECKPOINTS[family]
    checkpoint = Path(
        hf_hub_download(
            repo_id=record["repo_id"],
            revision=record["revision"],
            filename=record["filename"],
            cache_dir=os.environ.get("CELLSEG_BASELINE_CACHE"),
        )
    )
    checkpoint_hash = _file_sha256(checkpoint)
    assert checkpoint_hash == record["sha256"]
    image_path = ROOT / "cellseg_models_pytorch/utils/tests/data/HE.png"
    assert (
        _file_sha256(image_path)
        == "677d0b8828afc0ee2c6bc911945c2425baa32b68a6648bbc546b31706e37dd1e"
    )
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    image = minmax_normalize(
        cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    )
    if family == "cellvit":
        image = image[16:240, 16:240]
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        model = MODELS[family].from_pretrained(checkpoint, device=torch.device("cpu"))
        model.set_inference_mode(mixed_precision=False)
        path = model.export_onnx(
            tmp_path / "pinned.onnx", tensor.shape, dynamic_batch=dynamic_batch
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        assert [output.name for output in session.get_outputs()] == [
            "hv_map",
            "type_map",
        ]
        for batch in (1, 2, 3) if dynamic_batch else (1,):
            inputs = tensor.repeat(batch, 1, 1, 1)
            with torch.inference_mode():
                raw = model.model(inputs)["nuc"]
            expected = _with_masks(
                model, {"hv_map": raw.aux_map.numpy(), "type_map": raw.type_map.numpy()}
            )
            values = session.run(None, {"image": inputs.numpy()})
            actual = _with_masks(
                model, dict(zip(("hv_map", "type_map"), values, strict=True))
            )
            assert np.count_nonzero(expected["instances"]) > 0
            _assert_pinned_parity(family, checkpoint_hash, actual, expected)
    finally:
        torch.set_num_threads(previous_threads)
