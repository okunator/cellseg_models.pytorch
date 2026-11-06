"""Check the offline migration baseline and its measurement boundaries."""

import json
import sys

import numpy as np
import pytest
import torch

from tools import profile_postprocessing as profile


@pytest.mark.parametrize("family", ["cellpose", "stardist"])
def test_profile_pinned_cpu_predictions(family: str, monkeypatch) -> None:
    def unexpected_cuda(*args, **kwargs):
        pytest.fail("CPU profiling attempted to use CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_cuda)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        report = profile.profile_case(family, torch.device("cpu"), warmup=1, repeats=1)
    finally:
        torch.set_num_threads(previous_threads)
    assert report["instances"] == {"cellpose": 96, "stardist": 79}[family]
    assert report["device"] == "cpu"
    assert report["prepare_device_to_host_bytes"] == 0
    assert report["cuda_peak_allocated_bytes"] is None
    assert (
        report["prepared_array_bytes"]
        == 256 * 256 * (4 if family == "cellpose" else 34) * 4
    )
    for timing in report["timings_ms"].values():
        assert len(timing["samples"]) == 1
        assert timing["median"] == timing["samples"][0] > 0


@pytest.mark.parametrize(
    "changed",
    [
        [[0, 9, 9], [4, 4, 0]],  # identifiers may permute
        [[0, 1, 3], [2, 2, 0]],  # split
        [[0, 1, 1], [1, 1, 0]],  # merge
        [[1, 0, 0], [2, 2, 1]],  # background swapped
        [[0, 1, 2], [2, 2, 0]],  # moved boundary, same count
    ],
)
def test_partition_parity(changed: list[list[int]]) -> None:
    expected = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int32)
    reference = {"instances": expected, "classes": np.zeros_like(expected)}
    actual = np.array(changed, dtype=np.int32)
    if changed[0][1] == 9:
        profile.check_result((actual, reference["classes"]), reference)
    else:
        with pytest.raises(ValueError, match="Instance"):
            profile.check_result((actual, reference["classes"]), reference)


@pytest.mark.parametrize("change", ["class_values", "class_dtype", "instance_dtype"])
def test_parity_rejects_changed_classes_and_dtypes(change: str) -> None:
    expected = np.array([[0, 1], [2, 2]], dtype=np.int32)
    reference = {"instances": expected, "classes": expected}
    instances, classes = expected.copy(), expected.copy()
    if change == "class_values":
        classes[1, 1] = 3
    elif change == "class_dtype":
        classes = classes.astype(np.int64)
    else:
        instances = instances.astype(np.int64)
    with pytest.raises(ValueError, match="dtype|Class"):
        profile.check_result((instances, classes), reference)


def test_timing_synchronizes_before_and_after_work(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    times = iter([1.0, 1.05])
    monkeypatch.setattr(profile, "perf_counter", lambda: next(times))
    result, elapsed = profile.timed(
        lambda: events.append("work") or 7, torch.device("cuda:0")
    )
    assert events == ["sync", "work", "sync"]
    assert result == 7
    assert elapsed == pytest.approx(50)


@pytest.mark.parametrize(("warmup", "repeats"), [(0, 1), (1, 0), (-1, 1)])
def test_rejects_invalid_repetition_counts(warmup: int, repeats: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        profile.profile_case("cellpose", torch.device("cpu"), warmup, repeats)


def test_no_silent_device_fallback(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        profile.profile_case("cellpose", torch.device("cuda:0"))
    with pytest.raises(ValueError, match="Only explicit CPU and CUDA"):
        profile.profile_case("cellpose", torch.device("mps"))


def test_fixture_checksum_is_verified(tmp_path, monkeypatch) -> None:
    (tmp_path / "metadata.json").write_bytes(
        (profile.FIXTURES / "metadata.json").read_bytes()
    )
    (tmp_path / "cellpose.npz").write_bytes(b"not the reference")
    monkeypatch.setattr(profile, "FIXTURES", tmp_path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        profile.load_case("cellpose", torch.device("cpu"))


def test_cli_preserves_existing_report(tmp_path, monkeypatch) -> None:
    output = tmp_path / "report.json"
    output.write_text("preserve")
    monkeypatch.delenv("NUMBA_DISABLE_JIT", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["profile_postprocessing", "--family", "cellpose", "--output", str(output)],
    )
    with pytest.raises(SystemExit, match="2"):
        profile.main()
    assert output.read_text() == "preserve"


def test_cli_writes_json_report(tmp_path, monkeypatch) -> None:
    output = tmp_path / "report.json"
    monkeypatch.delenv("NUMBA_DISABLE_JIT", raising=False)
    monkeypatch.setattr(profile, "profile_case", lambda *args: {"family": "cellpose"})
    monkeypatch.setattr(
        sys,
        "argv",
        ["profile_postprocessing", "--family", "cellpose", "--output", str(output)],
    )
    previous_threads = torch.get_num_threads()
    try:
        profile.main()
    finally:
        torch.set_num_threads(previous_threads)
    report = json.loads(output.read_text())
    assert report["family"] == "cellpose"
    assert report["environment"]["torch_threads"] == 2
    assert len(report["environment"]["source_commit"]) == 40
    assert report["environment"]["gpu"] is None


def test_cli_rejects_disabled_jit(tmp_path, monkeypatch) -> None:
    output = tmp_path / "report.json"
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        ["profile_postprocessing", "--family", "cellpose", "--output", str(output)],
    )
    with pytest.raises(SystemExit, match="2"):
        profile.main()
    assert not output.exists()


@pytest.mark.cuda
@pytest.mark.parametrize("family", ["cellpose", "stardist"])
def test_profile_pinned_cuda_predictions(family: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    report = profile.profile_case(family, torch.device("cuda:0"), warmup=1, repeats=1)
    expected_bytes = 256 * 256 * (24 if family == "cellpose" else 140)
    assert report["prepare_device_to_host_bytes"] == expected_bytes
    assert report["cuda_peak_allocated_bytes"] > 0
    assert report["instances"] == {"cellpose": 96, "stardist": 79}[family]
