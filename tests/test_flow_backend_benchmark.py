"""Offline checks for the paired reconstruction benchmark."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cellseg_models_pytorch.transforms.functional import gen_flow_maps
from tools.benchmark_flow_backends import (
    compare_case,
    measure_pair,
    read_row,
    same_partition,
)


@pytest.mark.parametrize(
    ("actual", "matches"),
    [
        ([[0, 9, 9], [4, 4, 0]], True),
        ([[0, 1, 3], [2, 2, 0]], False),
        ([[0, 1, 1], [1, 1, 0]], False),
        ([[0, 1, 2], [2, 2, 0]], False),
        ([[1, 0, 0], [2, 2, 1]], False),
    ],
)
def test_comparison_detects_partition_changes(actual, matches: bool) -> None:
    reference = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int32)
    assert same_partition(np.array(actual), reference) is matches


def test_paired_timing_alternates_order_and_excludes_warmup() -> None:
    calls = []
    timings = measure_pair(
        {
            "original": lambda: calls.append("original"),
            "tensor": lambda: calls.append("tensor"),
        },
        warmup=1,
        repeats=2,
    )
    assert calls == ["original", "tensor", "tensor", "original", "original", "tensor"]
    assert all(len(timing["samples_ms"]) == 2 for timing in timings.values())
    with pytest.raises(ValueError, match="positive"):
        measure_pair({}, warmup=0, repeats=1)


def test_read_row_uses_global_index_across_row_groups(tmp_path) -> None:
    path = tmp_path / "rows.parquet"
    table = pa.table(
        {
            "id": ["a", "b", "c"],
            "image": [b"i"] * 3,
            "inst": [b"m"] * 3,
            "type": [b"t"] * 3,
        }
    )
    pq.write_table(table, path, row_group_size=2)
    file = pq.ParquetFile(path)
    assert read_row(file, 2)["id"] == "c"
    with pytest.raises(IndexError):
        read_row(file, 3)


def test_small_target_reconstruction_and_metrics_agree() -> None:
    truth = np.zeros((17, 19), dtype=np.int32)
    truth[5:12, 5:14] = 1
    mask = truth > 0
    result, arrays = compare_case(
        "cellpose", gen_flow_maps(truth), mask, truth * 2, truth, 0, 1, 1
    )
    assert result["endpoint_max_abs_diff"] == 0
    assert (
        result["seed_indices_equal"]
        and result["partition_equal"]
        and result["classes_equal"]
    )
    assert (
        result["outputs"]["original"]["objects"]
        == result["outputs"]["tensor"]["objects"]
        == 1
    )
    assert result["outputs"]["tensor"]["pq"] > 0.9
    np.testing.assert_array_equal(
        arrays["original_instances"], arrays["tensor_instances"]
    )
