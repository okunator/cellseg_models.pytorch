"""Keep the historical imports bound to the standalone implementation."""

import importlib

import cellseg_metrics
import pytest

import cellseg_models_pytorch.metrics as legacy


@pytest.mark.parametrize("module", ["", ".binary_metrics", ".functional"])
def test_existing_exports_keep_function_identity(module):
    old = importlib.import_module(f"cellseg_models_pytorch.metrics{module}")
    new = importlib.import_module(f"cellseg_metrics{module}")

    # Standalone releases can add APIs without raising the main package's minimum version.
    assert set(old.__all__) <= set(new.__all__)
    for name in old.__all__:
        assert getattr(old, name) is getattr(new, name)


@pytest.mark.parametrize(
    "name", ["_match_instances", "_absent_inds", "get_stats", "iou_score"]
)
def test_existing_functional_helpers_keep_identity(name):
    old = importlib.import_module("cellseg_models_pytorch.metrics.functional")
    new = importlib.import_module("cellseg_metrics.functional")
    assert getattr(old, name) is getattr(new, name)


def test_old_import_computes_the_same_instance_scores():
    import numpy as np

    true = np.array([[0, 7, 7, 21, 21]])
    pred = np.array([[0, 42, 42, 0, 0]])
    expected = {"dq": 2 / 3, "sq": 1.0, "pq": 2 / 3}

    assert legacy.panoptic_quality(true, pred) == cellseg_metrics.panoptic_quality(
        true, pred
    )
    assert legacy.panoptic_quality(true, pred) == pytest.approx(expected)
