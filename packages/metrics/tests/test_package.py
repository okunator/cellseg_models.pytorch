"""Protect the lightweight runtime and independently calculated overlap scores."""

import subprocess
import sys
from importlib.metadata import distribution

import numpy as np
import pytest
from packaging.requirements import Requirement

from cellseg_metrics import average_precision, panoptic_quality


def test_runtime_dependencies_and_version():
    import cellseg_metrics

    package = distribution("cellseg-metrics")
    assert package.version == cellseg_metrics.__version__
    assert {Requirement(requirement).name for requirement in package.requires} == {
        "numpy",
        "scipy",
    }


def test_distribution_retains_license_and_attribution():
    package = distribution("cellseg-metrics")
    for name in ("LICENSE", "NOTICE.md", "LICENSES/HoVer-Net.txt"):
        files = [
            path for path in package.files if str(path).endswith(f"/licenses/{name}")
        ]
        assert len(files) == 1, name
        assert files[0].locate().read_text().strip(), name


def test_import_and_computation_without_models_or_imaging():
    # A fresh process catches eager imports even in the full development environment.
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import importlib.abc
import sys

class BlockHeavyPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {
            'cellseg_models_pytorch', 'torch', 'torchvision', 'timm', 'cv2',
            'numba', 'skimage', 'geopandas', 'rasterio', 'shapely', 'mpire',
            'pyarrow', 'pandas', 'tables', 'albumentations',
        }
        if fullname.split('.')[0] in blocked:
            raise AssertionError(f'Metrics imported unrelated package {fullname}')

sys.meta_path.insert(0, BlockHeavyPackages())

import numpy as np
from cellseg_metrics import mean_average_precision, panoptic_quality, precision_recall_ap

true = np.array([[0, 7, 7, 21, 21]])
pred = np.array([[0, 42, 42, 3, 3]])
np.testing.assert_allclose(list(panoptic_quality(true, pred).values()), 1.0, atol=1e-6)
assert precision_recall_ap(true, pred, {3: 0.8, 42: 0.9}) == 1.0
assert mean_average_precision(true, pred, {3: 0.8, 42: 0.9}) == 1.0
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("transpose", [False, True])
def test_partial_overlap_with_missed_and_extra_nuclei(transpose):
    # One match has IoU 2/3; another reference nucleus is missed and one is extra.
    true = np.array([[0, 7, 7, 7, 0, 21, 21, 0]])
    pred = np.array([[0, 42, 42, 0, 0, 0, 0, 99]])
    if transpose:
        true, pred = true.T, pred.T
    originals = true.copy(), pred.copy()

    scores = panoptic_quality(true, pred)

    assert scores == pytest.approx({"dq": 0.5, "sq": 2 / 3, "pq": 1 / 3})
    assert average_precision(true, pred) == pytest.approx(0.5)
    np.testing.assert_array_equal(true, originals[0])
    np.testing.assert_array_equal(pred, originals[1])
