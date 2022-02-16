from pathlib import Path

import numpy as np
import pytest

from cellseg_models_pytorch.utils import FileHandler


def pytest_addoption(parser):
    parser.addoption("--cuda", action="store_true", default=False, help="run gpu tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: mark test as a gpu test")


def pytest_collection_modifyitems(config, items):
    only_cuda = pytest.mark.skip(reason="--cuda option runs only gpu tests")
    if config.getoption("--cuda"):
        for item in items:
            if "cuda" not in item.keywords:
                item.add_marker(only_cuda)
    else:
        skip_cuda = pytest.mark.skip(reason="need --cuda option to run")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)


@pytest.fixture(scope="package")
def img_sample() -> np.ndarray:
    """Read in test RGB img."""
    path = Path().resolve()
    return FileHandler.read_img(path / "cellseg_models_pytorch/utils/tests/data/HE.png")


@pytest.fixture(scope="package")
def inst_map() -> np.ndarray:
    """Return a dummy labelled segmentation mask."""
    inst_map = np.array(
        [
            [2, 2, 2, 1, 1, 1, 1, 0, 5, 5],
            [2, 2, 2, 1, 1, 1, 1, 0, 5, 5],
            [2, 2, 0, 0, 1, 1, 1, 0, 5, 5],
            [2, 0, 0, 0, 0, 0, 0, 4, 4, 0],
            [0, 0, 3, 3, 3, 0, 4, 4, 4, 0],
            [0, 3, 3, 3, 3, 0, 4, 4, 4, 0],
            [0, 3, 3, 3, 0, 0, 4, 4, 4, 0],
            [0, 3, 3, 0, 0, 0, 4, 4, 0, 0],
            [0, 0, 0, 0, 0, 0, 9, 9, 0, 0],
            [0, 8, 8, 8, 0, 0, 9, 9, 9, 0],
            [0, 8, 8, 8, 0, 0, 9, 9, 9, 0],
        ],
        dtype=int,
    )
    return inst_map


@pytest.fixture(scope="package")
def type_map() -> np.ndarray:
    """Return dummy (cell) type segmentation mask."""
    type_map = np.array(
        [
            [2, 2, 2, 1, 1, 1, 1, 0, 3, 3],
            [2, 2, 2, 1, 1, 1, 1, 0, 3, 3],
            [2, 2, 0, 0, 1, 1, 1, 0, 3, 3],
            [2, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 3, 3, 3, 0, 1, 1, 1, 0],
            [0, 3, 3, 3, 3, 0, 1, 1, 1, 0],
            [0, 3, 3, 3, 0, 0, 1, 1, 1, 0],
            [0, 3, 3, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 2, 2, 2, 0, 0, 1, 1, 1, 0],
            [0, 2, 2, 2, 0, 0, 1, 1, 1, 0],
        ],
        dtype=int,
    )
    return type_map


@pytest.fixture(scope="package")
def sem_map() -> np.ndarray:
    """Return a dummy semantic segmentation mask."""
    sem_map = np.array(
        [
            [2, 2, 2, 1, 1, 1, 1, 3, 3, 3],
            [2, 2, 2, 1, 1, 1, 1, 3, 3, 3],
            [2, 1, 2, 2, 1, 1, 1, 3, 3, 3],
            [2, 1, 0, 2, 3, 3, 3, 1, 1, 1],
            [2, 0, 2, 2, 3, 3, 1, 1, 1, 0],
            [2, 2, 2, 2, 3, 3, 1, 2, 1, 1],
            [2, 2, 2, 2, 3, 3, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 3, 1, 1, 0, 0],
            [3, 3, 3, 0, 3, 3, 1, 1, 0, 0],
            [3, 2, 2, 2, 3, 3, 1, 1, 1, 1],
            [3, 2, 3, 2, 3, 3, 1, 1, 1, 1],
        ],
        dtype=int,
    )
    return sem_map
