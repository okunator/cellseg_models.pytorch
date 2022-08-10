from pathlib import Path

import numpy as np
import pytest
import torch

from cellseg_models_pytorch.utils import FileHandler


def pytest_addoption(parser):
    parser.addoption(
        "--cuda",
        action="store_true",
        default=False,
        help="run gpu tests",
    )
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="run slow tests",
    )
    parser.addoption(
        "--optional",
        action="store_true",
        default=False,
        help="run tests that require optional packages",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: mark test as a gpu test")
    config.addinivalue_line("markers", "slow: mark test as a slow test")
    config.addinivalue_line("markers", "optional: mark test as an optional test")


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

    only_slow = pytest.mark.skip(reason="--slow option runs only slow tests")
    if config.getoption("--slow"):
        for item in items:
            if "slow" not in item.keywords:
                item.add_marker(only_slow)
    else:
        skip_slow = pytest.mark.skip(reason="need --slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    only_opt = pytest.mark.skip(reason="--optional option runs only optional tests")
    if config.getoption("--optional"):
        for item in items:
            if "optional" not in item.keywords:
                item.add_marker(only_opt)
    else:
        skip_opt = pytest.mark.skip(reason="need --optional option to run")
        for item in items:
            if "optional" in item.keywords:
                item.add_marker(skip_opt)


@pytest.fixture(scope="package")
def img_dir() -> Path:
    """Return a path to directory with a few test images."""
    path = Path().resolve()
    return path / "cellseg_models_pytorch//inference/tests/data"


@pytest.fixture(scope="package")
def type_map_tensor() -> torch.Tensor:
    """Return a dummy type map target tensor. Shape (8, 320, 320)."""
    path = Path().resolve()
    path = path / "cellseg_models_pytorch/training/tests/data/type_target_batch8.pt"
    return torch.load(path.as_posix())


@pytest.fixture(scope="package")
def sem_map_tensor() -> torch.Tensor:
    """Return a dummy semantic map target tensor. Shape (8, 320, 320)."""
    path = Path().resolve()
    path = path / "cellseg_models_pytorch/training/tests/data/sem_target_batch8.pt"
    return torch.load(path.as_posix())


@pytest.fixture(scope="package")
def img_sample() -> np.ndarray:
    """Read in test RGB img."""
    path = Path().resolve()
    return FileHandler.read_img(path / "cellseg_models_pytorch/utils/tests/data/HE.png")


@pytest.fixture(scope="package")
def hdf5db() -> Path:
    """Read in test RGB img."""
    path = Path().resolve()
    return path / "cellseg_models_pytorch/datasets/tests/data/tiny_test.h5"


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


@pytest.fixture(scope="package")
def true_sem() -> np.ndarray:
    true = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0],
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 4, 4, 0],
            [0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 4, 4, 4, 4],
            [0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4],
            [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 4, 4, 0],
            [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0],
        ],
        dtype=int,
    )

    return true


@pytest.fixture(scope="package")
def pred_sem() -> np.ndarray:
    pred = np.array(
        [
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0],
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 3, 3, 4, 4, 0, 0, 4, 0, 0, 0],
            [0, 0, 0, 3, 3, 3, 4, 4, 4, 4, 4, 0, 0, 0],
            [0, 0, 3, 3, 3, 3, 4, 4, 4, 4, 0, 0, 0, 0],
            [0, 3, 3, 3, 3, 3, 4, 4, 4, 4, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    return pred


@pytest.fixture(scope="package")
def tensor_sem_map() -> torch.LongTensor:
    """Return a dummy tensor of shape (2, 6, 6)."""
    t = torch.LongTensor(
        [
            [
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
                [2, 2, 2, 3, 3, 3],
                [2, 2, 2, 3, 3, 3],
                [2, 2, 2, 3, 3, 3],
            ],
            [
                [2, 2, 2, 0, 0, 0],
                [2, 2, 2, 0, 0, 0],
                [2, 2, 2, 0, 0, 0],
                [3, 3, 3, 1, 1, 1],
                [3, 3, 3, 1, 1, 1],
                [3, 3, 3, 1, 1, 1],
            ],
        ]
    )

    return t
