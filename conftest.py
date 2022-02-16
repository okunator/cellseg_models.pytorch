import pytest


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
