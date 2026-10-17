from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from cellseg_models_pytorch.wsi import cucim_reader
from cellseg_models_pytorch.wsi.bioio_reader import BioIOReader
from cellseg_models_pytorch.wsi.openslide_reader import OpenSlideReader
from cellseg_models_pytorch.wsi.reader import SlideReader


@pytest.fixture
def reader(tmp_path, monkeypatch):
    path = tmp_path / "slide.svs"
    path.touch()
    native = SimpleNamespace(
        resolutions={"level_dimensions": [(36, 20), (12, 5)]},
        read_region=Mock(
            side_effect=lambda *, size, **kwargs: np.full(
                (size[1], size[0], 3), 42, dtype=np.uint8
            )
        ),
    )
    monkeypatch.setattr(cucim_reader, "HAS_CUCIM", True)
    monkeypatch.setattr(cucim_reader, "CuImage", lambda path: native, raising=False)
    return SlideReader(path, backend="CUCIM"), native


def test_cucim_downsamples_sizes_and_pads_slide_edges(reader):
    reader, native = reader
    tile = reader.read_region((30, 16, 12, 8), level=1)
    native.read_region.assert_called_once_with(
        location=(30, 16), level=1, size=(2, 1), device="cpu"
    )
    assert tile.shape == (2, 4, 3)
    assert tile.dtype == np.uint8
    np.testing.assert_array_equal(tile[0, :2], 42)
    np.testing.assert_array_equal(tile[1], 255)
    np.testing.assert_array_equal(tile[:, 2:], 255)


def test_cucim_outside_slide_is_white_without_calling_decoder(reader):
    reader, native = reader
    tile = reader.read_region((40, 24, 12, 8), level=1)
    assert tile.shape == (2, 4, 3)
    np.testing.assert_array_equal(tile, 255)
    native.read_region.assert_not_called()


@pytest.mark.parametrize("bounds", [(-1, 0, 2, 2), (0, -1, 2, 2), (0, 0, 0, 2)])
def test_cucim_rejects_invalid_bounds_before_decoding(reader, bounds):
    reader, native = reader
    with pytest.raises(ValueError, match="Region"):
        reader.read_region(bounds)
    native.read_region.assert_not_called()


def test_cucim_rejects_non_cuda_tensor_requests(reader):
    reader, native = reader
    with pytest.raises(ValueError, match="CUDA"):
        reader.read_region((0, 0, 4, 4), device="cpu")
    native.read_region.assert_not_called()


@pytest.mark.parametrize("backend", [OpenSlideReader, BioIOReader])
def test_cpu_backends_reject_device_reads_instead_of_returning_host_pixels(backend):
    reader = backend.__new__(backend)
    with pytest.raises(ValueError, match="CPU NumPy arrays"):
        reader.read_region((0, 0, 4, 4), 0, device="cuda")


def test_cpu_reader_import_does_not_load_torch():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from cellseg_models_pytorch.wsi import SlideReader; assert callable(SlideReader); assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.cuda
def test_cucim_cuda_storage_lifetime_and_padding(reader):
    cp = pytest.importorskip("cupy")
    reader, native = reader
    allocations = []

    def decode(*, size, device, **kwargs):
        assert device == str(torch.device("cuda", torch.cuda.current_device()))
        pixels = cp.full((size[1], size[0], 3), 42, dtype=cp.uint8)
        allocations.append(pixels.data.ptr)
        return pixels

    native.read_region.side_effect = decode
    tile = reader.read_region((0, 0, 12, 8), device="cuda")
    assert tile.is_cuda and tile.dtype == torch.uint8
    assert tile.data_ptr() == allocations[0]
    # A later read must not invalidate the first decoder-owned allocation.
    padded = reader.read_region((30, 16, 12, 8), level=1, device="cuda")
    assert tile.shape == (8, 12, 3) and torch.all(tile == 42)
    assert padded.shape == (2, 4, 3) and padded.is_cuda
    assert torch.all(padded[0, :2] == 42)
    assert torch.all(padded[1] == 255) and torch.all(padded[:, 2:] == 255)
    outside = reader.read_region((40, 24, 12, 8), level=1, device="cuda")
    assert outside.is_cuda and torch.all(outside == 255)
