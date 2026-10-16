"""Exercise the WSI notebook with small offline tiles and the real stitcher."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from cellseg_models_pytorch.inference.tests.test_wsi_segmenter import (
    pipeline as pipeline,
)
from cellseg_models_pytorch.wsi import SlideReader
from cellseg_models_pytorch.wsi.tiles import get_region_from_array


@pytest.fixture
def tutorial(tmp_path, monkeypatch, pipeline):
    pytest.importorskip("openslide")
    pytest.importorskip("albumentations")
    model, reader, image, _ = pipeline
    previous_threads = torch.get_num_threads()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(plt, "show", Mock())
    path = Path(__file__).resolve().parents[1] / "docs/tutorials/wsi.ipynb"
    notebook = json.loads(path.read_text())
    cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    ns = {}
    exec(cells[0], ns)
    ns["slide_path"] = tmp_path / "data/slide.svs"
    ns["checkpoint_path"] = tmp_path / "model.safetensors"
    ns["output_dir"] = tmp_path / "results"
    ns["tile_size"] = 16
    classes = [
        "background",
        "neoplastic",
        "stromal",
        "inflammatory",
        "epithelial",
        "other",
        "unknown",
    ]
    save_file(
        {"fixture": torch.zeros(1)},
        ns["checkpoint_path"],
        metadata={"class_names": json.dumps(classes)},
    )
    reader.dimensions = (32, 32)
    reader.level_from_max_dimension = lambda size: 0
    canvas = np.zeros((32, 32, 3), dtype=np.uint8)
    for y in (4, 16):
        for x in (4, 16):
            canvas[y : y + 8, x : x + 8] = 255
    reader.read_level = lambda level: canvas
    reader.read_region = lambda bounds, level: get_region_from_array(canvas, bounds)
    reader.get_tissue_mask = lambda **kwargs: (128, np.ones((32, 32), dtype=np.uint8))
    reader.get_tile_coordinates = lambda **kwargs: SlideReader.get_tile_coordinates(
        reader, **kwargs
    )
    reader.get_annotated_thumbnail = lambda *args, **kwargs: (
        SlideReader.get_annotated_thumbnail(reader, *args, **kwargs)
    )
    ns["SlideReader"] = lambda *args, **kwargs: reader
    ns["CellPose"] = SimpleNamespace(from_pretrained=lambda *args, **kwargs: model)
    model.set_inference_mode = lambda **kwargs: None
    ns["PostProcessor"] = lambda *args, **kwargs: model.post_processor
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_content.return_value = [b"slide fixture"]
    download = Mock(return_value=response)
    monkeypatch.setattr(ns["requests"], "get", download)
    metadata = MagicMock()
    metadata.__enter__.return_value.properties = {"openslide.mpp-x": "0.252"}
    monkeypatch.setattr(ns["openslide"], "OpenSlide", lambda *args: metadata)
    yield ns, cells, download
    torch.set_num_threads(previous_threads)
    plt.close("all")


def test_wsi_notebook_writes_all_tiles_and_preserves_their_nuclei(tutorial):
    ns, cells, download = tutorial
    for source in cells[1:]:
        exec(source, ns)
    assert ns["slide_path"].read_bytes() == b"slide fixture"
    download.assert_called_once_with(ns["slide_url"], stream=True, timeout=120)
    assert len(ns["coordinates"]) == 9
    assert len(list((ns["output_dir"] / "tiles").glob("*.parquet"))) == 4
    assert len(ns["nuclei"]) == 4
    assert len(ns["seam_centres"]) == 12
    assert ns["nuclei"].uid.is_unique
    assert ns["nuclei"].geometry.is_valid.all()
    assert ns["nuclei"].crs is None
    assert ns["nuclei"].is_partial.tolist() == [False] * 4
    assert ns["counts"]["Nuclei"].to_list() == [0, 4, 0, 0, 0, 0, 0]
    # Smoothing changes centroids, but the four complete nuclei must remain 12 pixels apart.
    for values in (ns["nuclei"].geometry.centroid.x, ns["nuclei"].geometry.centroid.y):
        centers = np.sort(values.to_numpy())
        np.testing.assert_allclose(centers[2:] - centers[:2], 12, rtol=0, atol=1e-12)
    _, axis = plt.subplots()
    ns["plot_region"](axis, (0, 0, 16, 16), "Fixture crop")
    assert len(axis.collections[0].get_segments()) == 1
    exec(cells[1], ns)
    assert download.call_count == 1
    with pytest.raises(FileExistsError):
        exec(
            next(source for source in cells if source.startswith("output_dir.mkdir")),
            ns,
        )


def test_failed_download_is_not_published_as_a_slide(tutorial):
    ns, cells, download = tutorial
    download.return_value.iter_content.side_effect = RuntimeError("transfer failed")
    with pytest.raises(RuntimeError, match="transfer failed"):
        exec(cells[1], ns)
    assert not ns["slide_path"].exists()
