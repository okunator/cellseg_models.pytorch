from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import torch
from geopandas.testing import assert_geodataframe_equal
from scipy.ndimage import label

from cellseg_models_pytorch.decoders.multitask_decoder import SoftInstanceOutput
from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.inference.predictor import Predictor
from cellseg_models_pytorch.inference.wsi_segmenter import WsiSegmenter
from cellseg_models_pytorch.models.base._base_model_inst import BaseModelInst
from cellseg_models_pytorch.postproc import POSTPROC_LOOKUP
from cellseg_models_pytorch.utils.vectorize import gaussian_smooth, inst2gdf


class DenseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(self, image):
        assert image.dtype == torch.float32
        assert image.shape[1:] == (3, 16, 16)
        # Binary fixtures must stay binary, including all-background tiles.
        assert ((image == 0) | (image == 1)).all()
        foreground = image[:, :1]
        return {
            "nuc": SoftInstanceOutput(
                type_map=torch.cat((1 - foreground, foreground), dim=1),
                aux_map=torch.zeros_like(foreground),
                binary_map=foreground,
            ),
            "cyto": None,
            "tissue": None,
        }


def baseline_instances(prob, aux):
    return label(prob > 0.5)[0]


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setitem(POSTPROC_LOOKUP, "baseline", baseline_instances)
    model = BaseModelInst()
    model.model = DenseModel().eval()
    model.device = torch.device("cpu")
    model.predictor = Predictor(model.model, mixed_precision=False)
    model.post_processor = PostProcessor("baseline")
    model.inference_mode = True
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[4:12, 4:12] = 255
    calls = []

    def read_region(coordinates, level):
        calls.append((coordinates, level))
        return image.copy()

    reader = SimpleNamespace(name="slide.v1", read_region=read_region)
    return model, reader, image, calls


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("normalize", [False, True])
def test_wsi_pipeline_matches_serial_reference(
    tmp_path, pipeline, batch_size, normalize
):
    model, reader, image, calls = pipeline
    coordinates = [(0, 0, 16, 16), (16, 0, 16, 16), (0, 16, 16, 16)]
    transform = (
        (lambda *, image: {"image": image.astype(np.float32) / 255})
        if normalize
        else None
    )
    segmenter = WsiSegmenter(reader, model, 2, coordinates, batch_size, transform)
    destination = tmp_path / "nested" / "tiles"
    segmenter.segment(destination)
    assert segmenter._has_processed
    assert calls == [(coordinate, 2) for coordinate in coordinates]
    assert len(list(destination.glob("*.parquet"))) == 3
    soft = model.predict(image)
    mask, classes = model.post_process(soft)["nuc"][0]
    np.testing.assert_array_equal(mask > 0, image[..., 0] > 0)
    for x, y, width, height in coordinates:
        path = destination / f"slide.v1_x{x}-y{y}_w{width}-h{height}.parquet"
        actual = gpd.read_parquet(path)
        expected = inst2gdf(mask, classes, xoff=x, yoff=y, smooth_func=gaussian_smooth)
        assert_geodataframe_equal(actual, expected, normalize=True)


def test_wsi_refuses_existing_output(tmp_path, pipeline):
    model, reader, _, _ = pipeline
    path = tmp_path / "slide.v1_x0-y0_w16-h16.parquet"
    path.write_bytes(b"existing output")
    segmenter = WsiSegmenter(reader, model, 0, [(0, 0, 16, 16)])
    with pytest.raises(FileExistsError):
        segmenter.segment(tmp_path)
    assert path.read_bytes() == b"existing output"
    assert not segmenter._has_processed


def test_wsi_rejects_unsupported_maptype_before_reading(tmp_path, pipeline):
    model, reader, _, calls = pipeline
    segmenter = WsiSegmenter(reader, model, 0, [(0, 0, 16, 16)])
    with pytest.raises(ValueError, match="amap"):
        segmenter.segment(tmp_path, maptype="imap")
    assert not calls and not segmenter._has_processed


@pytest.mark.parametrize("cytoplasm", [False, True])
@pytest.mark.parametrize("tissue", [False, True])
def test_async_postprocessing_matches_serial_with_optional_heads(
    pipeline, cytoplasm, tissue
):
    from cellseg_models_pytorch.decoders.multitask_decoder import SoftSemanticOutput

    model, _, image, _ = pipeline
    batch = (
        torch.from_numpy(image.transpose(2, 0, 1)).float().div(255).repeat(2, 1, 1, 1)
    )
    outputs = model.predict(batch)
    outputs["cyto"] = outputs["nuc"] if cytoplasm else None
    outputs["tissue"] = (
        SoftSemanticOutput(torch.ones(2, 16, 16, dtype=torch.long)) if tissue else None
    )
    expected = model.post_processor.postproc_serial(outputs)
    actual = model.post_processor.postproc_parallel_async(outputs, n_jobs=2)
    for head in ("nuc", "cyto", "tissue"):
        if expected[head] is None:
            assert actual[head] is None
        else:
            np.testing.assert_array_equal(actual[head], expected[head])


def test_failed_wsi_run_is_not_marked_complete(tmp_path, pipeline, monkeypatch):
    model, reader, _, _ = pipeline
    segmenter = WsiSegmenter(reader, model, 0, [(0, 0, 16, 16)])

    def fail(*args, **kwargs):
        raise RuntimeError("postprocessing failed")

    monkeypatch.setattr(model, "post_process", fail)
    with pytest.raises(RuntimeError, match="postprocessing failed"):
        segmenter.segment(tmp_path)
    assert not segmenter._has_processed


def test_duplicate_tile_outputs_are_rejected(tmp_path, pipeline):
    model, reader, _, _ = pipeline
    segmenter = WsiSegmenter(reader, model, 0, [(0, 0, 16, 16)] * 2, batch_size=2)
    with pytest.raises(ValueError, match="duplicate output paths"):
        segmenter.segment(tmp_path)
    assert not segmenter._has_processed
    assert not list(tmp_path.glob("*.parquet"))


def test_overlapping_wsi_uses_source_tiles_and_preserves_complete_seam_nucleus(
    tmp_path, pipeline
):
    model, reader, _, _ = pipeline
    canvas = np.zeros((24, 24, 3), dtype=np.uint8)
    canvas[10:18, 10:18] = 255

    def read_region(coordinates, level):
        x, y, width, height = coordinates
        return canvas[y : y + height, x : x + width].copy()

    reader.read_region = read_region
    coordinates = [(0, 0, 16, 16), (8, 0, 16, 16), (0, 8, 16, 16), (8, 8, 16, 16)]
    segmenter = WsiSegmenter(reader, model, 0, coordinates, batch_size=2)
    segmenter.segment(tmp_path / "tiles")
    segmenter.merge_instances(
        tmp_path / "tiles", tmp_path / "nuclei.parquet", simplify_level=0
    )
    actual = gpd.read_parquet(tmp_path / "nuclei.parquet")
    assert len(actual) == 1
    geometry = actual.geometry.iloc[0]
    assert geometry.geom_type == "Polygon" and geometry.is_valid
    assert geometry.bounds[0] < 16 < geometry.bounds[2]
    assert geometry.bounds[1] < 16 < geometry.bounds[3]
    full_prediction = gpd.read_parquet(
        tmp_path / "tiles/slide.v1_x8-y8_w16-h16.parquet"
    )
    assert geometry.equals(full_prediction.geometry.iloc[0])


def test_overlap_merge_accepts_all_empty_tiles(tmp_path, pipeline):
    model, reader, _, _ = pipeline
    reader.read_region = lambda coordinates, level: np.zeros(
        (16, 16, 3), dtype=np.uint8
    )
    segmenter = WsiSegmenter(reader, model, 0, [(0, 0, 16, 16), (8, 0, 16, 16)])
    segmenter.segment(tmp_path / "tiles")
    assert not list((tmp_path / "tiles").glob("*.parquet"))
    segmenter.merge_instances(tmp_path / "tiles", tmp_path / "nuclei.parquet")
    assert gpd.read_parquet(tmp_path / "nuclei.parquet").empty


def test_merge_rejects_files_without_a_known_source_tile(tmp_path, pipeline):
    model, reader, _, _ = pipeline
    segmenter = WsiSegmenter(reader, model, 0, [(0, 0, 16, 16)])
    segmenter.segment(tmp_path / "tiles")
    unrelated = tmp_path / "tiles/other-slide.parquet"
    unrelated.write_bytes(b"unrelated data")
    with pytest.raises(ValueError, match="no source tile"):
        segmenter.merge_instances(tmp_path / "tiles", tmp_path / "nuclei.parquet")
    assert unrelated.read_bytes() == b"unrelated data"
    assert not (tmp_path / "nuclei.parquet").exists()


@pytest.mark.parametrize(
    "kwargs,expected_count",
    [
        ({}, 2),
        ({"filter_level": 0}, 2),
        ({"filter_level": None}, 2),
        ({"filter_level": 1, "pixel_size": 0.35}, 2),
        ({"filter_level": 2, "pixel_size": 0.35}, 0),
        ({"filter_level": 3, "pixel_size": 0.35}, 0),
        ({"filter_level": 2, "pixel_size": 0.5}, 2),
    ],
)
def test_wsi_optional_integer_filter_levels_and_pixel_size(
    tmp_path, pipeline, kwargs, expected_count
):
    model, reader, _, _ = pipeline
    segmenter = WsiSegmenter(reader, model, 0, [(0, 0, 16, 16), (16, 0, 16, 16)])
    segmenter.segment(tmp_path / "tiles")
    segmenter.merge_instances(tmp_path / "tiles", tmp_path / "nuclei.parquet", **kwargs)
    actual = gpd.read_parquet(tmp_path / "nuclei.parquet")
    assert len(actual) == expected_count
    assert len(list((tmp_path / "tiles").glob("*.parquet"))) == 2
