from collections.abc import Iterator
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import pytest

from cellseg_models_pytorch.wsi.data import SpotCoordinates, TileCoordinates
from cellseg_models_pytorch.wsi.reader import SlideReader


@pytest.mark.parametrize("kind", [TileCoordinates, SpotCoordinates])
def test_coordinate_collections_iterate_bounds_without_names(kind):
    bounds = [(0, 0, 8, 8), (8, 0, 8, 8)]
    if kind is TileCoordinates:
        collection = kind(bounds, width=8, height=8, overlap=0)
    else:
        collection = kind(bounds, ["spot_1", "spot_2"], np.ones((8, 16), np.uint8))
    assert len(collection) == 2
    assert list(collection) == bounds
    assert collection[1] == bounds[1]
    assert get_type_hints(kind)["coordinates"] == list[tuple[int, int, int, int]]
    assert (
        get_type_hints(kind.__iter__)["return"] == Iterator[tuple[int, int, int, int]]
    )


def test_detected_spots_keep_names_separate_and_scale_bounds_to_level_zero():
    mask = np.zeros((32, 32), dtype=np.uint8)
    for y in (4, 20):
        for x in (4, 20):
            mask[y : y + 8, x : x + 8] = 1
    reader = SlideReader.__new__(SlideReader)
    reader._reader = SimpleNamespace(dimensions=(64, 64))
    spots = reader.get_spot_coordinates(mask)
    assert isinstance(spots.coordinates, list)
    assert len(spots) == len(spots.spot_names) == 4
    assert set(spots.spot_names) == {"spot_1", "spot_2", "spot_3", "spot_4"}
    assert set(spots) == {(x * 2, y * 2, 16, 16) for y in (4, 20) for x in (4, 20)}
    assert spots.tissue_mask.shape == mask.shape


@pytest.mark.parametrize(
    "scale,divided,multiplied",
    [
        (2, (6, 9, 12, 15), (24, 36, 48, 60)),
        ((2, 3), (4, 9, 8, 15), (36, 36, 72, 60)),
        ((2.5, 1.5), (8, 7, 16, 12), (18, 45, 36, 75)),
    ],
)
def test_coordinate_helpers_use_height_width_scales(scale, divided, multiplied):
    from cellseg_models_pytorch.wsi.tiles import _divide_xywh, _multiply_xywh

    bounds = (12, 18, 24, 30)
    assert _divide_xywh(bounds, scale) == divided
    assert _multiply_xywh(bounds, scale) == multiplied


def test_array_region_scales_origin_size_and_padding_on_each_axis():
    from cellseg_models_pytorch.wsi.tiles import get_region_from_array

    image = np.arange(8 * 12, dtype=np.uint8).reshape(8, 12)
    actual = get_region_from_array(image, (9, 4, 12, 6), downsample=(2, 3))
    np.testing.assert_array_equal(actual, image[2:5, 3:7])
    padded = get_region_from_array(
        image, (30, 12, 12, 8), downsample=(2, 3), fill=255
    )
    expected = np.full((4, 4), 255, dtype=np.uint8)
    expected[:2, :2] = image[6:8, 10:12]
    np.testing.assert_array_equal(padded, expected)


def test_background_filter_uses_full_slide_mask_with_different_axis_scales():
    reader = SlideReader.__new__(SlideReader)
    reader._reader = SimpleNamespace(dimensions=(24, 48))
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[:6, 6:] = 1
    tiles = reader.get_tile_coordinates(
        width=24, height=12, tissue_mask=mask, max_background=0
    )
    assert list(tiles) == [(24, 0, 24, 12)]


def test_spots_scale_height_and_width_independently():
    reader = SlideReader.__new__(SlideReader)
    reader._reader = SimpleNamespace(dimensions=(64, 96))
    mask = np.zeros((32, 32), dtype=np.uint8)
    for y in (4, 20):
        for x in (4, 20):
            mask[y : y + 8, x : x + 8] = 1
    spots = reader.get_spot_coordinates(mask)
    assert set(spots) == {(x, y, 24, 16) for x in (12, 60) for y in (8, 40)}
    assert len(spots.spot_names) == 4
    np.testing.assert_array_equal(spots.tissue_mask, mask)


@pytest.mark.parametrize("highlight", [False, True])
def test_thumbnail_rectangle_scales_both_axes_including_highlight(highlight):
    pytest.importorskip("matplotlib")
    reader = SlideReader.__new__(SlideReader)
    reader._reader = SimpleNamespace(dimensions=(24, 48))
    bounds = [(12, 4, 16, 8)]
    coordinates = (
        TileCoordinates(bounds, width=16, height=8, overlap=0) if highlight else bounds
    )
    actual = np.asarray(
        reader.get_annotated_thumbnail(np.zeros((12, 12, 3), np.uint8), coordinates)
    )
    expected = np.zeros((12, 12, 3), np.uint8)
    color = (0, 0, 255) if highlight else (255, 0, 0)
    expected[2, 3:8] = color
    expected[6, 3:8] = color
    expected[2:7, 3] = color
    expected[2:7, 7] = color
    np.testing.assert_array_equal(actual, expected)


def test_openslide_region_keeps_level_zero_origin_and_scales_output_size():
    from unittest.mock import Mock

    from cellseg_models_pytorch.wsi.openslide_reader import OpenSlideReader

    reader = OpenSlideReader.__new__(OpenSlideReader)
    reader._OpenSlideReader__level_dimensions = {0: (40, 90), 1: (20, 30)}
    reader._OpenSlideReader__level_downsamples = {0: (1, 1), 1: (2, 3)}
    pixels = np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4)
    native = Mock()
    native.read_region.return_value = pixels
    reader._OpenSlideReader__reader = native
    actual = reader.read_region((9, 4, 12, 6), level=1)
    native.read_region.assert_called_once_with(location=(9, 4), level=1, size=(4, 3))
    np.testing.assert_array_equal(actual, pixels[..., :3])


def test_bioio_region_scales_output_size_and_preserves_requested_pixel_slice():
    from unittest.mock import MagicMock

    from cellseg_models_pytorch.wsi.bioio_reader import BioIOReader

    reader = BioIOReader.__new__(BioIOReader)
    reader._BioIOReader__level_dimensions = {0: (40, 90), 1: (20, 30)}
    reader._BioIOReader__level_downsamples = {0: (1, 1), 1: (2, 3)}
    pixels = np.arange(20 * 30 * 3, dtype=np.uint16).reshape(20, 30, 3)
    array = MagicMock()
    array.__getitem__.return_value = pixels[4:7, 9:13]
    reader._BioIOReader__reader = SimpleNamespace(xarray_data=array)
    actual = reader.read_region((9, 4, 12, 6), level=1)
    array.__getitem__.assert_called_once_with((0, 0, 0, range(4, 7), range(9, 13)))
    np.testing.assert_array_equal(actual, pixels[4:7, 9:13])
