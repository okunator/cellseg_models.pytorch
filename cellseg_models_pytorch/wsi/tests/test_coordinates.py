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
