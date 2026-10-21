from typing import get_type_hints

import numpy as np
import pytest

from cellseg_models_pytorch.wsi import get_sub_grids
from cellseg_models_pytorch.wsi.data import SpotCoordinates, TileCoordinates
from cellseg_models_pytorch.wsi.tiles import _get_allowed_dimensions


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ((1, 2, 3, 4), (4, 3)),
        ((8, 6, 6, 5), (4, 4)),
        ((12, 10, 2, 3), (0, 0)),
        ((20, 15, 3, 4), (0, 0)),
    ],
)
def test_allowed_dimensions_returns_clipped_height_and_width(
    bounds: tuple[int, int, int, int], expected: tuple[int, int]
) -> None:
    assert _get_allowed_dimensions(bounds, dimensions=(10, 12)) == expected


def test_allowed_dimensions_annotation_matches_two_axis_result() -> None:
    assert get_type_hints(_get_allowed_dimensions)["return"] == tuple[int, int]


@pytest.mark.parametrize("kind", ["list", "tiles", "spots"])
def test_sub_grids_preserves_global_indices_and_coordinate_collections(
    kind: str,
) -> None:
    bounds = [
        (0, 0, 8, 8),
        (8, 0, 8, 8),
        (16, 0, 8, 8),
        (0, 32, 8, 8),
        (8, 32, 8, 8),
        (0, 64, 8, 8),
    ]
    coordinates = bounds
    if kind == "tiles":
        coordinates = TileCoordinates(bounds, width=8, height=8, overlap=0)
    elif kind == "spots":
        coordinates = SpotCoordinates(
            bounds, [f"spot_{i}" for i in range(len(bounds))], np.ones((1, 1), np.uint8)
        )

    assert get_sub_grids(coordinates) == [bounds[:3], bounds[3:5]]
    assert get_sub_grids(coordinates, inds=(1,)) == [bounds[3:5]]
    assert get_sub_grids(coordinates, min_size=2) == [bounds[:3]]
    assert list(coordinates) == bounds
