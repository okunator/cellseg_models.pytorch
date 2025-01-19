from typing import List, Tuple

import geopandas as gpd
import networkx as nx
from libpysal.weights import W, fuzzy_contiguity
from shapely.geometry import box

__all__ = ["get_sub_grids"]


def get_sub_grids(
    coordinates: List[Tuple[int, int, int, int]],
    inds: Tuple[int, ...] = None,
    min_size: int = 1,
) -> List[List[Tuple[int, int, int, int]]]:
    """Get sub-grids based on connected components of the grid.

    Note:
        The order of the sub-grids is from ymin to ymax.

    Parameters:
        coordinates (List[Tuple[int, int, int, int]]):
            List of grid coordinates in (x, y, w, h) format.
        inds (Tuple[int, ...], default=None):
            Indices of the connected components to extract.
        min_size (int, default=1):
            Minimum size of the sub grid.

    Returns:
        List[List[Tuple[int, int, int, int]]]:
            Nested list of sub-grids in (x, y, w, h) format.
    """
    # convert to (xmin, ymin, xmax, ymax) format
    bbox_coords = [(x, y, x + w, y + h) for x, y, w, h in coordinates]

    # convert to shapely boxes
    box_polys = [box(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in bbox_coords]

    # Create GeoDataFrame from the grid
    grid = gpd.GeoDataFrame({"geometry": box_polys}, crs="+proj=cea")

    # get queen contiguity of the grid
    w = fuzzy_contiguity(
        grid,
        buffering=False,
        predicate="intersects",
        silence_warnings=True,
    )

    # get connected components of the grid
    G = w.to_networkx()
    sub_graphs = [
        W(nx.to_dict_of_lists(G.subgraph(c).copy())) for c in nx.connected_components(G)
    ]

    sub_graphs = [g for g in sub_graphs if len(g.neighbors) > min_size]

    if inds is not None:
        sub_graphs = [sub_graphs[i] for i in inds]

    sub_grids = []
    for g in sub_graphs:
        indices = list(g.neighbors.keys())
        sub_grids.append([coordinates.coordinates[i] for i in indices])

    return sub_grids
