from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import geopandas as gpd
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from shapely import (
    STRtree,
    area,
    boundary,
    bounds,
    distance,
    disjoint_subset_union_all,
    from_wkt,
    geos_version,
    intersection,
    length,
    to_wkt,
    union_all,
)
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

__all__ = ["InstMerger", "filter_nuclei"]

# Shapely stores nucleus polygons in NumPy object arrays.
_PolygonArray: TypeAlias = NDArray[np.object_]
_IndexArray: TypeAlias = NDArray[np.intp]
_FloatArray: TypeAlias = NDArray[np.float64]
_BoolArray: TypeAlias = NDArray[np.bool_]


def filter_nuclei(
    gdf: gpd.GeoDataFrame,
    pixel_size: float | None = None,
    filter_level: int | None = 0,
) -> gpd.GeoDataFrame:
    """Remove polygons that are both small and thin.

    An object is removed only when its area and thickness are both below the
    selected limits. Thickness is 2 * area / perimeter, not the nucleus diameter.
    Levels 1, 2 and 3 use (6 µm², 0.8 µm), (12 µm², 1.2 µm) and
    (20 µm², 1.6 µm), respectively. Small valid nuclei can also be removed.

    Args:
        gdf: Valid nucleus polygons and any class or metadata columns.
        pixel_size: Positive finite micrometers per coordinate pixel. Required
            when enabled; pixels must have equal physical height and width.
        filter_level: 0 or None disables filtering. Use 1 for the mildest filter,
            2 for the middle setting or 3 for the strongest.

    Returns:
        Retained rows in their original order, with outlines, IDs, index, attributes
        and CRS unchanged. Returns the input itself when disabled; does not modify it.

    Raises:
        ValueError: An enabled filter lacks valid pixel size, or the level is unknown.

    Example:
        >>> nuclei = gpd.GeoDataFrame(geometry=[box(0, 0, 4, 80), box(10, 0, 14, 40)])
        >>> len(filter_nuclei(nuclei, pixel_size=0.25, filter_level=2))
        1
    """
    if filter_level is None:
        return gdf
    if (
        not isinstance(filter_level, (int, np.integer))
        or isinstance(filter_level, (bool, np.bool_))
        or filter_level not in (0, 1, 2, 3)
    ):
        raise ValueError("filter level must be None or an integer from 0 to 3")
    if filter_level == 0:
        return gdf
    if pixel_size is None or not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("pixel_size must be positive finite micrometers per pixel")
    limits = {1: (6.0, 0.8), 2: (12.0, 1.2), 3: (20.0, 1.6)}
    maximum_area, maximum_thickness = limits[filter_level]
    geometries = gdf.geometry.to_numpy()
    physical_area = area(geometries) * pixel_size**2
    perimeter = length(geometries) * pixel_size
    thickness = np.divide(
        2 * physical_area,
        perimeter,
        out=np.zeros_like(physical_area),
        where=perimeter > 0,
    )
    remove = (physical_area < maximum_area) & (thickness < maximum_thickness)
    retained: gpd.GeoDataFrame = gdf.loc[~remove]
    return retained


def _union_footprint(geometries: _PolygonArray) -> BaseGeometry:
    """Combine polygon areas without counting overlaps twice."""
    # Separate groups can be combined faster on newer GEOS versions.
    union = disjoint_subset_union_all if geos_version >= (3, 12, 0) else union_all
    return union(geometries)


def _match_overlaps(
    geometries: _PolygonArray,
    areas: _FloatArray,
    partial: _BoolArray,
    left_ids: _IndexArray,
    right_ids: _IndexArray,
    common: _PolygonArray,
) -> tuple[_BoolArray, _FloatArray, _FloatArray]:
    """Identify strong matches between two overlapping tiles."""
    shared_areas = area(common)
    union_areas = areas[left_ids] + areas[right_ids] - shared_areas
    # IoU compares shared area with the combined area of both predictions.
    ious = np.divide(
        shared_areas,
        union_areas,
        out=np.zeros(len(left_ids)),
        where=union_areas > 0,
    )
    smaller = np.minimum(areas[left_ids], areas[right_ids])
    coverage = np.divide(
        shared_areas, smaller, out=np.zeros(len(left_ids)), where=smaller > 0
    )
    # A cut-off contour needs less overlap to match a whole nucleus.
    partial_smaller = (partial[left_ids] & (areas[left_ids] <= areas[right_ids])) | (
        partial[right_ids] & (areas[right_ids] <= areas[left_ids])
    )
    matches = (
        (ious >= 0.5)
        | (~partial[left_ids] & ~partial[right_ids] & (coverage >= 0.8))
        | (partial_smaller & (coverage >= 0.5))
    )
    alternatives = np.flatnonzero(
        (coverage >= 0.8) & (partial[left_ids] != partial[right_ids]) & ~partial_smaller
    )
    partial_ids = np.where(
        partial[left_ids[alternatives]], left_ids[alternatives], right_ids[alternatives]
    )
    complete_ids = np.where(
        partial[left_ids[alternatives]], right_ids[alternatives], left_ids[alternatives]
    )
    for index in np.unique(partial_ids):
        positions = partial_ids == index
        if matches[alternatives[positions]].all():
            continue
        # Several whole nuclei can replace one larger, cut-off prediction.
        observed = _union_footprint(geometries[complete_ids[positions]])
        if observed.area >= 0.5 * areas[index]:
            matches[alternatives[positions]] = True
    return matches, ious, shared_areas


def _choose_tile(
    views: dict[np.intp, list[int]],
    geometries: _PolygonArray,
    clearance: _FloatArray,
    partial: _BoolArray,
    fragments: _BoolArray,
    coordinate_tolerance: float | np.float64,
) -> tuple[np.intp, set[int], dict[np.intp, BaseGeometry]]:
    """Prefer a complete, well-supported view of the disputed nuclei."""
    # A tiny nested mask cannot stand in for a whole nucleus elsewhere.
    whole_views = {
        tile: rows for tile, rows in views.items() if not fragments[rows].any()
    }
    if not whole_views:
        whole_views = {
            tile: [index for index in rows if not fragments[index]]
            for tile, rows in views.items()
            if not fragments[rows].all()
        } or views
    complete_views = {
        tile: rows for tile, rows in whole_views.items() if not partial[rows].any()
    }
    eligible = complete_views or whole_views
    # Count covered area once, even when a tile's own predictions overlap.
    footprints: dict[np.intp, BaseGeometry] = {
        tile: geometries[rows[0]]
        if len(rows) == 1
        else _union_footprint(geometries[rows])
        for tile, rows in eligible.items()
    }
    footprint_areas = {tile: geom.area for tile, geom in footprints.items()}
    largest = max(footprint_areas.values())
    # Consider only tiles covering at least 80% of the largest prediction set.
    supported = [tile for tile in eligible if footprint_areas[tile] >= 0.8 * largest]

    priorities = ("clearance", "area") if complete_views else ("area", "clearance")
    values: Mapping[np.intp, float | np.float64]
    for priority in priorities:
        if len(supported) == 1:
            break
        if priority == "clearance":
            values = {tile: np.min(clearance[whole_views[tile]]) for tile in supported}
            tolerance = coordinate_tolerance
        else:
            values = footprint_areas
            # Longer outlines accumulate more area rounding error.
            tolerance = coordinate_tolerance * max(
                footprints[tile].length for tile in supported
            )
        best = max(values[tile] for tile in supported)
        supported = [tile for tile in supported if best - values[tile] <= tolerance]
    # Discarded fragments must not decide which retained source row wins.
    selected_tile = min(supported, key=lambda tile: whole_views[tile][0])
    selected = set(whole_views[selected_tile])
    return selected_tile, selected, footprints


class InstMerger:
    """Combine tiled nucleus polygons into one slide-wide result.

    Duplicate predictions keep a whole contour from one source tile. Unmatched
    nuclei are retained. Touching fragments can join across a shared tile edge.
    """

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        coordinates: list[tuple[int, int, int, int]],
        tile_indices: Sequence[int] | None = None,
    ) -> None:
        """Set up merging for polygons from known source tiles.

        Args:
            gdf: Nucleus polygons in level-zero slide pixels. Keep class_name and any
                other source attributes. Optional is_partial must be boolean, with
                no missing values, and marks objects cut off by tile edges.
            coordinates: Source-tile bounds as (x, y, width, height) in the same
                level-zero pixel coordinates as the polygons.
            tile_indices: One index into coordinates per input row. Required for
                overlapping tiles. Without overlap, the source can be inferred
                from each polygon's position.

        Raises:
            ValueError: Source indices are missing or invalid, or is_partial is not boolean.
        """
        self.grid = gpd.GeoDataFrame(
            geometry=[box(x, y, x + w, y + h) for x, y, w, h in coordinates]
        )
        self.gdf = gdf.reset_index(drop=True)
        if "is_partial" in self.gdf and self.gdf["is_partial"].dtype != bool:
            raise ValueError(
                "is_partial must contain boolean border flags without nulls"
            )
        neighbors = self.grid.sindex.query(self.grid.geometry, predicate="intersects")
        tiles = self.grid.geometry.to_numpy()
        self.seams = [
            (int(a), int(b), tiles[a].intersection(tiles[b]))
            for a, b in zip(*neighbors)
            if a < b
        ]
        overlapping = any(shared.area > 0 for _, _, shared in self.seams)
        # A slide location alone cannot identify which overlapping tile predicted it.
        if tile_indices is None:
            if overlapping:
                raise ValueError("Overlapping tiles require source tile_indices")
            self.owners: _IndexArray = np.full(len(self.gdf), -1, dtype=int)
            matches = self.grid.sindex.query(
                self.gdf.geometry.representative_point(), predicate="within"
            )
            self.owners[matches[0]] = matches[1]
        else:
            indices = np.asarray(tile_indices)
            if (
                indices.shape != (len(self.gdf),)
                or (indices.size and indices.dtype.kind not in "iu")
                or np.any(indices < 0)
                or np.any(indices >= len(self.grid))
            ):
                raise ValueError(
                    "tile_indices must contain one valid tile index per row"
                )
            self.owners = indices.astype(int)

    def merge(
        self,
        dst: str | Path | None = None,
        simplify_level: float = 1,
        precision: int | None = None,
        *,
        filter_level: int | None = 0,
        pixel_size: float | None = None,
    ) -> gpd.GeoDataFrame | None:
        """Merge polygons, then optionally simplify, filter and save them.

        Duplicates retain one source contour and its attributes. Joined fragments
        keep the largest fragment's attributes. Unmatched nuclei remain in the result.
        Fragments must share a boundary to join; a corner touch is not enough.

        Args:
            dst: New .parquet, .geojson or .feather file. None returns the table.
                Parent folders are created; an existing file is never overwritten.
            simplify_level: Outline simplification tolerance in pixels; 0 keeps
                the original outlines before any joining or rounding.
            precision: Decimal places for coordinate rounding, or None to keep them.
            filter_level: Optional small/thin-object filter, after simplification
                and rounding. 0 or None disables it; see filter_nuclei for limits.
            pixel_size: Micrometers per coordinate pixel when filtering. Pixels
                must have equal physical height and width.

        Returns:
            A GeoDataFrame, or None when saved. Input attributes and CRS are kept;
            uid is replaced with unique slide-wide IDs and also used as the index.
            Filtering does not renumber surviving IDs. Coordinates stay in pixels.

        Raises:
            ValueError: Output format or enabled filter settings are invalid.
            FileExistsError: The destination already exists.
        """
        path = Path(dst) if dst is not None else None
        if path is not None:
            if path.suffix not in {".parquet", ".geojson", ".feather"}:
                raise ValueError(f"Unsupported output format: {path.suffix}")
            if path.exists():
                raise FileExistsError(path)

        geometries = self.gdf.geometry.to_numpy()
        areas = area(geometries)
        tile_rows, clearance, partial = self._tile_detections(geometries)
        conflicts, touching = self._find_matches(geometries, areas, tile_rows, partial)
        fragments = self._preserve_unmatched(conflicts, geometries, areas)
        representatives, source_tiles = self._select_predictions(
            conflicts, geometries, areas, clearance, partial, fragments
        )
        merged = self._join_touching(
            touching, representatives, source_tiles, geometries, areas
        )
        if simplify_level:
            merged.geometry = merged.geometry.simplify(simplify_level)
        if precision is not None:
            merged.geometry = from_wkt(
                to_wkt(merged.geometry.to_numpy(), rounding_precision=precision)
            )
        merged["uid"] = np.arange(len(merged))
        merged = merged.set_index("uid", drop=False)
        merged = filter_nuclei(merged, pixel_size=pixel_size, filter_level=filter_level)

        if path is None:
            return merged
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            merged.to_parquet(path)
        elif path.suffix == ".geojson":
            merged.to_file(path, driver="GeoJSON")
        else:
            merged.to_feather(path)
        return None

    def _tile_detections(
        self, geometries: _PolygonArray
    ) -> tuple[list[_IndexArray], _FloatArray, _BoolArray]:
        """Group nucleus segmentation polygons by their source image tile.

        Each polygon outlines one nucleus segmented in one tile. Its coordinates
        are already in whole-slide pixels. Overlapping tiles can therefore
        provide several polygons for the same nucleus. Collect their source
        tiles and edge information so later steps can prefer a whole nucleus.

        Args:
            geometries: Nucleus segmentation polygons, one per row of self.gdf,
                in full-resolution slide pixel coordinates.

        Returns:
            A tuple (tile_rows, clearance, partial). tile_rows contains polygon
            indices grouped by source tile. clearance gives the distance to the
            nearest source-tile edge for polygons with a known source. partial
            flags polygons that may be cut off at a tile edge. Distances are in
            pixels; the last two arrays follow the input polygon order.
        """
        tiles = self.grid.geometry.to_numpy()
        clearance_by_row = np.zeros(len(geometries))
        partial = np.zeros(len(geometries), dtype=bool)
        known = np.flatnonzero(self.owners >= 0)

        # Keep each tile's rows together without changing their source order.
        ordered = known[np.argsort(self.owners[known], kind="stable")]
        counts = np.bincount(self.owners[known], minlength=len(tiles))
        tile_rows = np.split(ordered, np.cumsum(counts)[:-1]) if len(tiles) else []
        # Rectangle bounds give edge distances without measuring every outline.
        for rows, tile, (minx, miny, maxx, maxy) in zip(
            tile_rows, tiles, bounds(tiles)
        ):
            edges = bounds(geometries[rows])
            clearance = np.min(
                (
                    edges[:, 0] - minx,
                    edges[:, 1] - miny,
                    maxx - edges[:, 2],
                    maxy - edges[:, 3],
                ),
                axis=0,
            )
            outside = clearance < 0
            if outside.any():
                clearance[outside] = distance(geometries[rows[outside]], boundary(tile))
            clearance_by_row[rows] = clearance
        if "is_partial" in self.gdf:
            partial = self.gdf["is_partial"].to_numpy(dtype=bool)
        else:
            # Older files lack border flags; treat near-edge contours as cut off.
            partial[known] = clearance_by_row[known] < 1

        return tile_rows, clearance_by_row, partial

    def _find_matches(
        self,
        geometries: _PolygonArray,
        areas: _FloatArray,
        tile_rows: list[_IndexArray],
        partial: _BoolArray,
    ) -> tuple[nx.Graph[int], list[tuple[int, int]]]:
        """Find possible duplicate nucleus polygons between neighboring tiles.

        The same nucleus can have a segmentation polygon from each overlapping
        tile. Record polygons that share area so later steps can choose between
        them. For tiles that meet without overlap, record touching nucleus
        fragments that may need to be joined.

        Args:
            geometries: Nucleus segmentation polygons in whole-slide pixels,
                in the same order as the input rows.
            areas: Area of each input polygon in square pixels.
            tile_rows: Input polygon indices grouped by source tile, from
                _tile_detections.
            partial: One boolean per input polygon marking possibly cut-off nuclei.

        Returns:
            A tuple (conflicts, touching). conflicts is a graph of polygon pairs
            that might represent the same nucleus, including their overlap and
            match strength. touching is a list of polygon-index pairs that
            touch across a shared tile edge. All indices refer to input polygons.
        """
        # Reuse each tile's search index across all its neighbors.
        trees = [STRtree(geometries[rows]) for rows in tile_rows]
        # Edges connect competing predictions from different source tiles.
        conflicts: nx.Graph[int] = nx.Graph()
        touching: list[tuple[int, int]] = []
        for left, right, shared in self.seams:
            if shared.area == 0 and shared.length == 0:
                continue
            if not len(tile_rows[left]) or not len(tile_rows[right]):
                continue
            left_rows = tile_rows[left][
                trees[left].query(shared, predicate="intersects")
            ]
            pairs = trees[right].query(geometries[left_rows], predicate="intersects")
            left_ids, right_ids = left_rows[pairs[0]], tile_rows[right][pairs[1]]
            common = intersection(geometries[left_ids], geometries[right_ids])
            if shared.area > 0:
                matches, ious, shared_areas = _match_overlaps(
                    geometries, areas, partial, left_ids, right_ids, common
                )
                # Keep weak pairs too: several nuclei may jointly match another tile.
                candidates = np.flatnonzero(shared_areas > 0)
                conflicts.add_edges_from(
                    (
                        int(left_ids[index]),
                        int(right_ids[index]),
                        {
                            "weight": float(ious[index]),
                            "strong": bool(matches[index]),
                            "overlap": float(shared_areas[index]),
                        },
                    )
                    for index in candidates
                )
            else:
                contact = length(common) > 0
                touching.extend(
                    zip(left_ids[contact].tolist(), right_ids[contact].tolist())
                )

        return conflicts, touching

    def _preserve_unmatched(
        self, conflicts: nx.Graph[int], geometries: _PolygonArray, areas: _FloatArray
    ) -> _BoolArray:
        """Keep nucleus polygons that have no adequate match in another tile.

        A nucleus segmented in only one tile should survive stitching. Exclude
        its polygon from duplicate selection. Also flag small polygons that
        cover only a fragment of a larger nucleus in another tile, so later
        selection can prefer the whole nucleus polygon.

        Args:
            conflicts: Candidate nucleus polygon matches from _find_matches.
                Updated in place to exclude polygons with no adequate match.
            geometries: Nucleus segmentation polygons in whole-slide pixels,
                in the same order as the input rows.
            areas: Area of each input polygon in square pixels.

        Returns:
            A boolean array in input polygon order. True marks a small competing
            polygon that covers too little of a larger nucleus to replace it.
            Later selection uses these flags to prefer whole nucleus polygons.
        """
        unmatched = []
        fragments = np.zeros(len(geometries), dtype=bool)
        # A small matching fragment cannot replace a larger nucleus on its own.
        for index in conflicts:
            neighbors = conflicts[index]
            strong = {other for other, data in neighbors.items() if data["strong"]}
            undersized = {
                other
                for other in strong
                if neighbors[other]["overlap"] < 0.5 * areas[index]
            }
            if strong and not undersized:
                continue
            views: dict[np.intp, list[int]] = {}
            for other in neighbors:
                views.setdefault(self.owners[other], []).append(other)
            explained = False
            for rows in views.values():
                overlaps = [neighbors[other]["overlap"] for other in rows]
                shared_area = sum(overlaps)
                if shared_area < 0.5 * areas[index]:
                    fragments[list(undersized.intersection(rows))] = True
                    continue
                # Measure a union only when no single contour covers enough area.
                if len(rows) > 1 and max(overlaps) < 0.5 * areas[index]:
                    shared_area = (
                        geometries[index]
                        .intersection(_union_footprint(geometries[rows]))
                        .area
                    )
                if shared_area >= 0.5 * areas[index]:
                    explained = True
                else:
                    fragments[list(undersized.intersection(rows))] = True
            if not strong and not explained:
                unmatched.append(index)
        # Preserve unmatched nuclei without letting them decide between other tiles.
        conflicts.remove_nodes_from(unmatched)

        return fragments

    def _select_predictions(
        self,
        conflicts: nx.Graph[int],
        geometries: _PolygonArray,
        areas: _FloatArray,
        clearance: _FloatArray,
        partial: _BoolArray,
        fragments: _BoolArray,
    ) -> tuple[_IndexArray, dict[int, set[np.intp]]]:
        """Choose whole nucleus polygons from each group of competing tiles.

        Args:
            conflicts: Graph linking input polygon indices that may describe
                the same nucleus, with their overlap and match strength.
            geometries: Nucleus segmentation polygons in whole-slide pixels,
                in input row order.
            areas: Area of each input polygon in square pixels.
            clearance: Distance from each polygon to its source-tile edge in pixels.
            partial: Flags for input polygons that may be cut off at a tile edge.
            fragments: Flags for small polygons that cannot replace a larger nucleus.

        Returns:
            A tuple (representatives, source_tiles). representatives maps each
            input polygon index to the retained polygon index; kept polygons map
            to themselves. source_tiles is a new dictionary from retained polygon
            indices to represented source tile indices, populated when duplicates
            are replaced.
        """
        # Map each discarded row to the original row that replaces it.
        representatives = np.arange(len(geometries))
        # Joining later must account for every source tile a nucleus represents.
        source_tiles: dict[int, set[np.intp]] = {}
        # Only tie-breaking depends on coordinate precision, not overlap matching.
        coordinate_tolerance = (
            8 * np.spacing(np.abs(self.grid.total_bounds).max())
            if len(self.grid)
            else 0.0
        )
        for component in nx.connected_components(conflicts):
            remaining = set(component)
            while remaining:
                views: dict[np.intp, list[int]] = {}
                # A view is the set of predictions supplied by one tile.
                for index in sorted(remaining):
                    views.setdefault(self.owners[index], []).append(index)
                # Nothing competes with the remaining tile's own detections.
                if len(views) == 1:
                    break
                selected_tile, selected, footprints = _choose_tile(
                    views,
                    geometries,
                    clearance,
                    partial,
                    fragments,
                    coordinate_tolerance,
                )
                # A direct match must explain at least half of the rejected nucleus.
                rejected = {
                    other
                    for index in selected
                    for other, data in conflicts[index].items()
                    if data["strong"] and data["overlap"] >= 0.5 * areas[other]
                    if other in remaining and other not in selected
                }
                alternatives = remaining - selected - rejected
                if alternatives:
                    rejected.update(
                        self._covered_alternatives(
                            alternatives,
                            selected_tile,
                            views,
                            footprints,
                            conflicts,
                            geometries,
                            areas,
                        )
                    )
                for index in rejected:
                    neighbors = selected.intersection(conflicts[index])
                    if not neighbors:
                        neighbors = selected
                    winner = max(
                        neighbors,
                        key=lambda other: (
                            conflicts.get_edge_data(index, other, {}).get("weight", 0),
                            areas[other],
                            -other,
                        ),
                    )
                    representatives[index] = winner
                    source_tiles.setdefault(winner, {self.owners[winner]}).add(
                        self.owners[index]
                    )
                # Keep detections that the chosen view does not explain.
                remaining.difference_update(selected | rejected)

        return representatives, source_tiles

    def _covered_alternatives(
        self,
        alternatives: set[int],
        selected_tile: np.intp,
        views: dict[np.intp, list[int]],
        footprints: dict[np.intp, BaseGeometry],
        conflicts: nx.Graph[int],
        geometries: _PolygonArray,
        areas: _FloatArray,
    ) -> _IndexArray:
        """Find duplicate nucleus polygons supported by agreement between tiles.

        Args:
            alternatives: Input polygon indices still under consideration.
            selected_tile: Source tile chosen to supply the retained nuclei.
            views: Read-only map from tile indices to input polygon indices
                in the group being compared.
            footprints: Combined nucleus geometry for each tile in this group.
                Missing tile entries are added to this dictionary in place.
            conflicts: Graph of candidate nucleus polygon matches and their overlap.
            geometries: Nucleus segmentation polygons in whole-slide pixels,
                in input row order.
            areas: Area of each input polygon in square pixels.

        Returns:
            Input indices of alternative polygons that the selected tile's
            nuclei can replace, supported by agreement with other tiles.
        """
        for tile, rows in views.items():
            if tile not in footprints:
                footprints[tile] = (
                    geometries[rows[0]]
                    if len(rows) == 1
                    else _union_footprint(geometries[rows])
                )
        # Compare each tile's local predictions as a set, not just nucleus pairs.
        agreements: dict[np.intp, set[np.intp]] = {tile: set() for tile in views}
        view_tiles = list(views)
        for position, left in enumerate(view_tiles):
            for right in view_tiles[position + 1 :]:
                if len(views[left]) == len(views[right]) == 1:
                    edge = conflicts.get_edge_data(views[left][0], views[right][0], {})
                    agrees = edge.get("weight", 0) >= 0.5
                else:
                    shared_area = footprints[left].intersection(footprints[right]).area
                    union_area = (
                        footprints[left].area + footprints[right].area - shared_area
                    )
                    agrees = union_area > 0 and shared_area >= 0.5 * union_area
                if agrees:
                    agreements[left].add(right)
                    agreements[right].add(left)
        competing_tiles = {
            tile
            for tile in views
            if tile in agreements[selected_tile]
            # Require two supporting tiles before accepting an indirect match.
            or len(agreements[tile] & agreements[selected_tile]) >= 2
        }
        candidates: _IndexArray = np.array(
            sorted(
                index for index in alternatives if self.owners[index] in competing_tiles
            ),
            dtype=int,
        )
        # An agreeing tile may still contain extra nuclei that must survive.
        covered = area(intersection(geometries[candidates], footprints[selected_tile]))
        return candidates[covered >= 0.5 * areas[candidates]]

    def _join_touching(
        self,
        touching: list[tuple[int, int]],
        representatives: _IndexArray,
        source_tiles: dict[int, set[np.intp]],
        geometries: _PolygonArray,
        areas: _FloatArray,
    ) -> gpd.GeoDataFrame:
        """Join touching nucleus fragments and keep their largest member's metadata.

        Args:
            touching: Input polygon-index pairs from tiles that meet without overlap.
            representatives: Map from each input polygon index to its retained
                polygon index after duplicate selection.
            source_tiles: Dictionary mapping polygon or joined-group representatives
                to the source tile indices they represent. Updated in place as
                fragments join, to keep separate nuclei from one tile apart.
            geometries: Nucleus segmentation polygons in whole-slide pixels,
                in input row order.
            areas: Area of each input polygon in square pixels.

        Returns:
            A new GeoDataFrame of retained nucleus polygons with touching fragments
            joined. Each joined nucleus keeps its largest fragment's attributes.
        """
        groups: nx.utils.UnionFind[int] = nx.utils.UnionFind()
        graph: nx.Graph[int] = nx.Graph()
        for left, right in touching:
            a, b = int(representatives[left]), int(representatives[right])
            # The chosen whole contours may no longer touch where fragments did.
            if a == b:
                continue
            touches = geometries[a].intersection(geometries[b]).length > 0
            if not touches:
                continue
            group_a, group_b = groups[a], groups[b]
            if group_a == group_b:
                continue
            tiles_a = source_tiles.get(group_a, {self.owners[a]})
            tiles_b = source_tiles.get(group_b, {self.owners[b]})
            # Never join two detections that were already separate in one source tile.
            if tiles_a & tiles_b:
                continue
            groups.union(group_a, group_b)
            source_tiles[groups[a]] = tiles_a | tiles_b
            graph.add_edge(a, b)

        keep = representatives == np.arange(len(geometries))
        unions = {}
        for component in nx.connected_components(graph):
            indices = sorted(component)
            # Joined fragments inherit their largest member's class and attributes.
            best = max(indices, key=lambda i: (areas[i], -i))
            keep[indices] = False
            keep[best] = True
            unions[best] = union_all(geometries[indices])
        merged: gpd.GeoDataFrame = self.gdf.loc[keep].copy()
        for index, geometry in unions.items():
            # pandas-stubs omits Shapely geometries from DataFrame scalar assignments.
            merged.at[index, merged.geometry.name] = geometry  # type: ignore[assignment]
        merged = merged.reset_index(drop=True)
        return merged
