from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely import (
    STRtree,
    area,
    boundary,
    distance,
    from_wkt,
    intersection,
    length,
    to_wkt,
    union_all,
)
from shapely.geometry import box

__all__ = ["InstMerger"]


class InstMerger:
    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        coordinates: list[tuple[int, int, int, int]],
        tile_indices: Sequence[int] | None = None,
    ) -> None:
        """Reconcile nucleus predictions in slide-pixel coordinates.

        Args:
            gdf: Nucleus polygons in level-zero pixel coordinates, with class_name
                and optionally the boolean is_partial column from inst2gdf.
            coordinates: Tile bounds in level-zero (x, y, width, height) pixels.
            tile_indices: Source-tile index for each input row. Required when tiles
                overlap; centroid position cannot identify the source tile.

        Raises:
            ValueError: Source indices are missing for overlapping tiles or invalid.
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
            self.owners = np.full(len(self.gdf), -1, dtype=int)
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
    ) -> gpd.GeoDataFrame | None:
        """Match duplicate predictions and join physically touching tile fragments.

        Complete predictions compete at intersection-over-union >= 0.5 or
        coverage >= 0.8 of the smaller contour. An is_partial flag from
        vectorization identifies contours clipped by the
        input image border before smoothing; these may instead match at half the
        smaller polygon's area when the smaller contour is partial. Older
        exports use tile-edge distance < 1 pixel as a fallback. Unmatched
        predictions survive. Complete observations with >=80% smaller-contour
        coverage can also replace a larger partial alternative if their combined
        union area is at least half its area. A lone tiny detection cannot do so.
        Choose coherent observations from one source tile
        within each local conflict, preserving its separate nuclei. Prefer complete
        views, retaining those with at least 80% of their largest union area, then
        tile-edge clearance; if all views are cut, prefer area.
        Suppress only direct competitors, retaining the selected original contours
        and attributes. Distinct nuclei may overlap; intersections are not objects.

        Abutting joins are handled separately per tile pair and require actual
        boundary contact, including in grids that also contain overlapping tiles.
        No polygon is expanded to close a gap. Coordinates and CRS stay unchanged.

        Args:
            dst: New .parquet, .geojson or .feather file; None returns the result.
            simplify_level: Geometry simplification tolerance in pixels; zero disables it.
            precision: Optional number of decimal places for coordinate rounding.

        Returns:
            The merged GeoDataFrame when dst is None; otherwise writes the file
            and returns None. Each row has a unique slide-wide uid.

        Raises:
            ValueError: The destination extension is unsupported.
            FileExistsError: The destination already exists.
        """
        path = Path(dst) if dst is not None else None
        if path is not None:
            if path.suffix not in {".parquet", ".geojson", ".feather"}:
                raise ValueError(f"Unsupported output format: {path.suffix}")
            if path.exists():
                raise FileExistsError(path)

        geometries = self.gdf.geometry.to_numpy()
        tiles = self.grid.geometry.to_numpy()
        areas = area(geometries)
        context = np.zeros(len(geometries))
        cut = np.zeros(len(geometries), dtype=bool)
        known = np.flatnonzero(self.owners >= 0)
        context[known] = distance(
            geometries[known], boundary(tiles)[self.owners[known]]
        )
        if "is_partial" in self.gdf:
            cut = self.gdf["is_partial"].to_numpy(dtype=bool)
        else:
            # Older exports lack pre-smoothing border flags; retain uncertain contours.
            cut[known] = context[known] < 1

        # Reuse one spatial index per tile instead of rebuilding it for each seam.
        ordered = known[np.argsort(self.owners[known], kind="stable")]
        counts = np.bincount(self.owners[known], minlength=len(tiles))
        tile_rows = np.split(ordered, np.cumsum(counts)[:-1]) if len(tiles) else []
        trees = [STRtree(geometries[rows]) for rows in tile_rows]
        conflicts = nx.Graph()
        abutting = []
        for left, right, shared in self.seams:
            if shared.area == 0 and shared.length == 0:
                continue
            if not len(tile_rows[left]) or not len(tile_rows[right]):
                continue
            left_ids = tile_rows[left][
                trees[left].query(shared, predicate="intersects")
            ]
            pairs = trees[right].query(geometries[left_ids], predicate="intersects")
            a, b = left_ids[pairs[0]], tile_rows[right][pairs[1]]
            common = intersection(geometries[a], geometries[b])
            if shared.area > 0:
                shared_areas = area(common)
                union_areas = areas[a] + areas[b] - shared_areas
                ious = np.divide(
                    shared_areas,
                    union_areas,
                    out=np.zeros(len(a)),
                    where=union_areas > 0,
                )
                smaller = np.minimum(areas[a], areas[b])
                coverage = np.divide(
                    shared_areas, smaller, out=np.zeros(len(a)), where=smaller > 0
                )
                # Strong coverage also links nested and one-versus-many alternatives.
                partial_smaller = (cut[a] & (areas[a] <= areas[b])) | (
                    cut[b] & (areas[b] <= areas[a])
                )
                matches = (
                    (ious >= 0.5)
                    | (~cut[a] & ~cut[b] & (coverage >= 0.8))
                    | (partial_smaller & (coverage >= 0.5))
                )
                alternatives = np.flatnonzero(
                    (coverage >= 0.8) & (cut[a] != cut[b]) & ~partial_smaller
                )
                partial_ids = np.where(
                    cut[a[alternatives]], a[alternatives], b[alternatives]
                )
                complete_ids = np.where(
                    cut[a[alternatives]], b[alternatives], a[alternatives]
                )
                for index in np.unique(partial_ids):
                    positions = partial_ids == index
                    if matches[alternatives[positions]].all():
                        continue
                    # Several complete nuclei can explain one border-cut alternative.
                    observed = union_all(geometries[complete_ids[positions]])
                    if observed.area >= 0.5 * areas[index]:
                        matches[alternatives[positions]] = True
                conflicts.add_weighted_edges_from(
                    (int(i), int(j), float(iou))
                    for iou, i, j in zip(ious[matches], a[matches], b[matches])
                )
            else:
                touching = length(common) > 0
                abutting.extend(zip(a[touching].tolist(), b[touching].tolist()))

        representatives = np.arange(len(geometries))
        source_tiles = {}
        for component in nx.connected_components(conflicts):
            remaining = set(component)
            while remaining:
                views = {}
                for index in sorted(remaining):
                    views.setdefault(self.owners[index], []).append(index)
                complete_views = {
                    tile: rows for tile, rows in views.items() if not cut[rows].any()
                }
                eligible = complete_views or views
                # Union area avoids rewarding duplicate or overlapping masks by count.
                footprints = {
                    tile: areas[rows[0]]
                    if len(rows) == 1
                    else union_all(geometries[rows]).area
                    for tile, rows in eligible.items()
                }
                largest = max(footprints.values())
                # An interior fragment cannot displace a view of the whole nucleus.
                supported = [
                    tile for tile in eligible if footprints[tile] >= 0.8 * largest
                ]

                def view_rank(tile):
                    rows = views[tile]
                    complete = not cut[rows].any()
                    clearance = context[rows].min()
                    return (
                        complete,
                        clearance if complete else footprints[tile],
                        footprints[tile] if complete else clearance,
                        -rows[0],
                    )

                selected = set(views[max(supported, key=view_rank)])
                rejected = {
                    other
                    for index in selected
                    for other in conflicts[index]
                    if other in remaining and other not in selected
                }
                for index in rejected:
                    winner = max(
                        selected.intersection(conflicts[index]),
                        key=lambda other: (
                            conflicts[index][other]["weight"],
                            areas[other],
                            -other,
                        ),
                    )
                    representatives[index] = winner
                    source_tiles.setdefault(winner, {self.owners[winner]}).add(
                        self.owners[index]
                    )
                # A neighbour of a rejected alternative is not necessarily a duplicate.
                remaining.difference_update(selected | rejected)

        groups = nx.utils.UnionFind()

        def join(left: int, right: int) -> bool:
            a, b = groups[left], groups[right]
            if a == b:
                return False
            tiles_a = source_tiles.get(a, {self.owners[left]})
            tiles_b = source_tiles.get(b, {self.owners[right]})
            # Two nuclei predicted in one tile must remain separate objects.
            if tiles_a & tiles_b:
                return False
            groups.union(a, b)
            source_tiles[groups[left]] = tiles_a | tiles_b
            return True

        graph = nx.Graph()
        for left, right in abutting:
            a, b = representatives[left], representatives[right]
            # Recheck contact after replacing duplicates by their selected contours.
            if (
                a != b
                and geometries[a].intersection(geometries[b]).length > 0
                and join(int(a), int(b))
            ):
                graph.add_edge(int(a), int(b))

        keep = representatives == np.arange(len(geometries))
        unions = {}
        for component in nx.connected_components(graph):
            indices = sorted(component)
            best = max(indices, key=lambda i: (areas[i], -i))
            keep[indices] = False
            keep[best] = True
            unions[best] = union_all(geometries[indices])
        merged = self.gdf.loc[keep].copy()
        for index, geometry in unions.items():
            merged.at[index, merged.geometry.name] = geometry
        merged = merged.reset_index(drop=True)
        if simplify_level:
            merged.geometry = merged.geometry.simplify(simplify_level)
        if precision is not None:
            merged.geometry = from_wkt(
                to_wkt(merged.geometry.to_numpy(), rounding_precision=precision)
            )
        merged["uid"] = np.arange(len(merged))
        merged = merged.set_index("uid", drop=False)

        if path is None:
            return merged
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            merged.to_parquet(path)
        elif path.suffix == ".geojson":
            merged.to_file(path, driver="GeoJSON")
        else:
            merged.to_feather(path)
