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
    disjoint_subset_union_all,
    from_wkt,
    geos_version,
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

        Direct duplicates compete at intersection-over-union >= 0.5 or
        coverage >= 0.8 of the smaller contour. An is_partial flag from
        vectorization identifies contours clipped by the
        input image border before smoothing; these may instead match at half the
        smaller polygon's area when the smaller contour is partial. Older
        exports use tile-edge distance < 1 pixel as a fallback. Unmatched
        predictions survive. Complete observations with >=80% smaller-contour
        coverage can also replace a larger partial alternative if their combined
        union area is at least half its area. A lone tiny detection cannot do so.
        Compare intersecting local source-tile partitions.
        Their polygon unions must agree at IoU >= 0.5, directly or
        through two other source tiles. Keep whole nuclei from one view;
        an alternative needs >=50% coverage by the selected view to be removed.
        A single intermediate match cannot establish a transitive duplicate.
        Prefer views that cover at least half of strongly matched larger nuclei,
        then complete views with at least 80% of their largest union area, then
        tile-edge clearance; if all views are cut, prefer area.
        Retain the selected original contours and attributes, and unmatched
        detections from other tiles. Distinct nuclei may overlap;
        intersections are not objects.

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

        # Disjoint groups are common; older GEOS builds need the general union.
        footprint_union = (
            disjoint_subset_union_all if geos_version >= (3, 12, 0) else union_all
        )
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
                    observed = footprint_union(geometries[complete_ids[positions]])
                    if observed.area >= 0.5 * areas[index]:
                        matches[alternatives[positions]] = True
                # Disagreeing splits may agree as a set without any strong pair.
                candidates = np.flatnonzero(shared_areas > 0)
                conflicts.add_edges_from(
                    (
                        int(a[index]),
                        int(b[index]),
                        {
                            "weight": float(ious[index]),
                            "strong": bool(matches[index]),
                            "overlap": float(shared_areas[index]),
                        },
                    )
                    for index in candidates
                )
            else:
                touching = length(common) > 0
                abutting.extend(zip(a[touching].tolist(), b[touching].tolist()))

        unmatched = []
        fragments = np.zeros(len(geometries), dtype=bool)
        # Separate unmatched nuclei from views that only explain a small fragment.
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
            views = {}
            for other in neighbors:
                views.setdefault(self.owners[other], []).append(other)
            explained = False
            for rows in views.values():
                overlaps = [neighbors[other]["overlap"] for other in rows]
                shared_area = sum(overlaps)
                if shared_area < 0.5 * areas[index]:
                    fragments[list(undersized.intersection(rows))] = True
                    continue
                # One covering contour already proves coverage by the whole view.
                if len(rows) > 1 and max(overlaps) < 0.5 * areas[index]:
                    shared_area = (
                        geometries[index]
                        .intersection(footprint_union(geometries[rows]))
                        .area
                    )
                if shared_area >= 0.5 * areas[index]:
                    explained = True
                else:
                    fragments[list(undersized.intersection(rows))] = True
            if not strong and not explained:
                unmatched.append(index)
        # Weakly touched nuclei survive without changing which tile wins a dispute.
        conflicts.remove_nodes_from(unmatched)

        representatives = np.arange(len(geometries))
        source_tiles = {}
        for component in nx.connected_components(conflicts):
            remaining = set(component)
            while remaining:
                views = {}
                for index in sorted(remaining):
                    views.setdefault(self.owners[index], []).append(index)
                # No competing tile remains; keep all its original detections.
                if len(views) == 1:
                    break
                # A tiny nested mask cannot stand in for a whole nucleus elsewhere.
                whole_views = {
                    tile: rows
                    for tile, rows in views.items()
                    if not fragments[rows].any()
                }
                if not whole_views:
                    whole_views = {
                        tile: [index for index in rows if not fragments[index]]
                        for tile, rows in views.items()
                        if not fragments[rows].all()
                    } or views
                complete_views = {
                    tile: rows
                    for tile, rows in whole_views.items()
                    if not cut[rows].any()
                }
                eligible = complete_views or whole_views
                # Union area avoids rewarding duplicate or overlapping masks by count.
                footprints = {
                    tile: geometries[rows[0]]
                    if len(rows) == 1
                    else footprint_union(geometries[rows])
                    for tile, rows in eligible.items()
                }
                footprint_areas = {tile: geom.area for tile, geom in footprints.items()}
                largest = max(footprint_areas.values())
                # An interior fragment cannot displace a view of the whole nucleus.
                supported = [
                    tile for tile in eligible if footprint_areas[tile] >= 0.8 * largest
                ]

                def view_rank(tile):
                    rows = whole_views[tile]
                    complete = not cut[rows].any()
                    clearance = context[rows].min()
                    return (
                        complete,
                        clearance if complete else footprint_areas[tile],
                        footprint_areas[tile] if complete else clearance,
                        -rows[0],
                    )

                selected_tile = max(supported, key=view_rank)
                selected = set(whole_views[selected_tile])
                rejected = {
                    other
                    for index in selected
                    for other, data in conflicts[index].items()
                    if data["strong"] and data["overlap"] >= 0.5 * areas[other]
                    if other in remaining and other not in selected
                }
                alternatives = remaining - selected - rejected
                if alternatives:
                    for tile, rows in views.items():
                        if tile not in footprints:
                            footprints[tile] = (
                                geometries[rows[0]]
                                if len(rows) == 1
                                else footprint_union(geometries[rows])
                            )
                    agreements = {tile: set() for tile in views}
                    view_tiles = list(views)
                    for position, left in enumerate(view_tiles):
                        for right in view_tiles[position + 1 :]:
                            if len(views[left]) == len(views[right]) == 1:
                                edge = conflicts.get_edge_data(
                                    views[left][0], views[right][0], {}
                                )
                                agrees = edge.get("weight", 0) >= 0.5
                            else:
                                shared_area = (
                                    footprints[left]
                                    .intersection(footprints[right])
                                    .area
                                )
                                union_area = (
                                    footprints[left].area
                                    + footprints[right].area
                                    - shared_area
                                )
                                agrees = (
                                    union_area > 0 and shared_area >= 0.5 * union_area
                                )
                            if agrees:
                                agreements[left].add(right)
                                agreements[right].add(left)
                    competing_tiles = {
                        tile
                        for tile in views
                        if tile in agreements[selected_tile]
                        # Two corroborating views avoid following a lone broad bridge.
                        or len(agreements[tile] & agreements[selected_tile]) >= 2
                    }
                    candidates = np.array(
                        sorted(
                            index
                            for index in alternatives
                            if self.owners[index] in competing_tiles
                        ),
                        dtype=int,
                    )
                    covered = area(
                        intersection(geometries[candidates], footprints[selected_tile])
                    )
                    rejected.update(candidates[covered >= 0.5 * areas[candidates]])
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
