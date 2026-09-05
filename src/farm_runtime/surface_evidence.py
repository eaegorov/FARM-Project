"""Point-level, timestamp-balanced evidence on registered static surfaces.

Unknown depth and an absent segmentation are never background observations.
Disconnected surface components are query targets, not permission to prune.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from farm_runtime.proposal_geometry import project_evidence


def point_observation(points, mask, *, background_guard_pixels=3, **frame):
    """Separate exact foreground, definite visible background and uncertainty."""
    if background_guard_pixels < 0:
        raise ValueError("background guard must be nonnegative")
    has_mask = mask is not None and np.any(mask)
    grid = (
        np.zeros(frame["depth"].shape, bool) if mask is None else np.asarray(mask, bool)
    )
    if grid.shape != frame["depth"].shape:
        raise ValueError("mask and depth grids must match")
    evidence = project_evidence(
        points, grid, **frame, mask_is_dilated=True, include_point_indices=True
    )
    n = len(points)
    result = {
        k: np.zeros(n, bool)
        for k in ("positive", "negative", "visible", "occluded", "in_front_of_surface")
    }
    indices = evidence["surface_indices"]
    x, y = evidence["surface_pixels_xy"].T
    result["visible"][indices] = True
    result["occluded"][evidence["occluded_indices"]] = True
    result["in_front_of_surface"][evidence["in_front_of_surface_indices"]] = True
    if has_mask:
        result["positive"][indices] = grid[y, x]
        distance = distance_transform_edt(~grid)
        result["negative"][indices] = distance[y, x] > background_guard_pixels
    result["unknown"] = ~(result["positive"] | result["negative"])
    return result


def aggregate_timestamps(observations, point_count, source_timestamps=None):
    """One vote per physical timestamp; within-timestamp conflict stays explicit.

    A source sample counts as foreground in its own timestamp. Multiple cameras
    cannot multiply this evidence. Conservative rejection requires two negative
    timestamps, more negatives than positives, and no within-timestamp conflict.
    """
    grouped = {}
    for timestamp, evidence in observations:
        row = grouped.setdefault(
            str(timestamp),
            {k: np.zeros(point_count, bool) for k in ("positive", "negative")},
        )
        for k in row:
            values = np.asarray(evidence[k], bool)
            if values.shape != (point_count,):
                raise ValueError("point evidence shape mismatch")
            row[k] |= values
    if source_timestamps is not None:
        sources = np.asarray(source_timestamps).astype(str)
        if sources.shape != (point_count,):
            raise ValueError("source timestamp shape mismatch")
        for timestamp in np.unique(sources):
            row = grouped.setdefault(
                timestamp,
                {k: np.zeros(point_count, bool) for k in ("positive", "negative")},
            )
            row["positive"] |= sources == timestamp
    counts = {
        k: np.zeros(point_count, np.int32)
        for k in ("positive", "negative", "conflicted")
    }
    for row in grouped.values():
        conflict = row["positive"] & row["negative"]
        counts["positive"] += row["positive"] & ~conflict
        counts["negative"] += row["negative"] & ~conflict
        counts["conflicted"] += conflict
    counts["contradicted"] = (
        (counts["negative"] >= 2)
        & (counts["negative"] > counts["positive"])
        & (counts["conflicted"] == 0)
    )
    counts["corroborated"] = (
        (counts["positive"] >= 2)
        & (counts["negative"] == 0)
        & (counts["conflicted"] == 0)
    )
    return counts


def surface_components(points, source_timestamps, radii_m, *, neighbors=16):
    """Local spatial components used only to prioritize inspection and views.

    Mutual local radii avoid a coarse distant sample bridging separate surfaces.
    No assumption is made that a physical object has one connected component.
    """
    points = np.asarray(points, float)
    timestamps = np.asarray(source_timestamps).astype(str)
    radii = np.asarray(radii_m, float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or timestamps.shape != (len(points),)
        or radii.shape != (len(points),)
    ):
        raise ValueError("N x 3 points and N timestamps/radii required")
    if (
        not np.isfinite(points).all()
        or not np.isfinite(radii).all()
        or np.any(radii <= 0)
        or neighbors < 1
    ):
        raise ValueError("finite points, positive radii and neighbors required")
    if not len(points):
        return np.empty(0, np.int32), []
    distances, indices = cKDTree(points).query(
        points, k=min(neighbors + 1, len(points))
    )
    distances, indices = distances.reshape(len(points), -1), indices.reshape(
        len(points), -1
    )
    source = np.broadcast_to(np.arange(len(points))[:, None], indices.shape)
    keep = (source != indices) & (
        distances <= np.minimum(radii[:, None], radii[indices])
    )
    graph = coo_matrix(
        (np.ones(int(keep.sum()), bool), (source[keep], indices[keep])),
        shape=(len(points), len(points)),
    ).tocsr()
    count, labels = connected_components(graph, directed=False)
    weights = np.zeros(len(points))
    for timestamp in np.unique(timestamps):
        selected = timestamps == timestamp
        weights[selected] = 1 / selected.sum()
    rows = []
    for label in range(count):
        selected = labels == label
        rows.append(
            dict(
                id=label,
                points=int(selected.sum()),
                timestamp_weight=float(weights[selected].sum()),
                timestamps=sorted(set(timestamps[selected])),
                bounds_m=[
                    points[selected].min(axis=0).tolist(),
                    points[selected].max(axis=0).tolist(),
                ],
            )
        )
    rows.sort(key=lambda r: (-r["timestamp_weight"], r["id"]))
    for row in rows:
        row["role"] = (
            "dominant_surface" if row is rows[0] else "disconnected_unverified"
        )
        row["automatic_removal_allowed"] = False
    return labels, rows
