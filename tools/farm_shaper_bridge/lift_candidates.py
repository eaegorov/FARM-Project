"""Geometry proposals outside an initial OBB; ownership still needs renderer evidence.

Mask/depth surfaces define a search region, never a filled object volume. All
returned indices address the unchanged source PLY. No heldout data is read here.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def backproject_mask_surface(mask, depth, K, pose, *, stride, depth_min, depth_max):
    """Return metric surface samples and a bound for the skipped pixel spacing."""
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth)
    K, pose = np.asarray(K), np.asarray(pose)
    if mask.shape != depth.shape or depth.ndim != 2:
        raise ValueError("mask and depth must share one image grid")
    if stride < 1 or K.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("invalid sampling stride or camera")
    if (
        not np.isfinite(K).all()
        or not np.isfinite(pose).all()
        or min(K[0, 0], K[1, 1]) <= 0
    ):
        raise ValueError("invalid camera calibration")
    # One actual foreground sample per occupied image cell preserves thin
    # components that could fall between a regular sampling lattice.
    valid = mask & np.isfinite(depth) & (depth >= depth_min) & (depth <= depth_max)
    y, x = np.nonzero(valid)
    cells = (y // stride) * ((depth.shape[1] + stride - 1) // stride) + x // stride
    _, keep = np.unique(cells, return_index=True)
    y, x = y[keep], x[keep]
    z = depth[y, x].astype(np.float64)
    camera = np.column_stack(
        ((x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z)
    )
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    max_depth = float(z.max()) if z.size else 0.0
    spacing = float(stride * np.hypot(1 / K[0, 0], 1 / K[1, 1]) * max_depth)
    return world, max_depth, spacing


def surface_candidate_indices(
    means,
    radii,
    opacities,
    seeds,
    *,
    minimum_opacity,
    maximum_radius,
    radius_multiplier,
    surface_tolerance,
    voxel_size=0.01,
    chunk_size=250_000,
):
    """Find source Gaussian centers near observed surfaces, with scale allowance.

    This conservative search is deliberately broader than segmentation. The
    existing depth gate, positive/negative VJP and multi-timestamp arbitration
    decide membership later. Work memory is bounded by a chunk of source IDs.
    """
    means, seeds = np.asarray(means), np.asarray(seeds, dtype=np.float64)
    radii, opacities = np.asarray(radii), np.asarray(opacities)
    if means.ndim != 2 or means.shape[1] != 3 or seeds.ndim != 2 or seeds.shape[1] != 3:
        raise ValueError("means and seeds must be Nx3")
    if radii.shape != (len(means),) or opacities.shape != radii.shape:
        raise ValueError("source Gaussian attributes must be row aligned")
    if (
        voxel_size <= 0
        or chunk_size < 1
        or surface_tolerance < 0
        or maximum_radius <= 0
        or radius_multiplier < 0
    ):
        raise ValueError("invalid surface candidate policy")
    if not np.isfinite(seeds).all():
        raise ValueError("surface seeds must be finite")
    if not len(seeds):
        return np.empty(0, dtype=np.int64), {"input_seeds": 0, "voxel_seeds": 0}
    _, keep = np.unique(
        np.floor(seeds / voxel_size).astype(np.int64), axis=0, return_index=True
    )
    tree = cKDTree(seeds[np.sort(keep)])
    tolerance = surface_tolerance + np.sqrt(3.0) * voxel_size
    upper = tolerance + radius_multiplier * maximum_radius
    selected = []
    for start in range(0, len(means), chunk_size):
        end = min(start + chunk_size, len(means))
        local = np.flatnonzero(opacities[start:end] >= minimum_opacity)
        if not local.size:
            continue
        ids = start + local
        distance, _ = tree.query(
            means[ids], k=1, distance_upper_bound=upper, workers=-1
        )
        allowed = tolerance + radius_multiplier * np.minimum(radii[ids], maximum_radius)
        selected.append(ids[distance <= allowed])
    ids = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
    return ids, {
        "input_seeds": len(seeds),
        "voxel_seeds": len(keep),
        "surface_tolerance_m": float(tolerance),
        "maximum_search_radius_m": float(upper),
    }
