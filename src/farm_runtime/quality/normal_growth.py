"""Bounded surface-graph proposals; never assigns object labels.

Run after object-scope assembly. All coordinates and radii must share units.
Unknown space is eligible only when the caller explicitly supplies its domain.
Normals cannot separate coplanar neighboring objects: every proposal still
requires image/multiview evidence before acceptance into a mask.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class NormalGrowthPolicy:
    max_steps: int = 4
    max_candidates: int = 10000
    max_growth_ratio: float = 2.0
    max_unknown_hops: int = 1
    normal_angle_degrees: float = 30.0
    connection_spacing_multiplier: float = 1.75
    max_connection_voxels: float = 2.5
    normal_offset_spacing: float = 0.35
    normal_neighbors: int = 16
    normal_radius_voxels: float = 3.0
    max_pca_thickness: float = 0.15
    min_pca_tangent: float = 0.10
    max_voxels: int = 200000
    max_neighbors: int = 32

    def __post_init__(self):
        for name in ("max_steps", "max_candidates", "max_unknown_hops"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("normal_neighbors", "max_voxels", "max_neighbors"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.normal_neighbors < 6:
            raise ValueError("normal_neighbors must be at least six")
        for name in ("max_growth_ratio", "connection_spacing_multiplier",
                     "max_connection_voxels", "normal_offset_spacing",
                     "normal_radius_voxels", "max_pca_thickness",
                     "min_pca_tangent", "normal_angle_degrees"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.normal_angle_degrees < 90:
            raise ValueError("normal angle must be between zero and 90 degrees")
        if self.max_pca_thickness >= 1 or self.min_pca_tangent >= 1:
            raise ValueError("PCA ratios must be below one")


def _mask(value, count, name):
    array = np.asarray(value)
    if array.shape != (count,) or array.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean vector matching xyz")
    return array


def _edge_clear(a, b, voxel_size, blocked_cells):
    """Visit every voxel traversed by the segment, including short crossings."""
    cuts = [0.0, 1.0]
    for axis in range(3):
        delta = b[axis] - a[axis]
        if abs(delta) < 1e-14 * voxel_size:
            continue
        lo, hi = sorted((a[axis], b[axis]))
        planes = np.arange(np.floor(lo / voxel_size) + 1,
                           np.ceil(hi / voxel_size)) * voxel_size
        cuts.extend(((planes - a[axis]) / delta).tolist())
    cuts = np.unique(np.clip(cuts, 0.0, 1.0))
    return not any(
        tuple(np.floor((a + (b - a) * ((t0 + t1) / 2)) / voxel_size)
              .astype(np.int64)) in blocked_cells
        for t0, t1 in pairwise(cuts)
    )


def _voxel_normals(centers, tree, voxel_size, policy):
    """Local PCA rejects both volumetric clouds and line-like neighborhoods."""
    result = np.zeros_like(centers)
    reliable = np.zeros(len(centers), bool)
    _, neighbors = tree.query(
        centers, k=min(policy.normal_neighbors, len(centers)),
        distance_upper_bound=policy.normal_radius_voxels * voxel_size,
    )
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
    for i, row in enumerate(neighbors):
        local = centers[row[row < len(centers)]]
        if len(local) < 6:
            continue
        centered = local - local.mean(axis=0)
        values, vectors = np.linalg.eigh(centered.T @ centered / len(local))
        largest = values[-1]
        if (largest > np.finfo(float).eps * voxel_size ** 2
                and values[0] / largest <= policy.max_pca_thickness
                and values[1] / largest >= policy.min_pca_tangent):
            result[i], reliable[i] = vectors[:, 0], True
    return result, reliable


def graph_surface_candidates(
    xyz, seed_indices, *, positive_mask, unknown_mask, protected_mask=None,
    negative_mask=None, normals=None, normal_reliable=None,
    gaussian_radii=None, voxel_size=None, policy=NormalGrowthPolicy(),
):
    """Return source-row indices and audit, excluding seeds; inputs stay intact.

    positive_mask denotes usable object evidence, not accepted membership.
    unknown_mask is an explicitly bounded domain lacking such evidence.
    protected_mask denotes another owner or an unresolved strong conflict.
    Negative/protected occupied voxels are barriers, including between nodes.
    Supplied normals require caller reliability; zero/nonfinite normals never
    grow. Without normals, planar local voxel PCA estimates them conservatively.
    max_unknown_hops limits consecutive unsupported steps, not confidence.
    For large scenes pass a local source subset and map returned rows to global
    source IDs; max_voxels bounds the prototype's graph construction.
    """
    if not isinstance(policy, NormalGrowthPolicy):
        raise ValueError("NormalGrowthPolicy required")
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("xyz must be finite N by 3 coordinates")
    count = len(points)
    positive = _mask(positive_mask, count, "positive_mask")
    unknown = _mask(unknown_mask, count, "unknown_mask")
    if np.any(positive & unknown):
        raise ValueError("positive and unknown domains must be disjoint")
    zero = np.zeros(count, bool)
    protected = zero if protected_mask is None else _mask(
        protected_mask, count, "protected_mask")
    negative = zero if negative_mask is None else _mask(
        negative_mask, count, "negative_mask")
    seeds = np.asarray(seed_indices)
    if seeds.ndim != 1 or (seeds.size and seeds.dtype.kind not in "iu"):
        raise ValueError("seed_indices must be integer source rows")
    seeds = seeds.astype(np.int64)
    if (np.any(seeds < 0) or np.any(seeds >= count)
            or len(np.unique(seeds)) != len(seeds)):
        raise ValueError("unique in-range seed indices required")
    barrier = protected | negative
    if barrier[seeds].any():
        raise ValueError("seeds cannot be negative or protected")
    radii = np.zeros(count) if gaussian_radii is None else np.asarray(
        gaussian_radii, dtype=float)
    if radii.shape != (count,) or not np.isfinite(radii).all() or np.any(radii < 0):
        raise ValueError("nonnegative finite gaussian_radii matching xyz required")
    given_normals = None
    if normal_reliable is not None and normals is None:
        raise ValueError("normal_reliable requires supplied normals")
    if normals is not None:
        given_normals = np.asarray(normals, dtype=float)
        if given_normals.shape != points.shape:
            raise ValueError("normals must match xyz")
        lengths = np.linalg.norm(given_normals, axis=1)
        point_reliable = np.isfinite(given_normals).all(axis=1) & (lengths > 1e-12)
        if normal_reliable is not None:
            point_reliable &= _mask(normal_reliable, count, "normal_reliable")
        given_normals = np.divide(
            given_normals, lengths[:, None], out=np.zeros_like(given_normals),
            where=point_reliable[:, None],
        )
    if voxel_size is not None and (not np.isfinite(voxel_size) or voxel_size <= 0):
        raise ValueError("voxel_size must be finite and positive")
    audit = dict(
        schema="farm.normal-growth-candidates.v1", source_rows=count,
        seed_count=len(seeds), policy=asdict(policy),
        candidates_only=True, labels_assigned=False,
        normals_source="supplied" if normals is not None else "local_voxel_pca",
        positive_domain=int(positive.sum()), unknown_domain=int(unknown.sum()),
        protected_rows=int(protected.sum()), negative_rows=int(negative.sum()),
        candidates=0, candidate_hops=[], candidate_unknown_hops=[],
        limitations="Surface continuity is not physical object identity; "
                    "coplanar neighbors need image evidence. Voxel barriers "
                    "are conservative and may suppress thin surfaces.",
    )
    cap = min(policy.max_candidates, int(len(seeds) * policy.max_growth_ratio))
    if not len(seeds) or not cap or not policy.max_steps:
        return np.zeros(0, np.int64), dict(audit, stop_reason="empty_seed_or_budget")
    domain = (positive | unknown).copy()
    domain[seeds] = True
    rows = np.flatnonzero(domain | barrier)
    local_points = points[rows]
    if voxel_size is None:
        tree = cKDTree(local_points)
        sample = local_points[np.linspace(0, len(rows) - 1,
                                          min(2048, len(rows)), dtype=int)]
        distances, _ = tree.query(sample, k=min(8, len(rows)))
        distances = np.asarray(distances).reshape(len(sample), -1)
        distances = np.where(distances > 1e-12, distances, np.inf).min(axis=1)
        finite = distances[np.isfinite(distances)]
        if not len(finite):
            return np.zeros(0, np.int64), dict(audit, stop_reason="no_local_spacing")
        voxel_size = float(np.median(finite))
    voxel_size = float(voxel_size)
    cell_float = np.floor(local_points / voxel_size)
    if np.any(np.abs(cell_float) > np.iinfo(np.int64).max / 2):
        raise ValueError("coordinate/voxel ratio exceeds integer index range")
    cells, inverse = np.unique(cell_float.astype(np.int64), axis=0,
                               return_inverse=True)
    size = len(cells)
    if size > policy.max_voxels:
        raise ValueError("voxel budget exceeded; pass a bounded source subset")
    weights = np.bincount(inverse, minlength=size)
    centers = np.stack([np.bincount(inverse, weights=local_points[:, d],
                                   minlength=size) / weights for d in range(3)], axis=1)
    tree = cKDTree(centers)
    distances, _ = tree.query(centers, k=min(2, size))
    spacing = (distances[:, 1] if size > 1 else np.full(size, voxel_size))
    spacing = np.clip(spacing, 0.5 * voxel_size, 2 * voxel_size)
    blocked = np.zeros(size, bool)
    blocked[inverse[barrier[rows]]] = True
    blocked_cells = {tuple(cell) for cell in cells[blocked]}
    voxel_positive = np.zeros(size, bool)
    voxel_positive[inverse[positive[rows]]] = True
    seed_local = np.searchsorted(rows, seeds)
    seed_voxels = np.unique(inverse[seed_local])
    voxel_positive[seed_voxels] = True
    if given_normals is None:
        unit, reliable = _voxel_normals(centers, tree, voxel_size, policy)
    else:
        # Sign-invariant averaging; mixed orientations invalidate a voxel.
        outer = given_normals[rows, :, None] * given_normals[rows, None, :]
        scatter = np.stack([
            np.bincount(inverse, weights=outer[:, d, e], minlength=size)
            for d in range(3) for e in range(3)
        ], axis=1).reshape(size, 3, 3)
        values, vectors = np.linalg.eigh(scatter)
        total = values.sum(axis=1)
        reliable = (total > 0) & (values[:, -1] >= 0.9 * total)
        unit = vectors[:, :, -1]
    reliable &= ~blocked
    voxel_radius = np.zeros(size)
    np.maximum.at(voxel_radius, inverse, np.minimum(radii[rows], 2 * spacing[inverse]))
    max_edge = policy.max_connection_voxels * voxel_size
    hop = np.full(size, -1, dtype=int)
    unsupported = np.full(size, policy.max_unknown_hops + 1, dtype=int)
    frontier = seed_voxels[reliable[seed_voxels]]
    hop[frontier], unsupported[frontier] = 0, 0
    threshold = np.cos(np.deg2rad(policy.normal_angle_degrees))
    rejected = dict(normal=0, distance=0, normal_offset=0, barrier=0, unknown=0)
    for step in range(1, policy.max_steps + 1):
        pending = {}
        for parent in frontier:
            distance, near = tree.query(centers[parent], k=min(size, policy.max_neighbors + 1),
                                        distance_upper_bound=max_edge)
            for length, child in zip(np.atleast_1d(distance), np.atleast_1d(near)):
                if child >= size or child == parent or hop[child] >= 0:
                    continue
                if not reliable[child]:
                    rejected["barrier" if blocked[child] else "normal"] += 1
                    continue
                gap = 0 if voxel_positive[child] else unsupported[parent] + 1
                if gap > policy.max_unknown_hops:
                    rejected["unknown"] += 1
                    continue
                local_spacing = min(spacing[parent], spacing[child])
                limit = min(max_edge, max(
                    policy.connection_spacing_multiplier * local_spacing,
                    voxel_radius[parent] + voxel_radius[child],
                ))
                if length > limit:
                    rejected["distance"] += 1
                    continue
                if abs(np.dot(unit[parent], unit[child])) < threshold:
                    rejected["normal"] += 1
                    continue
                delta = centers[child] - centers[parent]
                offset = max(abs(np.dot(delta, unit[parent])),
                             abs(np.dot(delta, unit[child])))
                if offset > policy.normal_offset_spacing * local_spacing:
                    rejected["normal_offset"] += 1
                    continue
                if not _edge_clear(centers[parent], centers[child],
                                   voxel_size, blocked_cells):
                    rejected["barrier"] += 1
                    continue
                pending[child] = min(pending.get(child, gap), gap)
        frontier = np.asarray(sorted(pending), dtype=int)
        if not len(frontier):
            break
        hop[frontier] = step
        unsupported[frontier] = [pending[i] for i in frontier]
    eligible = domain[rows] & ~barrier[rows] & (hop[inverse] >= 0)
    eligible[seed_local] = False
    if given_normals is not None:
        eligible &= point_reliable[rows]
        eligible &= np.abs(np.einsum(
            "ij,ij->i", given_normals[rows], unit[inverse]
        )) >= threshold
    point_gaps = np.where(unknown[rows], np.maximum(1, unsupported[inverse]), 0)
    eligible &= point_gaps <= policy.max_unknown_hops
    selected = np.flatnonzero(eligible)
    order = np.lexsort((rows[selected], ~positive[rows[selected]], hop[inverse[selected]]))
    selected = selected[order[:cap]]
    # Source IDs stay sorted, with per-candidate diagnostics in the same order.
    selected = selected[np.argsort(rows[selected])]
    candidates = rows[selected].astype(np.int64)
    audit.update(
        voxel_size=voxel_size, voxels=size, reliable_voxels=int(reliable.sum()),
        blocked_voxels=int(blocked.sum()), reliable_seed_voxels=int(
            reliable[seed_voxels].sum()), reached_voxels=int((hop >= 0).sum()),
        candidates=len(candidates), candidates_before_cap=int(eligible.sum()),
        candidate_hops=hop[inverse[selected]].tolist(),
        candidate_unknown_hops=point_gaps[selected].tolist(),
        positive_candidates=int(positive[candidates].sum()),
        unknown_candidates=int(unknown[candidates].sum()),
        rejected_edges=rejected, stop_reason=(
            "candidate_budget" if eligible.sum() > cap else "bounded_graph_complete"),
    )
    return candidates, audit
