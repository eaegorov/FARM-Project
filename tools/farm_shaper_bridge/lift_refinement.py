"""Build-only spatial refinement for Gaussian instance claims.

The refinement is deliberately downstream of strong multi-timestamp claims and
upstream of the frozen held-out boundary.  It may add only previously unknown
source rows that have positive build evidence and are spatially connected to a
strong core.  It never opens held-out masks and never overwrites another object
or a strong unresolved conflict.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from tools.farm_shaper_bridge.common import UNKNOWN_ID, resolve_sparse_claims


def _membership_mask(sorted_values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Return exact membership without allocating a dense global bitmap."""

    values = np.asarray(sorted_values, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)
    if not len(values) or not len(query):
        return np.zeros(len(query), dtype=bool)
    positions = np.searchsorted(values, query)
    valid = positions < len(values)
    result = np.zeros(len(query), dtype=bool)
    result[valid] = values[positions[valid]] == query[valid]
    return result


def _rank_growth(
    local_indices: np.ndarray,
    item: Any,
    purity: np.ndarray,
    nearest_distance: np.ndarray,
) -> np.ndarray:
    """Stable evidence-first ordering used only for a conservative growth cap."""

    local = np.asarray(local_indices, dtype=np.int64)
    return np.lexsort((
        item.indices[local],
        nearest_distance,
        -item.positive_weight[local],
        -purity[local],
        -item.positive_timestamps[local].astype(np.int32),
    ))


def _connected_growth(
    means_m: np.ndarray,
    radius_m: np.ndarray,
    item: Any,
    core_local: np.ndarray,
    eligible_local: np.ndarray,
    purity: np.ndarray,
    policy: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grow at most N hops from core with adaptive Gaussian-scale edges."""

    minimum_neighbors = int(policy["minimum_core_neighbors"])
    maximum_steps = int(policy["maximum_growth_steps"])
    minimum_radius = float(policy["minimum_connection_radius_m"])
    maximum_radius = float(policy["maximum_connection_radius_m"])
    radius_multiplier = float(policy["connection_radius_multiplier"])
    growth_cap = int(np.floor(len(core_local) * float(policy["maximum_growth_ratio"])))
    if (
        growth_cap <= 0
        or len(core_local) < minimum_neighbors
        or not len(eligible_local)
    ):
        empty_i = np.zeros(0, dtype=np.int64)
        return empty_i, empty_i.astype(np.uint8), empty_i.astype(np.float32)

    connected = np.asarray(core_local, dtype=np.int64)
    remaining = np.asarray(eligible_local, dtype=np.int64)
    selected_parts: list[np.ndarray] = []
    step_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    selected_total = 0

    for step in range(1, maximum_steps + 1):
        capacity = growth_cap - selected_total
        if capacity <= 0 or not len(remaining) or len(connected) < minimum_neighbors:
            break
        tree = cKDTree(
            means_m[item.indices[connected]],
            compact_nodes=False,
            balanced_tree=False,
            copy_data=False,
        )
        distance, neighbor = tree.query(
            means_m[item.indices[remaining]],
            k=minimum_neighbors,
            distance_upper_bound=maximum_radius,
            workers=-1,
        )
        distance = np.asarray(distance, dtype=np.float64)
        neighbor = np.asarray(neighbor, dtype=np.int64)
        if minimum_neighbors == 1:
            distance = distance[:, None]
            neighbor = neighbor[:, None]
        valid_neighbor = neighbor < len(connected)
        safe_neighbor = np.where(valid_neighbor, neighbor, 0)
        neighbor_local = connected[safe_neighbor]
        candidate_radius = radius_m[item.indices[remaining]][:, None]
        neighbor_radius = radius_m[item.indices[neighbor_local]]
        connection_limit = np.clip(
            radius_multiplier * (candidate_radius + neighbor_radius),
            minimum_radius,
            maximum_radius,
        )
        valid_edge = valid_neighbor & np.isfinite(distance) & (distance <= connection_limit)
        accepted_mask = np.count_nonzero(valid_edge, axis=1) >= minimum_neighbors
        if not np.any(accepted_mask):
            break

        accepted_local = remaining[accepted_mask]
        accepted_distance = np.min(
            np.where(valid_edge[accepted_mask], distance[accepted_mask], np.inf),
            axis=1,
        ).astype(np.float32)
        order = _rank_growth(accepted_local, item, purity, accepted_distance)
        if len(order) > capacity:
            order = order[:capacity]
        accepted_local = accepted_local[order]
        accepted_distance = accepted_distance[order]
        selected_parts.append(accepted_local)
        step_parts.append(np.full(len(accepted_local), step, dtype=np.uint8))
        distance_parts.append(accepted_distance)
        selected_total += len(accepted_local)
        connected = np.concatenate((connected, accepted_local))
        chosen = np.zeros(len(remaining), dtype=bool)
        accepted_positions = np.flatnonzero(accepted_mask)[order]
        chosen[accepted_positions] = True
        remaining = remaining[~chosen]

    if not selected_parts:
        empty_i = np.zeros(0, dtype=np.int64)
        return empty_i, empty_i.astype(np.uint8), empty_i.astype(np.float32)
    return (
        np.concatenate(selected_parts),
        np.concatenate(step_parts),
        np.concatenate(distance_parts),
    )


def refine_connected_claims(
    means_m: np.ndarray,
    radius_m: np.ndarray,
    evidence: Mapping[int, Any],
    labels: np.ndarray,
    confidence: np.ndarray,
    support: np.ndarray,
    blocked_strong_indices: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Add conservative, build-evidenced rows connected to strong object cores."""

    policy = config["refinement"]
    object_rows: list[dict[str, Any]] = []
    if not bool(policy["enabled"]):
        return labels, confidence, support, {
            "enabled": False,
            "blocked_strong_conflicts": int(len(blocked_strong_indices)),
            "eligible": 0,
            "proposed": 0,
            "added": 0,
            "ambiguous": 0,
            "objects_grown": 0,
            "objects": object_rows,
        }

    proposals: list[dict[str, Any]] = []
    blocked = np.sort(np.asarray(blocked_strong_indices, dtype=np.int64))
    for object_id in sorted(evidence):
        item = evidence[object_id]
        local_labels = labels[item.indices]
        core_local = np.flatnonzero(local_labels == int(object_id)).astype(np.int64)
        purity = item.positive_weight / np.maximum(
            item.positive_weight + item.negative_weight,
            item.positive_weight + 1e-8,
        )
        visible_share = item.positive_weight / np.maximum(
            item.visible_weight, item.positive_weight + 1e-8
        )
        eligible = (
            (local_labels == UNKNOWN_ID)
            & ~_membership_mask(blocked, item.indices)
            & (item.positive_timestamps >= int(policy["minimum_positive_timestamps"]))
            & (item.negative_timestamps <= int(policy["maximum_negative_timestamps"]))
            & (
                item.positive_timestamps.astype(np.int32)
                >= item.negative_timestamps.astype(np.int32)
                + int(policy["minimum_timestamp_margin"])
            )
            & (item.positive_weight >= float(policy["minimum_positive_weight"]))
            & (purity >= float(policy["minimum_global_purity"]))
            & (visible_share >= float(policy["minimum_visible_share"]))
        )
        eligible_local = np.flatnonzero(eligible).astype(np.int64)
        grown_local, grown_step, grown_distance = _connected_growth(
            means_m,
            radius_m,
            item,
            core_local,
            eligible_local,
            purity,
            policy,
        )
        grown_indices = item.indices[grown_local]
        grown_support = item.positive_timestamps[grown_local].astype(np.uint16)
        support_fraction = grown_support.astype(np.float32) / max(
            item.build_timestamp_count, 1
        )
        grown_confidence = np.clip(
            0.55 * purity[grown_local]
            + 0.30 * np.minimum(support_fraction, 1.0)
            + 0.15 * np.minimum(visible_share[grown_local], 1.0),
            0.0,
            1.0,
        ).astype(np.float32)
        scores = grown_support.astype(np.float32) + float(
            policy["score_log_weight"]
        ) * np.log1p(item.positive_weight[grown_local])
        proposals.append({
            "object_id": int(object_id),
            "indices": grown_indices,
            "scores": scores.astype(np.float32),
            "confidence": grown_confidence,
            "support": grown_support,
        })
        object_rows.append({
            "object_id": int(object_id),
            "seed_gaussians": int(len(core_local)),
            "eligible_build_evidence": int(len(eligible_local)),
            "connected_proposals": int(len(grown_indices)),
            "added_gaussians": 0,
            "maximum_growth_step": int(grown_step.max()) if len(grown_step) else 0,
            "median_connection_distance_m": (
                float(np.median(grown_distance)) if len(grown_distance) else None
            ),
        })

    proposal_labels, proposal_confidence, proposal_support, conflict = resolve_sparse_claims(
        proposals,
        len(labels),
        minimum_margin=float(policy["minimum_winner_margin"]),
        minimum_ratio=float(policy["minimum_winner_ratio"]),
    )
    accepted = (proposal_labels >= 0) & (labels == UNKNOWN_ID)
    if np.any(_membership_mask(blocked, np.flatnonzero(accepted))):
        raise AssertionError("refinement attempted to promote a strong unresolved conflict")
    labels[accepted] = proposal_labels[accepted]
    confidence[accepted] = proposal_confidence[accepted]
    support[accepted] = proposal_support[accepted]
    added_ids, added_counts = np.unique(labels[accepted], return_counts=True) if np.any(accepted) else (
        np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)
    )
    added_by_id = {
        int(object_id): int(count)
        for object_id, count in zip(added_ids, added_counts, strict=True)
    }
    for row in object_rows:
        row["added_gaussians"] = added_by_id.get(int(row["object_id"]), 0)
    return labels, confidence, support, {
        "enabled": True,
        "heldout_masks_opened": False,
        "strong_labels_overwritten": 0,
        "blocked_strong_conflicts": int(len(blocked)),
        "eligible": int(sum(row["eligible_build_evidence"] for row in object_rows)),
        "proposed": int(sum(row["connected_proposals"] for row in object_rows)),
        "added": int(np.count_nonzero(accepted)),
        "ambiguous": int(conflict["ambiguous"]),
        "objects_grown": int(sum(row["added_gaussians"] > 0 for row in object_rows)),
        "objects": object_rows,
    }


def apply_provisional_geometry_gate(
    labels: np.ndarray,
    confidence: np.ndarray,
    support: np.ndarray,
    object_rows: Sequence[dict[str, Any]],
    gaussian_count: int,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply final object-size gates after refinement and update build rows."""

    policy = config["build"]
    assigned = labels[labels >= 0]
    ids, counts = np.unique(assigned, return_counts=True) if len(assigned) else (
        np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)
    )
    count_by_id = {
        int(object_id): int(count)
        for object_id, count in zip(ids, counts, strict=True)
    }
    maximum = int(float(policy["maximum_object_fraction"]) * gaussian_count)
    invalid = {
        int(row["object_id"])
        for row in object_rows
        if count_by_id.get(int(row["object_id"]), 0)
        < int(policy["minimum_object_gaussians"])
        or count_by_id.get(int(row["object_id"]), 0) > maximum
    }
    if invalid:
        remove = np.isin(labels, np.asarray(sorted(invalid), dtype=np.int32))
        labels[remove] = UNKNOWN_ID
        confidence[remove] = 0.0
        support[remove] = 0
    for row in object_rows:
        object_id = int(row["object_id"])
        final_count = 0 if object_id in invalid else count_by_id.get(object_id, 0)
        row["provisional_gaussians"] = int(final_count)
        row["geometry_gate"] = "REJECT" if object_id in invalid else (
            "PASS" if final_count else "NO_CLAIMS"
        )
    return labels, confidence, support, {
        "objects_geometry_rejected": int(len(invalid)),
        "minimum_object_gaussians": int(policy["minimum_object_gaussians"]),
        "maximum_object_gaussians": maximum,
        "rejected_object_ids": sorted(invalid),
    }
