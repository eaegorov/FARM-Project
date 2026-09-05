"""Anchor-aware validation of tracker identities across paired folds.

The global tracker association deliberately does not use FARM objects. This
module is a later bounded audit: it asks whether a shared candidate identity
in one object-paired train/heldout episode is supported by the metric voxel
cloud of that same canonical object. It never repairs or rewrites tracker
outputs, and it cannot promote the shadow local gate to a production gate.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "farm.anchor-cross-fold-reconciliation.v1"
ALLOWLIST_SCHEMA = "farm.gaussian-grouping-anchor-allowlist.v1"
_EPISODE_RE = re.compile(
    r"^object-(?P<object_id>[0-9]{6})__(?P<stream>.+)__(?P<split>train|heldout)$"
)


@dataclass(frozen=True)
class AnchorReconciliationPolicy:
    """Conservative, metric thresholds for the bounded audit."""

    minimum_anchor_points: int = 24
    maximum_anchor_distance_m: float = 0.50
    maximum_anchor_distance_scale: float = 0.35
    minimum_identity_to_anchor_diagonal_ratio: float = 0.20
    maximum_identity_to_anchor_diagonal_ratio: float = 2.00
    maximum_cross_fold_centroid_distance_m: float = 0.50
    maximum_cross_fold_centroid_distance_scale: float = 0.35
    maximum_cross_fold_shape_log_rms: float = 0.50
    require_exact_prompt_set_agreement: bool = True

    def validate(self) -> None:
        if int(self.minimum_anchor_points) < 3:
            raise ValueError("minimum_anchor_points must be at least 3")
        for name in (
            "maximum_anchor_distance_m",
            "maximum_anchor_distance_scale",
            "minimum_identity_to_anchor_diagonal_ratio",
            "maximum_identity_to_anchor_diagonal_ratio",
            "maximum_cross_fold_centroid_distance_m",
            "maximum_cross_fold_centroid_distance_scale",
            "maximum_cross_fold_shape_log_rms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            float(self.minimum_identity_to_anchor_diagonal_ratio)
            >= float(self.maximum_identity_to_anchor_diagonal_ratio)
        ):
            raise ValueError("identity/anchor diagonal ratio bounds are inconsistent")


def normalize_prompt(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _finite_vector(value: object, *, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite length-3 vector")
    return result


def robust_anchor_geometry(points_m: object) -> dict[str, Any]:
    """Summarize canonical metric support without reading an OBB."""

    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("anchor points must have shape Nx3")
    if points.shape[0] < 3 or not np.isfinite(points).all():
        raise ValueError("anchor points must contain at least three finite rows")
    center = np.median(points, axis=0)
    centered = points - center[None, :]
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, axes = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    projected = centered @ axes[:, order]
    extents = np.quantile(projected, 0.98, axis=0) - np.quantile(
        projected, 0.02, axis=0
    )
    extents = np.sort(np.maximum(extents, 1.0e-9))[::-1]
    diagonal = float(np.linalg.norm(extents))
    if not math.isfinite(diagonal) or diagonal <= 1.0e-9:
        raise ValueError("anchor support is geometrically degenerate")
    return {
        "point_count": int(points.shape[0]),
        "center_method": "coordinate-median",
        "center_m": center.tolist(),
        "shape_method": "sorted-robust-pca-q02-q98-extents",
        "shape_extents_m": extents.tolist(),
        "shape_diagonal_m": diagonal,
        "obb_used": False,
    }


def _validated_anchor(
    anchor: Mapping[str, Any], policy: AnchorReconciliationPolicy
) -> tuple[np.ndarray, np.ndarray, float]:
    point_count = int(anchor.get("point_count", -1))
    if point_count < int(policy.minimum_anchor_points):
        raise ValueError(
            f"canonical anchor has {point_count} points; "
            f"requires {policy.minimum_anchor_points}"
        )
    center = _finite_vector(anchor.get("center_m"), label="anchor center")
    extents = _finite_vector(
        anchor.get("shape_extents_m"), label="anchor shape extents"
    )
    if np.any(extents <= 0.0):
        raise ValueError("anchor shape extents must be positive")
    extents = np.sort(extents)[::-1]
    diagonal = float(np.linalg.norm(extents))
    if not math.isfinite(diagonal) or diagonal <= 1.0e-9:
        raise ValueError("anchor shape diagonal must be positive")
    return center, extents, diagonal


def _episode_parts(episode_id: object) -> tuple[int, str]:
    value = str(episode_id or "")
    match = _EPISODE_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid object-paired episode ID: {value!r}")
    return int(match.group("object_id")), str(match.group("split"))


def _identity_geometry(
    row: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    center = _finite_vector(row.get("center_m"), label="identity center")
    extents = _finite_vector(row.get("shape_extents_m"), label="identity shape")
    if np.any(extents <= 0.0):
        raise ValueError("identity shape extents must be positive")
    extents = np.sort(extents)[::-1]
    diagonal = float(np.linalg.norm(extents))
    if not math.isfinite(diagonal) or diagonal <= 1.0e-9:
        raise ValueError("identity shape diagonal must be positive")
    return center, extents, diagonal


def _prompt_set(row: Mapping[str, Any]) -> tuple[str, ...]:
    raw = row.get("prompts")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("identity prompts must be a sequence")
    return tuple(
        sorted({normalize_prompt(value) for value in raw if normalize_prompt(value)})
    )


def _individual_anchor_metrics(
    row: Mapping[str, Any],
    *,
    anchor_center: np.ndarray,
    anchor_diagonal: float,
    policy: AnchorReconciliationPolicy,
) -> dict[str, Any]:
    center, extents, diagonal = _identity_geometry(row)
    distance = float(np.linalg.norm(center - anchor_center))
    distance_scale = distance / anchor_diagonal
    diagonal_ratio = diagonal / anchor_diagonal
    checks = {
        "anchor_distance_m": distance <= float(policy.maximum_anchor_distance_m),
        "anchor_distance_scale": distance_scale
        <= float(policy.maximum_anchor_distance_scale),
        "identity_to_anchor_diagonal_ratio_min": diagonal_ratio
        >= float(policy.minimum_identity_to_anchor_diagonal_ratio),
        "identity_to_anchor_diagonal_ratio_max": diagonal_ratio
        <= float(policy.maximum_identity_to_anchor_diagonal_ratio),
    }
    return {
        "center_m": center.tolist(),
        "shape_extents_m": extents.tolist(),
        "shape_diagonal_m": diagonal,
        "anchor_distance_m": distance,
        "anchor_distance_scale": distance_scale,
        "identity_to_anchor_diagonal_ratio": diagonal_ratio,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _pair_metrics(
    train: Mapping[str, Any],
    heldout: Mapping[str, Any],
    *,
    anchor_center: np.ndarray,
    anchor_diagonal: float,
    policy: AnchorReconciliationPolicy,
) -> dict[str, Any]:
    train_anchor = _individual_anchor_metrics(
        train,
        anchor_center=anchor_center,
        anchor_diagonal=anchor_diagonal,
        policy=policy,
    )
    heldout_anchor = _individual_anchor_metrics(
        heldout,
        anchor_center=anchor_center,
        anchor_diagonal=anchor_diagonal,
        policy=policy,
    )
    train_center = np.asarray(train_anchor["center_m"], dtype=np.float64)
    heldout_center = np.asarray(heldout_anchor["center_m"], dtype=np.float64)
    cross_distance = float(np.linalg.norm(train_center - heldout_center))
    cross_distance_scale = cross_distance / anchor_diagonal
    train_shape = np.asarray(train_anchor["shape_extents_m"], dtype=np.float64)
    heldout_shape = np.asarray(heldout_anchor["shape_extents_m"], dtype=np.float64)
    shape_log_rms = float(
        np.sqrt(np.mean(np.square(np.log(train_shape / heldout_shape))))
    )
    train_prompts = _prompt_set(train)
    heldout_prompts = _prompt_set(heldout)
    prompt_agreement = bool(train_prompts) and train_prompts == heldout_prompts
    checks = {
        "train_metric_anchor_support": bool(train_anchor["passed"]),
        "heldout_metric_anchor_support": bool(heldout_anchor["passed"]),
        "cross_fold_centroid_distance_m": cross_distance
        <= float(policy.maximum_cross_fold_centroid_distance_m),
        "cross_fold_centroid_distance_scale": cross_distance_scale
        <= float(policy.maximum_cross_fold_centroid_distance_scale),
        "cross_fold_shape": shape_log_rms
        <= float(policy.maximum_cross_fold_shape_log_rms),
        "prompt_set_agreement": (
            prompt_agreement if policy.require_exact_prompt_set_agreement else True
        ),
    }
    return {
        "train": train_anchor,
        "heldout": heldout_anchor,
        "cross_fold_centroid_distance_m": cross_distance,
        "cross_fold_centroid_distance_scale": cross_distance_scale,
        "cross_fold_shape_log_rms": shape_log_rms,
        "train_prompts": list(train_prompts),
        "heldout_prompts": list(heldout_prompts),
        "checks": checks,
        "failed_checks": sorted(key for key, passed in checks.items() if not passed),
        "passed": all(checks.values()),
    }


def reconcile_object_pair(
    *,
    object_id: int,
    train_episode_id: str,
    heldout_episode_id: str,
    train_physical_timestamps: Sequence[str],
    heldout_physical_timestamps: Sequence[str],
    anchor: Mapping[str, Any],
    identities: Sequence[Mapping[str, Any]],
    policy: AnchorReconciliationPolicy | None = None,
) -> dict[str, Any]:
    """Validate one target pair and return a deterministic fail-closed row."""

    selected_policy = policy or AnchorReconciliationPolicy()
    selected_policy.validate()
    target = int(object_id)
    train_target, train_split = _episode_parts(train_episode_id)
    heldout_target, heldout_split = _episode_parts(heldout_episode_id)
    if (
        train_target != target
        or heldout_target != target
        or train_split != "train"
        or heldout_split != "heldout"
    ):
        raise ValueError("paired episodes do not belong to the same target object/splits")
    train_timestamps = tuple(sorted({str(value) for value in train_physical_timestamps}))
    heldout_timestamps = tuple(
        sorted({str(value) for value in heldout_physical_timestamps})
    )
    if not train_timestamps or not heldout_timestamps:
        raise ValueError("both folds require physical timestamps")
    overlap = sorted(set(train_timestamps).intersection(heldout_timestamps))
    if overlap:
        raise ValueError(
            "train/heldout physical timestamps overlap: " + ", ".join(overlap[:8])
        )
    anchor_center, _, anchor_diagonal = _validated_anchor(anchor, selected_policy)

    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "heldout": []}
    seen_keys: set[str] = set()
    expected_episode = {"train": train_episode_id, "heldout": heldout_episode_id}
    for raw in identities:
        row = dict(raw)
        key = str(row.get("identity_key") or "")
        if not key or key in seen_keys:
            raise ValueError(f"missing or duplicate identity key: {key!r}")
        seen_keys.add(key)
        episode_id = str(row.get("episode_id") or "")
        row_target, row_split = _episode_parts(episode_id)
        if row_target != target or episode_id != expected_episode[row_split]:
            raise ValueError(f"identity belongs outside target pair: {key}")
        if str(row.get("split") or "") != row_split:
            raise ValueError(f"identity split mismatch: {key}")
        if not isinstance(row.get("global_id"), int) or int(row["global_id"]) < 0:
            raise ValueError(f"identity global_id must be non-negative: {key}")
        _identity_geometry(row)
        _prompt_set(row)
        row["passed_shadow_gate"] = bool(row.get("passed_shadow_gate"))
        by_split[row_split].append(row)

    passed = {
        split: [row for row in rows if row["passed_shadow_gate"]]
        for split, rows in by_split.items()
    }
    train_by_gid: dict[int, list[dict[str, Any]]] = {}
    heldout_by_gid: dict[int, list[dict[str, Any]]] = {}
    for row in passed["train"]:
        if int(row["global_id"]) > 0:
            train_by_gid.setdefault(int(row["global_id"]), []).append(row)
    for row in passed["heldout"]:
        if int(row["global_id"]) > 0:
            heldout_by_gid.setdefault(int(row["global_id"]), []).append(row)

    candidate_rows: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    for global_id in sorted(set(train_by_gid).intersection(heldout_by_gid)):
        train_rows = sorted(train_by_gid[global_id], key=lambda row: row["identity_key"])
        heldout_rows = sorted(
            heldout_by_gid[global_id], key=lambda row: row["identity_key"]
        )
        if len(train_rows) != 1 or len(heldout_rows) != 1:
            candidate = {
                "raw_global_id": global_id,
                "train_identity_keys": [row["identity_key"] for row in train_rows],
                "heldout_identity_keys": [row["identity_key"] for row in heldout_rows],
                "passed": False,
                "failed_checks": ["non_unique_member_within_target_fold"],
            }
        else:
            train_row, heldout_row = train_rows[0], heldout_rows[0]
            metrics = _pair_metrics(
                train_row,
                heldout_row,
                anchor_center=anchor_center,
                anchor_diagonal=anchor_diagonal,
                policy=selected_policy,
            )
            candidate = {
                "raw_global_id": global_id,
                "train_identity_key": train_row["identity_key"],
                "heldout_identity_key": heldout_row["identity_key"],
                **metrics,
            }
        candidate_rows.append(candidate)
        if candidate["passed"]:
            eligible_rows.append(candidate)

    # Diagnostic only: find a plausible target pair whose raw IDs differ. This
    # explains #47-like misses but never creates an association.
    near_unreconciled: list[dict[str, Any]] = []
    for train_row in passed["train"]:
        for heldout_row in passed["heldout"]:
            same_nonzero = (
                int(train_row["global_id"]) > 0
                and int(train_row["global_id"]) == int(heldout_row["global_id"])
            )
            if same_nonzero:
                continue
            metrics = _pair_metrics(
                train_row,
                heldout_row,
                anchor_center=anchor_center,
                anchor_diagonal=anchor_diagonal,
                policy=selected_policy,
            )
            if metrics["passed"]:
                near_unreconciled.append(
                    {
                        "train_identity_key": train_row["identity_key"],
                        "heldout_identity_key": heldout_row["identity_key"],
                        "train_raw_global_id": int(train_row["global_id"]),
                        "heldout_raw_global_id": int(heldout_row["global_id"]),
                        **metrics,
                    }
                )

    if len(eligible_rows) == 1:
        accepted = eligible_rows[0]
        status = "accepted"
        reason = "unique_shared_id_with_joint_metric_anchor_support"
        accepted_mapping = {
            "raw_global_id": int(accepted["raw_global_id"]),
            "train_identity_key": str(accepted["train_identity_key"]),
            "heldout_identity_key": str(accepted["heldout_identity_key"]),
        }
    elif len(eligible_rows) > 1:
        status = "ambiguous"
        reason = "multiple_shared_ids_have_joint_metric_anchor_support"
        accepted_mapping = None
    elif near_unreconciled:
        status = "unresolved"
        reason = "metric_anchor_pair_exists_but_raw_cross_fold_id_is_not_shared"
        accepted_mapping = None
    elif candidate_rows:
        status = "rejected"
        reason = "shared_raw_ids_fail_metric_anchor_or_shape_gates"
        accepted_mapping = None
    else:
        status = "unresolved"
        reason = "no_shared_raw_id_within_target_pair"
        accepted_mapping = None

    return {
        "object_id": target,
        "status": status,
        "reason": reason,
        "train_episode_id": train_episode_id,
        "heldout_episode_id": heldout_episode_id,
        "timestamp_independence": {
            "train_count": len(train_timestamps),
            "heldout_count": len(heldout_timestamps),
            "overlap_count": 0,
            "verified": True,
        },
        "anchor": dict(anchor),
        "shadow_passed_identity_count": {
            "train": len(passed["train"]),
            "heldout": len(passed["heldout"]),
        },
        "shared_candidate_count": len(candidate_rows),
        "eligible_shared_candidate_count": len(eligible_rows),
        "candidates": candidate_rows,
        "near_anchor_unreconciled": near_unreconciled,
        "accepted_mapping": accepted_mapping,
    }


def policy_payload(policy: AnchorReconciliationPolicy) -> dict[str, Any]:
    policy.validate()
    return asdict(policy)
