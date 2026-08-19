"""Deterministic final-release policy for FARM presentation objects.

The acceptance layer is deliberately model-free.  It consumes the evidence
already persisted by duplicate resolution and semantic consensus, suppresses
only high-confidence duplicate *presentation* rows, and fails closed when a
residual overlap or label lacks enough structured provenance.  Metric
activation, geometry, features, and mask observations are never modified.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch

from scene_graph.captioning.evidence import evidence_event_id


PAIR_SOURCES = (
    "pair_evidence",
    "residual_strong_candidates",
    "review_only_conflicts",
)
PRESENTATION_MUTABLE_KEYS = frozenset(
    {
        "object_display_status",
        "object_duplicate_canonical_id",
        "object_duplicate_group_ids",
        "object_duplicate_reason",
    }
)
_UNRESOLVED_SENTINELS = frozenset({"", "unknown", "unresolved", "unresolved object"})
_STRUCTURED_FALLBACK_SOURCES = frozenset({"neutral_form_fallback"})
_SHARED_FORM_HYPERNYM_SOURCE = "explicit_shared_form_hypernym_consensus"
_VERIFIED_DISTINCT_DISPOSITIONS = frozenset(
    {"verified_distinct", "manually_verified_distinct"}
)


@dataclass(frozen=True)
class AcceptancePolicy:
    """Versioned, scene-agnostic acceptance thresholds."""

    schema: str = "farm.final-acceptance-policy.v1"
    candidate_multiplier: int = 64
    candidate_absolute_cap: int = 50_000
    max_sample_objects: int = 32
    minimum_sample_objects: int = 16
    max_crop_decode_attempts_per_object: int = 3
    uncertainty_threshold: float = 0.50
    high_uncertainty_threshold: float = 0.75
    uncertainty_rate_warning: float = 0.05


DEFAULT_POLICY = AcceptancePolicy()


def _normalise_category(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower())
    return " ".join(text.split())


def _as_list(value: object) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_int(value: object) -> int | None:
    number = _finite_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def runtime_budget_seconds(pre_qa_wall_seconds: object) -> float:
    """Return the bounded two-percent QA wall-time allowance."""
    wall = _finite_float(pre_qa_wall_seconds)
    if wall is None or wall < 0.0:
        wall = 0.0
    return min(90.0, max(30.0, 0.02 * wall))


def candidate_budget(presentation_count: int, policy: AcceptancePolicy = DEFAULT_POLICY) -> int:
    if presentation_count < 0:
        raise ValueError("presentation_count must be non-negative")
    return min(
        policy.candidate_absolute_cap,
        max(policy.candidate_multiplier, policy.candidate_multiplier * presentation_count),
    )


def presentation_indices(state: Mapping[str, Any]) -> tuple[list[int], list[str]]:
    """Return visible row indices using the canonical export predicate."""
    object_ids = _as_list(state.get("object_id"))
    size = len(object_ids)
    errors: list[str] = []
    if size == 0:
        return [], ["scene_state_has_no_object_ids"]

    active = _as_list(state.get("active"))
    if len(active) != size:
        errors.append("scene_state_active_not_row_aligned")
        active = [False] * size
    tiers = _as_list(state.get("object_semantic_tier"))
    if len(tiers) != size:
        errors.append("scene_state_semantic_tier_not_row_aligned")
        tiers = [""] * size
    statuses = _as_list(state.get("object_display_status"))
    if len(statuses) != size:
        errors.append("scene_state_display_status_not_row_aligned")
        statuses = ["canonical"] * size
    geometry = _as_list(state.get("object_geometry_status"))
    if len(geometry) != size:
        geometry = [""] * size
    compound_boxes = _as_list(state.get("object_compound_boxes"))
    if len(compound_boxes) != size:
        compound_boxes = [[] for _ in range(size)]

    visible: list[int] = []
    for index in range(size):
        tier = str(tiers[index] or "").strip().lower()
        display = str(statuses[index] or "").strip().lower()
        geometry_status = str(geometry[index] or "").strip().lower()
        boxes = compound_boxes[index]
        probable_compound = (
            "compound_geometry_probable" in geometry_status
            and "rejected" not in geometry_status
            and isinstance(boxes, (list, tuple))
            and len(boxes) == 2
        )
        if not display.endswith("_suppressed") and (
            (bool(active[index]) and tier in {"confirmed", "probable"})
            or probable_compound
        ):
            visible.append(index)
    return visible, errors


def presentation_ids(state: Mapping[str, Any]) -> tuple[set[int], list[str]]:
    indices, errors = presentation_indices(state)
    ids = _as_list(state.get("object_id"))
    result: set[int] = set()
    for index in indices:
        try:
            result.add(int(ids[index]))
        except (TypeError, ValueError):
            errors.append("scene_state_object_id_not_integer")
    if len(result) != len(indices):
        errors.append("scene_state_presentation_ids_not_unique")
    return result, sorted(set(errors))


def _raw_candidate_count(audit: Mapping[str, Any]) -> int:
    return sum(
        len(value) if isinstance(value := audit.get(source), list) else 0
        for source in PAIR_SOURCES
    )


def merge_pair_evidence(
    audit: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Union all persisted pair collections without trusting array order."""
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    errors: list[str] = []
    maximum_fields = {
        "feature_cosine",
        "first_inside_second",
        "second_inside_first",
        "first_near_second",
        "second_near_first",
        "common_frames",
        "compared_mask_frames",
        "mask_iou",
        "mask_containment",
        "mask_area_ratio",
    }
    minimum_fields = {"center_distance_m"}
    boolean_fields = {
        "semantic_compatible",
        "cannot_link",
        "co_visible",
        "three_d_corroborated",
    }

    for source in PAIR_SOURCES:
        rows = audit.get(source)
        if rows is None:
            continue
        if not isinstance(rows, list):
            errors.append(f"dedup_{source}_not_array")
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                errors.append("invalid_candidate_record")
                continue
            try:
                first_id = int(row["first_id"])
                second_id = int(row["second_id"])
            except (KeyError, TypeError, ValueError):
                errors.append("invalid_candidate_record")
                continue
            if first_id == second_id:
                errors.append("candidate_self_pair")
                continue
            first_id, second_id = sorted((first_id, second_id))
            record = pairs.setdefault(
                (first_id, second_id),
                {
                    "first_id": first_id,
                    "second_id": second_id,
                    "sources": set(),
                    "raw_relations": set(),
                    "dispositions": set(),
                    "reasons": set(),
                    "mask_image_ids": set(),
                },
            )
            record["sources"].add(source)
            for key in maximum_fields:
                number = _finite_float(row.get(key))
                if number is not None:
                    previous = _finite_float(record.get(key))
                    record[key] = number if previous is None else max(previous, number)
            for key in minimum_fields:
                number = _finite_float(row.get(key))
                if number is not None:
                    previous = _finite_float(record.get(key))
                    record[key] = number if previous is None else min(previous, number)
            for key in boolean_fields:
                if row.get(key) is True:
                    record[key] = True
                elif key not in record and row.get(key) is False:
                    record[key] = False
            relation = str(row.get("raw_relation") or row.get("relation") or "").strip().lower()
            if relation:
                record["raw_relations"].add(relation)
            disposition = str(row.get("disposition") or "").strip().lower()
            if disposition:
                record["dispositions"].add(disposition)
            reason = str(row.get("reason") or row.get("rejection_reason") or "").strip()
            if reason:
                record["reasons"].add(reason)
            image_id = _finite_int(row.get("mask_image_id"))
            if image_id is not None:
                record["mask_image_ids"].add(image_id)

    result: list[dict[str, Any]] = []
    for key in sorted(pairs):
        record = pairs[key]
        for field in (
            "sources",
            "raw_relations",
            "dispositions",
            "reasons",
            "mask_image_ids",
        ):
            record[field] = sorted(record[field])
        result.append(record)
    return result, sorted(set(errors))


def _state_object_maps(
    state: Mapping[str, Any],
) -> tuple[dict[int, int], dict[int, str], dict[int, float], list[str]]:
    ids = _as_list(state.get("object_id"))
    categories = _as_list(state.get("object_category"))
    dimensions = _as_list(state.get("object_box_dimensions_m"))
    errors: list[str] = []
    if len(categories) != len(ids):
        errors.append("scene_state_category_not_row_aligned")
        categories = [""] * len(ids)
    if len(dimensions) != len(ids):
        errors.append("scene_state_dimensions_not_row_aligned")
        dimensions = [[] for _ in ids]
    index_by_id: dict[int, int] = {}
    category_by_id: dict[int, str] = {}
    diagonal_by_id: dict[int, float] = {}
    for index, raw_id in enumerate(ids):
        try:
            object_id = int(raw_id)
        except (TypeError, ValueError):
            errors.append("scene_state_object_id_not_integer")
            continue
        if object_id in index_by_id:
            errors.append("scene_state_object_ids_not_unique")
            continue
        index_by_id[object_id] = index
        category_by_id[object_id] = _normalise_category(categories[index])
        try:
            values = [float(value) for value in dimensions[index]]
        except (TypeError, ValueError):
            values = []
        if len(values) == 3 and all(math.isfinite(value) and value > 0.0 for value in values):
            diagonal_by_id[object_id] = math.sqrt(sum(value * value for value in values))
    return index_by_id, category_by_id, diagonal_by_id, sorted(set(errors))


def classify_pair_records(
    records: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
    visible_ids: set[int],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Classify merged evidence into disjoint release dispositions."""
    _, categories, diagonals, errors = _state_object_maps(state)
    classified: list[dict[str, Any]] = []
    for source in records:
        row = dict(source)
        first_id = int(row["first_id"])
        second_id = int(row["second_id"])
        row["first_category"] = categories.get(first_id, "")
        row["second_category"] = categories.get(second_id, "")
        if first_id not in visible_ids or second_id not in visible_ids:
            row.update(
                {
                    "classification": "hidden_context",
                    "matched_rule": "endpoint_not_presentation_visible",
                    "normalized_center_distance": None,
                }
            )
            classified.append(row)
            continue

        first_diagonal = diagonals.get(first_id)
        second_diagonal = diagonals.get(second_id)
        center_distance = _finite_float(row.get("center_distance_m"))
        if first_diagonal is None or second_diagonal is None or center_distance is None:
            errors.append("visible_candidate_missing_finite_geometry")
            normalized_center = None
        else:
            normalized_center = center_distance / min(first_diagonal, second_diagonal)
        row["normalized_center_distance"] = normalized_center

        feature = _finite_float(row.get("feature_cosine")) or 0.0
        iou = _finite_float(row.get("mask_iou")) or 0.0
        containment = _finite_float(row.get("mask_containment")) or 0.0
        area_ratio = _finite_float(row.get("mask_area_ratio")) or 0.0
        near = max(
            _finite_float(row.get("first_near_second")) or 0.0,
            _finite_float(row.get("second_near_first")) or 0.0,
        )
        relations = {str(value).lower() for value in row.get("raw_relations") or []}
        dispositions = {str(value).lower() for value in row.get("dispositions") or []}
        raw_duplicate = any(value.startswith("duplicate") for value in relations)
        same_category = bool(
            row["first_category"]
            and row["first_category"] not in _UNRESOLVED_SENTINELS
            and row["first_category"] == row["second_category"]
        )
        verified_distinct = bool(dispositions & _VERIFIED_DISTINCT_DISPOSITIONS)
        finite_center = normalized_center is not None

        rule = ""
        if not verified_distinct and finite_center:
            if (
                raw_duplicate
                and feature >= 0.95
                and iou >= 0.70
                and containment >= 0.98
                and area_ratio >= 0.70
                and normalized_center <= 0.30
                and near >= 0.90
            ):
                rule = "resolver_duplicate_with_mask_geometry"
            elif (
                same_category
                and feature >= 0.95
                and iou >= 0.68
                and containment >= 0.93
                and area_ratio >= 0.68
                and normalized_center <= 0.30
                and near >= 0.95
            ):
                rule = "same_category_mask_geometry_consensus"
            elif (
                feature >= 0.90
                and iou >= 0.82
                and containment >= 0.965
                and area_ratio >= 0.85
                and normalized_center <= 0.35
                and near >= 0.95
            ):
                rule = "overwhelming_label_independent_overlap"

        ambiguity_support = (
            feature >= 0.90,
            finite_center and normalized_center <= 0.50,
            near >= 0.85,
            containment >= 0.90,
            iou >= 0.55,
            area_ratio >= 0.60,
        )
        ambiguous = (
            verified_distinct and rule != ""
        ) or (
            all(ambiguity_support[:4]) and (ambiguity_support[4] or ambiguity_support[5])
        ) or (
            raw_duplicate and sum(bool(value) for value in ambiguity_support) >= 4
        )

        if rule and not verified_distinct:
            classification = "auto_holdout"
            matched_rule = rule
        elif ambiguous or verified_distinct:
            classification = "release_blocking_ambiguous"
            matched_rule = (
                "verified_distinct_conflicts_with_duplicate_evidence"
                if verified_distinct else "strong_overlap_below_auto_threshold"
            )
        else:
            classification = "similar_distinct"
            matched_rule = "insufficient_duplicate_evidence"
        row.update(
            {
                "classification": classification,
                "matched_rule": matched_rule,
                "same_normalized_category": same_category,
                "max_near_fraction": near,
            }
        )
        classified.append(row)
    return classified, sorted(set(errors))


def _indexed_scalar(state: Mapping[str, Any], key: str, index: int) -> float:
    values = state.get(key)
    if isinstance(values, torch.Tensor) and values.ndim > 0 and index < int(values.shape[0]):
        number = _finite_float(values[index].item())
        return number or 0.0
    sequence = _as_list(values)
    if index < len(sequence):
        number = _finite_float(sequence[index])
        return number or 0.0
    return 0.0


def canonical_rank(state: Mapping[str, Any], index: int) -> tuple[float, int]:
    """Match the resolver's evidence-first canonical ranking and stable tie-break."""
    ids = _as_list(state.get("object_id"))
    tiers = _as_list(state.get("object_semantic_tier"))
    categories = _as_list(state.get("object_category"))
    tier = str(tiers[index] if index < len(tiers) else "").strip().lower()
    tier_score = {"confirmed": 3.0, "probable": 2.0, "geometry_only": 0.5}.get(tier, 0.0)
    category = _normalise_category(categories[index] if index < len(categories) else "")
    resolved = category not in _UNRESOLVED_SENTINELS
    score = (
        10.0 * tier_score
        + (3.0 if resolved else 0.0)
        + 2.0 * math.log1p(max(_indexed_scalar(state, "object_evidence_count", index), 0.0))
        + _indexed_scalar(state, "object_geometry_projected_box_iou", index)
        + _indexed_scalar(state, "object_geometry_inside_rate", index)
        + _indexed_scalar(state, "object_geometry_voxel_inside_rate", index)
    )
    object_id = int(ids[index])
    return score, -object_id


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, value: int) -> int:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, first: int, second: int) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[max(first_root, second_root)] = min(first_root, second_root)


def build_overlap_clusters(
    classified: Sequence[Mapping[str, Any]], state: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Build order-invariant components from auto and blocking overlap edges."""
    relevant = [
        row
        for row in classified
        if row.get("classification") in {"auto_holdout", "release_blocking_ambiguous"}
    ]
    dsu = _DisjointSet()
    for row in relevant:
        dsu.union(int(row["first_id"]), int(row["second_id"]))
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in relevant:
        grouped[dsu.find(int(row["first_id"]))].append(row)

    index_by_id, _, _, _ = _state_object_maps(state)
    clusters: list[dict[str, Any]] = []
    for root in sorted(grouped):
        edges = sorted(
            grouped[root], key=lambda row: (int(row["first_id"]), int(row["second_id"]))
        )
        members = sorted(
            {
                int(value)
                for row in edges
                for value in (row["first_id"], row["second_id"])
            }
        )
        blocking = any(
            row.get("classification") == "release_blocking_ambiguous" for row in edges
        )
        canonical_id: int | None = None
        suppressed_ids: list[int] = []
        canonical_score: float | None = None
        if not blocking:
            canonical_id = max(
                members,
                key=lambda object_id: canonical_rank(state, index_by_id[object_id]),
            )
            canonical_score = canonical_rank(state, index_by_id[canonical_id])[0]
            suppressed_ids = [value for value in members if value != canonical_id]
        clusters.append(
            {
                "cluster_id": len(clusters),
                "classification": (
                    "release_blocking_ambiguous" if blocking else "auto_holdout"
                ),
                "member_ids": members,
                "canonical_id": canonical_id,
                "suppressed_ids": suppressed_ids,
                "canonical_score": canonical_score,
                "edge_count": len(edges),
                "matched_rules": sorted({str(row.get("matched_rule") or "") for row in edges}),
                "edges": [dict(row) for row in edges],
            }
        )
    return clusters


def apply_presentation_holdouts(
    state: Mapping[str, Any], clusters: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Clone and alter only presentation metadata for auto-holdout clusters."""
    output = dict(state)
    ids = _as_list(state.get("object_id"))
    size = len(ids)
    index_by_id = {int(object_id): index for index, object_id in enumerate(ids)}

    original_status = _as_list(state.get("object_display_status"))
    statuses = list(original_status) if len(original_status) == size else ["canonical"] * size
    original_groups = _as_list(state.get("object_duplicate_group_ids"))
    groups = list(original_groups) if len(original_groups) == size else [[] for _ in range(size)]
    original_reasons = _as_list(state.get("object_duplicate_reason"))
    reasons = list(original_reasons) if len(original_reasons) == size else [""] * size
    original_canonical = state.get("object_duplicate_canonical_id")
    if isinstance(original_canonical, torch.Tensor) and original_canonical.numel() == size:
        canonical = original_canonical.detach().clone().reshape(-1)
    else:
        values = _as_list(original_canonical)
        if len(values) != size:
            values = [-1] * size
        canonical = torch.tensor(values, dtype=torch.int64)

    holdouts: list[dict[str, Any]] = []
    for cluster in sorted(clusters, key=lambda row: tuple(row.get("member_ids") or [])):
        if cluster.get("classification") != "auto_holdout":
            continue
        canonical_id = int(cluster["canonical_id"])
        member_ids = sorted(int(value) for value in cluster.get("member_ids") or [])
        rule = "+".join(str(value) for value in cluster.get("matched_rules") or [])
        for object_id in member_ids:
            index = index_by_id[object_id]
            groups[index] = member_ids
            if object_id == canonical_id:
                continue
            statuses[index] = "acceptance_duplicate_suppressed"
            canonical[index] = canonical_id
            reasons[index] = f"acceptance:{rule}"
            holdouts.append(
                {
                    "id": object_id,
                    "canonical_id": canonical_id,
                    "reason": reasons[index],
                }
            )

    output["object_display_status"] = statuses
    output["object_duplicate_canonical_id"] = canonical
    output["object_duplicate_group_ids"] = groups
    output["object_duplicate_reason"] = reasons
    return output, sorted(holdouts, key=lambda row: int(row["id"]))


def _eligible_candidate_votes(row: Mapping[str, Any]) -> tuple[dict[str, float], set[str]]:
    votes: dict[str, float] = defaultdict(float)
    structured_support: set[str] = set()
    seen_fingerprints: set[str] = set()
    candidates = [
        candidate
        for candidate in row.get("candidates") or []
        if isinstance(candidate, Mapping)
    ]
    by_event_id = {
        evidence_event_id(candidate): candidate
        for candidate in candidates
        if evidence_event_id(candidate)
    }
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        category = _normalise_category(candidate.get("category"))
        if not category or str(candidate.get("decision") or "").strip().lower() != "keep":
            continue
        source = str(candidate.get("source") or "").strip().lower()
        shared_form_supported = False
        if (
            source == _SHARED_FORM_HYPERNYM_SOURCE
            and candidate.get("confirmation_eligible") is False
        ):
            contract = candidate.get("label_contract") or {}
            raw_ids = candidate.get("derived_from_event_ids")
            contributor_ids = {
                str(value).strip()
                for value in raw_ids or []
                if str(value).strip()
            } if isinstance(raw_ids, list) else set()
            contributors = [
                by_event_id[event_id]
                for event_id in sorted(contributor_ids)
                if event_id in by_event_id
            ]
            contributor_nouns = {
                _normalise_category(
                    (value.get("label_contract") or {}).get("category")
                )
                for value in contributors
            }
            shared_form_supported = bool(
                len(contributor_ids) >= 2
                and len(contributors) == len(contributor_ids)
                and len(contributor_nouns) >= 2
                and _normalise_category(contract.get("category")) == category
                and str(contract.get("category_role") or "") == "whole_form"
                and str(contract.get("specificity") or "") == "generic_form"
                and str(candidate.get("maximum_semantic_tier") or "") == "probable"
                and all(
                    _normalise_category(
                        (value.get("label_contract") or {}).get("form_hypernym")
                    ) == category
                    for value in contributors
                )
            )
        if (
            candidate.get("confirmation_eligible") is True
            or source in _STRUCTURED_FALLBACK_SOURCES
            or shared_form_supported
        ):
            structured_support.add(category)
        if candidate.get("confirmation_eligible") is not True:
            continue
        fingerprint = str(
            candidate.get("evidence_fingerprint_sha256")
            or candidate.get("event_id")
            or ""
        ).strip()
        if not fingerprint or fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        confidence = _finite_float(candidate.get("confidence"))
        votes[category] += min(1.0, max(0.0, confidence if confidence is not None else 0.0))
    return dict(votes), structured_support


def _recovery_contract(row: Mapping[str, Any], final_category: str) -> tuple[bool, list[str]]:
    assessment = row.get("physical_form_recovery_assessment") or {}
    if not isinstance(assessment, Mapping) or not assessment.get("triggered"):
        return False, []
    guard = row.get("physical_form_recovery") or {}
    guard = guard if isinstance(guard, Mapping) else {}
    metric = assessment.get("metric_physical_assessment") or {}
    metric = metric if isinstance(metric, Mapping) else {}
    errors: list[str] = []
    if assessment.get("accepted") is not True:
        errors.append("rejected_recovery_visible")
    if str(assessment.get("action") or "") != "recovered_probable":
        errors.append("invalid_recovery_action")
    if _normalise_category(assessment.get("category")) != final_category:
        errors.append("recovery_category_mismatch")
    if str(assessment.get("topology") or "").strip().lower() != "standalone":
        errors.append("recovery_not_standalone")
    if assessment.get("complete_bounded") is not True:
        errors.append("recovery_not_complete_bounded")
    if str(assessment.get("maximum_semantic_tier") or "").strip().lower() != "probable":
        errors.append("recovery_tier_cap_invalid")
    if metric.get("hard_veto") is True:
        errors.append("recovery_metric_hard_veto")
    if _normalise_category(guard.get("visible_form_category")) != final_category:
        errors.append("recovery_visible_form_mismatch")
    if str(guard.get("topology") or "").strip().lower() != "standalone":
        errors.append("recovery_guard_not_standalone")
    if guard.get("complete_bounded") is not True:
        errors.append("recovery_guard_not_complete_bounded")
    if str(guard.get("carrier_category") or "").strip():
        errors.append("recovery_carrier_payload_conflict")
    payload = guard.get("payload_categories") or []
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)) and len(payload) > 0:
        errors.append("recovery_carrier_payload_conflict")
    return not errors, sorted(set(errors))


def _semantic_contract_errors(row: Mapping[str, Any], tier: str) -> list[str]:
    groups = _finite_int(row.get("semantic_independent_group_count"))
    views = _finite_int(row.get("semantic_unique_view_count"))
    overlap = _finite_float(row.get("semantic_max_support_overlap"))
    if tier == "confirmed":
        if not (
            row.get("semantic_confirmation_eligible") is True
            and groups is not None
            and groups >= 2
            and views is not None
            and views >= 6
            and overlap is not None
            and 0.0 <= overlap <= 0.25
        ):
            return ["confirmed_independence_contract_violation"]
    elif tier == "probable":
        if not (
            groups is not None
            and groups >= 1
            and views is not None
            and views >= 3
            and (overlap is None or 0.0 <= overlap <= 1.0)
        ):
            return ["probable_support_contract_violation"]
    else:
        return ["invalid_presentation_semantic_tier"]
    return []


def assess_label_quality(
    state: Mapping[str, Any],
    semantic_catalog: Sequence[Mapping[str, Any]],
    visible_ids: set[int],
    decodable_crop_counts: Mapping[int, int | None] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Assess every visible label using structured provenance only."""
    index_by_id, _, _, state_errors = _state_object_maps(state)
    semantic_by_id: dict[int, Mapping[str, Any]] = {}
    errors = list(state_errors)
    for row in semantic_catalog:
        if not isinstance(row, Mapping) or row.get("id") is None:
            errors.append("invalid_semantic_catalog_row")
            continue
        try:
            object_id = int(row["id"])
        except (TypeError, ValueError):
            errors.append("invalid_semantic_catalog_row")
            continue
        if object_id in semantic_by_id:
            errors.append("duplicate_semantic_catalog_id")
        semantic_by_id[object_id] = row

    categories = _as_list(state.get("object_category"))
    tiers = _as_list(state.get("object_semantic_tier"))
    dimensions = _as_list(state.get("object_box_dimensions_m"))
    assessments: list[dict[str, Any]] = []
    for object_id in sorted(visible_ids):
        index = index_by_id.get(object_id)
        if index is None:
            errors.append("presentation_id_missing_from_state")
            continue
        row = semantic_by_id.get(object_id)
        category = _normalise_category(categories[index] if index < len(categories) else "")
        tier = str(tiers[index] if index < len(tiers) else "").strip().lower()
        hard: list[str] = []
        reasons: list[str] = []
        if row is None:
            hard.append("missing_semantic_catalog_row")
            row = {}
        if category in _UNRESOLVED_SENTINELS:
            hard.append("unresolved_presentation_label")
        hard.extend(_semantic_contract_errors(row, tier))
        physical = row.get("physical_form_assessment") or {}
        if isinstance(physical, Mapping) and physical.get("hard_veto") is True:
            hard.append("physical_form_hard_veto_visible")

        recovery_supported, recovery_errors = _recovery_contract(row, category)
        hard.extend(recovery_errors)
        votes, structured_support = _eligible_candidate_votes(row)
        if recovery_supported:
            structured_support.add(category)
        if category not in structured_support:
            hard.append("presentation_label_has_no_structured_support")

        crop_count = None
        if decodable_crop_counts is not None:
            crop_count = decodable_crop_counts.get(object_id)
            if crop_count is None or int(crop_count) <= 0:
                hard.append("presentation_label_has_no_decodable_crop")
            elif int(crop_count) < 3:
                reasons.append("sparse_decodable_crop_evidence")

        group_count = _finite_int(row.get("semantic_independent_group_count"))
        unique_views = _finite_int(row.get("semantic_unique_view_count"))
        max_overlap = _finite_float(row.get("semantic_max_support_overlap"))
        total_vote = sum(votes.values())
        final_vote = votes.get(category, 0.0)
        vote_share = final_vote / total_vote if total_vote > 0.0 else None
        strongest_category = ""
        if votes:
            strongest_category = min(
                votes,
                key=lambda value: (-votes[value], value),
            )

        score = 0.0
        if tier == "probable":
            score += 0.20
            reasons.append("probable_semantic_tier")
        minimum_groups = 2 if tier == "confirmed" else 1
        if group_count == minimum_groups:
            score += 0.15
            reasons.append("support_at_minimum_group_count")
        view_margin = 9 if tier == "confirmed" else 6
        if unique_views is not None and unique_views < view_margin:
            score += 0.15
            reasons.append("support_below_view_margin")
        if vote_share is None:
            score += 0.25
            reasons.append("no_confirmation_eligible_category_vote")
        elif vote_share < 0.67:
            score += 0.25
            reasons.append("independent_category_vote_disagreement")
        if max_overlap is not None and max_overlap > 0.25:
            score += 0.15
            reasons.append("support_view_overlap_above_confirmation_limit")
        if recovery_supported:
            score += 0.10
            reasons.append("conservative_visible_form_recovery")
        score = min(1.0, score)

        volume = None
        if index < len(dimensions):
            try:
                values = [float(value) for value in dimensions[index]]
                if len(values) == 3 and all(math.isfinite(value) and value > 0 for value in values):
                    volume = math.prod(values)
            except (TypeError, ValueError):
                pass
        assessments.append(
            {
                "id": object_id,
                "category": category,
                "semantic_tier": tier,
                "hard_error_codes": sorted(set(hard)),
                "uncertainty_score": round(score, 6),
                "uncertainty_reasons": sorted(set(reasons)),
                "independent_group_count": group_count,
                "unique_view_count": unique_views,
                "max_support_overlap": max_overlap,
                "eligible_vote_share": vote_share,
                "strongest_eligible_category": strongest_category,
                "decodable_crop_count": crop_count,
                "metric_volume_m3": volume,
                "recovery_supported": recovery_supported,
            }
        )
    return assessments, sorted(set(errors))


def deterministic_label_sample(
    assessments: Sequence[Mapping[str, Any]],
    scene_id: str,
    policy: AcceptancePolicy = DEFAULT_POLICY,
) -> list[dict[str, Any]]:
    """Choose a stable, uncertainty-first sample with coarse strata coverage."""
    if not assessments:
        return []
    target = min(
        len(assessments),
        policy.max_sample_objects,
        max(policy.minimum_sample_objects, 2 * math.ceil(math.sqrt(len(assessments)))),
    )
    category_counts = Counter(str(row.get("category") or "") for row in assessments)
    volumes = sorted(
        float(row["metric_volume_m3"])
        for row in assessments
        if _finite_float(row.get("metric_volume_m3")) is not None
    )
    low = volumes[len(volumes) // 3] if volumes else 0.0
    high = volumes[(2 * len(volumes)) // 3] if volumes else 0.0

    def stable_hash(row: Mapping[str, Any]) -> str:
        return hashlib.sha256(f"{scene_id}:{int(row['id'])}".encode("utf-8")).hexdigest()

    def bucket(row: Mapping[str, Any]) -> tuple[str, str, str]:
        frequency = "rare" if category_counts[str(row.get("category") or "")] == 1 else "common"
        volume = _finite_float(row.get("metric_volume_m3"))
        size = "unknown"
        if volume is not None:
            size = "small" if volume <= low else "large" if volume >= high else "medium"
        return str(row.get("semantic_tier") or ""), frequency, size

    ordered = sorted(
        assessments,
        key=lambda row: (
            0 if row.get("hard_error_codes") else 1,
            -float(row.get("uncertainty_score") or 0.0),
            stable_hash(row),
        ),
    )
    selected: list[Mapping[str, Any]] = []
    seen_ids: set[int] = set()
    seen_buckets: set[tuple[str, str, str]] = set()
    for row in ordered:
        key = bucket(row)
        if key in seen_buckets:
            continue
        selected.append(row)
        seen_ids.add(int(row["id"]))
        seen_buckets.add(key)
        if len(selected) == target:
            break
    for row in ordered:
        if len(selected) == target:
            break
        if int(row["id"]) not in seen_ids:
            selected.append(row)
            seen_ids.add(int(row["id"]))
    return [dict(row) for row in selected]


def _classification_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("classification") or "") for row in rows)
    return {
        key: int(counts.get(key, 0))
        for key in (
            "auto_holdout",
            "release_blocking_ambiguous",
            "similar_distinct",
            "hidden_context",
        )
    }


def _status(errors: Sequence[Mapping[str, Any]], warnings: Sequence[Mapping[str, Any]]) -> str:
    if errors:
        return "FAIL"
    if warnings:
        return "WARN"
    return "PASS"


def run_acceptance_policy(
    state: Mapping[str, Any],
    dedup_audit: Mapping[str, Any],
    semantic_catalog: Sequence[Mapping[str, Any]],
    *,
    scene_id: str,
    decodable_crop_counts: Mapping[int, int | None] | None,
    pre_qa_wall_seconds: object = None,
    elapsed_seconds: float = 0.0,
    apply_holdouts: bool = True,
    policy: AcceptancePolicy = DEFAULT_POLICY,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply one presentation-only holdout pass and compute release status."""
    visible_before, visibility_errors = presentation_ids(state)
    budget = candidate_budget(len(visible_before), policy)
    raw_candidates = _raw_candidate_count(dedup_audit)
    structural_codes = list(visibility_errors)
    records: list[dict[str, Any]] = []
    classified_before: list[dict[str, Any]] = []
    clusters_before: list[dict[str, Any]] = []
    output_state = dict(state)
    holdouts: list[dict[str, Any]] = []

    if raw_candidates > budget:
        structural_codes.append("acceptance_candidate_budget_exhausted")
    else:
        records, merge_errors = merge_pair_evidence(dedup_audit)
        structural_codes.extend(merge_errors)
        classified_before, pair_errors = classify_pair_records(records, state, visible_before)
        structural_codes.extend(pair_errors)
        clusters_before = build_overlap_clusters(classified_before, state)
        if apply_holdouts and not structural_codes:
            output_state, holdouts = apply_presentation_holdouts(state, clusters_before)

    visible_after, after_visibility_errors = presentation_ids(output_state)
    structural_codes.extend(after_visibility_errors)
    classified_after: list[dict[str, Any]] = []
    clusters_after: list[dict[str, Any]] = []
    if not structural_codes or records:
        classified_after, pair_errors = classify_pair_records(records, output_state, visible_after)
        structural_codes.extend(pair_errors)
        clusters_after = build_overlap_clusters(classified_after, output_state)

    labels, label_schema_errors = assess_label_quality(
        output_state,
        semantic_catalog,
        visible_after,
        decodable_crop_counts,
    )
    structural_codes.extend(label_schema_errors)
    label_hard_errors = [
        {"id": int(row["id"]), "codes": list(row.get("hard_error_codes") or [])}
        for row in labels
        if row.get("hard_error_codes")
    ]
    remaining_auto = [
        row for row in clusters_after if row.get("classification") == "auto_holdout"
    ]
    remaining_blockers = [
        row
        for row in clusters_after
        if row.get("classification") == "release_blocking_ambiguous"
    ]

    errors: list[dict[str, Any]] = [
        {"code": code} for code in sorted(set(structural_codes))
    ]
    if remaining_auto:
        errors.append(
            {
                "code": "auto_duplicate_cluster_remains_presentation_visible",
                "count": len(remaining_auto),
            }
        )
    if remaining_blockers:
        errors.append(
            {
                "code": "release_blocking_ambiguous_overlap_cluster",
                "count": len(remaining_blockers),
            }
        )
    if label_hard_errors:
        errors.append(
            {
                "code": "presentation_label_hard_errors",
                "count": len(label_hard_errors),
                "objects": label_hard_errors,
            }
        )

    warnings: list[dict[str, Any]] = []
    high_uncertainty = [
        row
        for row in labels
        if float(row.get("uncertainty_score") or 0.0) >= policy.high_uncertainty_threshold
    ]
    uncertain = [
        row
        for row in labels
        if float(row.get("uncertainty_score") or 0.0) >= policy.uncertainty_threshold
    ]
    uncertain_rate = len(uncertain) / len(labels) if labels else 0.0
    if high_uncertainty:
        warnings.append(
            {"code": "high_label_uncertainty", "count": len(high_uncertainty)}
        )
    elif uncertain_rate > policy.uncertainty_rate_warning:
        warnings.append(
            {
                "code": "label_uncertainty_rate_above_threshold",
                "count": len(uncertain),
                "rate": uncertain_rate,
                "threshold": policy.uncertainty_rate_warning,
            }
        )
    sparse_crops = [
        row
        for row in labels
        if row.get("decodable_crop_count") is not None
        and 0 < int(row["decodable_crop_count"]) < 3
    ]
    if sparse_crops:
        warnings.append(
            {"code": "sparse_presentation_crop_evidence", "count": len(sparse_crops)}
        )
    wall_budget = runtime_budget_seconds(pre_qa_wall_seconds)
    if elapsed_seconds > wall_budget:
        warnings.append(
            {
                "code": "acceptance_runtime_budget_exceeded",
                "elapsed_seconds": elapsed_seconds,
                "budget_seconds": wall_budget,
            }
        )

    report = {
        "schema": "farm.final-acceptance.v1",
        "scene_id": scene_id,
        "status": _status(errors, warnings),
        "policy": asdict(policy),
        "counts": {
            "presentation_before": len(visible_before),
            "presentation_after": len(visible_after),
            "raw_candidate_records": raw_candidates,
            "merged_candidate_pairs": len(records),
            "auto_holdout_clusters_initial": sum(
                row.get("classification") == "auto_holdout" for row in clusters_before
            ),
            "blocking_clusters_initial": sum(
                row.get("classification") == "release_blocking_ambiguous"
                for row in clusters_before
            ),
            "held_objects": len(holdouts),
            "remaining_auto_clusters": len(remaining_auto),
            "remaining_blocking_clusters": len(remaining_blockers),
            "label_hard_errors": len(label_hard_errors),
            "uncertain_labels": len(uncertain),
            "high_uncertainty_labels": len(high_uncertainty),
        },
        "candidate_budget": {
            "limit": budget,
            "observed": raw_candidates,
            "complete": raw_candidates <= budget,
        },
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "budget_seconds": wall_budget,
            "budget_exceeded": elapsed_seconds > wall_budget,
            "gpu_required": False,
            "model_calls": 0,
            "automatic_holdout_passes": (
                1
                if apply_holdouts
                and raw_candidates <= budget
                and not structural_codes
                else 0
            ),
        },
        "classification_before": _classification_counts(classified_before),
        "classification_after": _classification_counts(classified_after),
        "holdouts": holdouts,
        "errors": errors,
        "warnings": warnings,
        "label_sample_ids": [
            int(row["id"])
            for row in deterministic_label_sample(labels, scene_id, policy)
        ],
    }
    cluster_payload = [dict(row) for row in clusters_before]
    return output_state, report, cluster_payload, labels


def refresh_runtime_status(
    report: Mapping[str, Any], elapsed_seconds: float
) -> dict[str, Any]:
    """Finalize measured runtime without rerunning the policy."""
    output = dict(report)
    runtime = dict(output.get("runtime") or {})
    budget = float(runtime.get("budget_seconds") or 30.0)
    runtime.update(
        {
            "elapsed_seconds": elapsed_seconds,
            "budget_exceeded": elapsed_seconds > budget,
        }
    )
    warnings = [
        dict(row)
        for row in output.get("warnings") or []
        if row.get("code") != "acceptance_runtime_budget_exceeded"
    ]
    if elapsed_seconds > budget:
        warnings.append(
            {
                "code": "acceptance_runtime_budget_exceeded",
                "elapsed_seconds": elapsed_seconds,
                "budget_seconds": budget,
            }
        )
    output["runtime"] = runtime
    output["warnings"] = warnings
    output["status"] = _status(output.get("errors") or [], warnings)
    return output
