"""Fail-closed validation for SAM3 proposals independent of legacy mask coverage.

The primary FARM refinement route deliberately preserves a legacy projected
seed. That is safe for normal refinement, but creates a catch-22 when the seed
geometry itself is the defect being rescued. This module validates an explicit
second-chance proposal using a different contract:

* an external, provenance-bearing concept prompt establishes semantic identity;
* tiny overlap with projected metric support selects the physical instance,
  but coverage of the legacy support is advisory only;
* image/depth/competition/shape gates remain hard;
* at least two distinct physical timestamps must agree through bidirectional
  RGB-D reprojection and metric voxel consensus.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from farm_runtime.full_colmap_rescue import (
    backproject_mask_points,
    mask_shape_metrics,
    multiview_voxel_consensus,
    project_world_points_seed,
    voxel_downsample_points,
)

PROMPT_SCHEMA = "farm.sam3-concept-prompts.v1"
PLAN_PROMPT_BINDING_SCHEMA = "farm.sam3-concept-prompt-selection-binding.v1"
SEMANTIC_AUDIT_SCHEMA = "farm.whole-object-evidence-audit.v1"
_GENERIC_ONLY_PROMPTS = {
    "device",
    "equipment",
    "item",
    "machine",
    "object",
    "thing",
    "unit",
    "unknown object",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_category(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _bind_acceptance_prompt_to_semantic_audit(
    object_id: int,
    prompt: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    evidence_path_raw = str(provenance.get("evidence_path") or "").strip()
    if not evidence_path_raw:
        raise ValueError("acceptance prompts require provenance.evidence_path")
    evidence_path = Path(evidence_path_raw).expanduser().resolve(strict=True)
    expected_sha256 = str(provenance.get("evidence_sha256") or "")
    actual_sha256 = _file_sha256(evidence_path)
    if _SHA256.fullmatch(expected_sha256) is None or expected_sha256 != actual_sha256:
        raise ValueError("semantic evidence SHA-256 does not match immutable artifact")
    artifact = json.loads(evidence_path.read_text(encoding="utf-8"))
    objects = artifact.get("objects") if isinstance(artifact, Mapping) else None
    rows = []
    for row in objects or []:
        if not isinstance(row, Mapping):
            continue
        try:
            row_object_id = int(row.get("object_id", -1))
        except (TypeError, ValueError):
            continue
        if row_object_id == int(object_id):
            rows.append(row)
    if len(rows) != 1:
        raise ValueError(
            "semantic evidence must contain exactly one matching object row"
        )
    row = rows[0]
    adjudication = row.get("semantic_identity_re_adjudication")
    if not isinstance(adjudication, Mapping):
        raise ValueError("semantic evidence lacks identity re-adjudication")
    canonical = _normalize_category(adjudication.get("canonical_category"))
    initial = _normalize_category(adjudication.get("initial_category"))
    verification = _normalize_category(adjudication.get("verification_category"))
    independence = adjudication.get("evidence_independence")
    pose = (
        independence.get("pose_diversity")
        if isinstance(independence, Mapping)
        else None
    )
    identity_consensus = row.get("identity_consensus")
    confirmed_category = (
        _normalize_category(identity_consensus.get("confirmed_category"))
        if isinstance(identity_consensus, Mapping)
        else ""
    )
    conflicting_categories = (
        identity_consensus.get("conflicting_categories")
        if isinstance(identity_consensus, Mapping)
        else None
    )
    group = (
        identity_consensus.get("group_diagnostics", {}).get(canonical)
        if isinstance(identity_consensus, Mapping)
        and isinstance(identity_consensus.get("group_diagnostics"), Mapping)
        else None
    )
    reason_codes = {
        str(value)
        for value in adjudication.get("reason_codes", [])
        if isinstance(value, str)
    }
    confirmed_categories = (
        {
            _normalize_category(value)
            for value in identity_consensus.get("confirmed_categories", [])
            if _normalize_category(value)
        }
        if isinstance(identity_consensus, Mapping)
        else set()
    )
    gates = {
        "exact_semantic_audit_schema": str(artifact.get("schema") or "")
        == SEMANTIC_AUDIT_SCHEMA,
        "exact_adjudication_schema": str(adjudication.get("schema") or "")
        == "farm.independent-blind-label-consensus.v1",
        "adjudication_accepted": bool(adjudication.get("accepted"))
        and str(adjudication.get("status") or "") == "accepted",
        "raw_label_identity_contract": str(
            adjudication.get("identity_contract_source") or ""
        )
        == "raw_label_contract",
        "exact_safe_canonical_noun_reason": (
            "exact_safe_canonical_noun_agreement" in reason_codes
        ),
        "exact_safe_category_agreement": bool(canonical)
        and canonical == initial == verification,
        "identity_consensus_exact_category": confirmed_category == canonical,
        "identity_consensus_single_confirmed_category": confirmed_categories
        == {canonical},
        "identity_consensus_has_no_conflicts": isinstance(
            conflicting_categories, list
        )
        and not conflicting_categories,
        "prompt_exactly_matches_canonical_category": _normalize_category(prompt)
        == canonical,
        "independent_evidence": isinstance(independence, Mapping)
        and bool(independence.get("independent")),
        "pose_independent_evidence": isinstance(pose, Mapping)
        and bool(pose.get("pose_independent")),
        "identity_consensus_confirmation_ready": isinstance(group, Mapping)
        and bool(group.get("confirmation_ready")),
        "prompt_frozen_before_target_selection": bool(
            provenance.get("prompt_frozen_before_target_selection")
        ),
    }
    if not all(gates.values()):
        failed = [key for key, passed in gates.items() if not passed]
        raise ValueError(
            "semantic evidence does not authorize an acceptance prompt: "
            + ", ".join(failed)
        )
    return {
        "evidence_path": str(evidence_path),
        "evidence_sha256": actual_sha256,
        "evidence_schema": str(artifact.get("schema") or ""),
        "canonical_category": canonical,
        "semantic_evidence_gates": gates,
        "semantic_identity_re_adjudication": {
            "schema": str(adjudication.get("schema") or ""),
            "status": str(adjudication.get("status") or ""),
            "accepted": bool(adjudication.get("accepted")),
            "initial_category": initial,
            "verification_category": verification,
            "canonical_category": canonical,
            "initial_event_id": str(adjudication.get("initial_event_id") or ""),
            "verification_event_id": str(
                adjudication.get("verification_event_id") or ""
            ),
        },
    }


def load_concept_prompt_manifest(path: Path) -> dict[int, dict[str, Any]]:
    """Load explicit Qwen/VLM prompts without falling back to state labels."""

    resolved = path.expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != PROMPT_SCHEMA:
        raise ValueError(f"concept prompt manifest must use schema {PROMPT_SCHEMA!r}")
    rows = payload.get("objects")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "concept prompt manifest must contain a non-empty objects list"
        )
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("concept prompt rows must be mappings")
        object_id = int(row.get("object_id", -1))
        if object_id < 0 or object_id in output:
            raise ValueError(
                "concept prompt object IDs must be unique and non-negative"
            )
        prompt = " ".join(str(row.get("prompt") or "").strip().split())
        if not prompt or len(prompt) > 160:
            raise ValueError("concept prompts must contain 1-160 normalized characters")
        normalized_prompt = _normalize_category(prompt)
        prompt_tokens = re.findall(r"[a-z0-9]+", normalized_prompt)
        if (
            normalized_prompt in _GENERIC_ONLY_PROMPTS
            or (prompt_tokens and prompt_tokens[-1] in _GENERIC_ONLY_PROMPTS)
        ):
            raise ValueError(
                f"generic concept prompt is not an identity anchor: {prompt!r}"
            )
        provenance = row.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or not str(provenance.get("source") or "").strip()
        ):
            raise ValueError(
                "every concept prompt requires provenance.source (for example a "
                "frozen Qwen independent-blind review)"
            )
        authorization = str(row.get("authorization") or "").strip()
        if authorization not in {"acceptance", "diagnostic_only"}:
            raise ValueError(
                "concept prompt authorization must be acceptance or diagnostic_only"
            )
        identity_anchor_authorized = authorization == "acceptance"
        bound_evidence: dict[str, Any] | None = None
        if identity_anchor_authorized:
            bound_evidence = _bind_acceptance_prompt_to_semantic_audit(
                object_id, prompt, provenance
            )
        output[object_id] = {
            "object_id": object_id,
            "prompt": prompt,
            "provenance": dict(provenance),
            "authorization": authorization,
            "identity_anchor_authorized": identity_anchor_authorized,
            "bound_semantic_evidence": bound_evidence,
        }
    return output


def validate_acceptance_prompt_plan_binding(
    plan: Mapping[str, Any],
    prompts: Mapping[int, Mapping[str, Any]],
    prompt_manifest_path: Path,
    view_role: str | None = None,
) -> dict[str, Any]:
    """Bind acceptance prompts to frozen train-fit or unseen-heldout selection."""

    acceptance_ids = sorted(
        int(object_id)
        for object_id, row in prompts.items()
        if bool(row.get("identity_anchor_authorized"))
    )
    actual_manifest_sha256 = _file_sha256(
        prompt_manifest_path.expanduser().resolve(strict=True)
    )
    if not acceptance_ids:
        return {
            "acceptance_binding_required": False,
            "acceptance_object_ids": [],
            "prompt_manifest_sha256": actual_manifest_sha256,
            "verified": True,
        }

    binding = plan.get("concept_prompt_selection_contract")
    if not isinstance(binding, Mapping):
        raise ValueError(
            "acceptance prompts require plan.concept_prompt_selection_contract"
        )
    policy = plan.get("policy")
    if not isinstance(policy, Mapping):
        policy = {}
    plan_ids = {
        int(row.get("object_id", -1))
        for row in plan.get("objects", [])
        if isinstance(row, Mapping)
    }
    expected_sha256 = str(binding.get("prompt_manifest_sha256") or "")
    selection_role = str(binding.get("selection_role") or "")
    if selection_role == "train_fit_candidate":
        role_gates = {
            "train_plan_not_heldout_reference": policy.get(
                "heldout_reference_only"
            ) is False,
            "train_plan_not_post_adaptation_unseen": policy.get(
                "post_adaptation_unseen_heldout"
            ) is False,
            "train_state_fit_authorized": policy.get("state_fit_authorized") is True,
            "train_view_role": view_role == "train",
            "fit_candidate_authorized": binding.get("fit_candidate_authorized")
            is True,
        }
    elif selection_role == "post_adaptation_unseen_heldout":
        role_gates = {
            "heldout_reference_only": policy.get("heldout_reference_only") is True,
            "post_adaptation_unseen_heldout": policy.get(
                "post_adaptation_unseen_heldout"
            ) is True,
            "heldout_state_fit_forbidden": policy.get("state_fit_authorized")
            is False,
            "heldout_view_role": view_role == "heldout-reference",
            "fit_candidate_forbidden": binding.get("fit_candidate_authorized")
            is False,
        }
    else:
        role_gates = {"selection_role": False}
    gates = {
        "binding_schema": str(binding.get("schema") or "")
        == PLAN_PROMPT_BINDING_SCHEMA,
        "exact_prompt_manifest_sha256": _SHA256.fullmatch(expected_sha256)
        is not None
        and expected_sha256 == actual_manifest_sha256,
        "selection_frozen_after_prompt_manifest": bool(
            binding.get("selection_frozen_after_prompt_manifest")
        ),
        "category_agnostic_view_ranking": binding.get(
            "category_agnostic_view_ranking"
        )
        is True,
        "target_conditioned_view_selection_disabled": binding.get(
            "target_conditioned_view_selection"
        )
        is False,
        "prompt_not_used_for_view_ranking": binding.get(
            "prompt_used_for_view_ranking"
        )
        is False,
        "plan_policy_category_agnostic": policy.get(
            "category_agnostic_view_ranking"
        )
        is True,
        "plan_policy_target_conditioned_disabled": policy.get(
            "target_conditioned_view_selection"
        )
        is False,
        "release_authorized_is_false": binding.get("release_authorized") is False,
        "acceptance_objects_present_in_plan": set(acceptance_ids).issubset(
            plan_ids
        ),
        **role_gates,
    }
    if not all(gates.values()):
        failed = [key for key, passed in gates.items() if not passed]
        raise ValueError(
            "acceptance prompt/plan binding failed: " + ", ".join(failed)
        )
    return {
        "acceptance_binding_required": True,
        "acceptance_object_ids": acceptance_ids,
        "prompt_manifest_sha256": actual_manifest_sha256,
        "verified": True,
        "gates": gates,
        "plan_binding": dict(binding),
        "selection_role": selection_role,
        "fit_candidate_authorized": selection_role == "train_fit_candidate",
        "evaluation_only": selection_role == "post_adaptation_unseen_heldout",
        "release_authorized": False,
    }


def select_unique_identity_compatible_mapping_mask(
    candidate: np.ndarray,
    identity_anchor: np.ndarray,
    mapping_masks: Iterable[np.ndarray],
    depth: np.ndarray,
    *,
    minimum_anchor_pixels: int = 3,
    minimum_anchor_hit_margin: int = 3,
    minimum_anchor_hit_ratio: float = 1.5,
    maximum_centroid_distance_fraction: float = 0.35,
    maximum_depth_median_difference_m: float = 0.12,
) -> tuple[int | None, dict[str, Any]]:
    """Identify at most one mapping mask as the same physical instance.

    The exclusion is geometry-only: absolute projected-anchor support, a
    strict margin over the runner-up, 2D centroid agreement and observed-depth
    agreement must all pass. Ambiguity returns ``None`` so every mapping mask
    remains a hard negative.
    """

    candidate_mask = np.asarray(candidate, dtype=bool)
    anchor = np.asarray(identity_anchor, dtype=bool)
    depth_array = np.asarray(depth, dtype=np.float32)
    if candidate_mask.shape != anchor.shape or anchor.shape != depth_array.shape:
        raise ValueError("candidate, identity anchor and depth shapes must match")
    if minimum_anchor_pixels < 1 or minimum_anchor_hit_margin < 1:
        raise ValueError("anchor support and margin minima must be positive")
    if minimum_anchor_hit_ratio <= 1.0:
        raise ValueError("anchor hit ratio must be greater than one")
    if maximum_centroid_distance_fraction <= 0.0:
        raise ValueError("centroid distance fraction must be positive")
    if maximum_depth_median_difference_m <= 0.0:
        raise ValueError("depth median tolerance must be positive")

    valid_depth = np.isfinite(depth_array) & (depth_array > 0.05) & (depth_array < 80)

    def centroid(mask: np.ndarray) -> np.ndarray:
        ys, xs = np.nonzero(mask)
        if not ys.size:
            return np.asarray([np.nan, np.nan], dtype=np.float64)
        return np.asarray([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)

    candidate_center = centroid(candidate_mask)
    candidate_ys, candidate_xs = np.nonzero(candidate_mask)
    if candidate_ys.size:
        candidate_diagonal = float(
            math.hypot(
                int(candidate_xs.max()) - int(candidate_xs.min()) + 1,
                int(candidate_ys.max()) - int(candidate_ys.min()) + 1,
            )
        )
    else:
        candidate_diagonal = 1.0
    candidate_depth_values = depth_array[candidate_mask & valid_depth]
    candidate_depth = (
        float(np.median(candidate_depth_values))
        if candidate_depth_values.size
        else float("nan")
    )

    diagnostics: list[dict[str, Any]] = []
    masks = [np.asarray(mask, dtype=bool) for mask in mapping_masks]
    for index, mask in enumerate(masks):
        if mask.shape != candidate_mask.shape:
            raise ValueError("all mapping masks must match candidate shape")
        anchor_hits = int(np.logical_and(mask, anchor).sum())
        intersection = int(np.logical_and(mask, candidate_mask).sum())
        union = int(np.logical_or(mask, candidate_mask).sum())
        mapping_center = centroid(mask)
        centroid_distance = float(np.linalg.norm(mapping_center - candidate_center))
        normalized_centroid_distance = centroid_distance / max(candidate_diagonal, 1.0)
        mapping_depth_values = depth_array[mask & valid_depth]
        mapping_depth = (
            float(np.median(mapping_depth_values))
            if mapping_depth_values.size
            else float("nan")
        )
        depth_delta = (
            abs(mapping_depth - candidate_depth)
            if math.isfinite(mapping_depth) and math.isfinite(candidate_depth)
            else float("inf")
        )
        iou = float(intersection / max(union, 1))
        candidate_recall = float(intersection / max(int(candidate_mask.sum()), 1))
        local_gates = {
            "absolute_anchor_support": anchor_hits >= int(minimum_anchor_pixels),
            "concept_geometry_agreement": iou >= 0.25 or candidate_recall >= 0.50,
            "centroid_closeness": normalized_centroid_distance
            <= float(maximum_centroid_distance_fraction),
            "depth_closeness": depth_delta <= float(maximum_depth_median_difference_m),
        }
        diagnostics.append(
            {
                "mapping_mask_index": index,
                "identity_anchor_overlap_pixels": anchor_hits,
                "concept_iou": iou,
                "concept_recall": candidate_recall,
                "normalized_centroid_distance": (
                    normalized_centroid_distance
                    if math.isfinite(normalized_centroid_distance)
                    else None
                ),
                "candidate_depth_median_m": (
                    candidate_depth if math.isfinite(candidate_depth) else None
                ),
                "mapping_depth_median_m": (
                    mapping_depth if math.isfinite(mapping_depth) else None
                ),
                "depth_median_difference_m": (
                    depth_delta if math.isfinite(depth_delta) else None
                ),
                "local_gates": local_gates,
                "locally_compatible": all(local_gates.values()),
            }
        )
    ordered = sorted(
        diagnostics,
        key=lambda row: (
            int(row["identity_anchor_overlap_pixels"]),
            float(row["concept_iou"]),
        ),
        reverse=True,
    )
    top = ordered[0] if ordered else None
    runner_hits = (
        int(ordered[1]["identity_anchor_overlap_pixels"]) if len(ordered) > 1 else 0
    )
    top_hits = int(top["identity_anchor_overlap_pixels"]) if top is not None else 0
    uniqueness = {
        "absolute_hit_margin": top_hits - runner_hits >= int(minimum_anchor_hit_margin),
        "relative_hit_margin": top_hits
        >= float(minimum_anchor_hit_ratio) * max(runner_hits, 1),
    }
    selected = (
        int(top["mapping_mask_index"])
        if top is not None
        and bool(top["locally_compatible"])
        and all(uniqueness.values())
        else None
    )
    return selected, {
        "mapping_masks": len(masks),
        "selected_mapping_mask_index": selected,
        "top_anchor_hits": top_hits,
        "runner_up_anchor_hits": runner_hits,
        "uniqueness_gates": uniqueness,
        "diagnostics": diagnostics,
        "policy": "unique_metric_anchor_margin_centroid_and_depth",
    }


def evaluate_second_chance_candidate(
    mask: np.ndarray,
    identity_anchor: np.ndarray,
    competing_masks: np.ndarray,
    depth: np.ndarray,
    frame: Mapping[str, Any],
    *,
    identity_world_points: np.ndarray | None = None,
    confidence: float,
    minimum_confidence: float,
    maximum_competing_fraction: float,
    minimum_anchor_pixels: int = 3,
    minimum_depth_valid_fraction: float = 0.50,
    maximum_image_area_fraction: float = 0.45,
    maximum_border_fraction: float = 0.20,
    minimum_metric_points: int = 24,
    maximum_identity_surface_distance_m: float = 0.10,
    minimum_identity_surface_points: int = 12,
    minimum_identity_surface_fraction: float = 0.20,
) -> tuple[dict[str, Any], np.ndarray]:
    """Apply local safety gates while treating legacy coverage as advisory."""

    candidate = np.asarray(mask, dtype=bool)
    anchor = np.asarray(identity_anchor, dtype=bool)
    competing = np.asarray(competing_masks, dtype=bool)
    depth_array = np.asarray(depth, dtype=np.float32)
    if not (candidate.shape == anchor.shape == competing.shape == depth_array.shape):
        raise ValueError("candidate, anchor, competition and depth shapes must match")
    if not 0.0 <= float(minimum_confidence) <= 1.0:
        raise ValueError("minimum confidence must be in [0, 1]")
    if not 0.0 <= float(maximum_competing_fraction) <= 1.0:
        raise ValueError("maximum competing fraction must be in [0, 1]")
    if minimum_anchor_pixels < 1 or minimum_metric_points < 1:
        raise ValueError("anchor and metric point minima must be positive")
    if not 0.0 <= float(minimum_identity_surface_fraction) <= 1.0:
        raise ValueError("identity surface fraction must be in [0, 1]")

    pixels = int(candidate.sum())
    anchor_pixels = int(np.logical_and(candidate, anchor).sum())
    anchor_total = int(anchor.sum())
    competing_pixels = int(np.logical_and(candidate, competing).sum())
    valid_depth = np.isfinite(depth_array) & (depth_array > 0.05) & (depth_array < 80)
    valid_pixels = int(np.logical_and(candidate, valid_depth).sum())
    border = np.zeros_like(candidate)
    border[:3] = border[-3:] = True
    border[:, :3] = border[:, -3:] = True
    border_pixels = int(np.logical_and(candidate, border).sum())
    shape = mask_shape_metrics(candidate)
    world_points = voxel_downsample_points(
        backproject_mask_points(candidate & valid_depth, depth_array, frame),
        voxel_size_m=0.02,
        max_points=5_000,
    )
    identity_points = np.asarray(
        (
            identity_world_points
            if identity_world_points is not None
            else np.zeros((0, 3), dtype=np.float32)
        ),
        dtype=np.float32,
    ).reshape(-1, 3)
    identity_points = identity_points[np.isfinite(identity_points).all(axis=1)]
    identity_points = voxel_downsample_points(
        identity_points,
        voxel_size_m=0.02,
        max_points=5_000,
    )
    near_identity = 0
    if world_points.size and identity_points.size:
        squared_limit = float(maximum_identity_surface_distance_m) ** 2
        for start in range(0, int(world_points.shape[0]), 256):
            chunk = world_points[start : start + 256].astype(np.float64)
            distances = np.sum(
                (chunk[:, None, :] - identity_points[None, :, :]) ** 2,
                axis=2,
            )
            near_identity += int(np.any(distances <= squared_limit, axis=1).sum())
    identity_surface_fraction = float(
        near_identity / max(int(world_points.shape[0]), 1)
    )

    image_area_fraction = float(pixels / max(candidate.size, 1))
    depth_valid_fraction = float(valid_pixels / max(pixels, 1))
    competing_fraction = float(competing_pixels / max(pixels, 1))
    border_fraction = float(border_pixels / max(pixels, 1))
    anchor_fraction_of_candidate = float(anchor_pixels / max(pixels, 1))
    legacy_anchor_coverage = float(anchor_pixels / max(anchor_total, 1))
    gates = {
        "sam3_concept_confidence": float(confidence) >= float(minimum_confidence),
        "nonempty_candidate": pixels >= 24,
        "identity_anchor_hit": anchor_pixels >= int(minimum_anchor_pixels),
        "depth_valid": depth_valid_fraction >= float(minimum_depth_valid_fraction),
        "image_area": image_area_fraction <= float(maximum_image_area_fraction),
        "image_border": border_fraction <= float(maximum_border_fraction),
        "dominant_component": float(shape["dominant_component_fraction"]) >= 0.92,
        "significant_components": int(shape["significant_component_count"]) <= 2,
        "competing_mask_exclusion": competing_fraction
        <= float(maximum_competing_fraction),
        "metric_backprojection": int(world_points.shape[0])
        >= int(minimum_metric_points),
        "frontmost_metric_identity_surface": near_identity
        >= int(minimum_identity_surface_points)
        and identity_surface_fraction >= float(minimum_identity_surface_fraction),
    }
    score = (
        0.30 * float(np.clip(confidence, 0.0, 1.0))
        + 0.20 * depth_valid_fraction
        + 0.15 * float(shape["dominant_component_fraction"])
        + 0.15 * min(1.0, anchor_fraction_of_candidate / 0.05)
        + 0.10 * (1.0 - min(1.0, competing_fraction))
        + 0.10 * (1.0 - min(1.0, border_fraction))
    )
    audit = {
        "passed_local_gates": all(bool(value) for value in gates.values()),
        "quality_gates": gates,
        "rejection_reasons": [key for key, passed in gates.items() if not bool(passed)],
        "confidence": float(confidence),
        "candidate_pixels": pixels,
        "identity_anchor_pixels": anchor_pixels,
        "identity_anchor_fraction_of_candidate": anchor_fraction_of_candidate,
        "legacy_anchor_coverage_advisory": legacy_anchor_coverage,
        "legacy_anchor_coverage_is_hard_gate": False,
        "depth_valid_fraction": depth_valid_fraction,
        "competing_mask_fraction": competing_fraction,
        "image_area_fraction": image_area_fraction,
        "border_fraction": border_fraction,
        "metric_points": int(world_points.shape[0]),
        "metric_identity_surface_points": int(near_identity),
        "metric_identity_surface_fraction": identity_surface_fraction,
        "minimum_identity_surface_fraction": float(minimum_identity_surface_fraction),
        "maximum_identity_surface_distance_m": float(
            maximum_identity_surface_distance_m
        ),
        "local_score": float(score),
        **shape,
    }
    return audit, world_points


def _cycle_recall(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    projected, projection = project_world_points_seed(
        np.asarray(source["world_points"], dtype=np.float32),
        target["frame"],
        np.asarray(target["depth"], dtype=np.float32),
        dilation_pixels=2,
        minimum_projected_points=8,
    )
    pixels = int(projected.sum())
    overlap = int(np.logical_and(projected, target["mask"]).sum())
    recall = float(overlap / max(pixels, 1))
    return recall, {
        **projection,
        "cycle_seed_pixels": pixels,
        "cycle_overlap_pixels": overlap,
        "cycle_recall": recall,
    }


def select_cross_view_consistent_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    minimum_physical_timestamps: int = 2,
    minimum_consensus_points: int = 48,
    minimum_cycle_recall: float = 0.45,
    maximum_centroid_distance_m: float = 0.45,
    consensus_voxel_size_m: float = 0.04,
    maximum_exact_combinations: int = 4096,
) -> tuple[set[str], dict[str, Any]]:
    """Choose the strongest all-pairs-consistent multi-timestamp proposal clique."""

    if minimum_physical_timestamps < 2:
        raise ValueError("second chance requires at least two physical timestamps")
    if minimum_consensus_points < 1:
        raise ValueError("minimum consensus points must be positive")
    if not 0.0 <= minimum_cycle_recall <= 1.0:
        raise ValueError("minimum cycle recall must be in [0, 1]")
    if maximum_centroid_distance_m <= 0.0 or consensus_voxel_size_m <= 0.0:
        raise ValueError("metric consistency thresholds must be positive")
    if maximum_exact_combinations < 1:
        raise ValueError("maximum exact combinations must be positive")

    rows = [dict(row) for row in candidates]
    keys = [str(row.get("candidate_key") or "") for row in rows]
    if any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("second-chance candidate keys must be unique and non-empty")
    local = [
        row
        for row in rows
        if bool(row.get("passed_local_gates"))
        and str(row.get("physical_timestamp") or "")
        and np.asarray(row.get("world_points"), dtype=np.float32)
        .reshape(-1, 3)
        .shape[0]
        > 0
    ]
    pairwise: list[dict[str, Any]] = []
    compatible: dict[frozenset[str], bool] = {}
    for left, right in itertools.combinations(local, 2):
        left_key = str(left["candidate_key"])
        right_key = str(right["candidate_key"])
        different_timestamp = str(left["physical_timestamp"]) != str(
            right["physical_timestamp"]
        )
        left_points = np.asarray(left["world_points"], dtype=np.float32).reshape(-1, 3)
        right_points = np.asarray(right["world_points"], dtype=np.float32).reshape(
            -1, 3
        )
        centroid_distance = float(
            np.linalg.norm(
                np.median(left_points, axis=0) - np.median(right_points, axis=0)
            )
        )
        forward, forward_audit = _cycle_recall(left, right)
        reverse, reverse_audit = _cycle_recall(right, left)
        passed = (
            different_timestamp
            and centroid_distance <= float(maximum_centroid_distance_m)
            and forward >= float(minimum_cycle_recall)
            and reverse >= float(minimum_cycle_recall)
        )
        compatible[frozenset((left_key, right_key))] = passed
        pairwise.append(
            {
                "left": left_key,
                "right": right_key,
                "different_physical_timestamp": different_timestamp,
                "centroid_distance_m": centroid_distance,
                "forward": forward_audit,
                "reverse": reverse_audit,
                "passed": passed,
                "rejection_reasons": [
                    name
                    for name, value in {
                        "same_physical_timestamp": different_timestamp,
                        "centroid_distance": centroid_distance
                        <= float(maximum_centroid_distance_m),
                        "forward_cycle": forward >= float(minimum_cycle_recall),
                        "reverse_cycle": reverse >= float(minimum_cycle_recall),
                    }.items()
                    if not value
                ],
            }
        )

    by_timestamp: dict[str, list[dict[str, Any]]] = {}
    for row in local:
        by_timestamp.setdefault(str(row["physical_timestamp"]), []).append(row)
    timestamp_groups: list[list[dict[str, Any]]] = []
    for timestamp in sorted(by_timestamp):
        timestamp_groups.append(
            sorted(
                by_timestamp[timestamp],
                key=lambda row: (
                    float(row.get("local_score") or 0.0),
                    str(row["candidate_key"]),
                ),
                reverse=True,
            )
        )
    exact_search_space = 1
    for group in timestamp_groups:
        exact_search_space *= len(group) + 1
    exact_search_space -= 1
    if exact_search_space > int(maximum_exact_combinations):
        return set(), {
            "status": "rejected_exact_search_bound",
            "accepted": False,
            "rejection_reasons": ["maximum_exact_combinations"],
            "input_candidates": len(rows),
            "locally_accepted_candidates": len(local),
            "exact_search_space": int(exact_search_space),
            "maximum_exact_combinations": int(maximum_exact_combinations),
            "evaluated_exact_combinations": 0,
            "pairwise_cycle": pairwise,
        }

    ranked_clusters: list[
        tuple[tuple[int, int, float, tuple[str, ...]], list[dict], np.ndarray, dict]
    ] = []
    failed_clusters: list[
        tuple[tuple[int, int, float, tuple[str, ...]], list[dict], np.ndarray, dict]
    ] = []
    evaluated_combinations = 0
    choices = [[None, *group] for group in timestamp_groups]
    for selection in itertools.product(*choices):
        cluster = [row for row in selection if row is not None]
        if len(cluster) < int(minimum_physical_timestamps):
            continue
        if not all(
            compatible.get(
                frozenset((str(left["candidate_key"]), str(right["candidate_key"]))),
                False,
            )
            for left, right in itertools.combinations(cluster, 2)
        ):
            continue
        evaluated_combinations += 1
        support, consensus = multiview_voxel_consensus(
            [np.asarray(row["world_points"], dtype=np.float32) for row in cluster],
            voxel_size_m=float(consensus_voxel_size_m),
            minimum_views=2,
            max_points=20_000,
        )
        rank = (
            len(cluster),
            int(support.shape[0]),
            float(sum(float(row.get("local_score") or 0.0) for row in cluster)),
            tuple(sorted(str(row["candidate_key"]) for row in cluster)),
        )
        entry = (rank, cluster, support, consensus)
        failed_clusters.append(entry)
        if int(support.shape[0]) >= int(minimum_consensus_points):
            ranked_clusters.append(entry)

    candidates_to_rank = ranked_clusters or failed_clusters
    if candidates_to_rank:
        _, best, support, consensus = max(
            candidates_to_rank, key=lambda row: row[0]
        )
    else:
        best = []
        support = np.zeros((0, 3), dtype=np.float32)
        consensus = {
            "input_views": 0,
            "minimum_views": 2,
            "support_points": 0,
            "voxel_size_m": float(consensus_voxel_size_m),
        }
    selected_timestamps = {str(row["physical_timestamp"]) for row in best}
    accepted = len(selected_timestamps) >= int(minimum_physical_timestamps) and int(
        support.shape[0]
    ) >= int(minimum_consensus_points)
    selected = {str(row["candidate_key"]) for row in best} if accepted else set()
    if accepted:
        status = "accepted_cross_view_cycle_consensus"
        reasons: list[str] = []
    else:
        status = "rejected_cross_view_cycle_consensus"
        reasons = []
        if len(selected_timestamps) < int(minimum_physical_timestamps):
            reasons.append("independent_physical_timestamps")
        if int(support.shape[0]) < int(minimum_consensus_points):
            reasons.append("metric_voxel_consensus")
    return selected, {
        "status": status,
        "accepted": accepted,
        "rejection_reasons": reasons,
        "input_candidates": len(rows),
        "locally_accepted_candidates": len(local),
        "exact_search_space": int(exact_search_space),
        "maximum_exact_combinations": int(maximum_exact_combinations),
        "evaluated_exact_combinations": int(evaluated_combinations),
        "selected_candidate_keys": sorted(selected),
        "selected_physical_timestamps": sorted(selected_timestamps) if accepted else [],
        "independent_physical_timestamps": len(selected_timestamps) if accepted else 0,
        "minimum_physical_timestamps": int(minimum_physical_timestamps),
        "support_points": int(support.shape[0]) if accepted else 0,
        "minimum_consensus_points": int(minimum_consensus_points),
        "consensus_voxel_size_m": float(consensus_voxel_size_m),
        "minimum_cycle_recall": float(minimum_cycle_recall),
        "maximum_centroid_distance_m": float(maximum_centroid_distance_m),
        "pairwise_cycle": pairwise,
        "voxel_consensus": consensus,
    }
