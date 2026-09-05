#!/usr/bin/env python3
"""Reconcile FARM captions with mask-grounded multi-view evidence.

The reconciler is scene- and class-agnostic. A saved FARM caption, independent
re-caption passes, detector proposals and a final paired-crop audit are treated
as model evidence. Human object ids and hand-written class corrections are
neither accepted nor used.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.semantics.review_farm_object_crops import (
    choose_crops,
    crop_candidates,
    parse_json_text,
    request_review,
)
from scene_graph.captioning.evidence import (
    crop_evidence_manifest,
    evidence_pose_diversity,
    load_frame_pose_index,
    partition_crop_candidates,
    select_independent_evidence,
)
from scene_graph.captioning.label_contract import (
    OPEN_VOCABULARY_JSON_INSTRUCTIONS,
    assess_open_vocabulary_label,
    has_open_vocabulary_contract_evidence,
)
from scene_graph.captioning.physical_guard import (
    PHYSICAL_RECOVERY_MAX_VIEW_OVERLAP,
    PHYSICAL_RECOVERY_MIN_TOTAL_UNIQUE_VIEWS,
    PHYSICAL_RECOVERY_MIN_VIEWS_PER_EVENT,
    assess_guard_visible_form_recovery,
    assess_physical_form,
    assess_physical_recovery,
)
from scene_graph.map_update.mask_observations import resolve_object_mask_observations


FINAL_PROMPT = """You are the final blind object-captioning stage of a 3D scene
memory pipeline. Each image is a paired view of the same automatically tracked
object: the left panel is the ordinary FARM object crop and the right panel is
the mask-grounded target. The right target pixels decide identity; the left
panel is context for scale and placement only.

You have not been given any proposed label. Decide composition and boundedness
before identity. Do not identify a nearby object or infer function from scene
context. This compatibility path is disabled by default in standard runs.
""" + OPEN_VOCABULARY_JSON_INSTRUCTIONS

PHYSICAL_GUARD_PROMPT = """Audit whether the proposed noun below is physically
compatible with the outlined target as one complete bounded object. Each image
pairs an ordinary crop (left) with its mask-grounded target (right); only right
target pixels decide. Check metric shape, gravity orientation, and whether
diagnostic parts belong to the target rather than a neighbour. This call is a
candidate-conditioned veto only: do not propose, imply, or emit any replacement
noun and do not preserve the proposal merely for consistency.

Return JSON only with keys category, description, attributes, confidence,
decision, topology, complete_bounded, shape_profile, diagnostic_view_count.
category MUST be exactly one of: compatible, contradiction,
part_only, insufficient. Use contradiction for an incompatible physical form,
part_only when the mask is only a component of the proposed whole, and
insufficient when the available views cannot decide. description states the
visible physical evidence; attributes lists diagnostic parts; confidence is
0..1 and decision is keep. topology MUST be standalone_whole,
carrier_payload, standalone_component, attached_component, partial_unbounded,
background_surface, mixed_targets or unknown. complete_bounded is true only
when all target pixels form one complete bounded object. shape_profile MUST be
planar, elongated_rigid, elongated_flexible, compact_volumetric, open_frame,
lattice, articulated, irregular or unknown. diagnostic_view_count is the
number of target views that independently show the stated diagnostic evidence.

Proposed noun: {category}
"""

PHYSICAL_RECOVERY_PROMPT = """Perform an unconditioned open-vocabulary review
of only the mask-grounded target. No previous noun is supplied. Decide
composition and boundedness before identity, and abstain for fragments or
mixed targets. This optional compatibility path is disabled by default.
""" + OPEN_VOCABULARY_JSON_INSTRUCTIONS

SOURCE_WEIGHTS = {
    "blind_pass_a": 2.0,
    "verification_pass": 2.0,
    "blind_pass_b": 2.0,
    "blind_pass_c": 2.0,
    "multi_source_fusion": 1.0,
    "neutral_form_fallback": 0.35,
    "prior_farm_caption": 1.25,
    "paired_crop_blind_adjudicator": 4.0,
}
UNRESOLVED = {"", "unknown", "unresolved object", "object", "item", "thing"}
DERIVED_EVIDENCE_SOURCES = {
    "strict_two_pass",
    "multi_source_fusion",
    "semantic_reconciliation",
    "semantic_consensus",
}


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-state", type=Path, required=True)
    parser.add_argument("--candidate-review", type=Path, required=True)
    parser.add_argument("--detector-state", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--model", default="qwen3-vl-8b")
    parser.add_argument("--crops-per-object", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.72)
    parser.add_argument(
        "--blind-adjudication",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compatibility-only extra blind adjudication; standard runs disable it.",
    )
    parser.add_argument(
        "--enable-recovery-calls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt in to two extra blind recovery calls after a hard physical veto.",
    )
    return parser.parse_args()


def normalize_category(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    words = [word[:-1] if word.endswith("s") and len(word) > 4 else word for word in text.split()]
    return " ".join(words)


def category_related(first: object, second: object) -> float:
    a, b = normalize_category(first), normalize_category(second)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    left, right = set(a.split()), set(b.split())
    return 0.72 if left <= right or right <= left else 0.0


def _load_state(path: Path) -> dict:
    payload = torch.load(path.expanduser(), map_location="cpu", weights_only=False)
    state = payload.get("state", payload) if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not isinstance(state.get("object_id"), torch.Tensor):
        raise ValueError(f"Invalid FARM state: {path}")
    return state


def _state_rows(state: dict) -> dict[int, dict]:
    object_ids = [int(value) for value in state["object_id"].detach().cpu().tolist()]
    categories = list(state.get("object_category") or [])
    descriptions = list(state.get("object_caption") or [])
    attributes = list(state.get("object_key_attributes") or [])
    dimensions = state.get("object_box_dimensions_m")
    dimensions_np = dimensions.detach().cpu().numpy() if isinstance(dimensions, torch.Tensor) else None
    result = {}
    for index, object_id in enumerate(object_ids):
        result[object_id] = {
            "category": str(categories[index] if index < len(categories) else "").strip(),
            "description": str(descriptions[index] if index < len(descriptions) else "").strip(),
            "attributes": attributes[index] if index < len(attributes) and isinstance(attributes[index], list) else [],
            "metric_dimensions_m": (
                np.asarray(dimensions_np[index], dtype=float).tolist()
                if dimensions_np is not None and index < len(dimensions_np)
                else None
            ),
        }
    return result


def _detector_rows(state: dict) -> dict[int, dict[str, float]]:
    object_ids = [int(value) for value in state["object_id"].detach().cpu().tolist()]
    values = list(state.get("object_detection_category_conf") or [])
    result = {}
    for index, object_id in enumerate(object_ids):
        row = values[index] if index < len(values) and isinstance(values[index], dict) else {}
        result[object_id] = {
            str(label).strip().lower(): float(score)
            for label, score in row.items()
            if str(label).strip() and np.isfinite(float(score))
        }
    return result


def _informative(value: object) -> bool:
    return normalize_category(value) not in UNRESOLVED


def _candidate_event_id(candidate: dict, source: str) -> str:
    return str(
        candidate.get("event_id")
        or candidate.get("evidence_event_id")
        or candidate.get("request_id")
        or source
    ).strip()


def _positive_candidate(candidate: dict, minimum: float) -> bool:
    try:
        confidence = float(candidate.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return False
    return bool(
        str(candidate.get("decision") or "unknown").strip().lower() == "keep"
        and _informative(candidate.get("category"))
        and np.isfinite(confidence)
        and confidence >= float(minimum)
    )


def _candidate_evidence(row: dict, prior: dict, minimum: float = 0.0) -> list[dict]:
    evidence = []
    seen: set[tuple[str, str]] = set()
    raw_candidates = list(row.get("semantic_evidence") or []) + list(
        row.get("candidates") or []
    )
    for candidate in raw_candidates:
        source = str(candidate.get("source") or "")
        if source == "strict_two_pass":
            continue
        category = str(candidate.get("category") or "").strip().lower()
        event_id = _candidate_event_id(candidate, source)
        key = (event_id, normalize_category(category))
        if not _positive_candidate(candidate, minimum) or key in seen:
            continue
        seen.add(key)
        evidence.append({
            **candidate,
            "source": source,
            "event_id": event_id,
            "category": category,
        })
    category = str(prior.get("category") or "").strip().lower()
    # A strict current-run label contract supersedes the legacy saved caption.
    # The prior remains useful only for backward-compatible old artifacts and
    # must never vote against or promote a current open-vocabulary proposal.
    if not has_open_vocabulary_contract_evidence(row) and _informative(category):
        evidence.append({
            "source": "prior_farm_caption",
            "event_id": f"object:{int(row.get('id', -1))}:prior_farm_caption",
            "category": category,
            "description": prior.get("description") or "",
            "attributes": prior.get("attributes") or [],
            "confidence": 0.95,
            "decision": "keep",
            "confirmation_eligible": False,
            "confirmation_ineligibility_reason": "legacy_state_has_no_crop_provenance",
        })
    return evidence


def independent_support(category: str, evidence: list[dict]) -> list[dict]:
    """Return distinct positive inference events supporting ``category``.

    Candidate-informed fusion/final rows are derived summaries, not new model
    observations. They may help select a hypothesis but cannot promote its tier.
    """

    candidates: list[dict] = []
    for candidate in evidence:
        source = str(candidate.get("source") or "").strip()
        if source in DERIVED_EVIDENCE_SOURCES or not _positive_candidate(candidate, 0.0):
            continue
        if category_related(category, candidate.get("category")) <= 0.0:
            continue
        event_id = _candidate_event_id(candidate, source)
        if event_id:
            candidates.append({**candidate, "event_id": event_id})
    selected, _ = select_independent_evidence(candidates)
    if selected:
        return selected
    # One correlated or provenance-incomplete positive event may retain a
    # probable label, but can never reach confirmed.
    return [
        max(candidates, key=lambda item: float(item.get("confidence") or 0.0))
    ] if candidates else []


def _hypotheses(evidence: list[dict]) -> list[str]:
    result: list[str] = []
    for item in evidence:
        category = str(item.get("category") or "").strip().lower()
        if _informative(category) and normalize_category(category) not in {normalize_category(x) for x in result}:
            result.append(category)
    return result


def evidence_scores(hypotheses: list[str], evidence: list[dict], detector: dict[str, float]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for hypothesis in hypotheses:
        score = sum(
            SOURCE_WEIGHTS.get(str(item.get("source") or ""), 1.0)
            * category_related(hypothesis, item.get("category"))
            for item in evidence
        )
        score += sum(
            0.9 * min(1.0, max(0.0, float(confidence))) * category_related(hypothesis, label)
            for label, confidence in detector.items()
        )
        scores[hypothesis] = float(score)
    return scores


def needs_adjudication(prior_category: str, row: dict, evidence: list[dict]) -> bool:
    labels = {normalize_category(item.get("category")) for item in evidence if _informative(item.get("category"))}
    selected, prior = normalize_category(row.get("category")), normalize_category(prior_category)
    return bool((prior and selected and prior != selected) or len(labels) > 1)


def _paired_crop(raw_encoded: bytes, grounded_encoded: bytes) -> bytes | None:
    raw = cv2.imdecode(np.frombuffer(raw_encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    grounded = cv2.imdecode(np.frombuffer(grounded_encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if raw is None or grounded is None:
        return None
    panel_h, panel_w, header = 320, 356, 32
    raw = cv2.resize(raw, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    grounded = cv2.resize(grounded, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((panel_h + header, panel_w * 2 + 8, 3), 18, dtype=np.uint8)
    canvas[header:, :panel_w] = raw
    canvas[header:, panel_w + 8:] = grounded
    cv2.putText(canvas, "FARM CROP", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 238, 244), 1, cv2.LINE_AA)
    cv2.putText(canvas, "MASK-GROUNDED TARGET", (panel_w + 16, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 238, 244), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    return bytes(encoded) if ok else None


def paired_crops(
    paths: list[Path],
    requested: int,
    source: str = "paired_crop_blind_adjudicator",
) -> tuple[list[tuple[Path, bytes]], dict]:
    candidates = crop_candidates(paths)
    partitioned, partition = partition_crop_candidates(candidates, source)
    partitioned = sorted(partitioned, key=lambda item: len(item[1]), reverse=True)
    grounded = choose_crops(partitioned, requested)
    output = []
    for path, grounded_encoded in grounded:
        try:
            with np.load(path, allow_pickle=False) as payload:
                raw_encoded = bytes(np.asarray(payload["crop_jpeg_bytes"], dtype=np.uint8))
            paired = _paired_crop(raw_encoded, grounded_encoded)
            if paired:
                output.append((path, paired))
        except (OSError, KeyError, ValueError):
            continue
    return output, partition


def physical_guard_event(
    url: str,
    model: str,
    row: dict,
    category: str,
    crops: list[tuple[Path, bytes]],
    partition: dict,
    frame_pose_index: dict,
) -> dict:
    """Run a candidate-conditioned veto that cannot add semantic support."""

    prompt = PHYSICAL_GUARD_PROMPT.format(category=category)
    reviewed = request_review(
        url,
        model,
        row,
        crops,
        prompt=prompt,
    )
    verdict = normalize_category(reviewed.get("review_category"))
    verdict = verdict.replace(" ", "_")
    if verdict not in {"compatible", "contradiction", "part_only", "insufficient"}:
        verdict = "insufficient"
    structured = parse_json_text(str(reviewed.get("review_raw_response") or ""))
    topology = normalize_category(structured.get("topology")).replace(" ", "_")
    if topology not in {
        "standalone_whole", "carrier_payload", "standalone_component",
        "attached_component", "partial_unbounded", "background_surface",
        "mixed_targets", "unknown",
    }:
        topology = "unknown"
    try:
        diagnostic_view_count = int(structured.get("diagnostic_view_count") or 0)
    except (TypeError, ValueError):
        diagnostic_view_count = 0
    manifest = crop_evidence_manifest(
        int(row.get("id", -1)),
        crops,
        partition=partition,
        frame_pose_index=frame_pose_index,
        object_position_world_m=row.get("position_world_m"),
    )
    return {
        "source": "paired_physical_form_guard",
        "event_id": f"object:{int(row.get('id', -1))}:paired_physical_form_guard",
        "verdict": verdict,
        "category": verdict,
        "proposed_category": str(category).strip().lower(),
        "description": str(reviewed.get("review_description") or "").strip(),
        "diagnostic_parts": [
            str(value).strip()
            for value in (reviewed.get("review_attributes") or [])
            if str(value).strip()
        ][:8],
        "confidence": float(reviewed.get("review_confidence") or 0.0),
        "decision": str(reviewed.get("review_decision") or "unknown"),
        "supported_view_count": len(manifest.get("crop_image_ids") or []),
        # Additive empty compatibility fields make the veto-only contract
        # explicit for older consumers: this conditioned call contributes no noun.
        "visible_form_category": "unknown",
        "visible_form_confidence": 0.0,
        "topology": topology,
        "complete_bounded": (
            structured.get("complete_bounded")
            if isinstance(structured.get("complete_bounded"), bool)
            else None
        ),
        "shape_profile": normalize_category(
            structured.get("shape_profile") or "unknown"
        ).replace(" ", "_"),
        "diagnostic_view_count": max(0, diagnostic_view_count),
        "carrier_category": "",
        "payload_categories": [],
        "mobility": "unknown",
        "model_id": str(model),
        "prompt_id": "paired_physical_form_guard.v1",
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "confirmation_eligible": False,
        "confirmation_ineligibility_reason": "candidate_conditioned_veto_only",
        **manifest,
    }


def physical_recovery_event(
    url: str,
    model: str,
    row: dict,
    crops: list[tuple[Path, bytes]],
    partition: dict,
    frame_pose_index: dict,
    *,
    source: str = "physical_form_recovery",
) -> dict:
    """Run one blind form-only inference on an auditable view fold."""

    if source not in {
        "physical_form_recovery",
        "physical_form_recovery_verification",
    }:
        raise ValueError(f"Unsupported physical recovery source: {source}")

    reviewed = request_review(
        url,
        model,
        # Keep the request genuinely blind: neither the object id nor the
        # rejected noun/class reaches prompt construction. Metric dimensions
        # are permitted only as a physical-scale sanity check.
        {"metric_dimensions_m": row.get("metric_dimensions_m")},
        crops,
        prompt=PHYSICAL_RECOVERY_PROMPT,
        strict_open_vocabulary=True,
    )
    attributes = [
        str(value).strip()
        for value in (reviewed.get("review_attributes") or [])
        if str(value).strip()
    ][:8]
    normalized_attributes = {
        normalize_category(value).replace(" ", "_") for value in attributes
    }
    manifest = crop_evidence_manifest(
        int(row.get("id", -1)),
        crops,
        partition=partition,
        frame_pose_index=frame_pose_index,
        object_position_world_m=row.get("position_world_m"),
    )
    event = {
        "source": source,
        "event_id": f"object:{int(row.get('id', -1))}:{source}",
        "category": str(reviewed.get("review_category") or "unknown").strip().lower(),
        "description": str(reviewed.get("review_description") or "").strip(),
        "attributes": attributes,
        "confidence": float(reviewed.get("review_confidence") or 0.0),
        "decision": str(reviewed.get("review_decision") or "unknown").strip().lower(),
        "standalone_bounded": "standalone_bounded" in normalized_attributes,
        "component_only": "component_only" in normalized_attributes,
        "raw_response": str(reviewed.get("review_raw_response") or ""),
        "model_id": str(model),
        "prompt_id": f"{source}.v1",
        "prompt_sha256": hashlib.sha256(
            PHYSICAL_RECOVERY_PROMPT.encode("utf-8")
        ).hexdigest(),
        "confirmation_eligible": False,
        "confirmation_ineligibility_reason": "neutral_form_recovery_max_probable",
        **manifest,
    }
    if isinstance(reviewed.get("review_label_contract"), dict):
        event["label_contract"] = dict(reviewed["review_label_contract"])
    return event


def independent_recovery_crop_contract(
    object_id: int,
    primary_crops: list[tuple[Path, bytes]],
    primary_partition: dict,
    verification_crops: list[tuple[Path, bytes]],
    verification_partition: dict,
    frame_pose_index: dict,
    object_position_world_m: object,
) -> dict:
    """Check view independence before spending a verification VLM request."""

    primary = crop_evidence_manifest(
        object_id, primary_crops, partition=primary_partition,
        frame_pose_index=frame_pose_index,
        object_position_world_m=object_position_world_m,
    )
    verification = crop_evidence_manifest(
        object_id, verification_crops, partition=verification_partition,
        frame_pose_index=frame_pose_index,
        object_position_world_m=object_position_world_m,
    )
    primary_views = set(primary.get("crop_image_ids") or [])
    verification_views = set(verification.get("crop_image_ids") or [])
    overlap = (
        len(primary_views & verification_views)
        / min(len(primary_views), len(verification_views))
        if primary_views and verification_views
        else None
    )
    reasons: list[str] = []
    if primary.get("crop_image_ids_complete") is not True:
        reasons.append("recovery_primary_view_provenance_incomplete")
    if verification.get("crop_image_ids_complete") is not True:
        reasons.append("recovery_verification_view_provenance_incomplete")
    if len(primary_views) < PHYSICAL_RECOVERY_MIN_VIEWS_PER_EVENT:
        reasons.append("recovery_primary_insufficient_unique_views")
    if len(verification_views) < PHYSICAL_RECOVERY_MIN_VIEWS_PER_EVENT:
        reasons.append("recovery_verification_insufficient_unique_views")
    if len(primary_views | verification_views) < PHYSICAL_RECOVERY_MIN_TOTAL_UNIQUE_VIEWS:
        reasons.append("recovery_insufficient_total_unique_views")
    if overlap is not None and overlap > PHYSICAL_RECOVERY_MAX_VIEW_OVERLAP:
        reasons.append("recovery_view_overlap_above_threshold")
    if (
        primary.get("evidence_fingerprint_sha256")
        == verification.get("evidence_fingerprint_sha256")
    ):
        reasons.append("recovery_same_evidence_fingerprint")
    pose_diversity = evidence_pose_diversity(
        {**primary, "source": "physical_form_recovery"},
        {**verification, "source": "physical_form_recovery_verification"},
    )
    if not pose_diversity.get("pose_independent"):
        reasons.append(f"recovery_{pose_diversity.get('reason')}")
    return {
        "eligible": not reasons,
        "reason_codes": sorted(set(reasons)),
        "primary_manifest": primary,
        "verification_manifest": verification,
        "view_overlap_coefficient": overlap,
        "unique_view_count": len(primary_views | verification_views),
        "pose_diversity": pose_diversity,
    }


def _best_matching_evidence(category: str, evidence: list[dict]) -> dict | None:
    matches = [item for item in evidence if category_related(category, item.get("category")) >= 0.99]
    return max(
        matches,
        key=lambda item: (SOURCE_WEIGHTS.get(str(item.get("source") or ""), 1.0), float(item.get("confidence") or 0.0)),
    ) if matches else None


def _final_row(row: dict, chosen: dict, evidence: list[dict], scores: dict[str, float], provenance: str) -> dict:
    category = str(chosen.get("review_category") or chosen.get("category") or "").strip().lower()
    description = str(chosen.get("review_description") or chosen.get("description") or "").strip()
    attributes = chosen.get("review_attributes") or chosen.get("attributes") or []
    confidence = float(chosen.get("review_confidence") or chosen.get("confidence") or 0.0)
    support = independent_support(category, evidence)
    _, independence = select_independent_evidence([
        candidate
        for candidate in evidence
        if _positive_candidate(candidate, 0.0)
        and category_related(category, candidate.get("category")) > 0.0
    ])
    if not support:
        return _unresolved_row(
            row,
            evidence,
            scores,
            "no_independent_positive_semantic_evidence",
        )
    tier = "confirmed" if len(support) >= 2 else "probable"
    return {
        **row,
        "category": category,
        "description": description or f"multi-view {category}",
        "attributes": [str(value).strip() for value in attributes if str(value).strip()][:8],
        "semantic_tier": tier,
        "semantic_status": (
            "multi_view_semantic_reconciled"
            if tier == "confirmed"
            else "single_independent_semantic_evidence"
        ),
        "review_decision": "keep",
        "review_confidence": confidence,
        "semantic_gate_reason": provenance,
        "reconciliation_evidence": evidence,
        "semantic_evidence_count": len(support),
        "semantic_evidence_event_ids": [item["event_id"] for item in support],
        "semantic_independence": independence,
        "reconciliation_scores": scores,
    }


def _unresolved_row(
    row: dict,
    evidence: list[dict],
    scores: dict[str, float],
    reason: str = "semantic_identity_unresolved",
) -> dict:
    return {
        **row,
        "category": "unresolved object",
        "description": "Metric multi-view object with unresolved semantic identity.",
        "attributes": [],
        "semantic_tier": "geometry_only",
        "semantic_status": "geometry_valid_semantics_unresolved",
        "review_decision": "unknown",
        "review_confidence": 0.0,
        "semantic_gate_reason": reason,
        "reconciliation_evidence": evidence,
        "semantic_evidence_count": 0,
        "semantic_evidence_event_ids": [],
        "reconciliation_scores": scores,
    }


def _strict_resolved_row(row: dict, evidence: list[dict]) -> dict:
    """Carry a validated ensemble result forward without weighted relabeling."""

    assessment = assess_open_vocabulary_label(
        row.get("label_contract"), minimum_confidence=0.85
    )
    requested_tier = str(row.get("semantic_tier") or "geometry_only")
    if not assessment.get("usable") or requested_tier == "geometry_only":
        unresolved = _unresolved_row(
            row,
            evidence,
            {},
            "strict_open_vocabulary_contract_unresolved",
        )
        unresolved["label_contract"] = dict(row.get("label_contract") or {})
        unresolved["label_contract_assessment"] = assessment
        return unresolved
    maximum = str(assessment.get("maximum_semantic_tier") or "probable")
    tier = (
        "confirmed"
        if requested_tier == "confirmed" and maximum == "confirmed"
        else "probable"
    )
    event_ids = [
        _candidate_event_id(event, str(event.get("source") or ""))
        for event in evidence
        if isinstance(event, dict) and event.get("label_contract")
    ]
    return {
        **row,
        "category": str(assessment.get("category") or "").strip().lower(),
        "description": str(
            assessment.get("description") or row.get("description") or ""
        ).strip(),
        "semantic_tier": tier,
        "semantic_status": (
            "strict_open_vocabulary_confirmed"
            if tier == "confirmed"
            else "strict_open_vocabulary_probable"
        ),
        "review_decision": "keep",
        "review_confidence": float(assessment.get("confidence") or 0.0),
        "semantic_gate_reason": str(
            row.get("semantic_gate_reason") or "strict_open_vocabulary_contract"
        ),
        "reconciliation_evidence": evidence,
        "semantic_evidence_count": len(event_ids),
        "semantic_evidence_event_ids": event_ids,
        "reconciliation_scores": {},
        "label_contract": dict(assessment.get("contract") or {}),
        "label_contract_assessment": assessment,
    }


def _strict_current_resolution(
    row: dict,
    evidence: list[dict],
    scores: dict[str, float],
) -> dict | None:
    """Resolve strict-current rows without entering legacy weighted paths."""

    if isinstance(row.get("label_contract"), dict):
        return _strict_resolved_row(row, evidence)
    if not has_open_vocabulary_contract_evidence(row):
        return None
    unresolved = _unresolved_row(
        row,
        evidence,
        scores,
        "strict_open_vocabulary_contract_unresolved",
    )
    unresolved["strict_open_vocabulary_seen"] = True
    return unresolved


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    prior_state = _load_state(args.prior_state)
    prior_rows = _state_rows(prior_state)
    detector_rows = _detector_rows(_load_state(args.detector_state))
    payload = json.loads(args.candidate_review.read_text(encoding="utf-8"))
    rows = list(payload.get("objects") or [])
    mask_dir = args.mask_dir.expanduser().resolve()
    mask_index = resolve_object_mask_observations(
        prior_state,
        mask_dir,
        include_object_ids=[int(row["id"]) for row in rows],
    )
    frame_pose_index = load_frame_pose_index(args.frames_json.expanduser().resolve())
    minimum = float(np.clip(args.min_confidence, 0.0, 1.0))
    prepared = []
    for row in rows:
        object_id = int(row["id"])
        prior = prior_rows.get(object_id, {})
        detector = detector_rows.get(object_id, {})
        evidence = _candidate_evidence(row, prior, minimum)
        hypotheses = _hypotheses(evidence)
        prepared.append({
            "row": row,
            "prior": prior,
            "detector": detector,
            "evidence": evidence,
            "hypotheses": hypotheses,
            "scores": evidence_scores(hypotheses, evidence, detector),
            "adjudicate": bool(
                args.blind_adjudication
                and needs_adjudication(
                    str(prior.get("category") or ""), row, evidence
                )
            ),
        })

    def resolve(item: dict) -> dict:
        row, prior = item["row"], item["prior"]
        strict_current_seen = has_open_vocabulary_contract_evidence(row)
        resolved = _strict_current_resolution(
            row, item["evidence"], item["scores"]
        )
        if item["adjudicate"] and not strict_current_seen:
            crops, partition = paired_crops(
                mask_index.for_object_id(int(row["id"])),
                max(1, int(args.crops_per_object)),
            )
            if crops:
                reviewed = request_review(
                    args.vllm_url,
                    args.model,
                    {**row, "metric_dimensions_m": prior.get("metric_dimensions_m")},
                    crops,
                    prompt=FINAL_PROMPT,
                    strict_open_vocabulary=True,
                )
                label_assessment = assess_open_vocabulary_label(
                    reviewed.get("review_label_contract"),
                    minimum_confidence=minimum,
                )
                if (
                    label_assessment.get("usable")
                ):
                    blind = {
                        "source": "paired_crop_blind_adjudicator",
                        "event_id": (
                            f"object:{int(row['id'])}:paired_crop_blind_adjudicator"
                        ),
                        "category": reviewed["review_category"],
                        "description": reviewed.get("review_description") or "",
                        "attributes": reviewed.get("review_attributes") or [],
                        "confidence": reviewed.get("review_confidence") or 0.0,
                        "decision": "keep",
                        "label_contract": dict(
                            reviewed.get("review_label_contract") or {}
                        ),
                        "model_id": str(args.model),
                        "prompt_id": "paired_crop_blind_adjudicator.v1",
                        "prompt_sha256": hashlib.sha256(
                            FINAL_PROMPT.encode("utf-8")
                        ).hexdigest(),
                        "confirmation_eligible": True,
                        **crop_evidence_manifest(
                            int(row["id"]), crops, partition=partition,
                            frame_pose_index=frame_pose_index,
                            object_position_world_m=row.get("position_world_m"),
                        ),
                    }
                    item["evidence"].append(blind)
                    hypotheses = _hypotheses(item["evidence"])
                    scores = evidence_scores(hypotheses, item["evidence"], item["detector"])
                    category = max(hypotheses, key=lambda value: scores.get(value, 0.0))
                    chosen = _best_matching_evidence(category, item["evidence"]) or blind
                    resolved = _final_row(
                        row, chosen, item["evidence"], scores,
                        "blind_paired_crop_and_weighted_model_consensus",
                    )

        if resolved is None and item["hypotheses"]:
            category = max(item["hypotheses"], key=lambda value: item["scores"].get(value, 0.0))
            chosen = _best_matching_evidence(category, item["evidence"]) or {
                "category": category, "confidence": 0.75, "decision": "keep"
            }
            resolved = _final_row(
                row,
                chosen,
                item["evidence"],
                item["scores"],
                "weighted_automatic_model_evidence",
            )
        if resolved is None:
            resolved = _unresolved_row(
                row,
                item["evidence"],
                item["scores"],
            )

        metric_dimensions = row.get("metric_dimensions_m") or prior.get(
            "metric_dimensions_m"
        )
        resolved["metric_dimensions_m"] = metric_dimensions
        model_guard: dict | None = None
        guard_crops: list[tuple[Path, bytes]] = []
        guard_partition: dict = {}
        if resolved.get("semantic_tier") != "geometry_only":
            guard_crops, guard_partition = paired_crops(
                mask_index.for_object_id(int(row["id"])),
                max(1, int(args.crops_per_object)),
                source="paired_physical_form_guard",
            )
            if guard_crops:
                model_guard = physical_guard_event(
                    args.vllm_url,
                    args.model,
                    {**row, "metric_dimensions_m": metric_dimensions},
                    str(resolved.get("category") or ""),
                    guard_crops,
                    guard_partition,
                    frame_pose_index,
                )
        physical = assess_physical_form(
            resolved.get("category"),
            metric_dimensions,
            model_guard=model_guard,
        )
        resolved["physical_form_guard"] = model_guard or {}
        resolved["physical_form_assessment"] = physical

        if physical.get("hard_veto") and physical.get("verdict") in {
            "contradiction",
            "part_only",
        }:
            rejected = {
                "category": str(resolved.get("category") or ""),
                "description": str(resolved.get("description") or ""),
                "attributes": list(resolved.get("attributes") or []),
                "review_confidence": float(
                    resolved.get("review_confidence") or 0.0
                ),
                "semantic_tier": str(resolved.get("semantic_tier") or ""),
                "semantic_gate_reason": str(
                    resolved.get("semantic_gate_reason") or ""
                ),
                "physical_form_assessment": physical,
            }
            recovery: dict = {}
            recovery_verification: dict = {}
            recovery_preflight: dict = {
                "eligible": False,
                "reason_codes": ["recovery_primary_not_eligible"],
            }
            guard_recovery = assess_guard_visible_form_recovery(
                physical, metric_dimensions
            )
            if (
                args.enable_recovery_calls
                and not guard_recovery["accepted"]
                and guard_crops
            ):
                recovery = physical_recovery_event(
                    args.vllm_url,
                    args.model,
                    {**row, "metric_dimensions_m": metric_dimensions},
                    guard_crops,
                    guard_partition,
                    frame_pose_index,
                )
                preliminary = assess_physical_recovery(
                    recovery,
                    physical,
                    rejected_category=rejected["category"],
                    metric_dimensions_m=metric_dimensions,
                )
                primary_metric = assess_physical_form(
                    recovery.get("category"), metric_dimensions
                )
                if (
                    preliminary.get("primary", {}).get("valid")
                    and bool(primary_metric.get("metrics"))
                    and not primary_metric.get("hard_veto")
                ):
                    verification_crops, verification_partition = paired_crops(
                        mask_index.for_object_id(int(row["id"])),
                        max(1, int(args.crops_per_object)),
                        source="physical_form_recovery_verification",
                    )
                    recovery_preflight = independent_recovery_crop_contract(
                        int(row["id"]),
                        guard_crops,
                        guard_partition,
                        verification_crops,
                        verification_partition,
                        frame_pose_index,
                        row.get("position_world_m"),
                    )
                    if recovery_preflight["eligible"]:
                        recovery_verification = physical_recovery_event(
                            args.vllm_url,
                            args.model,
                            {**row, "metric_dimensions_m": metric_dimensions},
                            verification_crops,
                            verification_partition,
                            frame_pose_index,
                            source="physical_form_recovery_verification",
                        )
            recovery_assessment = assess_physical_recovery(
                recovery,
                physical,
                verification=recovery_verification,
                rejected_category=rejected["category"],
                metric_dimensions_m=metric_dimensions,
            )
            resolved["physical_form_rejected_semantics"] = rejected
            resolved["physical_form_rejected_assessment"] = physical
            resolved["physical_form_recovery"] = recovery
            resolved["physical_form_recovery_verification"] = (
                recovery_verification
            )
            resolved["physical_form_recovery_verification_preflight"] = (
                recovery_preflight
            )
            resolved["physical_form_recovery_assessment"] = recovery_assessment
            resolved["physical_form_guard_recovery_assessment"] = guard_recovery
            if recovery_assessment["accepted"]:
                recovered_category = str(recovery_assessment["category"])
                resolved["category"] = recovered_category
                resolved["description"] = f"multi-view visible {recovered_category}"
                primary_attributes = {
                    normalize_category(value).replace(" ", "_"): str(value)
                    for value in (recovery.get("attributes") or [])
                    if normalize_category(value).replace(" ", "_")
                    not in {"standalone_bounded", "component_only"}
                }
                verification_attributes = {
                    normalize_category(value).replace(" ", "_"): str(value)
                    for value in (recovery_verification.get("attributes") or [])
                    if normalize_category(value).replace(" ", "_")
                    not in {"standalone_bounded", "component_only"}
                }
                resolved["attributes"] = [
                    primary_attributes[key]
                    for key in sorted(
                        set(primary_attributes) & set(verification_attributes)
                    )
                ][:8]
                resolved["review_confidence"] = float(
                    recovery_assessment.get("confidence") or 0.0
                )
                resolved["review_decision"] = "keep"
                resolved["semantic_tier"] = "probable"
                resolved["semantic_status"] = "physical_form_recovered_probable"
                resolved["semantic_gate_reason"] = (
                    "independent_blind_physical_form_recovery_agreement"
                )
                recovery_events = [recovery, recovery_verification]
                resolved["reconciliation_evidence"] = list(
                    resolved.get("reconciliation_evidence") or []
                ) + recovery_events
                resolved["semantic_evidence_count"] = 2
                resolved["semantic_evidence_event_ids"] = [
                    event["event_id"] for event in recovery_events
                ]
                resolved["semantic_independent_group_count"] = 2
                resolved["semantic_unique_view_count"] = int(
                    recovery_assessment.get("unique_view_count") or 0
                )
                resolved["semantic_max_support_overlap"] = (
                    recovery_assessment.get("view_overlap_coefficient")
                )
                resolved["semantic_confirmation_eligible"] = False
                resolved["semantic_independence"] = {
                    "reason": "independent_recovery_events_agree",
                    "selected_event_ids": [
                        event["event_id"] for event in recovery_events
                    ],
                    "chosen_independent_strength": 2,
                    "selected_unique_view_count": int(
                        recovery_assessment.get("unique_view_count") or 0
                    ),
                    "max_support_overlap_coefficient": (
                        recovery_assessment.get("view_overlap_coefficient")
                    ),
                    "confirmation_eligible": False,
                    "confirmation_ready": False,
                    "tier_reason": (
                        "independent_blind_physical_form_recovery_agreement"
                    ),
                }
                resolved["physical_form_assessment"] = assess_physical_form(
                    recovered_category, metric_dimensions
                )
            else:
                resolved["category"] = "unresolved object"
                resolved["description"] = (
                    "Metric object hidden because the previous noun contradicted "
                    "physical evidence and no standalone visible form was recovered."
                )
                resolved["attributes"] = []
                resolved["review_confidence"] = 0.0
                resolved["review_decision"] = "unknown"
                resolved["semantic_tier"] = "geometry_only"
                resolved["semantic_status"] = (
                    "physical_component_or_contradiction_suppressed"
                )
                resolved["semantic_gate_reason"] = (
                    "physical_form_recovery_failed_closed"
                )
                resolved["semantic_evidence_count"] = 0
                resolved["semantic_evidence_event_ids"] = []
        elif (
            resolved.get("semantic_tier") == "confirmed"
            and not physical["confirmation_allowed"]
        ):
            resolved["semantic_tier"] = "probable"
            resolved["semantic_status"] = "physical_form_confirmation_withheld"
            resolved["semantic_gate_reason"] = "physical_form_confirmation_withheld"
        return resolved

    results: list[dict | None] = [None] * len(prepared)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {executor.submit(resolve, item): index for index, item in enumerate(prepared)}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            index = futures[future]
            results[index] = future.result()
            row = results[index]
            print(f"[semantic-reconcile] {completed}/{len(results)} object={row['id']} category={row['category']}")

    catalog = [row for row in results if row is not None]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "farm.semantic-reconciliation.v1",
        "status": "PASS",
        "objects": len(catalog),
        "adjudicated_conflicts": int(sum(bool(item["adjudicate"]) for item in prepared)),
        "blind_adjudication_enabled": bool(args.blind_adjudication),
        "recovery_calls_enabled": bool(args.enable_recovery_calls),
        "physical_form_guarded": int(sum(bool(row.get("physical_form_guard")) for row in catalog)),
        "physical_form_hard_vetoes": int(sum(
            bool((row.get("physical_form_rejected_assessment") or row.get(
                "physical_form_assessment"
            ) or {}).get("hard_veto"))
            for row in catalog
        )),
        "physical_form_recoveries_attempted": int(sum(
            bool(row.get("physical_form_recovery_assessment")) for row in catalog
        )),
        "physical_form_recovery_verifications_attempted": int(sum(
            bool(row.get("physical_form_recovery_verification")) for row in catalog
        )),
        "physical_form_recoveries_accepted": int(sum(
            bool((row.get("physical_form_recovery_assessment") or {}).get("accepted"))
            for row in catalog
        )),
        "physical_form_recoveries_suppressed": int(sum(
            (row.get("physical_form_recovery_assessment") or {}).get("action")
            == "suppress_geometry_only"
            for row in catalog
        )),
        "category_counts": dict(Counter(str(row["category"]) for row in catalog).most_common()),
        "tier_counts": dict(Counter(str(row["semantic_tier"]) for row in catalog)),
        "policy": "automatic paired raw-crop and mask-grounded multi-view model adjudication; no object ids or manual class hints",
        "duration_seconds": time.perf_counter() - started,
        "mask_observation_contract": mask_index.diagnostics,
        "sources": {
            "prior_state": str(args.prior_state.resolve()),
            "candidate_review": str(args.candidate_review.resolve()),
            "detector_state": str(args.detector_state.resolve()),
            "mask_dir": str(mask_dir),
        },
    }
    write_json_atomic(output_dir / "semantic_reconciled_catalog.json", catalog)
    write_json_atomic(output_dir / "semantic_reconciliation_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
