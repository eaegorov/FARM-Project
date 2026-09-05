#!/usr/bin/env python3
"""Build an evidence-tiered semantic catalog without scene-specific vocabulary."""

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
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scripts.semantics.review_farm_object_crops import (
    choose_crops,
    crop_candidates,
    hydrate_rows_from_state,
    request_review,
    resolve_world_up_contract,
)
from scene_graph.captioning.evidence import (
    crop_evidence_manifest,
    evidence_event_id,
    evidence_view_ids,
    gravity_upright_evidence_manifest,
    load_frame_pose_index,
    partition_crop_candidates,
    select_independent_evidence,
)
from scene_graph.captioning.label_contract import (
    OPEN_VOCABULARY_JSON_INSTRUCTIONS,
    SHARED_FORM_HYPERNYM_SOURCE,
    adaptive_alternate_plan,
    assess_open_vocabulary_identity,
    assess_open_vocabulary_label,
    event_label_assessment,
    has_open_vocabulary_contract_evidence,
    normalize_open_noun,
    shared_explicit_form_hypernym,
)
from scene_graph.map_update.mask_observations import resolve_object_mask_observations

try:
    from scene_graph.map_update.filtering import UNINFORMATIVE_YOLOE_LABELS
except ImportError:
    UNINFORMATIVE_YOLOE_LABELS = set()


BLIND_RECHECK_PROMPT = """The images are automatically selected masked views of
one proposed physical 3D entity. Perform an independent open-vocabulary review
without access to any previous label. The outlined natural-colour target pixels
are primary evidence. Stable target-linked context such as mounting, support,
scale, placement, and repeated interaction may disambiguate identity, but scene
type or a neighbouring object alone may never determine the noun. Explicitly
judge whether the mask covers the complete visible target before identity.
""" + OPEN_VOCABULARY_JSON_INSTRUCTIONS


TIEBREAKER_PROMPT = """The images are independently selected mask-grounded
views of one proposed physical 3D entity. Perform a fresh open-vocabulary review
without access to any previous label. Use target-owned pixels across views as
primary evidence and stable target-linked placement/mounting/scale as supporting
evidence. If the supplied context is insufficient, or the pixels do not support
one bounded identity, return unknown. Explicitly audit visible mask completeness.
""" + OPEN_VOCABULARY_JSON_INSTRUCTIONS


FUSION_PROMPT = """Deprecated candidate-conditioned fusion. It is retained only
for replay compatibility and is not used by the default standard call planner.
If explicitly enabled, it must not invent a noun absent from blind target evidence.
""" + OPEN_VOCABULARY_JSON_INSTRUCTIONS


NEUTRAL_FORM_PROMPT = """Deprecated forced-neutral fallback. It is retained only
for artifact compatibility and is not called by the default standard planner.
Abstain when no complete bounded physical head noun is supported.
""" + OPEN_VOCABULARY_JSON_INSTRUCTIONS


CATEGORY_NORMALIZATION = {
    "palet": "pallet",
    "racking": "shelf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument(
        "--scene-state", type=Path,
        help="Raw mapping state containing per-object YOLOE category evidence.",
    )
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument(
        "--geometry-report",
        type=Path,
        help="Optional geometry audit supplying $.up.vector for crop orientation.",
    )
    parser.add_argument(
        "--world-up-vector",
        default=None,
        metavar="X,Y,Z",
        help=(
            "Exact world up. Priority: CLI, geometry report up.vector, default 0,-1,0."
        ),
    )
    parser.add_argument(
        "--natural-context-panel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compose upright MASK and CONTEXT panels for alternate blind review.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--model", default="qwen3-vl-8b")
    parser.add_argument("--crops-per-object", type=int, default=6)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--probable-confidence", type=float, default=0.90)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def normalize_category(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def informative(candidate: dict | None, min_confidence: float) -> bool:
    if not candidate:
        return False
    category = normalize_category(candidate.get("category"))
    return bool(
        str(candidate.get("decision") or "unknown").lower() == "keep"
        and category
        and category != "unknown"
        and category not in UNINFORMATIVE_YOLOE_LABELS
        and float(candidate.get("confidence") or 0.0) >= min_confidence
    )


def candidate_from_review(prefix: str, row: dict) -> dict | None:
    explicit_events = [
        dict(event)
        for event in (row.get("semantic_evidence") or [])
        if isinstance(event, dict)
    ]

    def explicit(*sources: str) -> dict | None:
        wanted = set(sources)
        return next(
            (
                event
                for event in explicit_events
                if str(event.get("source") or "") in wanted
            ),
            None,
        )

    if prefix == "final":
        value = {
            "source": "strict_two_pass",
            "category": row.get("review_category"),
            "description": row.get("review_description"),
            "attributes": row.get("review_attributes") or [],
            "confidence": row.get("review_confidence", 0.0),
            "decision": row.get("review_decision", "unknown"),
            "confirmation_eligible": False,
            "confirmation_ineligibility_reason": "derived_two_pass_summary",
        }
    elif prefix == "initial":
        manifest = explicit(
            "initial_blind_review",
            "contextual_dual_panel_review",
            "blind_pass_a",
        )
        if manifest is not None:
            return manifest
        raw = row.get("review_initial")
        if not isinstance(raw, dict):
            return None
        value = {
            "source": "blind_pass_a",
            "category": raw.get("category"),
            "description": raw.get("description"),
            "attributes": raw.get("attributes") or [],
            "confidence": raw.get("confidence", 0.0),
            "decision": raw.get("decision", "unknown"),
            "confirmation_eligible": False,
            "confirmation_ineligibility_reason": "legacy_missing_crop_provenance",
        }
    elif prefix == "verification":
        manifest = explicit("independent_verification", "verification_pass")
        if manifest is not None:
            return manifest
        raw_text = str(row.get("verification_raw_response") or "").strip()
        if not raw_text.startswith("{"):
            return None
        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError:
            return None
        value = {
            "source": "verification_pass",
            "category": raw.get("category"),
            "description": raw.get("description"),
            "attributes": raw.get("attributes") or [],
            "confidence": raw.get("confidence", 0.0),
            "decision": raw.get("decision", "unknown"),
            "confirmation_eligible": False,
            "confirmation_ineligibility_reason": "candidate_conditioned_or_legacy_verification",
        }
    else:
        raise ValueError(prefix)
    return value


def request_candidate(
    url: str,
    model: str,
    row: dict,
    crops: list,
    prompt: str,
    source: str,
    *,
    partition: dict | None = None,
    confirmation_eligible: bool = True,
    ineligibility_reason: str = "",
    derived_from_event_ids: list[str] | None = None,
    frame_pose_index: dict | None = None,
    world_up_vector: object = (0.0, -1.0, 0.0),
    world_up_source: str = "default:[0,-1,0]",
) -> dict:
    result = request_review(
        url, model, row, crops, prompt=prompt,
        strict_open_vocabulary=True,
    )
    event = {
        "source": source,
        "event_id": f"object:{int(row.get('id', -1))}:{source}",
        "category": result.get("review_category"),
        "description": result.get("review_description"),
        "attributes": result.get("review_attributes") or [],
        "confidence": result.get("review_confidence", 0.0),
        "decision": result.get("review_decision", "unknown"),
        "raw_response": result.get("review_raw_response", ""),
        "model_id": str(model),
        "prompt_id": f"{source}.v1",
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "confirmation_eligible": bool(confirmation_eligible),
    }
    if isinstance(result.get("review_label_contract"), dict):
        event["label_contract"] = dict(result["review_label_contract"])
    if ineligibility_reason:
        event["confirmation_ineligibility_reason"] = str(ineligibility_reason)
    if derived_from_event_ids:
        event["derived_from_event_ids"] = sorted(set(derived_from_event_ids))
    event.update(
        crop_evidence_manifest(
            int(row.get("id", -1)),
            crops,
            partition=partition,
            frame_pose_index=frame_pose_index,
            object_position_world_m=row.get("position_world_m"),
        )
    )
    event["gravity_upright_normalization"] = gravity_upright_evidence_manifest(
        crops,
        frame_pose_index=frame_pose_index,
        world_up_vector=world_up_vector,
        world_up_source=world_up_source,
    )
    return event


def alternate_blind_fold_audit(
    initial_event: dict | None,
    alternate_manifest: dict,
    *,
    frame_pose_index: dict | None = None,
) -> dict:
    """Prove that blind pass C uses no source image from the initial pass."""

    initial_ids = evidence_view_ids(initial_event or {})
    alternate_ids = evidence_view_ids(alternate_manifest)
    initial_set = set(initial_ids or ())
    alternate_set = set(alternate_ids or ())
    overlap = sorted(initial_set.intersection(alternate_set))

    def source_identity(image_id: int) -> str | None:
        pose = (frame_pose_index or {}).get(int(image_id))
        if not isinstance(pose, dict):
            return None
        value = str(pose.get("source_image") or pose.get("rgb_path") or "").strip()
        return value or None

    initial_source_rows = {
        image_id: source_identity(image_id) for image_id in sorted(initial_set)
    }
    alternate_source_rows = {
        image_id: source_identity(image_id) for image_id in sorted(alternate_set)
    }
    source_complete = bool(
        initial_source_rows
        and alternate_source_rows
        and all(initial_source_rows.values())
        and all(alternate_source_rows.values())
    )
    initial_sources = {value for value in initial_source_rows.values() if value}
    alternate_sources = {value for value in alternate_source_rows.values() if value}
    source_overlap = sorted(initial_sources.intersection(alternate_sources))
    reasons: list[str] = []
    if initial_ids is None:
        reasons.append("initial_crop_provenance_incomplete")
    if alternate_ids is None:
        reasons.append("alternate_crop_provenance_incomplete")
    if not initial_set:
        reasons.append("initial_crop_image_ids_empty")
    if not alternate_set:
        reasons.append("alternate_crop_image_ids_empty")
    if not source_complete:
        reasons.append("source_image_identity_provenance_incomplete")
    if overlap:
        reasons.append("blind_fold_source_image_overlap")
    if source_overlap:
        reasons.append("blind_fold_physical_source_overlap")
    eligible = not reasons
    return {
        "schema": "farm.semantic-blind-fold-separation.v1",
        "status": "disjoint" if eligible else "unavailable",
        "eligible": eligible,
        "initial_event_id": evidence_event_id(initial_event or {}),
        "initial_source": str((initial_event or {}).get("source") or ""),
        "initial_image_ids": sorted(initial_set),
        "alternate_image_ids": sorted(alternate_set),
        "overlap_image_ids": overlap,
        "source_image_identity_provenance_complete": source_complete,
        "initial_source_images": sorted(initial_sources),
        "alternate_source_images": sorted(alternate_sources),
        "overlap_source_images": source_overlap,
        "reason_codes": reasons or ["disjoint_source_image_ids"],
    }


def apply_alternate_fold_gate(call_plan: dict, fold_audit: dict) -> dict:
    """Disable an otherwise useful model call when folds are not disjoint."""

    output = dict(call_plan)
    output["request_alternate_before_fold_gate"] = bool(output.get("request_alternate"))
    output["blind_fold_separation"] = dict(fold_audit)
    if fold_audit.get("eligible") is not True:
        output["request_alternate"] = False
        output["reason"] = "alternate_blind_fold_not_provably_disjoint"
    return output


def load_detector_votes(path: Path | None) -> dict[int, dict[str, float]]:
    if path is None:
        return {}
    import torch

    payload = torch.load(path.expanduser(), map_location="cpu", weights_only=False)
    state = payload.get("state", payload) if isinstance(payload, dict) else {}
    object_ids = state.get("object_id")
    rows = state.get("object_detection_category_conf") or []
    if object_ids is None:
        return {}
    ids = [int(value) for value in object_ids.detach().cpu().tolist()]
    result: dict[int, dict[str, float]] = {}
    for index, object_id in enumerate(ids):
        if index >= len(rows) or not isinstance(rows[index], dict):
            continue
        votes = {
            str(label).strip().lower(): float(confidence)
            for label, confidence in rows[index].items()
            if str(label).strip() and np.isfinite(float(confidence))
        }
        if votes:
            result[object_id] = votes
    return result


def build_fusion_prompt(row: dict, candidates: list[dict]) -> str:
    visual = []
    for candidate in candidates:
        category = str(candidate.get("category") or "").strip()
        if category and category.lower() != "unknown" and category not in visual:
            visual.append(category)
    detector = sorted(
        (row.get("detector_category_votes") or {}).items(),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return (
        FUSION_PROMPT.rstrip()
        + "\n\nBlind visual observations: "
        + (", ".join(visual) if visual else "none stable")
        + ".\nDetector proposals: "
        + (", ".join(f"{label} ({float(score):.2f})" for label, score in detector) if detector else "none")
        + "."
    )


def fusion_support(category: object, candidates: list[dict], detector: dict) -> str:
    target = normalize_category(category)
    if not target:
        return "none"
    visual = {normalize_category(value.get("category")) for value in candidates}
    proposals = {normalize_category(value) for value in detector}
    if target in visual or target in proposals:
        return "exact"
    target_tokens = set(target.split())
    for value in visual | proposals:
        tokens = set(value.split())
        if target_tokens and tokens and (target_tokens <= tokens or tokens <= target_tokens):
            return "hierarchical"
    return "novel"


def best_description(row: dict, candidates: list[dict]) -> str:
    ordered = sorted(
        candidates,
        key=lambda value: float(value.get("confidence") or 0.0),
        reverse=True,
    )
    for candidate in ordered:
        description = str(candidate.get("description") or "").strip()
        if description:
            return description
    return str(row.get("review_description") or "").strip()


def independent_evidence_decision(
    candidates: list[dict],
    negative_sources: list[str],
    probable_confidence: float,
) -> tuple[str, dict | None, str]:
    """Resolve only genuinely independent semantic votes.

    A candidate-informed arbiter is not an independent vote. Repeated prompts
    to the same model are correlated evidence, not an independent sensor.
    Therefore an explicit ``unknown`` prevents later same-model guesses from
    becoming a presentation label, even when those guesses repeat each other.
    """

    by_category: dict[str, list[dict]] = {}
    for candidate in candidates:
        category = normalize_category(candidate.get("category"))
        if category:
            by_category.setdefault(category, []).append(candidate)
    independently_supported: list[tuple[str, list[dict]]] = []
    for category, events in by_category.items():
        selected, diagnostics = select_independent_evidence(events)
        if diagnostics["confirmation_ready"]:
            independently_supported.append((category, selected))
    if len(independently_supported) == 1 and not negative_sources:
        agreed, selected = independently_supported[0]
        chosen = max(
            (
                value
                for value in selected
                if normalize_category(value.get("category")) == agreed
            ),
            key=lambda value: float(value.get("confidence") or 0.0),
        )
        return "confirmed", chosen, "independent_disjoint_view_consensus"
    distinct = set(by_category)
    if (
        len(distinct) == 1
        and not negative_sources
        and max(float(value.get("confidence") or 0.0) for value in candidates)
        >= probable_confidence
    ):
        chosen = max(candidates, key=lambda value: float(value.get("confidence") or 0.0))
        return "probable", chosen, "correlated_or_single_high_confidence_evidence"
    reason = (
        "conflicting_independent_evidence"
        if len(distinct) > 1 or len(independently_supported) > 1
        else "negative_independent_evidence"
        if negative_sources
        else "semantic_identity_unresolved"
    )
    return "geometry_only", None, reason


def canonical_open_vocabulary_event(event: dict, result: dict) -> dict:
    """Materialize the deterministic contract assessment on an evidence event."""

    output = dict(event)
    original = normalize_open_noun(
        (result.get("contract") or {}).get("category")
    )
    category = normalize_open_noun(result.get("category"))
    contract = dict(result.get("contract") or {})
    contract["proposed_category"] = original
    contract["category"] = category
    contract["category_role"] = result.get("category_role")
    if category != original and result.get("category_role") == "whole_form":
        contract["specificity"] = "generic_form"
    output.update({
        "category": category,
        "description": result.get("description") or event.get("description") or "",
        "attributes": list(result.get("attributes") or []),
        "confidence": float(result.get("confidence") or 0.0),
        "label_contract": contract,
        "maximum_semantic_tier": result.get("maximum_semantic_tier"),
    })
    return output


def assessed_open_vocabulary_event(
    event: dict, *, minimum_confidence: float
) -> tuple[dict | None, dict]:
    """Return a semantic candidate while keeping mask completeness separate."""

    result = event_label_assessment(
        event, minimum_confidence=minimum_confidence
    )
    identity_only = False
    if not result.get("usable"):
        identity = assess_open_vocabulary_identity(
            event.get("label_contract"),
            minimum_confidence=minimum_confidence,
        )
        if identity.get("usable"):
            result = identity
            identity_only = True
    if not result.get("usable"):
        return None, result
    canonical = canonical_open_vocabulary_event(event, result)
    if identity_only:
        canonical.update({
            "confirmation_eligible": False,
            "confirmation_ineligibility_reason": (
                "partial_or_component_identity_not_whole_object_safe"
            ),
            "maximum_semantic_tier": "probable",
        })
    return canonical, result


def adaptive_open_vocabulary_decision(
    events: list[dict],
    probable_confidence: float,
) -> tuple[str, dict | None, str, list[str]]:
    """Resolve strict label-contract events without inferred synonyms.

    Exact independent head nouns may confirm. Different head nouns can share
    only a form hypernym explicitly emitted by both events, and that result is
    capped at probable. Candidate-conditioned or malformed events never add a
    noun.
    """

    assessed: list[tuple[dict, dict]] = []
    negative_sources: list[str] = []
    for event in events:
        if not isinstance(event.get("label_contract"), dict):
            continue
        canonical, result = assessed_open_vocabulary_event(
            event, minimum_confidence=0.85
        )
        if canonical is not None:
            assessed.append((canonical, result))
        else:
            negative_sources.append(str(event.get("source") or "blind_event"))
    if not assessed:
        return (
            "geometry_only", None, "open_vocabulary_identity_unresolved",
            sorted(set(negative_sources)),
        )

    by_category: dict[str, list[tuple[dict, dict]]] = {}
    for event, result in assessed:
        by_category.setdefault(
            normalize_open_noun(result.get("category")), []
        ).append((event, result))

    if len(by_category) > 1:
        contracts = [result["contract"] for _, result in assessed]
        shared = ""
        if len(contracts) >= 2:
            shared = shared_explicit_form_hypernym(contracts[0], contracts[1])
            for contract in contracts[2:]:
                if not shared or normalize_open_noun(
                    contract.get("form_hypernym")
                ) != shared:
                    shared = ""
                    break
        if shared:
            event, result = max(
                assessed, key=lambda pair: float(pair[1].get("confidence") or 0.0)
            )
            chosen = {
                **event,
                "source": SHARED_FORM_HYPERNYM_SOURCE,
                "category": shared,
                "description": f"multi-view visible {shared}",
                "attributes": [],
                "confidence": min(
                    float(value.get("confidence") or 0.0)
                    for _, value in assessed
                ),
                "confirmation_eligible": False,
                "maximum_semantic_tier": "probable",
                "confirmation_ineligibility_reason": (
                    "explicit_shared_form_hypernym_max_probable"
                ),
            }
            contributor_event_ids = sorted({
                evidence_event_id(value)
                for value, _ in assessed
                if evidence_event_id(value)
            })
            contributor_categories = sorted({
                normalize_open_noun(value.get("category"))
                for value, _ in assessed
                if normalize_open_noun(value.get("category"))
            })
            derivation_digest = hashlib.sha256(
                "\n".join(contributor_event_ids).encode("utf-8")
            ).hexdigest()
            chosen.update({
                "event_id": f"shared-form-hypernym:{derivation_digest[:24]}",
                "derived_from_event_ids": contributor_event_ids,
                "derived_from_categories": contributor_categories,
                "derivation_policy": "identical_explicit_form_hypernym",
            })
            contract = dict(result["contract"])
            contract.update({
                "category": shared,
                "category_role": "whole_form",
                "specificity": "generic_form",
            })
            chosen["label_contract"] = contract
            return (
                "probable", chosen, "explicit_shared_form_hypernym",
                sorted(set(negative_sources)),
            )
        return (
            "geometry_only", None, "conflicting_open_vocabulary_head_nouns",
            sorted(set(negative_sources)),
        )

    _category, values = next(iter(by_category.items()))
    candidates = [event for event, _ in values]
    selected, diagnostics = select_independent_evidence(candidates)
    chosen_event, chosen_result = max(
        values, key=lambda pair: float(pair[1].get("confidence") or 0.0)
    )
    confirmation_allowed = all(
        result.get("maximum_semantic_tier") == "confirmed"
        for event, result in values
        if event in selected
    )
    if (
        diagnostics.get("confirmation_ready")
        and len(selected) >= 2
        and confirmation_allowed
        and not negative_sources
    ):
        return (
            "confirmed", chosen_event, "independent_disjoint_view_consensus",
            [],
        )
    highest = max(float(result.get("confidence") or 0.0) for _, result in values)
    if len(values) >= 2 or highest >= float(probable_confidence):
        reason = (
            "correlated_exact_open_vocabulary_agreement"
            if len(values) >= 2
            else "single_high_confidence_open_vocabulary_label"
        )
        return (
            "probable", chosen_event, reason, sorted(set(negative_sources))
        )
    return (
        "geometry_only", None, "single_open_vocabulary_label_below_probable_gate",
        sorted(set(negative_sources)),
    )


def neutral_fallback_allowed(candidates: list[dict]) -> bool:
    """Neutral form is allowed only when no informative vote survived."""

    return not candidates


def output_row(
    row: dict,
    chosen: dict | None,
    tier: str,
    candidates: list[dict],
    reason: str,
    negative_sources: list[str] | None = None,
    semantic_evidence: list[dict] | None = None,
) -> dict:
    candidates = list(candidates)
    evidence_rows = list(semantic_evidence or candidates)
    if (
        reason == "explicit_shared_form_hypernym"
        and isinstance(chosen, dict)
        and chosen.get("source") == SHARED_FORM_HYPERNYM_SOURCE
    ):
        chosen_id = evidence_event_id(chosen)
        if all(evidence_event_id(value) != chosen_id for value in candidates):
            candidates.append(chosen)
        if all(evidence_event_id(value) != chosen_id for value in evidence_rows):
            evidence_rows.append(chosen)
    category = str((chosen or {}).get("category") or "unresolved object").strip().lower()
    category = CATEGORY_NORMALIZATION.get(normalize_category(category), category)
    # Keep the description coupled to the winning class.  Selecting the most
    # confident description across all candidates could pair a rejected label
    # (for example workbench) with the accepted physical class (box).
    description = str((chosen or {}).get("description") or "").strip()
    if not description:
        description = best_description(row, candidates)
    output = {
        "id": int(row["id"]),
        "observations": int(row.get("observations") or 0),
        "position_world_m": row.get("position_world_m"),
        "cov6": row.get("cov6"),
        "metric_dimensions_m": row.get("metric_dimensions_m"),
        "category": category if tier != "geometry_only" else "unresolved object",
        "description": description or "Metric multi-view object with unresolved semantic identity.",
        "attributes": [
            str(value).strip()
            for value in ((chosen or {}).get("attributes") or [])
            if str(value).strip()
        ][:8],
        "semantic_tier": tier,
        "semantic_status": {
            "confirmed": "vlm_ensemble_confirmed",
            "probable": "vlm_single_pass_probable",
            "geometry_only": "geometry_valid_semantics_unresolved",
        }[tier],
        "review_decision": "keep" if tier in {"confirmed", "probable"} else "unknown",
        "review_confidence": float((chosen or {}).get("confidence") or 0.0),
        "semantic_gate_reason": reason,
        "negative_evidence_sources": sorted(set(negative_sources or [])),
        "candidates": candidates,
        "semantic_evidence": evidence_rows,
    }
    if has_open_vocabulary_contract_evidence(row) or (
        has_open_vocabulary_contract_evidence({
            "candidates": candidates,
            "semantic_evidence": output["semantic_evidence"],
        })
    ):
        output["strict_open_vocabulary_seen"] = True
    label_contract = (chosen or {}).get("label_contract")
    if isinstance(label_contract, dict):
        output["label_contract_schema"] = label_contract.get("schema")
        output["label_contract"] = dict(label_contract)
        output["category_role"] = label_contract.get("category_role")
        output["form_hypernym"] = label_contract.get("form_hypernym")
        output["topology"] = label_contract.get("topology")
        output["complete_bounded"] = label_contract.get("complete_bounded")
        output["carrier_category"] = label_contract.get("carrier_category")
        output["payload_categories"] = list(
            label_contract.get("payload_categories") or []
        )
    return output


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    payload = json.loads(args.review.expanduser().read_text(encoding="utf-8"))
    geometry_payload = (
        json.loads(args.geometry_report.expanduser().read_text(encoding="utf-8"))
        if args.geometry_report
        else None
    )
    world_up_contract = resolve_world_up_contract(
        args.world_up_vector, geometry_payload
    )
    rows = list(payload.get("objects") or [])
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.mask_dir.expanduser().resolve()
    frame_pose_index = load_frame_pose_index(args.frames_json.expanduser().resolve())
    min_conf = float(np.clip(args.min_confidence, 0.0, 1.0))
    probable_conf = float(np.clip(args.probable_confidence, min_conf, 1.0))
    detector_votes = load_detector_votes(args.scene_state)
    if args.scene_state:
        wrapper = torch.load(
            args.scene_state.expanduser(), map_location="cpu", weights_only=False
        )
        scene_state = (
            wrapper.get("state")
            if isinstance(wrapper, dict) and isinstance(wrapper.get("state"), dict)
            else wrapper
        )
        if not isinstance(scene_state, dict):
            raise TypeError(f"Unsupported scene state: {args.scene_state}")
    else:
        scene_state = {"object_id": [int(row["id"]) for row in rows]}
    metric_context_hydration = hydrate_rows_from_state(rows, scene_state)
    mask_index = resolve_object_mask_observations(
        scene_state,
        mask_dir,
        include_object_ids=[int(row["id"]) for row in rows],
    )
    arbitration_times: list[float] = []
    alternate_fold_audits: list[dict] = []

    alternate_prompt = TIEBREAKER_PROMPT
    if args.natural_context_panel:
        alternate_prompt += (
            "\n\nEach image contains the same upright observation twice: MASK "
            "on the left preserves target-owned pixels, while CONTEXT on the "
            "right preserves natural surroundings and outlines the target in "
            "yellow. Use context only for mounting, support, placement and scale; "
            "never transfer a neighbour's identity to the target."
        )

    def process(row: dict) -> dict:
        row = dict(row)
        row["detector_category_votes"] = detector_votes.get(int(row["id"]), {})
        all_crops = crop_candidates(
            mask_index.for_object_id(int(row["id"])),
            frames_json=args.frames_json,
            natural_context_panel=bool(args.natural_context_panel),
            frame_pose_index=frame_pose_index,
            world_up_vector=world_up_contract["vector"],
            world_up_source=world_up_contract["source"],
        )

        def crops_for(source: str) -> tuple[list, dict]:
            partitioned, partition = partition_crop_candidates(all_crops, source)
            partitioned = sorted(
                partitioned, key=lambda item: len(item[1]), reverse=True
            )
            return (
                choose_crops(partitioned, max(1, int(args.crops_per_object))),
                partition,
            )

        initial = candidate_from_review("initial", row)
        # Old review artifacts did not preserve the first-pass event. Keep a
        # compatibility fallback, but never count the derived summary as an
        # independent second observation.
        if initial is None:
            initial = candidate_from_review("final", row)
        semantic_evidence = [initial] if initial is not None else []

        alternate_crops, alternate_partition = crops_for("blind_pass_c")
        alternate_manifest = {
            "source": "blind_pass_c",
            "event_id": f"object:{int(row.get('id', -1))}:blind_pass_c",
            "confirmation_eligible": True,
        }
        alternate_manifest.update(
            crop_evidence_manifest(
                int(row.get("id", -1)),
                alternate_crops,
                partition=alternate_partition,
                frame_pose_index=frame_pose_index,
                object_position_world_m=row.get("position_world_m"),
            )
        )
        alternate_manifest["gravity_upright_normalization"] = (
            gravity_upright_evidence_manifest(
                alternate_crops,
                frame_pose_index=frame_pose_index,
                world_up_vector=world_up_contract["vector"],
                world_up_source=world_up_contract["source"],
            )
        )
        fold_audit = alternate_blind_fold_audit(
            initial,
            alternate_manifest,
            frame_pose_index=frame_pose_index,
        )
        alternate_fold_audits.append(fold_audit)
        if fold_audit["eligible"] is not True:
            alternate_manifest["confirmation_eligible"] = False
            alternate_manifest["confirmation_ineligibility_reason"] = (
                "alternate_blind_fold_not_provably_disjoint"
            )
        call_plan = apply_alternate_fold_gate(
            adaptive_alternate_plan(
                initial,
                alternate_manifest,
                probable_confidence=probable_conf,
            ),
            fold_audit,
        )
        if alternate_crops and call_plan["request_alternate"]:
            alternate_started = time.perf_counter()
            alternate = request_candidate(
                args.vllm_url,
                args.model,
                row,
                alternate_crops,
                alternate_prompt,
                "blind_pass_c",
                partition=alternate_partition,
                frame_pose_index=frame_pose_index,
                world_up_vector=world_up_contract["vector"],
                world_up_source=world_up_contract["source"],
            )
            alternate["seconds"] = time.perf_counter() - alternate_started
            arbitration_times.append(float(alternate["seconds"]))
            semantic_evidence.append(alternate)

        if any(
            isinstance(value.get("label_contract"), dict)
            for value in semantic_evidence
        ):
            tier, chosen, reason, negative_sources = (
                adaptive_open_vocabulary_decision(
                    semantic_evidence,
                    probable_confidence=probable_conf,
                )
            )
            normalized_evidence = []
            candidates = []
            for value in semantic_evidence:
                normalized, _assessment = assessed_open_vocabulary_event(
                    value, minimum_confidence=min_conf
                )
                if normalized is None:
                    normalized_evidence.append(value)
                else:
                    normalized_evidence.append(normalized)
                    candidates.append(normalized)
            semantic_evidence = normalized_evidence
        else:
            negative_sources = [
                str(value.get("source") or "review")
                for value in semantic_evidence
                if not informative(value, min_conf)
            ]
            candidates = [
                value
                for value in semantic_evidence
                if informative(value, min_conf)
            ]
            tier, chosen, reason = independent_evidence_decision(
                candidates, negative_sources, probable_conf
            )

        result = output_row(
            row,
            chosen,
            tier,
            candidates,
            reason,
            negative_sources,
            semantic_evidence,
        )
        result["adaptive_call_plan"] = call_plan
        result["alternate_preflight_manifest"] = alternate_manifest
        result["alternate_fold_audit"] = fold_audit
        result["alternate_requested"] = bool(
            alternate_crops and call_plan["request_alternate"]
        )
        return result

    results: list[dict | None] = [None] * len(rows)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(args.workers))
    ) as executor:
        future_map = {
            executor.submit(process, row): index for index, row in enumerate(rows)
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_map), start=1
        ):
            index = future_map[future]
            results[index] = future.result()
            current = results[index]
            print(
                f"[semantic-ensemble] {completed}/{len(rows)} "
                f"object={current['id']} tier={current['semantic_tier']} "
                f"category={current['category']}",
                flush=True,
            )

    catalog = [value for value in results if value is not None]
    tier_counts = Counter(str(row["semantic_tier"]) for row in catalog)
    category_counts = Counter(
        str(row["category"])
        for row in catalog
        if row["semantic_tier"] != "geometry_only"
    )
    fold_status_counts = Counter(
        str(value.get("status") or "unknown") for value in alternate_fold_audits
    )
    fold_reason_counts = Counter(
        str(reason)
        for value in alternate_fold_audits
        for reason in value.get("reason_codes") or []
    )
    report = {
        "schema": "farm.semantic-ensemble.v1",
        "policy": (
            "strict open-vocabulary mask-grounded pass A for every object; one "
            "alternate blind crop fold only for unresolved/low-confidence or "
            "pose-confirmable candidates; exact independent head nouns may "
            "confirm, shared explicit form hypernyms are capped at probable, "
            "and no conditioned fusion or forced-neutral noun is used"
        ),
        "source_review": str(args.review.expanduser().resolve()),
        "geometry_report": (
            str(args.geometry_report.expanduser().resolve())
            if args.geometry_report
            else None
        ),
        "gravity_upright_normalization": world_up_contract,
        "natural_context_panel": bool(args.natural_context_panel),
        "object_count": len(catalog),
        "tier_counts": dict(tier_counts),
        "named_count": int(tier_counts["confirmed"] + tier_counts["probable"]),
        "category_counts": dict(category_counts.most_common()),
        "crops_per_recheck": max(1, int(args.crops_per_object)),
        "mask_observation_contract": mask_index.diagnostics,
        "metric_context_hydration": metric_context_hydration,
        "alternate_blind_fold_audit": {
            "schema": "farm.semantic-blind-fold-separation.v1",
            "status_counts": dict(sorted(fold_status_counts.items())),
            "reason_counts": dict(sorted(fold_reason_counts.items())),
            "policy": (
                "blind_pass_c is never requested when initial and alternate "
                "source image IDs overlap or either provenance is incomplete"
            ),
        },
        "timing": {
            "duration_seconds": time.perf_counter() - started,
            "input_initial_events": len(rows),
            "alternate_blind_request_total_seconds": float(sum(arbitration_times)),
            "alternate_blind_requests": len(arbitration_times),
            "stage_model_requests": len(arbitration_times),
            "conditioned_fusion_requests": 0,
            "forced_neutral_requests": 0,
            "workers": max(1, int(args.workers)),
        },
        "objects": catalog,
    }
    (output_dir / "semantic_ensemble_review.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "semantic_tiered_catalog.json").write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in report.items() if key != "objects"}
    (output_dir / "semantic_ensemble_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
