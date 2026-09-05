#!/usr/bin/env python3
"""Select non-regressive FARM semantics from independent automatic runs.

This stage contains no scene vocabulary, object ids or hand-authored label
overrides. It treats an earlier FARM caption, an optional prior accepted run,
the mask-grounded ensemble, the blind paired-crop pass and YOLOE proposals as
independent evidence channels.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scene_graph.captioning.evidence import (
    evidence_event_id as contract_evidence_event_id,
    evidence_view_ids,
    select_independent_evidence,
)
from scene_graph.captioning.label_contract import (
    SHARED_FORM_HYPERNYM_SOURCE,
    assess_open_vocabulary_label,
    has_open_vocabulary_contract_evidence,
    normalize_open_noun,
)
from scene_graph.captioning.physical_guard import (
    assess_physical_form,
    assess_physical_recovery,
)


UNKNOWN = {"", "unknown", "unresolved object", "object", "item", "thing"}
DERIVED_EVIDENCE_SOURCES = {
    "strict_two_pass",
    "multi_source_fusion",
    "semantic_reconciliation",
    "semantic_consensus",
}
EVIDENCE_SOURCE_ALIASES = {
    "initial_blind_review": "initial_blind_review",
    "blind_pass_a": "initial_blind_review",
    "independent_verification": "independent_verification",
    "verification_pass": "independent_verification",
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
    prior = parser.add_mutually_exclusive_group(required=True)
    prior.add_argument("--prior-state", type=Path)
    prior.add_argument(
        "--prior-catalog", type=Path,
        help="Current-run two-pass review catalog used as the initial FARM vote.",
    )
    parser.add_argument("--reference-state", type=Path)
    parser.add_argument("--ensemble-review", type=Path, required=True)
    parser.add_argument("--blind-catalog", type=Path, required=True)
    parser.add_argument("--detector-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def normalize(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    return " ".join(word[:-1] if word.endswith("s") and len(word) > 4 else word for word in text.split())


def related(first: object, second: object) -> bool:
    a, b = normalize(first), normalize(second)
    if not a or not b:
        return False
    if a == b:
        return True
    left, right = set(a.split()), set(b.split())
    return bool(left <= right or right <= left)


def informative(value: object) -> bool:
    return normalize(value) not in UNKNOWN


def accepted_category(row: dict) -> str:
    """Return a usable top-level category without turning rejected rows positive."""

    decision = row.get("review_decision", row.get("decision"))
    if decision is not None and str(decision).strip().lower() != "keep":
        return ""
    if str(row.get("semantic_tier") or "").strip().lower() == "geometry_only":
        return ""
    category = str(row.get("category") or row.get("review_category") or "").strip().lower()
    return category if informative(category) else ""


def evidence_event_id(candidate: dict) -> str:
    return contract_evidence_event_id(candidate)


def positive_evidence(candidate: dict) -> bool:
    if isinstance(candidate.get("label_contract"), dict):
        assessment = assess_open_vocabulary_label(
            candidate.get("label_contract"), minimum_confidence=0.85
        )
        return bool(
            assessment.get("usable")
            and normalize_open_noun(assessment.get("category"))
            == normalize_open_noun(candidate.get("category"))
        )
    try:
        confidence = float(candidate.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return False
    return bool(
        str(candidate.get("decision") or "unknown").strip().lower() == "keep"
        and informative(candidate.get("category"))
        and np.isfinite(confidence)
    )


def category_supported_by_event(category: object, candidate: dict) -> bool:
    """Use exact head-noun support for strict events, legacy matching otherwise."""

    if isinstance(candidate.get("label_contract"), dict):
        assessment = assess_open_vocabulary_label(
            candidate.get("label_contract"), minimum_confidence=0.85
        )
        return bool(
            assessment.get("usable")
            and normalize_open_noun(category)
            == normalize_open_noun(assessment.get("category"))
        )
    return related(category, candidate.get("category"))


def _legacy_review_evidence(row: dict, object_id: int) -> list[dict]:
    """Recover request provenance from v2 review files written before manifests."""

    result: list[dict] = []
    initial = row.get("review_initial")
    if isinstance(initial, dict):
        result.append({
            "source": "initial_blind_review",
            "event_id": f"object:{object_id}:initial_blind_review",
            "category": initial.get("category") or "unknown",
            "description": initial.get("description") or "",
            "attributes": initial.get("attributes") or [],
            "confidence": initial.get("confidence") or 0.0,
            "decision": initial.get("decision") or "unknown",
        })
    raw_verification = str(row.get("verification_raw_response") or "").strip()
    if raw_verification.startswith("{"):
        try:
            verification = json.loads(raw_verification)
        except json.JSONDecodeError:
            verification = {}
        if isinstance(verification, dict):
            result.append({
                "source": "independent_verification",
                "event_id": f"object:{object_id}:independent_verification",
                "category": verification.get("category") or "unknown",
                "description": verification.get("description") or "",
                "attributes": verification.get("attributes") or [],
                "confidence": verification.get("confidence") or 0.0,
                "decision": verification.get("decision") or "unknown",
            })
    return result


def load_state(path: Path) -> dict:
    payload = torch.load(path.expanduser(), map_location="cpu", weights_only=False)
    state = payload.get("state", payload) if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not isinstance(state.get("object_id"), torch.Tensor):
        raise ValueError(f"Invalid FARM state: {path}")
    return state


def state_rows(state: dict) -> dict[int, dict]:
    ids = [int(value) for value in state["object_id"].detach().cpu().tolist()]
    active = state.get("active")
    active_values = active.detach().cpu().numpy().astype(bool) if isinstance(active, torch.Tensor) else np.ones(len(ids), bool)
    categories = list(state.get("object_category") or [])
    descriptions = list(state.get("object_caption") or [])
    attributes = list(state.get("object_key_attributes") or [])
    return {
        object_id: {
            "category": str(categories[index] if index < len(categories) else "").strip().lower(),
            "description": str(descriptions[index] if index < len(descriptions) else "").strip(),
            "attributes": attributes[index] if index < len(attributes) and isinstance(attributes[index], list) else [],
            "active": bool(active_values[index]),
        }
        for index, object_id in enumerate(ids)
    }


def review_rows(path: Path) -> dict[int, dict]:
    """Load accepted current-run FARM reviews without scene vocabulary."""

    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    values = payload.get("objects") if isinstance(payload, dict) else payload
    result: dict[int, dict] = {}
    for row in values or []:
        if not isinstance(row, dict) or "id" not in row:
            continue
        category = str(
            row.get("review_category") or row.get("category") or ""
        ).strip().lower()
        decision = str(
            row.get("review_decision") or row.get("decision") or "unknown"
        ).strip().lower()
        if decision != "keep" or not informative(category):
            category = ""
        object_id = int(row["id"])
        explicit_evidence = [
            dict(candidate)
            for candidate in (row.get("semantic_evidence") or [])
            if isinstance(candidate, dict)
        ]
        evidence = explicit_evidence or _legacy_review_evidence(row, object_id)
        if not evidence and category:
            # A legacy accepted review with no request manifest is one event,
            # never enough by itself to claim confirmation.
            evidence = [{
                "source": "legacy_accepted_review",
                "event_id": f"object:{object_id}:legacy_accepted_review",
                "category": category,
                "description": row.get("review_description") or row.get("description") or "",
                "attributes": row.get("review_attributes") or row.get("attributes") or [],
                "confidence": row.get("review_confidence") or row.get("confidence") or 0.0,
                "decision": "keep",
            }]
        result[object_id] = {
            "category": category,
            "description": str(
                row.get("review_description") or row.get("description") or ""
            ).strip(),
            "attributes": row.get("review_attributes") or row.get("attributes") or [],
            "review_confidence": float(
                row.get("review_confidence") or row.get("confidence") or 0.0
            ),
            "review_gate_reason": str(row.get("review_gate_reason") or ""),
            "evidence": evidence,
            "active": True,
        }
    return result


def detector_rows(state: dict) -> dict[int, dict[str, float]]:
    ids = [int(value) for value in state["object_id"].detach().cpu().tolist()]
    raw = list(state.get("object_detection_category_conf") or [])
    return {
        object_id: {
            str(label).strip().lower(): float(score)
            for label, score in (raw[index] if index < len(raw) and isinstance(raw[index], dict) else {}).items()
            if str(label).strip() and np.isfinite(float(score))
        }
        for index, object_id in enumerate(ids)
    }


def detector_support(category: str, votes: dict[str, float], threshold: float = 0.30) -> float:
    return max((float(score) for label, score in votes.items() if related(category, label)), default=0.0) if informative(category) else 0.0


def candidate_source_count(row: dict, category: str) -> int:
    sources = {
        str(candidate.get("source") or "")
        for candidate in row.get("candidates") or []
        if positive_evidence(candidate) and related(category, candidate.get("category"))
    }
    return len(sources)


def strong_ensemble_candidate(row: dict, minimum_sources: int = 3) -> str:
    """Return a repeated non-fallback category from distinct model passes."""

    by_category: dict[str, set[str]] = {}
    display: dict[str, str] = {}
    for candidate in row.get("candidates") or []:
        source = str(candidate.get("source") or "")
        category = str(candidate.get("category") or "").strip().lower()
        key = normalize(category)
        if (
            not key
            or not informative(category)
            or source in {"neutral_form_fallback", "strict_two_pass"}
            or str(candidate.get("decision") or "unknown").lower() != "keep"
        ):
            continue
        by_category.setdefault(key, set()).add(source)
        display.setdefault(key, category)
    if not by_category:
        return ""
    key, sources = max(by_category.items(), key=lambda item: len(item[1]))
    return display[key] if len(sources) >= int(minimum_sources) else ""


def choose_category(
    prior: dict,
    reference: dict,
    ensemble: dict,
    blind: dict,
    detector: dict[str, float],
) -> tuple[str, str]:
    old = accepted_category(prior)
    # Geometry activity and semantic evidence are separate contracts. A label
    # remains usable as an independent caption vote when an older run hid the
    # object for geometry/presentation reasons; it never reactivates the row.
    ref = accepted_category(reference)
    current = accepted_category(ensemble)
    blind_category = accepted_category(blind)

    # The current strict contract is authoritative over legacy captions and
    # detector token overlap. Reconciliation may veto it; weighted stages may
    # not silently substitute a different physical head noun.
    strict_current_seen = any(
        has_open_vocabulary_contract_evidence(row)
        for row in (blind, ensemble)
    )
    for row, provenance in (
        (blind, "strict_open_vocabulary_reconciled"),
        (ensemble, "strict_open_vocabulary_ensemble"),
    ):
        if not isinstance(row.get("label_contract"), dict):
            continue
        assessment = assess_open_vocabulary_label(
            row.get("label_contract"), minimum_confidence=0.85
        )
        category = accepted_category(row)
        if (
            assessment.get("usable")
            and normalize_open_noun(category)
            == normalize_open_noun(assessment.get("category"))
        ):
            return category, provenance
        # The reconciled row is authoritative when it carries the contract.
        # Do not fall through to an ensemble/prior noun after its topology,
        # boundedness, confidence, or physical guard failed closed.
        if row is blind:
            return "unresolved object", "strict_open_vocabulary_failed_closed"
    if strict_current_seen:
        return "unresolved object", "strict_open_vocabulary_failed_closed"

    stable = bool(informative(old) and informative(ref) and related(old, ref))
    if stable:
        # Two newly generated visual channels may supersede a stable caption.
        if informative(current) and related(current, blind_category) and not related(current, old):
            return current, "two_new_multiview_channels_override_stable_prior"
        # A current label supported by detector proposals may replace an old
        # label only when the old category has no detector support at all.
        if (
            informative(current)
            and not related(current, old)
            and detector_support(current, detector) >= 0.30
            and detector_support(old, detector) < 0.30
            and candidate_source_count(ensemble, current) >= 2
        ):
            return current, "ensemble_and_detector_override_unsupported_stable_prior"
        return old, "stable_cross_run_farm_caption_preserved"

    # First-run consensus has no earlier accepted scene state. Preserve the
    # current two-pass FARM vote unless two new visual channels agree on a
    # replacement. A repeated category from three distinct non-fallback passes
    # can recover a label that a later generic-form fallback would erase.
    if not informative(ref):
        if informative(old) and informative(current) and related(old, current):
            return current, "current_ensemble_confirms_two_pass_farm_vote"
        if informative(old) and informative(blind_category) and related(old, blind_category):
            return old, "blind_paired_crop_confirms_two_pass_farm_vote"
        if informative(current) and informative(blind_category) and related(current, blind_category):
            return current, "ensemble_and_blind_paired_crop_agree"
        repeated = strong_ensemble_candidate(ensemble)
        if informative(repeated):
            return repeated, "three_distinct_model_passes_agree"
        gate = str(ensemble.get("semantic_gate_reason") or "").lower()
        if informative(old) and "neutral" in gate:
            return old, "two_pass_vote_protected_from_neutral_fallback"
        if informative(old):
            return old, "current_run_two_pass_farm_vote_preserved"

    # If the historical runs disagree, prefer the new ensemble when it agrees
    # with either the accepted reference or the blind target audit.
    if informative(current) and (
        (informative(ref) and related(current, ref))
        or (informative(blind_category) and related(current, blind_category))
    ):
        return current, "current_ensemble_agrees_with_independent_channel"
    if informative(current):
        return current, "current_mask_grounded_ensemble"
    if informative(blind_category):
        return blind_category, "blind_paired_crop_fallback"
    if informative(old):
        return old, "prior_farm_caption_fallback"
    return "unresolved object", "no_supported_semantic_identity"


def collect_semantic_evidence(
    prior: dict,
    reference: dict,
    ensemble: dict,
    blind: dict,
) -> list[dict]:
    """Collect original inference events and remove derived/correlated copies."""

    candidates: list[dict] = []
    for values in (
        prior.get("evidence") or [],
        reference.get("evidence") or [],
        ensemble.get("semantic_evidence") or [],
        ensemble.get("candidates") or [],
        blind.get("reconciliation_evidence") or [],
        [blind.get("physical_form_recovery")]
        if blind.get("physical_form_recovery")
        else [],
        [blind.get("physical_form_recovery_verification")]
        if blind.get("physical_form_recovery_verification")
        else [],
    ):
        candidates.extend(dict(value) for value in values if isinstance(value, dict))
    by_event: dict[str, dict] = {}
    for candidate in candidates:
        source = str(candidate.get("source") or "").strip()
        event_id = evidence_event_id(candidate)
        if (
            source in DERIVED_EVIDENCE_SOURCES
            or not event_id
            or not positive_evidence(candidate)
        ):
            continue
        candidate = {**candidate, "event_id": event_id}
        current = by_event.get(event_id)
        rank = (
            evidence_view_ids(candidate) is not None,
            len(evidence_view_ids(candidate) or ()),
            float(candidate.get("confidence") or 0.0),
        )
        current_rank = (
            evidence_view_ids(current) is not None,
            len(evidence_view_ids(current) or ()),
            float(current.get("confidence") or 0.0),
        ) if current is not None else (False, -1, -1.0)
        if current is None or rank > current_rank:
            by_event[event_id] = candidate
    return [by_event[event_id] for event_id in sorted(by_event)]


def physical_recovery_override(blind: dict) -> dict | None:
    """Return an authoritative recovery/suppression action, when triggered."""

    rejected = blind.get("physical_form_rejected_semantics") or {}
    rejected_assessment = (
        blind.get("physical_form_rejected_assessment")
        or rejected.get("physical_form_assessment")
        or {}
    )
    recovery = blind.get("physical_form_recovery") or {}
    verification = blind.get("physical_form_recovery_verification") or {}
    stored = blind.get("physical_form_recovery_assessment") or {}
    if not rejected_assessment and not stored:
        return None
    assessment = assess_physical_recovery(
        recovery,
        rejected_assessment,
        verification=verification,
        rejected_category=rejected.get("category") or "",
        metric_dimensions_m=blind.get("metric_dimensions_m"),
    )
    if not assessment.get("triggered"):
        return None
    return {
        "action": "recover" if assessment.get("accepted") else "suppress",
        "category": str(assessment.get("category") or "unresolved object"),
        "recovery": recovery,
        "verification": verification,
        "rejected": rejected,
        "assessment": assessment,
    }


def _event_confidence(candidate: dict) -> float:
    try:
        value = float(candidate.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def _independent_strength(events: list[dict]) -> tuple[int, list[dict], dict]:
    selected, diagnostics = select_independent_evidence(events)
    if selected:
        return len(selected), selected, diagnostics
    if events:
        chosen = max(events, key=_event_confidence)
        return 1, [chosen], diagnostics
    return 0, [], diagnostics


def _shared_form_contributors(
    category: str,
    candidate: dict,
    evidence: list[dict],
) -> set[str]:
    """Validate the two-event derivation of a probable-only shared form."""

    if (
        str(candidate.get("source") or "") != SHARED_FORM_HYPERNYM_SOURCE
        or candidate.get("confirmation_eligible") is not False
    ):
        return set()
    assessment = assess_open_vocabulary_label(
        candidate.get("label_contract"), minimum_confidence=0.85
    )
    if (
        not assessment.get("usable")
        or normalize_open_noun(assessment.get("category"))
        != normalize_open_noun(category)
        or assessment.get("category_role") != "whole_form"
        or candidate.get("maximum_semantic_tier") != "probable"
    ):
        return set()
    raw_ids = candidate.get("derived_from_event_ids")
    if not isinstance(raw_ids, list):
        return set()
    contributor_ids = {str(value).strip() for value in raw_ids if str(value).strip()}
    if len(contributor_ids) < 2:
        return set()
    by_id = {
        evidence_event_id(value): value
        for value in evidence
        if value is not candidate
    }
    contributors = [by_id[event_id] for event_id in sorted(contributor_ids) if event_id in by_id]
    if len(contributors) != len(contributor_ids):
        return set()
    head_nouns: set[str] = set()
    for contributor in contributors:
        contract = contributor.get("label_contract")
        result = assess_open_vocabulary_label(contract, minimum_confidence=0.85)
        if (
            not result.get("usable")
            or normalize_open_noun((contract or {}).get("form_hypernym"))
            != normalize_open_noun(category)
        ):
            return set()
        head_nouns.add(normalize_open_noun(result.get("category")))
    return contributor_ids if len(head_nouns) >= 2 else set()


def _correlation_groups(events: list[dict], diagnostics: dict) -> list[dict]:
    """Serialize connected correlated components for downstream QA."""

    event_by_id = {evidence_event_id(event): event for event in events}
    parent = {event_id: event_id for event_id in event_by_id}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for overlap in diagnostics.get("pairwise_overlaps") or []:
        if overlap.get("independent"):
            continue
        left = str(overlap.get("left_event_id") or "")
        right = str(overlap.get("right_event_id") or "")
        if left in parent and right in parent:
            union(left, right)

    components: dict[str, list[str]] = {}
    for event_id in sorted(parent):
        components.setdefault(find(event_id), []).append(event_id)
    output = []
    for event_ids in components.values():
        view_ids = sorted(set().union(*(
            set(evidence_view_ids(event_by_id[event_id]) or ())
            for event_id in event_ids
        )))
        digest = hashlib.sha256("\n".join(event_ids).encode("utf-8")).hexdigest()
        output.append({
            "correlation_group_id": f"semantic-group-{digest[:16]}",
            "event_ids": event_ids,
            "sources": sorted({
                str(event_by_id[event_id].get("source") or "")
                for event_id in event_ids
            }),
            "crop_image_ids": view_ids,
            "correlated_repeat_count": max(0, len(event_ids) - 1),
        })
    return sorted(output, key=lambda row: row["correlation_group_id"])


def semantic_tier_assessment(
    category: str,
    evidence: list[dict],
    *,
    physical_assessment: dict | None = None,
) -> tuple[str, list[dict], dict]:
    shared_contributors = {
        id(candidate): _shared_form_contributors(category, candidate, evidence)
        for candidate in evidence
        if str(candidate.get("source") or "") == SHARED_FORM_HYPERNYM_SOURCE
    }
    supporting = [
        candidate
        for candidate in evidence
        if category_supported_by_event(category, candidate)
        and (
            str(candidate.get("source") or "") != SHARED_FORM_HYPERNYM_SOURCE
            or bool(shared_contributors.get(id(candidate)))
        )
    ] if informative(category) else []
    confirmation_candidates = [
        candidate for candidate in supporting if _event_confidence(candidate) >= 0.85
    ]
    strength, selected, diagnostics = _independent_strength(
        confirmation_candidates
    )

    competing: dict[str, list[dict]] = {}
    covered_contributors = set().union(*(
        shared_contributors.get(id(candidate), set())
        for candidate in supporting
    )) if supporting else set()
    for candidate in evidence:
        if evidence_event_id(candidate) in covered_contributors:
            continue
        hypothesis = str(candidate.get("category") or "").strip().lower()
        key = normalize(hypothesis)
        if (
            not informative(hypothesis)
            or category_supported_by_event(category, candidate)
            or _event_confidence(candidate) < 0.85
        ):
            continue
        competing.setdefault(key, []).append(candidate)
    competing_strengths = {
        key: _independent_strength(values)[0]
        for key, values in competing.items()
    }
    maximum_competing = max(competing_strengths.values(), default=0)
    outvoted_or_tied = bool(maximum_competing and maximum_competing >= strength)
    physical = dict(physical_assessment or {"confirmation_allowed": True})

    strict_maximum_allows_confirmation = all(
        not isinstance(candidate.get("label_contract"), dict)
        or assess_open_vocabulary_label(
            candidate.get("label_contract"), minimum_confidence=0.85
        ).get("maximum_semantic_tier") == "confirmed"
        for candidate in selected
    )

    if outvoted_or_tied:
        tier, tier_support = "geometry_only", []
        reason = "independent_category_tie_or_outvote"
    elif (
        diagnostics.get("confirmation_ready")
        and strength >= 2
        and strict_maximum_allows_confirmation
        and physical.get("confirmation_allowed") is True
    ):
        tier, tier_support = "confirmed", selected
        reason = "independent_disjoint_view_consensus"
    elif supporting:
        tier = "probable"
        tier_support = selected or [max(supporting, key=_event_confidence)]
        reason = (
            "physical_form_confirmation_withheld"
            if physical.get("confirmation_allowed") is not True
            else str(diagnostics.get("reason") or "single_semantic_evidence")
        )
    else:
        tier, tier_support = "geometry_only", []
        reason = "no_positive_semantic_evidence"

    selected_views = sorted(set().union(*(
        set(evidence_view_ids(event) or ()) for event in selected
    ))) if selected else []
    selected_event_ids = {evidence_event_id(event) for event in selected}
    overlaps = [
        float(row["overlap_coefficient"])
        for row in diagnostics.get("pairwise_overlaps") or []
        if row.get("overlap_coefficient") is not None
        and str(row.get("left_event_id") or "") in selected_event_ids
        and str(row.get("right_event_id") or "") in selected_event_ids
    ]
    assessment = {
        **diagnostics,
        "tier_reason": reason,
        "chosen_independent_strength": int(strength),
        "competing_independent_strengths": competing_strengths,
        "maximum_competing_independent_strength": int(maximum_competing),
        "outvoted_or_tied": outvoted_or_tied,
        "selected_unique_view_count": len(selected_views),
        "selected_unique_view_ids": selected_views,
        "max_support_overlap_coefficient": max(overlaps, default=None),
        "correlation_groups": _correlation_groups(
            confirmation_candidates, diagnostics
        ),
        "confirmation_eligible": tier == "confirmed",
    }
    return tier, tier_support, assessment


def semantic_tier_for(category: str, evidence: list[dict]) -> tuple[str, list[dict]]:
    """Backward-compatible wrapper around provenance-aware tiering."""

    tier, supporting, _ = semantic_tier_assessment(category, evidence)
    return tier, supporting


def semantic_payload(
    category: str,
    source: str,
    prior: dict,
    ensemble: dict,
    blind: dict,
    evidence: list[dict],
) -> dict:
    if not informative(category):
        return {
            "category": "unresolved object",
            "description": "Metric multi-view object with unresolved semantic identity.",
            "attributes": [],
            "confidence": 0.0,
        }
    candidates = []
    if source.startswith("stable") or source.endswith("stable_prior") or source == "prior_farm_caption_fallback":
        candidates.append(prior)
    candidates.extend(evidence)
    candidates.extend(list(ensemble.get("candidates") or []) + [ensemble, blind, prior])
    chosen = next((
        row for row in candidates
        if positive_evidence(row)
        and category_supported_by_event(category, row)
        and str(row.get("description") or "").strip()
    ), {})
    return {
        "category": category,
        "description": str(chosen.get("description") or f"multi-view {category}").strip(),
        "attributes": [str(value).strip() for value in (chosen.get("attributes") or []) if str(value).strip()][:8],
        "confidence": float(
            chosen.get("review_confidence") or chosen.get("confidence") or 0.95
        ),
    }


def main() -> int:
    args = parse_args()
    prior_rows = (
        state_rows(load_state(args.prior_state))
        if args.prior_state is not None
        else review_rows(args.prior_catalog)
    )
    reference_rows = state_rows(load_state(args.reference_state)) if args.reference_state else {}
    detectors = detector_rows(load_state(args.detector_state))
    ensemble_payload = json.loads(args.ensemble_review.read_text(encoding="utf-8"))
    ensemble_rows = {int(row["id"]): row for row in ensemble_payload.get("objects") or []}
    blind_payload = json.loads(args.blind_catalog.read_text(encoding="utf-8"))
    blind_values = blind_payload.get("objects") if isinstance(blind_payload, dict) else blind_payload
    blind_rows = {int(row["id"]): row for row in blind_values or []}

    output = []
    reasons = Counter()
    for object_id in sorted(ensemble_rows):
        prior = prior_rows.get(object_id, {})
        reference = reference_rows.get(object_id, {})
        ensemble = ensemble_rows[object_id]
        blind = blind_rows.get(object_id, {})
        recovery_override = physical_recovery_override(blind)
        if recovery_override and recovery_override["action"] == "recover":
            category = str(recovery_override["category"])
            reason = "independent_blind_physical_form_recovery_agreement"
        elif recovery_override and recovery_override["action"] == "suppress":
            category = "unresolved object"
            reason = "physical_form_recovery_failed_closed"
        else:
            category, reason = choose_category(
                prior,
                reference,
                ensemble,
                blind,
                detectors.get(object_id, {}),
            )
        evidence = collect_semantic_evidence(prior, reference, ensemble, blind)
        metric_dimensions = ensemble.get("metric_dimensions_m") or blind.get(
            "metric_dimensions_m"
        )
        if recovery_override:
            physical = (
                assess_physical_form(category, metric_dimensions)
                if recovery_override["action"] == "recover"
                else dict(
                    blind.get("physical_form_rejected_assessment")
                    or blind.get("physical_form_assessment")
                    or {}
                )
            )
        else:
            physical = assess_physical_form(
                category,
                metric_dimensions,
                model_guard=blind.get("physical_form_guard") or {},
            )
        tier_evidence = (
            [
                recovery_override["recovery"], recovery_override["verification"]
            ]
            if recovery_override and recovery_override["action"] == "recover"
            else []
            if recovery_override and recovery_override["action"] == "suppress"
            else evidence
        )
        tier, supporting, independence = semantic_tier_assessment(
            category,
            tier_evidence,
            physical_assessment=physical,
        )
        if recovery_override and recovery_override["action"] == "recover":
            tier = "probable"
            supporting = list(tier_evidence)
            recovery_assessment = recovery_override["assessment"]
            independence.update({
                "reason": "independent_recovery_events_agree",
                "selected_event_ids": [
                    evidence_event_id(event) for event in supporting
                ],
                "chosen_independent_strength": len(supporting),
                "selected_unique_view_count": int(
                    recovery_assessment.get("unique_view_count")
                    or len(recovery_override["recovery"].get("crop_image_ids") or [])
                ),
                "max_support_overlap_coefficient": (
                    recovery_assessment.get("view_overlap_coefficient")
                ),
                "confirmation_ready": False,
            })
            independence["confirmation_eligible"] = False
            independence["tier_reason"] = (
                "independent_blind_physical_form_recovery_agreement"
            )
        elif recovery_override and recovery_override["action"] == "suppress":
            tier = "geometry_only"
            supporting = []
            independence["confirmation_eligible"] = False
            independence["tier_reason"] = "physical_form_recovery_failed_closed"
        selection_reason = reason
        reason = str(independence.get("tier_reason") or reason)
        if tier == "geometry_only":
            category = "unresolved object"
        semantics = semantic_payload(category, reason, prior, ensemble, blind, evidence)
        if recovery_override and recovery_override["action"] == "recover":
            recovery = recovery_override["recovery"]
            verification = recovery_override["verification"]
            primary_attributes = {
                normalize(value).replace(" ", "_"): str(value)
                for value in (recovery.get("attributes") or [])
                if normalize(value).replace(" ", "_")
                not in {"standalone_bounded", "component_only"}
            }
            verification_attributes = {
                normalize(value).replace(" ", "_"): str(value)
                for value in (verification.get("attributes") or [])
                if normalize(value).replace(" ", "_")
                not in {"standalone_bounded", "component_only"}
            }
            semantics = {
                "category": category,
                "description": f"multi-view visible {category}",
                "attributes": [
                    primary_attributes[key]
                    for key in sorted(
                        set(primary_attributes) & set(verification_attributes)
                    )
                ][:8],
                "confidence": float(
                    recovery_override["assessment"].get("confidence") or 0.0
                ),
            }
        reasons[reason] += 1
        output.append({
            **ensemble,
            **semantics,
            "semantic_tier": tier,
            "semantic_status": (
                "physical_form_recovered_probable"
                if recovery_override and recovery_override["action"] == "recover"
                else "physical_component_or_contradiction_suppressed"
                if recovery_override and recovery_override["action"] == "suppress"
                else {
                "confirmed": "independent_semantic_consensus",
                "probable": "single_independent_semantic_evidence",
                "geometry_only": "geometry_valid_semantics_unresolved",
                }[tier]
            ),
            "review_decision": "keep" if tier != "geometry_only" else "unknown",
            "review_confidence": semantics["confidence"],
            "semantic_gate_reason": reason,
            "semantic_category_selection_reason": selection_reason,
            "semantic_evidence": evidence,
            "semantic_evidence_count": len({
                evidence_event_id(candidate) for candidate in supporting
            }),
            "semantic_evidence_event_ids": sorted({
                evidence_event_id(candidate) for candidate in supporting
            }),
            "semantic_independent_group_count": int(
                independence.get("chosen_independent_strength") or 0
            ),
            "semantic_unique_view_count": int(
                independence.get("selected_unique_view_count") or 0
            ),
            "semantic_max_support_overlap": independence.get(
                "max_support_overlap_coefficient"
            ),
            "semantic_confirmation_eligible": bool(
                independence.get("confirmation_eligible")
            ),
            "semantic_ineligibility_reasons": sorted({
                str(value)
                for value in (
                    list(independence.get("incomplete_provenance_event_ids") or [])
                    + list(independence.get("under_observed_event_ids") or [])
                    + list(independence.get("confirmation_ineligible_event_ids") or [])
                    + ([reason] if tier != "confirmed" else [])
                )
                if str(value)
            }),
            "semantic_correlation_groups": independence.get(
                "correlation_groups"
            ) or [],
            "semantic_independence": independence,
            "physical_form_assessment": physical,
            "physical_form_rejected_semantics": (
                recovery_override.get("rejected") if recovery_override else {}
            ),
            "physical_form_rejected_assessment": (
                blind.get("physical_form_rejected_assessment") or {}
            ),
            "physical_form_recovery": (
                recovery_override.get("recovery") if recovery_override else {}
            ),
            "physical_form_recovery_verification": (
                recovery_override.get("verification")
                if recovery_override else {}
            ),
            "physical_form_recovery_assessment": (
                recovery_override.get("assessment") if recovery_override else {}
            ),
            "metric_dimensions_m": metric_dimensions,
            "consensus_inputs": {
                "prior": prior.get("category") or "",
                "reference": reference.get("category") or "",
                "ensemble": ensemble.get("category") or "",
                "blind": blind.get("category") or "",
            },
        })

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    invalid_confirmed = [
        int(row["id"])
        for row in output
        if row["semantic_tier"] == "confirmed"
        and (
            not row.get("semantic_confirmation_eligible")
            or int(row.get("semantic_independent_group_count") or 0) < 2
            or int(row.get("semantic_unique_view_count") or 0) < 6
            or (
                row.get("semantic_max_support_overlap") is not None
                and float(row["semantic_max_support_overlap"]) > 0.25
            )
            or (row.get("physical_form_assessment") or {}).get(
                "confirmation_allowed"
            ) is not True
        )
    ]
    report = {
        "schema": "farm.semantic-consensus.v1",
        "status": "FAIL" if invalid_confirmed else "PASS",
        "objects": len(output),
        "resolved": sum(row["semantic_tier"] != "geometry_only" for row in output),
        "tier_counts": dict(Counter(row["semantic_tier"] for row in output)),
        "reasons": dict(reasons),
        "independence_qa": {
            "confirmed_independent": sum(
                row["semantic_tier"] == "confirmed" for row in output
            ),
            "correlated_repeat_only": sum(
                row["semantic_tier"] == "probable"
                and str((row.get("semantic_independence") or {}).get("reason"))
                == "correlated_view_evidence"
                for row in output
            ),
            "outvoted_labels": sum(
                bool((row.get("semantic_independence") or {}).get("outvoted_or_tied"))
                for row in output
            ),
            "physical_contradictions": sum(
                bool((row.get("physical_form_assessment") or {}).get("hard_veto"))
                for row in output
            ),
            "incomplete_provenance_objects": sum(
                bool((row.get("semantic_independence") or {}).get(
                    "incomplete_provenance_event_ids"
                ))
                for row in output
            ),
            "invalid_confirmed_object_ids": invalid_confirmed,
            "physical_form_recoveries_accepted": sum(
                bool((row.get("physical_form_recovery_assessment") or {}).get("accepted"))
                for row in output
            ),
            "physical_form_recovery_verifications": sum(
                bool(row.get("physical_form_recovery_verification"))
                for row in output
            ),
            "physical_form_recoveries_suppressed": sum(
                (row.get("physical_form_recovery_assessment") or {}).get("action")
                == "suppress_geometry_only"
                for row in output
            ),
        },
        "policy": (
            "automatic class-agnostic consensus; confirmed requires two "
            "provenance-complete disjoint-view events, at least three views per "
            "event and six unique views total, no tied/outvoted category, and no "
            "physical-form veto; no manual object ids or label overrides"
        ),
    }
    write_json_atomic(output_dir / "semantic_consensus_catalog.json", output)
    write_json_atomic(output_dir / "semantic_consensus_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
