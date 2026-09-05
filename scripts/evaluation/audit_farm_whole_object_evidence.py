#!/usr/bin/env python3
"""Build Lucida-style per-object evidence bundles and fail-closed readiness gates."""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from scene_graph.captioning.evidence import (
    evidence_event_id,
    evidence_view_ids,
    select_independent_evidence,
)
from scene_graph.captioning.label_contract import (
    DEFAULT_MIN_VISIBLE_TARGET_COVERAGE,
    PUBLICATION_FORBIDDEN_GENERIC_NOUNS,
    assess_open_vocabulary_whole_identity,
    assess_whole_object_readiness,
    independent_blind_whole_identity_consensus,
    normalize_open_noun,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _linked_gaussian_counts(
    build_manifest_path: Path,
    candidate_manifest_path: Path,
    heldout_payload: Mapping[str, Any],
) -> tuple[dict[int, int], dict[str, Any]]:
    """Load counts only when heldout candidate cryptographically binds the build."""

    build_path = build_manifest_path.expanduser().resolve(strict=True)
    candidate_path = candidate_manifest_path.expanduser().resolve(strict=True)
    build = json.loads(build_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if build.get("schema_version") != "farm.gaussian-lift.build.v1":
        raise ValueError("unsupported Gaussian lift build manifest schema")
    if build.get("status") != "frozen_pending_heldout":
        raise ValueError("Gaussian lift build is not frozen_pending_heldout")
    if candidate.get("schema") != "farm.frozen-heldout-candidate.v1":
        raise ValueError("unsupported frozen heldout candidate schema")
    if candidate.get("status") != "frozen":
        raise ValueError("heldout candidate is not frozen")
    qc_candidate = (
        (heldout_payload.get("provenance") or {}).get("candidate_manifest") or {}
    )
    candidate_sha = _sha256(candidate_path)
    if str(qc_candidate.get("sha256") or "") != candidate_sha:
        raise ValueError("heldout QC does not bind the supplied candidate manifest")
    if (
        (heldout_payload.get("contract") or {}).get(
            "all_consumed_inputs_sha256_verified_and_rechecked"
        )
        is not True
    ):
        raise ValueError("heldout QC input hashes were not verified")
    build_sha = _sha256(build_path)
    consumed = (candidate.get("provenance") or {}).get("consumed_input_hashes") or {}
    if not isinstance(consumed, Mapping) or build_sha not in set(map(str, consumed.values())):
        raise ValueError("heldout candidate does not bind the supplied build manifest")

    build_rows = build.get("objects")
    candidate_rows = candidate.get("objects")
    if not isinstance(build_rows, list) or not isinstance(candidate_rows, list):
        raise ValueError("Gaussian manifests are missing object rows")
    candidate_ids = {
        int(row["object_id"])
        for row in candidate_rows
        if isinstance(row, Mapping) and "object_id" in row
    }
    counts: dict[int, int] = {}
    for row in build_rows:
        if not isinstance(row, Mapping) or "object_id" not in row:
            raise ValueError("Gaussian build contains an invalid object row")
        object_id = int(row["object_id"])
        count = int(row.get("provisional_gaussians") or 0)
        if row.get("geometry_gate") != "PASS" or count <= 0:
            raise ValueError(f"Gaussian build object {object_id} is not count-eligible")
        if object_id in counts:
            raise ValueError(f"duplicate Gaussian build object {object_id}")
        counts[object_id] = count
    if set(counts) != candidate_ids:
        raise ValueError("Gaussian build and heldout candidate object sets differ")
    return counts, {
        "status": "verified",
        "build_manifest": str(build_path),
        "build_manifest_sha256": build_sha,
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": candidate_sha,
        "object_count": len(counts),
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("objects"), list):
        rows = payload["objects"]
    else:
        raise TypeError(f"unsupported object catalog schema: {path}")
    if not all(isinstance(row, dict) and "id" in row for row in rows):
        raise TypeError(f"catalog contains invalid object rows: {path}")
    return [dict(row) for row in rows]


def _finite_vector(value: object, length: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return []
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return []
    return result if all(math.isfinite(item) for item in result) else []


def _semantic_events(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in row.get("semantic_evidence") or []:
        if not isinstance(raw, dict) or not isinstance(raw.get("label_contract"), dict):
            continue
        event = dict(raw)
        event_id = evidence_event_id(event)
        if not event_id or event_id in seen:
            continue
        seen.add(event_id)
        result.append(event)
    return result


def _event_summary(event: Mapping[str, Any]) -> dict[str, Any]:
    contract = dict(event.get("label_contract") or {})
    return {
        "event_id": evidence_event_id(event),
        "source": str(event.get("source") or ""),
        "confirmation_eligible": event.get("confirmation_eligible") is not False,
        "category": normalize_open_noun(contract.get("category")),
        "form_hypernym": normalize_open_noun(contract.get("form_hypernym")),
        "topology": str(contract.get("topology") or "unknown"),
        "complete_bounded": contract.get("complete_bounded"),
        "context_sufficient": contract.get("context_sufficient"),
        "visible_target_coverage": contract.get("visible_target_coverage"),
        "missing_visible_parts": list(contract.get("missing_visible_parts") or []),
        "included_non_target": list(contract.get("included_non_target") or []),
        "diagnostic_parts": list(contract.get("diagnostic_parts") or []),
        "diagnostic_view_count": int(contract.get("diagnostic_view_count") or 0),
        "confidence": float(contract.get("confidence") or 0.0),
        "image_ids": list(evidence_view_ids(event) or ()),
        "evidence_fingerprint_sha256": str(
            event.get("evidence_fingerprint_sha256") or ""
        ),
    }


def _group_assessments(
    events: list[dict[str, Any]],
    *,
    minimum_identity_confidence: float,
    minimum_visible_target_coverage: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    identity_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    whole_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        contract = event["label_contract"]
        identity = assess_open_vocabulary_whole_identity(
            contract, minimum_confidence=minimum_identity_confidence
        )
        whole = assess_whole_object_readiness(
            contract,
            minimum_confidence=minimum_identity_confidence,
            minimum_visible_target_coverage=minimum_visible_target_coverage,
        )
        if identity.get("usable"):
            value = dict(event)
            value["category"] = identity["category"]
            value["confidence"] = identity["confidence"]
            identity_groups[str(identity["category"])].append(value)
        if identity.get("usable") and whole.get("ready"):
            value = dict(event)
            value["category"] = whole["category"]
            value["confidence"] = whole["confidence"]
            whole_groups[str(whole["category"])].append(value)
    return dict(identity_groups), dict(whole_groups)


def _independent_blind_re_adjudication(
    events: list[dict[str, Any]], minimum_confidence: float
) -> dict[str, Any]:
    """Re-evaluate immutable raw contracts without trusting derived review fields."""

    initial = [
        event for event in events
        if str(event.get("source") or "") in {
            "initial_blind_review", "contextual_dual_panel_review",
        }
    ]
    verification = [
        event for event in events
        if str(event.get("source") or "") == "independent_verification"
    ]
    if len(events) != 2 or len(initial) != 1 or len(verification) != 1:
        return {
            "schema": "farm.independent-blind-label-consensus.v1",
            "status": "unresolved",
            "accepted": False,
            "canonical_category": "",
            "reason_codes": ["exact_current_blind_event_pair_missing"],
            "identity_contract_source": "raw_label_contract",
            "mask_scope_incomplete": False,
            "mask_scope_reason_codes": ["mask_scope_unknown"],
            "geometry_refinement_required": False,
            "inpainting_eligible": False,
        }
    return independent_blind_whole_identity_consensus(
        initial[0], verification[0],
        minimum_confidence=float(minimum_confidence),
    )


def _consensus(groups: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    confirmed: list[str] = []
    for category, events in sorted(groups.items()):
        _, report = select_independent_evidence(events)
        diagnostics[category] = report
        if report.get("confirmation_ready"):
            confirmed.append(category)
    probable = ""
    if len(groups) == 1:
        probable = next(iter(groups))
    return {
        "confirmed_category": confirmed[0] if len(confirmed) == 1 else "",
        "confirmed_categories": confirmed,
        "probable_category": probable,
        "conflicting_categories": sorted(groups) if len(groups) > 1 else [],
        "group_diagnostics": diagnostics,
    }


def _refinement_signals(
    events: list[dict[str, Any]], minimum_visible_target_coverage: float
) -> list[str]:
    reasons: set[str] = set()
    for event in events:
        contract = event.get("label_contract") or {}
        topology = str(contract.get("topology") or "unknown")
        if topology in {"partial_unbounded", "attached_component"}:
            reasons.add(f"topology_{topology}")
        if contract.get("complete_bounded") is False:
            reasons.add("not_complete_bounded")
        coverage = contract.get("visible_target_coverage")
        if coverage is not None and float(coverage) < minimum_visible_target_coverage:
            reasons.add("visible_target_coverage_below_threshold")
        if contract.get("missing_visible_parts"):
            reasons.add("missing_visible_parts")
        if contract.get("included_non_target"):
            reasons.add("included_non_target")
    return sorted(reasons)


def _whole_instance_part_scope_signals(
    events: list[dict[str, Any]], row: Mapping[str, Any]
) -> list[str]:
    """Require explicit mask evidence for thin diagnostic appendages.

    VLM visibility claims identify the object but do not prove that a lifted
    mask contains every thin hose/cable/cord/tube. A later mask-specific gate
    may set ``whole_instance_part_coverage_verified`` only after checking the
    diagnostic parts in independent views.
    """

    if row.get("whole_instance_part_coverage_verified") is True:
        return []
    thin_terms = {"cable", "cord", "hose", "nozzle", "tube", "wire"}
    diagnostic_tokens = {
        token
        for event in events
        for part in (event.get("label_contract") or {}).get(
            "diagnostic_parts", []
        )
        for token in re.findall(r"[a-z]+", str(part).lower())
    }
    return (
        ["whole_instance_thin_appendage_coverage_unverified"]
        if diagnostic_tokens & thin_terms
        else []
    )


def _inpainting_gate(
    status: str,
    heldout: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if status != "whole_ready":
        return {"tier": "not_ready", "reason": "whole_object_gate_not_passed"}
    if not heldout:
        return {"tier": "needs_gaussian_lift", "reason": "missing_heldout_gaussian_qc"}
    if heldout.get("status") != "verified":
        return {
            "tier": "not_ready",
            "reason": f"heldout_gaussian_status_{heldout.get('status') or 'unknown'}",
        }
    if "provisional_gaussians" not in heldout:
        return {
            "tier": "needs_gaussian_count_evidence",
            "reason": "verified_heldout_qc_missing_linked_gaussian_count",
        }
    summaries = heldout.get("summaries") or {}
    median_iou = float((summaries.get("iou") or {}).get("median") or 0.0)
    minimum_precision = float(
        (summaries.get("precision") or {}).get("minimum") or 0.0
    )
    minimum_recall = float((summaries.get("recall") or {}).get("minimum") or 0.0)
    gaussian_count = int(heldout.get("provisional_gaussians") or 0)
    metrics = {
        "median_iou": median_iou,
        "minimum_precision": minimum_precision,
        "minimum_recall": minimum_recall,
        "provisional_gaussians": gaussian_count,
    }
    if (
        median_iou >= 0.65
        and minimum_precision >= 0.65
        and minimum_recall >= 0.70
        and gaussian_count >= 500
    ):
        return {"tier": "recommended", "reason": "strict_heldout_gate_passed", **metrics}
    return {"tier": "conditional", "reason": "verified_but_strict_margin_low", **metrics}


def audit_object(
    row: Mapping[str, Any],
    *,
    heldout: Mapping[str, Any] | None = None,
    published: bool = False,
    minimum_identity_confidence: float = 0.85,
    minimum_refinement_confidence: float = 0.90,
    minimum_visible_target_coverage: float = DEFAULT_MIN_VISIBLE_TARGET_COVERAGE,
) -> dict[str, Any]:
    events = _semantic_events(row)
    identity_groups, whole_groups = _group_assessments(
        events,
        minimum_identity_confidence=minimum_identity_confidence,
        minimum_visible_target_coverage=minimum_visible_target_coverage,
    )
    identity = _consensus(identity_groups)
    whole = _consensus(whole_groups)
    signals = _refinement_signals(events, minimum_visible_target_coverage)
    part_scope_signals = _whole_instance_part_scope_signals(events, row)
    re_adjudication = _independent_blind_re_adjudication(
        events, minimum_identity_confidence
    )
    identity_category = (
        identity["confirmed_category"] or identity["probable_category"]
    )
    identity_confidence = max(
        (
            float(event.get("confidence") or 0.0)
            for event in identity_groups.get(identity_category, [])
        ),
        default=0.0,
    )
    if whole["confirmed_category"] and not whole["conflicting_categories"]:
        status = "whole_ready"
        category = whole["confirmed_category"]
        reasons = ["independent_whole_object_consensus"]
    elif (
        identity_category
        and not identity["conflicting_categories"]
        and identity_confidence >= minimum_refinement_confidence
        and signals
    ):
        status = "refine"
        category = identity_category
        reasons = signals
    elif identity_category and not identity["conflicting_categories"]:
        status = "needs_independent_evidence"
        category = identity_category
        reasons = (
            signals
            if signals
            else ["whole_object_consensus_not_independently_verified"]
        )
    else:
        status = "reject"
        category = ""
        reasons = (
            ["conflicting_identity_labels"]
            if identity["conflicting_categories"]
            else ["identity_unresolved"]
        )

    if str(row.get("semantic_gate_reason") or "") == (
        "contextual_identity_quarantined"
    ):
        status = "reject"
        category = ""
        reasons = ["contextual_identity_quarantined"]
    quarantine_re_adjudicated = bool(
        re_adjudication.get("accepted") is True
        and re_adjudication.get("mask_scope_incomplete") is True
        and str(row.get("review_gate_reason") or "")
        == "independent_blind_consensus_failed"
    )
    if (
        row.get("semantic_quarantined") is True
        or str(row.get("semantic_publication_action") or "")
        == "quarantine_unresolved_semantics"
    ) and not quarantine_re_adjudicated:
        status = "reject"
        category = ""
        reasons = ["independent_blind_semantics_quarantined"]
    legacy_consensus = row.get("independent_blind_consensus")
    if (
        str(row.get("semantic_publication_action") or "")
        == "preserve_prior_label"
        and (
            not isinstance(legacy_consensus, Mapping)
            or legacy_consensus.get("accepted") is not True
        )
    ):
        status = "reject"
        category = ""
        reasons = ["legacy_unverified_prior_semantics_quarantined"]

    center = _finite_vector(
        row.get("position_world_m", row.get("center_m")), 3
    )
    dimensions = _finite_vector(
        row.get("metric_dimensions_m", row.get("dimensions_m")), 3
    )
    rotation = _finite_vector(row.get("wxyz"), 4)
    selected_contracts = [
        event["label_contract"]
        for event in identity_groups.get(identity_category, [])
    ]
    cue_contract = max(
        selected_contracts,
        key=lambda item: float(item.get("confidence") or 0.0),
        default={},
    )
    bundle = {
        "schema": "farm.object-evidence-bundle.v1",
        "object_id": int(row["id"]),
        "status": status,
        "category": category,
        "referring_cue": {
            "description": str(cue_contract.get("description") or ""),
            "diagnostic_parts": list(cue_contract.get("diagnostic_parts") or []),
            "shape_profile": str(cue_contract.get("shape_profile") or "unknown"),
        },
        "representative_3d_box": {
            "center_m": center,
            "dimensions_m": dimensions,
            "wxyz": rotation,
        },
        "published_in_baseline": bool(published),
        "identity_confidence": identity_confidence,
        "identity_consensus": identity,
        "whole_object_consensus": whole,
        "semantic_identity_re_adjudication": re_adjudication,
        "semantic_quarantine_re_adjudicated": quarantine_re_adjudicated,
        "refinement_signals": sorted(set(signals + part_scope_signals)),
        "decision_reasons": reasons,
        "view_evidence": [_event_summary(event) for event in events],
        "heldout_gaussian_qc": dict(heldout or {}),
    }
    scope_reasons = sorted(set(
        signals
        + part_scope_signals
        + (
            list(re_adjudication.get("mask_scope_reason_codes") or [])
            if re_adjudication.get("mask_scope_incomplete") is True else []
        )
    ))
    geometry_refinement_required = bool(
        status == "refine"
        or re_adjudication.get("geometry_refinement_required") is True
        or scope_reasons
    )
    bundle["mask_scope"] = {
        "status": "incomplete" if scope_reasons else "complete",
        "reason_codes": scope_reasons or ["mask_scope_complete"],
    }
    bundle["geometry_refinement"] = {
        "required": geometry_refinement_required,
        "route": (
            "mask_geometry_refinement"
            if geometry_refinement_required else "none"
        ),
    }
    bundle["inpainting"] = (
        {
            "tier": "not_ready",
            "reason": "mask_scope_incomplete",
            "blocked_by": scope_reasons,
        }
        if scope_reasons
        else _inpainting_gate(status, heldout)
    )
    return bundle


def semantic_status_matrix_row(
    source: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, Any]:
    """Flatten one audit result without weakening any release gate."""

    action = str(source.get("semantic_publication_action") or "")
    review_decision = str(source.get("review_decision") or "").lower()
    if action == "publish_independent_blind_consensus" and review_decision == "keep":
        current_label = normalize_open_noun(source.get("review_category"))
        current_label_source = "published_independent_blind_review"
    else:
        current_label = normalize_open_noun(source.get("category"))
        current_label_source = (
            "preserved_prior_unverified"
            if action == "preserve_prior_label" else "catalog"
        )

    events = _semantic_events(source)
    raw_categories = [
        normalize_open_noun((event.get("label_contract") or {}).get("category"))
        for event in events
    ]
    concrete_categories = [
        value for value in raw_categories
        if value not in PUBLICATION_FORBIDDEN_GENERIC_NOUNS
    ]
    generic_or_unknown = any(
        value in PUBLICATION_FORBIDDEN_GENERIC_NOUNS
        for value in raw_categories
    )
    category_conflict = len(set(concrete_categories)) > 1
    consensus = dict(bundle.get("semantic_identity_re_adjudication") or {})
    consensus_accepted = consensus.get("accepted") is True
    source_quarantine = bool(
        source.get("semantic_quarantined") is True
        or action == "quarantine_unresolved_semantics"
    )
    quarantine_re_adjudicated = bool(
        bundle.get("semantic_quarantine_re_adjudicated") is True
    )
    effective_quarantine = bool(
        (source_quarantine and not quarantine_re_adjudicated)
        or not consensus_accepted
    )
    mask_scope = dict(bundle.get("mask_scope") or {})
    mask_scope_incomplete = mask_scope.get("status") == "incomplete"
    routes: list[str] = []
    if not consensus_accepted:
        routes.append("semantic_verification")
    for route in source.get("semantic_resolution_review_routes") or []:
        if route == "relationship_verification" and route not in routes:
            routes.append(route)
    if mask_scope_incomplete:
        routes.append("mask_geometry_refinement")
    if not routes:
        routes.append("none")
    label_publishable = bool(
        consensus_accepted
        and not mask_scope_incomplete
        and bundle.get("status") == "whole_ready"
        and not effective_quarantine
    )
    inpainting = dict(bundle.get("inpainting") or {})
    return {
        "object_id": int(source["id"]),
        "current_label": current_label or "unresolved object",
        "current_label_source": current_label_source,
        "raw_blind_nouns": raw_categories,
        "exact_noun_consensus": {
            "accepted": consensus_accepted,
            "category": str(consensus.get("canonical_category") or ""),
            "initial_category": str(consensus.get("initial_category") or ""),
            "verification_category": str(
                consensus.get("verification_category") or ""
            ),
            "reason_codes": list(consensus.get("reason_codes") or []),
            "identity_contract_source": str(
                consensus.get("identity_contract_source") or ""
            ),
            "evidence_independence": dict(
                consensus.get("evidence_independence") or {}
            ),
        },
        "generic_or_unknown_raw_noun": generic_or_unknown,
        "category_conflict": category_conflict,
        "source_quarantine": source_quarantine,
        "source_quarantine_re_adjudicated": quarantine_re_adjudicated,
        "effective_quarantine": effective_quarantine,
        "audit_status": str(bundle.get("status") or "reject"),
        "mask_scope": mask_scope,
        "routes": routes,
        "primary_route": routes[0],
        "label_publishable": label_publishable,
        "inpainting_ready": inpainting.get("tier") == "recommended",
        "inpainting": inpainting,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heldout-qc", type=Path)
    parser.add_argument("--gaussian-build-manifest", type=Path)
    parser.add_argument("--heldout-candidate-manifest", type=Path)
    parser.add_argument("--published-catalog", type=Path)
    parser.add_argument("--minimum-identity-confidence", type=float, default=0.85)
    parser.add_argument("--minimum-refinement-confidence", type=float, default=0.90)
    parser.add_argument(
        "--minimum-visible-target-coverage",
        type=float,
        default=DEFAULT_MIN_VISIBLE_TARGET_COVERAGE,
    )
    args = parser.parse_args()
    started = time.perf_counter()
    catalog_path = args.catalog.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    rows = _load_rows(catalog_path)
    heldout_by_id: dict[int, dict[str, Any]] = {}
    if args.heldout_qc:
        heldout_path = args.heldout_qc.expanduser().resolve(strict=True)
        heldout_payload = json.loads(heldout_path.read_text(encoding="utf-8"))
        heldout_by_id = {
            int(row["object_id"]): dict(row)
            for row in heldout_payload.get("objects") or []
            if isinstance(row, dict) and "object_id" in row
        }
    gaussian_count_evidence: dict[str, Any] = {"status": "not_supplied"}
    if bool(args.gaussian_build_manifest) != bool(args.heldout_candidate_manifest):
        raise ValueError(
            "--gaussian-build-manifest and --heldout-candidate-manifest "
            "must be supplied together"
        )
    if args.gaussian_build_manifest:
        if not args.heldout_qc:
            raise ValueError("linked Gaussian counts require --heldout-qc")
        counts_by_id, gaussian_count_evidence = _linked_gaussian_counts(
            args.gaussian_build_manifest,
            args.heldout_candidate_manifest,
            heldout_payload,
        )
        for object_id, count in counts_by_id.items():
            if object_id not in heldout_by_id:
                raise ValueError(
                    f"Gaussian object {object_id} is absent from heldout QC"
                )
            heldout_by_id[object_id]["provisional_gaussians"] = int(count)
    published_ids: set[int] = set()
    if args.published_catalog:
        published_ids = {
            int(row["id"])
            for row in _load_rows(args.published_catalog.expanduser().resolve(strict=True))
        }

    bundles = [
        audit_object(
            row,
            heldout=heldout_by_id.get(int(row["id"])),
            published=int(row["id"]) in published_ids,
            minimum_identity_confidence=float(args.minimum_identity_confidence),
            minimum_refinement_confidence=float(args.minimum_refinement_confidence),
            minimum_visible_target_coverage=float(
                args.minimum_visible_target_coverage
            ),
        )
        for row in rows
    ]
    bundles.sort(key=lambda row: int(row["object_id"]))
    counts = Counter(str(row["status"]) for row in bundles)
    inpainting_counts = Counter(str(row["inpainting"]["tier"]) for row in bundles)
    artifact = {
        "schema": "farm.whole-object-evidence-audit.v1",
        "source_catalog": str(catalog_path),
        "source_catalog_sha256": _sha256(catalog_path),
        "thresholds": {
            "minimum_identity_confidence": float(args.minimum_identity_confidence),
            "minimum_refinement_confidence": float(args.minimum_refinement_confidence),
            "minimum_visible_target_coverage": float(
                args.minimum_visible_target_coverage
            ),
            "whole_ready_requires_two_pose_independent_disjoint_view_events": True,
            "inpainting_median_iou": 0.65,
            "inpainting_minimum_precision": 0.65,
            "inpainting_minimum_recall": 0.70,
            "inpainting_minimum_gaussians": 500,
        },
        "object_count": len(bundles),
        "status_counts": dict(sorted(counts.items())),
        "inpainting_tier_counts": dict(sorted(inpainting_counts.items())),
        "gaussian_count_evidence": gaussian_count_evidence,
        "timing_seconds": time.perf_counter() - started,
        "objects": bundles,
    }
    _atomic_json(output_dir / "whole_object_audit.json", artifact)
    _atomic_json(
        output_dir / "object_evidence_bundles.json",
        {"schema": "farm.object-evidence-bundle-set.v1", "objects": bundles},
    )
    source_by_id = {int(row["id"]): row for row in rows}
    status_matrix = [
        semantic_status_matrix_row(source_by_id[int(bundle["object_id"])], bundle)
        for bundle in bundles
    ]
    _atomic_json(
        output_dir / "semantic_status_matrix.json",
        {
            "schema": "farm.semantic-status-matrix.v1",
            "source_catalog": str(catalog_path),
            "source_catalog_sha256": _sha256(catalog_path),
            "object_count": len(status_matrix),
            "policy": {
                "single_catalog_only": True,
                "two_independent_raw_blind_contracts_required": True,
                "generic_unknown_or_conflict_is_not_publishable": True,
                "mask_scope_is_separate_from_semantic_identity": True,
                "inpainting_ready_requires_strict_whole_and_heldout_gate": True,
            },
            "objects": status_matrix,
        },
    )
    candidates = [
        row for row in bundles
        if row["inpainting"]["tier"] in {"recommended", "conditional"}
    ]
    _atomic_json(
        output_dir / "inpainting_candidates.json",
        {"schema": "farm.inpainting-candidate-set.v1", "objects": candidates},
    )
    for status in ("whole_ready", "refine", "needs_independent_evidence", "reject"):
        values = [
            str(row["object_id"]) for row in bundles if row["status"] == status
        ]
        (output_dir / f"{status}_ids.txt").write_text(
            "\n".join(values) + ("\n" if values else ""), encoding="utf-8"
        )
    print(json.dumps({
        "schema": artifact["schema"],
        "object_count": len(bundles),
        "status_counts": artifact["status_counts"],
        "inpainting_tier_counts": artifact["inpainting_tier_counts"],
        "output_dir": str(output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
