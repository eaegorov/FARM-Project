#!/usr/bin/env python3
"""Apply full-COLMAP rescue with separate existence and label gates.

SAM3, multi-view RGB-D geometry and a blind whole-object VLM decision establish
that an object exists. Candidate-conditioned verification may publish a label,
but cannot erase validated geometry; uncertain labels remain explicitly unresolved.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scene_graph.captioning.label_contract import assess_open_vocabulary_label  # noqa: E402


GEOMETRY_KEYS = (
    "active",
    "object_box_centers_m",
    "object_box_dimensions_m",
    "object_box_wxyz",
    "object_geometry_status",
    "object_geometry_inside_rate",
    "object_geometry_reprojection_error",
    "object_geometry_valid_observations",
    "object_geometry_consistent_observations",
    "object_geometry_projected_box_iou",
    "object_geometry_box_support_rate",
    "object_geometry_voxel_inside_rate",
    "object_geometry_voxel_volume_expansion",
    "object_geometry_voxel_extent_candidate",
    "object_geometry_orientation_confidence",
    "object_geometry_gravity_tilt_degrees",
    "object_geometry_orientation_mode",
    "object_mask_observations",
    "object_image_ids",
    "viewpoint_image_ids",
)


def _load(path: Path) -> tuple[Any, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state") if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"unsupported scene state: {path}")
    return payload, state


def _array(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _copy_row(destination: dict, source: dict, key: str, index: int, count: int) -> None:
    value = source.get(key)
    target = destination.get(key)
    if isinstance(value, torch.Tensor) and isinstance(target, torch.Tensor):
        if value.ndim >= 1 and target.ndim >= 1 and value.shape[0] == target.shape[0] == count:
            target[index] = value[index]
            return
    if isinstance(value, (list, tuple)) and isinstance(target, (list, tuple)):
        if len(value) == len(target) == count:
            mutable = list(target)
            mutable[index] = copy.deepcopy(value[index])
            destination[key] = mutable
            return
    raise ValueError(f"cannot copy row-aligned rescue field {key!r}")


def _assessed_label(row: Mapping[str, Any], minimum_confidence: float) -> dict[str, Any]:
    contract = row.get("review_label_contract")
    if not isinstance(contract, Mapping):
        return {
            "usable": False,
            "category": "unresolved object",
            "description": "",
            "attributes": [],
            "reason_codes": ["label_contract_missing"],
        }
    return assess_open_vocabulary_label(
        contract, minimum_confidence=float(minimum_confidence)
    )


def _initial_vlm_geometry_gate(
    row: Mapping[str, Any], minimum_confidence: float
) -> tuple[bool, list[str]]:
    """Validate blind whole-object evidence independently of label verification."""

    reasons: list[str] = []
    events = [
        event for event in row.get("semantic_evidence") or []
        if isinstance(event, Mapping)
        and str(event.get("source") or "") == "initial_blind_review"
    ]
    if len(events) != 1:
        return False, ["vlm_initial_blind_evidence_missing_or_duplicated"]
    event = events[0]
    if str(event.get("decision") or "").lower() != "keep":
        reasons.append("vlm_initial_decision_not_keep")
    confidence = float(event.get("confidence") or 0.0)
    if not np.isfinite(confidence) or confidence < minimum_confidence:
        reasons.append("vlm_initial_confidence_below_threshold")
    if event.get("confirmation_eligible") is not True:
        reasons.append("vlm_initial_evidence_not_confirmation_eligible")
    contract = event.get("label_contract")
    if not isinstance(contract, Mapping) or not bool(contract.get("contract_valid")):
        reasons.append("vlm_initial_label_contract_invalid")
    else:
        if contract.get("complete_bounded") is not True:
            reasons.append("vlm_initial_not_complete_bounded_object")
        if str(contract.get("topology") or "") not in {
            "standalone_whole", "carrier_payload"
        }:
            reasons.append("vlm_initial_topology_not_whole")
        if int(contract.get("diagnostic_view_count") or 0) < 2:
            reasons.append("vlm_initial_insufficient_diagnostic_views")
    image_ids = {
        int(value) for value in event.get("crop_image_ids") or []
        if isinstance(value, (int, np.integer)) or str(value).isdigit()
    }
    if len(image_ids) < 2:
        reasons.append("vlm_initial_multiview_evidence_missing")
    return not reasons, sorted(set(reasons))


def _vlm_gate(row: Mapping[str, Any], minimum_confidence: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if str(row.get("review_decision") or "").lower() != "keep":
        reasons.append("vlm_decision_not_keep")
    confidence = float(row.get("review_confidence") or 0.0)
    if not np.isfinite(confidence) or confidence < minimum_confidence:
        reasons.append("vlm_confidence_below_threshold")
    contract = row.get("review_label_contract")
    assessment = _assessed_label(row, minimum_confidence)
    if not bool(assessment.get("usable")):
        reasons.extend(
            f"vlm_label_{value}"
            for value in assessment.get("reason_codes") or ["assessment_failed"]
        )
    if not isinstance(contract, Mapping) or not bool(contract.get("contract_valid")):
        reasons.append("vlm_label_contract_invalid")
    else:
        if contract.get("complete_bounded") is not True:
            reasons.append("vlm_not_complete_bounded_object")
        if str(contract.get("topology") or "") not in {
            "standalone_whole", "carrier_payload"
        }:
            reasons.append("vlm_topology_not_whole")
        if int(contract.get("diagnostic_view_count") or 0) < 2:
            reasons.append("vlm_insufficient_diagnostic_views")
    evidence = [event for event in row.get("semantic_evidence") or [] if isinstance(event, Mapping)]
    sources = {str(event.get("source") or "") for event in evidence}
    if not {"initial_blind_review", "independent_verification"}.issubset(sources):
        reasons.append("vlm_two_pass_verification_missing")
    if any(str(event.get("decision") or "").lower() != "keep" for event in evidence):
        reasons.append("vlm_evidence_contains_nonkeep")
    image_ids = {
        int(value)
        for event in evidence
        for value in (event.get("crop_image_ids") or [])
        if isinstance(value, (int, np.integer)) or str(value).isdigit()
    }
    if len(image_ids) < 2:
        reasons.append("vlm_multiview_evidence_missing")
    return not reasons, sorted(set(reasons))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--refined-state", type=Path, required=True)
    parser.add_argument("--refinement", type=Path, required=True)
    parser.add_argument("--geometry-report", type=Path, required=True)
    parser.add_argument("--review-report", type=Path, required=True)
    parser.add_argument("--prior-catalog", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-catalog", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--minimum-vlm-confidence", type=float, default=0.70)
    args = parser.parse_args()
    started = time.perf_counter()

    source_path = args.source_state.expanduser().resolve(strict=True)
    refined_path = args.refined_state.expanduser().resolve(strict=True)
    source_payload, source_state = _load(source_path)
    _, refined_state = _load(refined_path)
    payload = copy.deepcopy(source_payload)
    state = payload["state"] if isinstance(payload, dict) and isinstance(payload.get("state"), dict) else payload
    if not isinstance(state, dict):
        raise TypeError("copied source state is invalid")
    source_ids = _array(state["object_id"]).astype(np.int64).reshape(-1)
    refined_ids = _array(refined_state["object_id"]).astype(np.int64).reshape(-1)
    if not np.array_equal(source_ids, refined_ids):
        raise ValueError("source and refined state object IDs differ")
    count = int(source_ids.size)
    index_by_id = {int(value): index for index, value in enumerate(source_ids.tolist())}

    refinement_path = args.refinement.expanduser().resolve(strict=True)
    refinement = json.loads(refinement_path.read_text(encoding="utf-8"))
    if refinement.get("schema") != "farm.full-colmap-mask-refinement.v1" or refinement.get("status") != "PASS":
        raise ValueError("invalid mask-refinement report")
    refined_candidates = {
        int(row["object_id"]): row
        for row in refinement.get("objects") or []
        if isinstance(row, dict) and bool(row.get("accepted"))
    }
    geometry_path = args.geometry_report.expanduser().resolve(strict=True)
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if geometry.get("schema") != "farm.object-geometry-audit.v3":
        raise ValueError("invalid geometry report")
    geometry_by_id = {
        int(row["object_id"]): row
        for row in geometry.get("objects") or []
        if isinstance(row, dict)
    }
    review_path = args.review_report.expanduser().resolve(strict=True)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("schema") != "farm.open-vocabulary-crop-review.v2":
        raise ValueError("invalid VLM review report")
    if not bool(review.get("verification_enabled")):
        raise ValueError("full-COLMAP rescue requires the second VLM verification pass")
    if int(review.get("request_error_count") or 0) != 0:
        raise ValueError("VLM rescue review contains transport failures")
    review_by_id = {
        int(row["id"]): row for row in review.get("objects") or [] if isinstance(row, dict)
    }

    if len(refined_state.get("images") or []) < len(source_state.get("images") or []):
        raise ValueError("refined state lost source images")
    state["images"] = copy.deepcopy(refined_state.get("images") or [])
    state["image_positions"] = copy.deepcopy(refined_state.get("image_positions") or [])

    statuses = ["not_planned"] * count
    evidence_rows: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    accepted_ids: list[int] = []
    label_verified_by_id: dict[int, bool] = {}
    decisions: list[dict[str, Any]] = []
    for object_id in sorted(refined_candidates):
        index = index_by_id.get(object_id)
        reasons: list[str] = []
        label_reasons: list[str] = []
        geometry_row = geometry_by_id.get(object_id)
        if geometry_row is None or str(geometry_row.get("status") or "") != "geometry_pass":
            reasons.append("geometry_not_pass")
        review_row = review_by_id.get(object_id)
        label_verified = False
        if review_row is None:
            reasons.append("vlm_review_missing")
        else:
            existence_pass, existence_reasons = _initial_vlm_geometry_gate(
                review_row, float(args.minimum_vlm_confidence)
            )
            if not existence_pass:
                reasons.extend(existence_reasons)
            label_verified, label_reasons = _vlm_gate(
                review_row, float(args.minimum_vlm_confidence)
            )
        accepted = index is not None and not reasons
        if accepted:
            for key in GEOMETRY_KEYS:
                _copy_row(state, refined_state, key, index, count)
            if label_verified:
                assessment = _assessed_label(review_row, float(args.minimum_vlm_confidence))
                category = str(assessment["category"])
                description = str(assessment.get("description") or "")
                attributes = [
                    str(value).strip() for value in assessment.get("attributes") or []
                    if str(value).strip()
                ][:8]
                semantic_tier = "probable"
                semantic_status = "full_colmap_sam3_vlm_rescued"
                statuses[index] = "accepted_label_verified"
            else:
                category = "unresolved object"
                description = (
                    "Validated whole physical object with multi-view SAM3/RGB-D geometry; "
                    "the candidate-conditioned label audit remained uncertain."
                )
                attributes = []
                semantic_tier = "geometry_only"
                semantic_status = "full_colmap_sam3_geometry_rescued"
                statuses[index] = "accepted_geometry_only"
            for key, value in (
                ("object_category", category),
                ("object_caption", description),
                ("object_key_attributes", attributes),
                ("object_semantic_tier", semantic_tier),
                ("object_semantic_status", semantic_status),
            ):
                values = list(state.get(key) or ["" for _ in range(count)])
                if len(values) != count:
                    raise ValueError(f"semantic field {key!r} is not object-aligned")
                values[index] = value
                state[key] = values
            evidence_rows[index] = copy.deepcopy(review_row.get("semantic_evidence") or [])
            accepted_ids.append(object_id)
            label_verified_by_id[object_id] = bool(label_verified)
        elif index is not None:
            statuses[index] = "rejected_preserved_original"
        decisions.append({
            "object_id": object_id,
            "accepted": bool(accepted),
            "label_verified": bool(label_verified),
            "status": statuses[index] if index is not None else "unknown_object_id",
            "reasons": sorted(set(reasons)),
            "label_reasons": sorted(set(label_reasons)),
            "refined_views": int(refined_candidates[object_id].get("accepted_views") or 0),
            "geometry_status": str((geometry_row or {}).get("status") or "missing"),
            "vlm_category": str((review_row or {}).get("review_category") or "unknown"),
            "vlm_confidence": float((review_row or {}).get("review_confidence") or 0.0),
        })
    state["object_full_colmap_rescue_status"] = statuses
    state["object_full_colmap_rescue_semantic_evidence"] = evidence_rows

    prior_catalog_path = args.prior_catalog.expanduser().resolve(strict=True)
    prior_catalog = json.loads(prior_catalog_path.read_text(encoding="utf-8"))
    if not isinstance(prior_catalog, list):
        raise TypeError("prior catalog must be a JSON array")
    catalog_by_id = {int(row["id"]): copy.deepcopy(row) for row in prior_catalog if isinstance(row, dict)}
    for object_id in accepted_ids:
        index = index_by_id[object_id]
        review_row = review_by_id[object_id]
        label_verified = bool(label_verified_by_id.get(object_id))
        if label_verified:
            assessment = _assessed_label(review_row, float(args.minimum_vlm_confidence))
            category = str(assessment["category"])
            description = str(assessment.get("description") or "")
            attributes = list(assessment.get("attributes") or [])[:8]
            semantic_tier = "probable"
            semantic_status = "full_colmap_sam3_vlm_rescued"
            semantic_gate_reason = "geometry_sam3_and_two_pass_label_agreement"
        else:
            category = "unresolved object"
            description = (
                "Validated whole physical object with multi-view SAM3/RGB-D geometry; "
                "semantic identity requires review."
            )
            attributes = []
            semantic_tier = "geometry_only"
            semantic_status = "full_colmap_sam3_geometry_rescued"
            semantic_gate_reason = "geometry_sam3_and_blind_whole_object_vlm"
        row = catalog_by_id.get(object_id, {"id": object_id})
        row.update({
            "observations": len(set(map(int, state["object_image_ids"][index] or []))),
            "position_world_m": _array(state["object_box_centers_m"])[index].astype(float).tolist(),
            "cov6": _array(state["cov6"])[index].astype(float).tolist(),
            "metric_dimensions_m": _array(state["object_box_dimensions_m"])[index].astype(float).tolist(),
            "category": category,
            "description": description,
            "attributes": attributes,
            "semantic_tier": semantic_tier,
            "semantic_status": semantic_status,
            "review_decision": "keep" if label_verified else "unknown",
            "review_confidence": float(review_row.get("review_confidence") or 0.0),
            "semantic_gate_reason": semantic_gate_reason,
            "negative_evidence_sources": [],
            "candidates": list(row.get("candidates") or []) + list(review_row.get("semantic_evidence") or []),
        })
        catalog_by_id[object_id] = row
    catalog = [catalog_by_id[key] for key in sorted(catalog_by_id)]

    if isinstance(payload, dict) and isinstance(payload.get("state"), dict):
        payload["state"] = state
        payload["saved_unix_s"] = time.time()
        payload.setdefault("meta", {})["full_colmap_rescue_acceptance"] = {
            "accepted_object_ids": accepted_ids,
            "policy": "sam3_geometry_and_blind_vlm_existence_with_separate_label_verification",
            "review_sha256": _sha256(review_path),
        }
    else:
        payload = state
    output_state = args.output_state.expanduser().resolve()
    output_state.parent.mkdir(parents=True, exist_ok=True)
    temporary_state = output_state.with_suffix(".pt.tmp")
    torch.save(payload, temporary_state)
    temporary_state.replace(output_state)
    output_catalog = args.output_catalog.expanduser().resolve()
    output_catalog.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_catalog, catalog)

    report = {
        "schema": "farm.full-colmap-rescue-acceptance.v1",
        "status": "PASS",
        "policy": {
            "rest3d_inspired_vlm_mask_refinement": True,
            "sam3_mask_refinement": True,
            "multi_view_rgbd_geometry_required": True,
            "blind_vlm_whole_object_review_required": True,
            "candidate_conditioned_verification_required_for_label_publication": True,
            "candidate_conditioned_verification_can_erase_geometry": False,
            "uncertain_label_becomes_geometry_only": True,
            "uncertain_existence_preserves_original": True,
            "minimum_vlm_confidence": float(args.minimum_vlm_confidence),
        },
        "candidate_objects": len(refined_candidates),
        "accepted_objects": len(accepted_ids),
        "accepted_object_ids": accepted_ids,
        "label_verified_objects": sum(label_verified_by_id.values()),
        "label_verified_object_ids": [
            object_id for object_id in accepted_ids if label_verified_by_id.get(object_id)
        ],
        "geometry_only_objects": sum(
            not label_verified_by_id.get(object_id, False) for object_id in accepted_ids
        ),
        "geometry_only_object_ids": [
            object_id for object_id in accepted_ids if not label_verified_by_id.get(object_id, False)
        ],
        "rejected_objects": len(refined_candidates) - len(accepted_ids),
        "decisions": decisions,
        "timing_seconds": time.perf_counter() - started,
        "hashes": {
            "source_state_sha256": _sha256(source_path),
            "refined_state_sha256": _sha256(refined_path),
            "refinement_sha256": _sha256(refinement_path),
            "geometry_report_sha256": _sha256(geometry_path),
            "review_report_sha256": _sha256(review_path),
            "output_state_sha256": _sha256(output_state),
            "output_catalog_sha256": _sha256(output_catalog),
        },
    }
    output_report = args.output_report.expanduser().resolve()
    output_report.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_report, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
