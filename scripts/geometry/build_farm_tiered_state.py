#!/usr/bin/env python3
"""Merge direct metric objects, semantic evidence tiers, and assemblies."""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from farm_pipeline.final_acceptance import (  # noqa: E402
    validated_assembly_semantic_row,
)


def ensure_list(state: dict, key: str, size: int, default):
    value = state.get(key)
    if not isinstance(value, list):
        value = list(value) if isinstance(value, tuple) else []
    if len(value) < size:
        value.extend(
            default() if callable(default) else default
            for _ in range(size - len(value))
        )
    state[key] = value
    return value


def load_state(path: Path) -> tuple[dict, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
        raise ValueError(f"Unsupported scene state: {path}")
    return payload, payload["state"]


def _canonical_member_tuple(value: object) -> tuple[int, ...]:
    if value is None or isinstance(value, (str, bytes)):
        return ()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    try:
        return tuple(sorted({int(member) for member in value}))
    except (TypeError, ValueError):
        return ()


def assembly_tiered_eligibility(
    *,
    geometry_active: bool,
    geometry_status: str,
    state_member_ids: object,
    review_member_ids: object,
    review_decision: str,
    review_category: str,
) -> tuple[bool, str]:
    """Strict AND gate: semantics may annotate, never override geometry."""
    state_members = _canonical_member_tuple(state_member_ids)
    review_members = _canonical_member_tuple(review_member_ids)
    if not geometry_active:
        return False, "geometry_inactive"
    if str(geometry_status) != "assembly_geometry_pass":
        return False, "geometry_status_not_pass"
    if not state_members or state_members != review_members:
        return False, "member_provenance_mismatch"
    if str(review_decision).strip().lower() != "keep":
        return False, "review_decision_not_keep"
    category = str(review_category).strip()
    if not category or category.lower() == "unknown":
        return False, "semantic_category_unresolved"
    return True, "eligible"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembled-state", type=Path, required=True)
    parser.add_argument("--direct-catalog", type=Path, required=True)
    parser.add_argument("--assembly-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    assembled_path = args.assembled_state.expanduser().resolve()
    direct_catalog_path = args.direct_catalog.expanduser().resolve()
    assembly_review_path = args.assembly_review.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    payload, state = load_state(assembled_path)
    object_ids = state["object_id"].detach().cpu().tolist()
    index_by_id = {
        int(object_id): index for index, object_id in enumerate(object_ids)
    }
    size = len(object_ids)

    geometry_active = state["active"].detach().cpu().to(torch.bool).clone()
    geometry_statuses = ensure_list(state, "object_geometry_status", size, "not_evaluated")
    assembly_members = ensure_list(state, "object_assembly_member_ids", size, list)
    active = torch.zeros((size,), dtype=torch.bool)
    categories = ensure_list(state, "object_category", size, "")
    captions = ensure_list(state, "object_caption", size, "")
    attributes = ensure_list(state, "object_key_attributes", size, list)
    supercategories = ensure_list(state, "object_supercategory", size, "")
    decisions = ensure_list(state, "object_caption_decision", size, "")
    statuses = ensure_list(state, "object_semantic_status", size, "inactive")
    tiers = ensure_list(state, "object_semantic_tier", size, "inactive")
    confidences = torch.zeros((size,), dtype=torch.float32)
    skipped_direct: list[dict] = []
    skipped_assemblies: list[dict] = []
    annotated_inactive_assemblies: list[int] = []

    direct = json.loads(direct_catalog_path.read_text(encoding="utf-8"))
    for row in direct:
        index = index_by_id.get(int(row["id"]))
        if index is None:
            continue
        if not bool(geometry_active[index]) or str(geometry_statuses[index]) != "geometry_pass":
            skipped_direct.append(
                {
                    "id": int(row["id"]),
                    "reason": "direct_geometry_not_pass",
                    "geometry_status": str(geometry_statuses[index]),
                }
            )
            continue
        active[index] = True
        categories[index] = str(row.get("category") or "unresolved object")
        captions[index] = str(
            row.get("description")
            or "Metric multi-view object with unresolved semantic identity."
        )
        attributes[index] = [
            str(value) for value in (row.get("attributes") or [])
        ]
        supercategories[index] = "open-vocabulary object"
        decisions[index] = "keep"
        statuses[index] = str(
            row.get("semantic_status")
            or "geometry_valid_semantics_unresolved"
        )
        tiers[index] = str(row.get("semantic_tier") or "geometry_only")
        confidences[index] = float(row.get("review_confidence") or 0.0)

    assembly_payload = json.loads(
        assembly_review_path.read_text(encoding="utf-8")
    )
    if (
        not isinstance(assembly_payload, dict)
        or assembly_payload.get("schema") != "farm.open-vocabulary-crop-review.v2"
        or not isinstance(assembly_payload.get("objects"), list)
    ):
        raise ValueError("Unsupported assembly review contract")
    minimum_assembly_confidence = assembly_payload.get("min_confidence", 0.65)
    assembly_index_by_members: dict[tuple[int, ...], int] = {}
    for index, members in enumerate(assembly_members):
        key = _canonical_member_tuple(members)
        if key:
            assembly_index_by_members[key] = index

    for row in assembly_payload.get("objects") or []:
        review_members = _canonical_member_tuple(row.get("member_object_ids"))
        index = assembly_index_by_members.get(review_members)
        review_id = int(row.get("id", -1))
        id_index = index_by_id.get(review_id)
        if id_index is not None and _canonical_member_tuple(assembly_members[id_index]) != review_members:
            skipped_assemblies.append(
                {"id": review_id, "reason": "review_id_member_mismatch", "review_members": list(review_members)}
            )
            continue
        if index is None:
            skipped_assemblies.append(
                {"id": review_id, "reason": "assembly_member_tuple_not_found", "review_members": list(review_members)}
            )
            continue
        eligible, reason = assembly_tiered_eligibility(
            geometry_active=bool(geometry_active[index]),
            geometry_status=str(geometry_statuses[index]),
            state_member_ids=assembly_members[index],
            review_member_ids=row.get("member_object_ids"),
            review_decision=str(row.get("review_decision") or ""),
            review_category=str(row.get("review_category") or ""),
        )
        if not eligible:
            skipped_assemblies.append(
                {
                    "id": review_id,
                    "reason": reason,
                    "geometry_status": str(geometry_statuses[index]),
                    "review_members": list(review_members),
                }
            )
            category = str(row.get("review_category") or "").strip()
            semantic_keep = str(row.get("review_decision") or "").strip().lower() == "keep"
            exact_provenance = _canonical_member_tuple(assembly_members[index]) == review_members and bool(review_members)
            if (
                str(geometry_statuses[index]) == "assembly_compound_geometry_probable"
                and semantic_keep
                and category
                and category.lower() != "unknown"
                and exact_provenance
            ):
                # Preserve useful semantics for an explicitly non-metric
                # presentation layer without allowing semantic activation.
                categories[index] = category
                captions[index] = str(row.get("review_description") or "")
                attributes[index] = [str(value) for value in (row.get("review_attributes") or [])]
                supercategories[index] = "open-vocabulary probable compound"
                decisions[index] = "keep"
                statuses[index] = "vlm_compound_geometry_probable"
                tiers[index] = "geometry_probable"
                confidences[index] = float(row.get("review_confidence") or 0.0)
                annotated_inactive_assemblies.append(int(object_ids[index]))
            continue
        semantic_row, semantic_reason = validated_assembly_semantic_row(
            state,
            row,
            minimum_confidence=minimum_assembly_confidence,
        )
        if semantic_row is None:
            skipped_assemblies.append(
                {
                    "id": review_id,
                    "reason": semantic_reason,
                    "geometry_status": str(geometry_statuses[index]),
                    "review_members": list(review_members),
                }
            )
            continue
        active[index] = True
        categories[index] = str(semantic_row["category"])
        captions[index] = str(semantic_row.get("description") or "")
        attributes[index] = [
            str(value) for value in (semantic_row.get("attributes") or [])
        ]
        supercategories[index] = "open-vocabulary assembly"
        decisions[index] = "keep"
        statuses[index] = str(semantic_row["semantic_status"])
        # An assembly has one blind review event. It is useful structured
        # evidence, but never an independent confirmation pair.
        tiers[index] = "probable"
        confidences[index] = float(semantic_row["review_confidence"])

    state["active"] = active
    state["object_review_confidence"] = confidences
    payload["saved_unix_s"] = time.time()
    tier_counts = Counter(
        tiers[index] for index in range(size) if bool(active[index])
    )
    payload.setdefault("meta", {})["tiered_presentation"] = {
        "assembled_state": str(assembled_path),
        "direct_catalog": str(direct_catalog_path),
        "assembly_review": str(assembly_review_path),
        "active_metric_objects": int(active.sum().item()),
        "tier_counts": dict(tier_counts),
        "default_viewer_layer": "confirmed_plus_probable",
        "activation_invariant": (
            "geometry_pass AND validated_structured_semantic_review "
            "AND exact_member_provenance"
        ),
        "skipped_direct": skipped_direct,
        "skipped_assemblies": skipped_assemblies,
        "annotated_inactive_compound_assemblies": annotated_inactive_assemblies,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        json.dumps(
            payload["meta"]["tiered_presentation"],
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
