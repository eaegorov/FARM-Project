#!/usr/bin/env python3
"""Publish an explicitly reviewed subset of contextual Qwen label proposals."""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


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


def _load_state(path: Path) -> tuple[Any, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state") if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"unsupported scene state: {path}")
    return payload, state


def _parse_ids(value: str) -> list[int]:
    values = [part.strip() for part in str(value).split(",") if part.strip()]
    result = [int(part) for part in values]
    if not result or len(result) != len(set(result)) or any(item < 0 for item in result):
        raise ValueError("--object-ids must contain unique non-negative comma-separated IDs")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--source-presentation-catalog", type=Path, required=True)
    parser.add_argument("--candidate-catalog", type=Path, required=True)
    parser.add_argument("--object-ids", required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-catalog", type=Path, required=True)
    parser.add_argument("--output-presentation-catalog", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()

    source_state_path = args.source_state.expanduser().resolve(strict=True)
    payload, source_state = _load_state(source_state_path)
    payload = copy.deepcopy(payload)
    state = payload["state"] if isinstance(payload, dict) and isinstance(payload.get("state"), dict) else payload
    object_ids = np.asarray(state["object_id"], dtype=np.int64).reshape(-1)
    index_by_id = {int(value): index for index, value in enumerate(object_ids.tolist())}
    if len(index_by_id) != object_ids.size:
        raise ValueError("scene-state object IDs are not unique")

    source_catalog_path = args.source_catalog.expanduser().resolve(strict=True)
    source_presentation_path = args.source_presentation_catalog.expanduser().resolve(strict=True)
    candidate_path = args.candidate_catalog.expanduser().resolve(strict=True)
    full_catalog = json.loads(source_catalog_path.read_text(encoding="utf-8"))
    presentation = json.loads(source_presentation_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    for name, rows in (("source catalog", full_catalog), ("presentation catalog", presentation), ("candidate catalog", candidates)):
        if not isinstance(rows, list):
            raise TypeError(f"{name} must be a JSON array")
    full_by_id = {int(row["id"]): copy.deepcopy(row) for row in full_catalog}
    presentation_by_id = {int(row["id"]): copy.deepcopy(row) for row in presentation}
    candidate_by_id = {int(row["id"]): row for row in candidates}
    selected_ids = _parse_ids(args.object_ids)
    unknown = sorted(set(selected_ids) - set(index_by_id))
    not_presented = sorted(set(selected_ids) - set(presentation_by_id))
    missing_candidates = sorted(set(selected_ids) - set(candidate_by_id))
    if unknown or not_presented or missing_candidates:
        raise ValueError(
            f"invalid publish IDs: unknown={unknown}, not_presented={not_presented}, "
            f"missing_candidates={missing_candidates}"
        )

    count = int(object_ids.size)

    def state_rows(name: str, width: int) -> np.ndarray:
        value = state.get(name)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (count, width):
            raise ValueError(
                f"scene-state field {name!r} is not object-aligned Nx{width}"
            )
        return array

    centers = state_rows("object_box_centers_m", 3)
    dimensions = state_rows("object_box_dimensions_m", 3)
    rotations = state_rows("object_box_wxyz", 4)
    cov6 = state_rows("cov6", 6)
    presentation_indices = np.asarray(
        [index_by_id[object_id] for object_id in presentation_by_id], dtype=np.int64
    )
    if (
        not np.isfinite(centers[presentation_indices]).all()
        or not np.isfinite(dimensions[presentation_indices]).all()
        or not np.isfinite(rotations[presentation_indices]).all()
        or not np.isfinite(cov6[presentation_indices]).all()
        or np.any(dimensions[presentation_indices] <= 0.0)
        or np.any(np.linalg.norm(rotations[presentation_indices], axis=1) <= 1e-8)
    ):
        raise ValueError("presentation objects contain invalid OBB geometry")
    evidence_rows = list(state.get("object_contextual_semantic_evidence") or [[] for _ in range(count)])
    if len(evidence_rows) != count:
        evidence_rows = [[] for _ in range(count)]
    decisions = []
    for object_id in selected_ids:
        candidate = candidate_by_id[object_id]
        category = str(candidate.get("category") or "").strip().lower()
        tier = str(candidate.get("semantic_tier") or "geometry_only")
        contract = candidate.get("label_contract")
        if category in {"", "unknown", "unresolved object"}:
            raise ValueError(f"object {object_id} has no publishable category")
        if tier not in {"probable", "confirmed"}:
            raise ValueError(f"object {object_id} has non-publishable tier {tier!r}")
        if not isinstance(contract, dict) or contract.get("contract_valid") is not True:
            raise ValueError(f"object {object_id} lacks a valid strict label contract")
        index = index_by_id[object_id]
        prior_category = str(state["object_category"][index])
        description = str(candidate.get("description") or "")
        attributes = [str(value).strip() for value in candidate.get("attributes") or [] if str(value).strip()][:8]
        for key, value in (
            ("object_category", category),
            ("object_caption", description),
            ("object_key_attributes", attributes),
            ("object_semantic_tier", tier),
            ("object_semantic_status", "contextual_qwen_reviewed"),
        ):
            values = list(state.get(key) or ["" for _ in range(count)])
            if len(values) != count:
                raise ValueError(f"semantic field {key!r} is not object-aligned")
            values[index] = value
            state[key] = values
        evidence_rows[index] = copy.deepcopy(candidate.get("semantic_evidence") or candidate.get("candidates") or [])
        update = {
            "category": category,
            "description": description,
            "attributes": attributes,
            "semantic_tier": tier,
            "semantic_status": "contextual_qwen_reviewed",
            "semantic_gate_reason": "explicit_experiment_publish_allowlist_after_visual_audit",
            "label_contract": copy.deepcopy(contract),
            "candidates": copy.deepcopy(candidate.get("candidates") or []),
        }
        full_by_id.setdefault(object_id, {"id": object_id}).update(update)
        presentation_by_id[object_id].update(update)
        decisions.append({
            "object_id": object_id,
            "prior_category": prior_category,
            "published_category": category,
            "semantic_tier": tier,
            "diagnostic_view_count": int(contract.get("diagnostic_view_count") or 0),
            "confidence": float(contract.get("confidence") or 0.0),
        })

    # Catalog geometry must come from the final state, not from old
    # presentation rows. This publishes OBB and semantic changes through one
    # state-bound handoff.
    for object_id in presentation_by_id:
        row = full_by_id.setdefault(object_id, {"id": object_id})
        index = index_by_id[object_id]
        row.update({
            "position_world_m": centers[index].astype(float).tolist(),
            "metric_dimensions_m": dimensions[index].astype(float).tolist(),
            "cov6": cov6[index].astype(float).tolist(),
        })
    for object_id, row in presentation_by_id.items():
        index = index_by_id[object_id]
        row.update({
            "center_m": centers[index].astype(float).tolist(),
            "dimensions_m": dimensions[index].astype(float).tolist(),
            "wxyz": rotations[index].astype(float).tolist(),
        })

    state["object_contextual_semantic_evidence"] = evidence_rows
    if isinstance(payload, dict) and isinstance(payload.get("state"), dict):
        payload["state"] = state
        payload["saved_unix_s"] = time.time()
        payload.setdefault("meta", {})["contextual_qwen_publish"] = {
            "object_ids": selected_ids,
            "candidate_catalog_sha256": _sha256(candidate_path),
        }
    else:
        payload = state

    output_state = args.output_state.expanduser().resolve()
    output_state.parent.mkdir(parents=True, exist_ok=True)
    temporary_state = output_state.with_suffix(output_state.suffix + ".tmp")
    torch.save(payload, temporary_state)
    temporary_state.replace(output_state)
    output_catalog = args.output_catalog.expanduser().resolve()
    output_presentation = args.output_presentation_catalog.expanduser().resolve()
    output_catalog.parent.mkdir(parents=True, exist_ok=True)
    output_presentation.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_catalog, [full_by_id[key] for key in sorted(full_by_id)])
    _atomic_json(output_presentation, [presentation_by_id[int(row["id"])] for row in presentation])

    report = {
        "schema": "farm.contextual-qwen-label-publish.v1",
        "status": "PASS",
        "published_objects": len(selected_ids),
        "published_object_ids": selected_ids,
        "decisions": decisions,
        "policy": {
            "strict_contract_required": True,
            "probable_or_confirmed_required": True,
            "explicit_visual_audit_allowlist": True,
            "unlisted_labels_preserved_bit_for_bit": True,
        },
        "timing_seconds": time.perf_counter() - started,
        "hashes": {
            "source_state_sha256": _sha256(source_state_path),
            "candidate_catalog_sha256": _sha256(candidate_path),
            "output_state_sha256": _sha256(output_state),
            "output_catalog_sha256": _sha256(output_catalog),
            "output_presentation_catalog_sha256": _sha256(output_presentation),
        },
    }
    _atomic_json(args.output_report.expanduser().resolve(), report)
    print(json.dumps({key: value for key, value in report.items() if key != "decisions"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
