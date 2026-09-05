#!/usr/bin/env python3
"""Export one conservative presentation OBB for selected compound objects.

Compound FARM objects retain their component boxes and consensus evidence in
the source state. This exporter uses the already-audited floor-aligned parent
box from compound diagnostics as a single presentation box. It does not alter
active flags, semantic labels, source evidence, or metric status.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import copy
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


def _finite_box(record: object) -> dict | None:
    if not isinstance(record, dict):
        return None
    center = np.asarray(record.get("center_m"), dtype=np.float64).reshape(-1)
    dimensions = np.asarray(
        record.get("dimensions_m", record.get("dimensions_lwh_m")),
        dtype=np.float64,
    ).reshape(-1)
    wxyz = np.asarray(record.get("wxyz"), dtype=np.float64).reshape(-1)
    if center.shape != (3,) or dimensions.shape != (3,) or wxyz.shape != (4,):
        return None
    if not np.isfinite(center).all() or not np.isfinite(dimensions).all() or not np.isfinite(wxyz).all():
        return None
    if np.any(dimensions <= 0.0) or float(np.linalg.norm(wxyz)) <= 1.0e-8:
        return None
    result = copy.deepcopy(record)
    result["center_m"] = center.tolist()
    result["dimensions_lwh_m"] = dimensions.tolist()
    result.pop("dimensions_m", None)
    result["wxyz"] = (wxyz / float(np.linalg.norm(wxyz))).tolist()
    return result


def unify_compound_boxes(
    payload: dict,
    object_ids: list[int] | None,
    *,
    min_gaussian_support: float = 0.90,
) -> dict:
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
        raise ValueError("Expected a wrapped FARM state dictionary")
    state = payload["state"]
    ids_raw = state.get("object_id")
    ids = (
        ids_raw.detach().cpu().numpy().astype(np.int64, copy=False)
        if isinstance(ids_raw, torch.Tensor)
        else np.asarray(ids_raw, dtype=np.int64)
    )
    index_by_id = {int(value): index for index, value in enumerate(ids.tolist())}
    boxes_all = state.get("object_compound_boxes") or []
    diagnostics_all = state.get("object_compound_geometry_diagnostics") or []
    statuses = state.get("object_geometry_status") or []
    rows: list[dict] = []

    requested_ids = (
        sorted(
            int(ids[index])
            for index in range(len(ids))
            if "compound_geometry_probable"
            in str(statuses[index] if index < len(statuses) else "").strip().lower()
            and "rejected"
            not in str(statuses[index] if index < len(statuses) else "").strip().lower()
            and index < len(boxes_all)
            and isinstance(boxes_all[index], (list, tuple))
            and len(boxes_all[index]) >= 2
        )
        if object_ids is None
        else sorted(set(int(value) for value in object_ids))
    )

    for object_id in requested_ids:
        index = index_by_id.get(object_id)
        if index is None:
            rows.append({"object_id": object_id, "accepted": False, "reason": "object_id_not_found"})
            continue
        status = str(statuses[index] if index < len(statuses) else "").strip().lower()
        current = boxes_all[index] if index < len(boxes_all) else []
        diagnostics = diagnostics_all[index] if index < len(diagnostics_all) else {}
        compound = diagnostics.get("compound", {}) if isinstance(diagnostics, dict) else {}
        parent = _finite_box(compound.get("parent_box") if isinstance(compound, dict) else None)
        support = float(parent.get("gaussian_supported_rate", 0.0)) if parent is not None else 0.0
        accepted = bool(
            "compound_geometry_probable" in status
            and "rejected" not in status
            and isinstance(current, (list, tuple))
            and len(current) >= 2
            and parent is not None
            and math.isfinite(support)
            and support >= float(min_gaussian_support)
        )
        reason = "audited_parent_box" if accepted else "unified_box_guard_failed"
        if accepted:
            original = copy.deepcopy(current)
            boxes_all[index] = [parent]
            rows.append(
                {
                    "object_id": object_id,
                    "accepted": True,
                    "reason": reason,
                    "source_component_count": len(original),
                    "source_components": original,
                    "unified_box": parent,
                    "gaussian_supported_rate": support,
                    "metric_status_changed": False,
                }
            )
        else:
            rows.append(
                {
                    "object_id": object_id,
                    "accepted": False,
                    "reason": reason,
                    "source_component_count": len(current) if isinstance(current, (list, tuple)) else 0,
                    "gaussian_supported_rate": support,
                }
            )

    state["object_compound_boxes"] = boxes_all
    accepted_rows = [row for row in rows if row["accepted"]]
    state["object_compound_presentation_unified_ids"] = sorted(
        int(row["object_id"]) for row in accepted_rows
    )
    return {
        "schema": "farm.unified-compound-presentation.v1",
        "selection_mode": "all_eligible" if object_ids is None else "explicit_ids",
        "requested_objects": len(requested_ids),
        "accepted_objects": len(accepted_rows),
        "min_gaussian_support": float(min_gaussian_support),
        "objects": rows,
    }


def _atomic_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--object-id", type=int, action="append")
    selection.add_argument("--all-eligible", action="store_true")
    parser.add_argument("--min-gaussian-support", type=float, default=0.90)
    args = parser.parse_args()

    source = args.scene_state.expanduser().resolve()
    output = args.output_state.expanduser().resolve()
    if source == output:
        raise ValueError("Refusing to overwrite the source scene state")
    started = time.perf_counter()
    original = torch.load(source, map_location="cpu", weights_only=False)
    payload = copy.deepcopy(original)
    report = unify_compound_boxes(
        payload,
        None if args.all_eligible else args.object_id,
        min_gaussian_support=float(args.min_gaussian_support),
    )
    if args.object_id and report["accepted_objects"] != len(set(args.object_id)):
        raise RuntimeError("Not every requested compound passed the unified-box guards")
    report.update(
        {
            "source_scene_state": str(source),
            "output_scene_state": str(output),
            "duration_seconds": time.perf_counter() - started,
        }
    )
    payload.setdefault("meta", {})["unified_compound_presentation"] = {
        "schema": report["schema"],
        "source": str(source),
        "object_ids": [
            int(row["object_id"]) for row in report["objects"] if row["accepted"]
        ],
        "metric_status_changed": False,
    }
    _atomic_save(payload, output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "accepted": report["accepted_objects"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
