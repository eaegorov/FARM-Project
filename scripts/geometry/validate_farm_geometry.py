#!/usr/bin/env python3
"""Category- and scene-agnostic structural QA for a FARM metric state."""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def _array(value: Any, dtype: Any | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _state(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    wrapper = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(wrapper, dict):
        raise ValueError("scene state must be a dictionary")
    state = wrapper.get("state") if isinstance(wrapper.get("state"), dict) else wrapper
    if not isinstance(state, dict):
        raise ValueError("scene state payload is missing a state dictionary")
    return wrapper, state


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalise(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("resolved up must be a finite three-vector")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        raise ValueError("resolved up must be non-zero")
    return value / norm


def _wxyz_rotation(value: Any) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(-1)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("OBB quaternion must be a finite wxyz four-vector")
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > 2.0e-2:
        raise ValueError(f"OBB quaternion must be unit length (norm={norm:.6f})")
    w, x, y, z = quaternion / max(norm, 1.0e-12)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def validate_state(
    state: Mapping[str, Any],
    *,
    resolved_up: Sequence[float],
    direct_state: Mapping[str, Any] | None = None,
    allowed_demoted_direct_ids: Sequence[int] = (),
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    means = _array(state.get("means", np.empty((0, 3))), np.float64)
    cov6 = _array(state.get("cov6", np.empty((0, 6))), np.float64)
    if means.ndim != 2 or means.shape[1:] != (3,):
        errors.append("means must have shape [N,3]")
        means = np.empty((0, 3), dtype=np.float64)
    count = int(means.shape[0])
    if cov6.shape != (count, 6):
        errors.append("cov6 must have shape [N,6]")
    if not np.isfinite(means).all():
        errors.append("means contains non-finite values")
    if cov6.size and not np.isfinite(cov6).all():
        errors.append("cov6 contains non-finite values")

    object_ids = _array(state.get("object_id", np.arange(count)), np.int64).reshape(-1)
    active = _array(state.get("active", np.zeros(count)), bool).reshape(-1)
    dimensions = _array(
        state.get("object_box_dimensions_m", np.full((count, 3), np.nan)), np.float64
    )
    box_centers = _array(
        state.get("object_box_centers_m", np.full((count, 3), np.nan)), np.float64
    )
    quaternions = _array(
        state.get(
            "object_box_wxyz",
            np.column_stack((np.ones(count), np.zeros((count, 3)))),
        ),
        np.float64,
    )
    statuses = list(state.get("object_geometry_status") or [""] * count)
    members = list(state.get("object_assembly_member_ids") or [[] for _ in range(count)])
    for label, length in (
        ("object_id", len(object_ids)),
        ("active", len(active)),
        ("object_geometry_status", len(statuses)),
        ("object_assembly_member_ids", len(members)),
    ):
        if length != count:
            errors.append(f"{label} length {length} != object count {count}")
    if dimensions.shape != (count, 3):
        errors.append("object_box_dimensions_m must have shape [N,3]")
    if box_centers.shape != (count, 3):
        errors.append("object_box_centers_m must have shape [N,3]")
    if quaternions.shape != (count, 4):
        errors.append("object_box_wxyz must have shape [N,4]")
    if len(set(int(value) for value in object_ids.tolist())) != len(object_ids):
        errors.append("object_id values are not unique")

    active_indices = np.flatnonzero(active) if len(active) == count else np.empty(0, dtype=int)
    if len(active_indices) == 0:
        errors.append("no active metric objects remain")
    if dimensions.shape == (count, 3) and len(active_indices):
        active_dimensions = dimensions[active_indices]
        if not np.isfinite(active_dimensions).all() or np.any(active_dimensions <= 0):
            errors.append("active objects contain non-finite or non-positive metric dimensions")
    if box_centers.shape == (count, 3) and len(active_indices):
        if not np.isfinite(box_centers[active_indices]).all():
            errors.append("active objects contain non-finite refined OBB centres")
    if quaternions.shape == (count, 4) and len(active_indices):
        active_quaternions = quaternions[active_indices]
        norms = np.linalg.norm(active_quaternions, axis=1)
        if not np.isfinite(active_quaternions).all() or np.any(np.abs(norms - 1.0) > 2e-2):
            errors.append("active object quaternions are non-finite or non-unit")

    allowed_active = {"geometry_pass", "assembly_geometry_pass"}
    bad_statuses = sorted({
        str(statuses[index])
        for index in active_indices
        if index < len(statuses) and str(statuses[index]) not in allowed_active
    })
    if bad_statuses:
        errors.append("active objects have unvalidated geometry statuses: " + ", ".join(bad_statuses))

    expected_up = _normalise(resolved_up)
    state_up_raw = state.get("object_geometry_up_vector")
    if state_up_raw is None:
        errors.append("state does not persist object_geometry_up_vector")
        up_alignment = None
    else:
        try:
            state_up = _normalise(state_up_raw)
            up_alignment = float(np.dot(expected_up, state_up))
            if up_alignment < 0.999:
                errors.append(
                    f"state up vector disagrees with preflight resolved_up (dot={up_alignment:.6f})"
                )
        except ValueError as exc:
            up_alignment = None
            errors.append(str(exc))

    covered_direct: set[int] = set()
    if direct_state is not None:
        direct_ids = _array(direct_state.get("object_id", []), np.int64).reshape(-1)
        direct_active = _array(direct_state.get("active", []), bool).reshape(-1)
        if len(direct_ids) != len(direct_active):
            errors.append("direct reference object_id/active lengths differ")
        else:
            expected_direct = {
                int(direct_ids[index]) for index in np.flatnonzero(direct_active)
            }
            active_ids = {
                int(object_ids[index]) for index in active_indices if index < len(object_ids)
            }
            covered_direct.update(expected_direct & active_ids)
            for index in active_indices:
                if index >= len(members):
                    continue
                try:
                    covered_direct.update(
                        int(value) for value in (members[index] or []) if int(value) in expected_direct
                    )
                except (TypeError, ValueError):
                    errors.append(f"assembly member provenance is invalid at row {index}")
            documented_demotions = {int(value) for value in allowed_demoted_direct_ids}
            assembly_ids = {
                int(object_ids[index])
                for index in range(min(count, len(members)))
                if members[index]
            }
            direct_demotions = documented_demotions & expected_direct
            assembly_demotions = documented_demotions & assembly_ids
            unknown_demotions = documented_demotions - expected_direct - assembly_ids
            if unknown_demotions:
                errors.append(
                    "surface report contains IDs absent from the direct reference: "
                    + ", ".join(map(str, sorted(unknown_demotions)))
                )
            missing = sorted(expected_direct - covered_direct - direct_demotions)
            if missing:
                errors.append(
                    f"{len(missing)} direct active objects are neither retained nor represented by an active assembly"
                )

    compound_boxes = list(state.get("object_compound_boxes") or [[] for _ in range(count)])
    if len(compound_boxes) not in {0, count}:
        errors.append("object_compound_boxes length differs from object count")
    probable = 0
    for index, status in enumerate(statuses[:count]):
        if "compound_geometry_probable" not in str(status) or "rejected" in str(status):
            continue
        probable += 1
        if bool(active[index]):
            errors.append(f"probable compound row {index} must remain inactive before final tiering")
        records = compound_boxes[index] if index < len(compound_boxes) else []
        if len(records) != 2:
            errors.append(f"probable compound row {index} does not contain exactly two component boxes")
            continue
        for child_index, record in enumerate(records):
            try:
                if not isinstance(record, Mapping):
                    raise ValueError("record is not a mapping")
                center = np.asarray(
                    record.get("center_m", record.get("center")), dtype=np.float64
                ).reshape(3)
                child_dimensions = np.asarray(
                    record.get("dimensions_lwh_m", record.get("dimensions_m", record.get("dimensions"))),
                    dtype=np.float64,
                ).reshape(3)
                if not np.isfinite(center).all():
                    raise ValueError("centre is non-finite")
                if not np.isfinite(child_dimensions).all() or np.any(child_dimensions <= 0.0):
                    raise ValueError("dimensions are non-finite or non-positive")
                rotation = _wxyz_rotation(record.get("wxyz"))
                alignment = float(np.dot(rotation[:, 2], expected_up))
                if alignment < 0.999:
                    raise ValueError(f"up alignment is {alignment:.6f}, expected >= 0.999")
            except (TypeError, ValueError) as exc:
                errors.append(f"probable compound row {index} child {child_index} is invalid: {exc}")

    if count and len(active_indices) / count < 0.01:
        warnings.append("fewer than 1% of allocated tracks remain active")
    return {
        "schema": "farm.geometry-structural-qa.v1",
        "status": "PASS" if not errors else "FAIL",
        "object_count": count,
        "active_metric_objects": int(len(active_indices)),
        "direct_objects_covered": int(len(covered_direct)),
        "documented_surface_demotions": int(len(set(map(int, allowed_demoted_direct_ids)))),
        "documented_direct_demotions": int(len(locals().get("direct_demotions", set()))),
        "documented_assembly_demotions": int(len(locals().get("assembly_demotions", set()))),
        "probable_compounds": probable,
        "resolved_up": expected_up.round(9).tolist(),
        "up_alignment": up_alignment,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--direct-state", type=Path)
    parser.add_argument("--resolved-context", type=Path, required=True)
    parser.add_argument(
        "--surface-report", type=Path,
        help="Optional enforce-mode support audit documenting legitimate direct-object demotions.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        _, state = _state(args.scene_state.resolve())
        direct = _state(args.direct_state.resolve())[1] if args.direct_state else None
        context = json.loads(args.resolved_context.read_text(encoding="utf-8"))
        demoted: list[int] = []
        if args.surface_report:
            surface = json.loads(args.surface_report.read_text(encoding="utf-8"))
            if str(surface.get("status", "")).upper() not in {"PASS", ""}:
                raise ValueError("surface support report is not PASS")
            raw_demoted = surface.get("demoted_active_objects", [])
            if not isinstance(raw_demoted, list):
                raise ValueError("surface report demoted_active_objects must be a list")
            demoted = [int(value) for value in raw_demoted]
        report = validate_state(
            state,
            resolved_up=context.get("resolved_up"),
            direct_state=direct,
            allowed_demoted_direct_ids=demoted,
        )
    except Exception as exc:
        report = {
            "schema": "farm.geometry-structural-qa.v1",
            "status": "FAIL",
            "errors": [f"{type(exc).__name__}: {exc}"],
            "warnings": [],
        }
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
