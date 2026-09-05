"""Fail-closed role contracts for full-COLMAP mask refinement.

Train views may update scene state. Heldout views may only produce frozen
reference masks for evaluation. Keeping this check in one dependency-free
module lets refinement, merge and apply enforce the same boundary.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

PLAN_SCHEMA = "farm.full-colmap-rescue-plan.v1"
FOLD_SCHEMA = "farm.full-colmap-rgbd-fold.v1"
REFINEMENT_SCHEMA = "farm.full-colmap-mask-refinement.v1"
TRAIN_VIEW_ROLE = "train"
HELDOUT_VIEW_ROLE = "heldout-reference"
TRAIN_REFINEMENT_ROLE = "train_fit"
HELDOUT_REFINEMENT_ROLE = "heldout_reference_only"


def _rows(payload: Mapping[str, Any], field: str) -> list[dict[str, Any]]:
    value = payload.get(field)
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON list")
    if any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{field} contains a non-object row")
    return [dict(row) for row in value]


def validate_frames_view_role(
    frames: Mapping[str, Any], view_role: str, *, legacy_train_allowed: bool = True
) -> dict[str, Any]:
    """Validate that every input frame belongs to the requested immutable fold."""

    if view_role not in {TRAIN_VIEW_ROLE, HELDOUT_VIEW_ROLE}:
        raise ValueError(f"unsupported full-COLMAP view role: {view_role!r}")
    expected_split = "heldout" if view_role == HELDOUT_VIEW_ROLE else "train"
    expected_fit = view_role == TRAIN_VIEW_ROLE
    frame_rows = _rows(frames, "frames")
    if not frame_rows:
        raise ValueError("full-COLMAP frames manifest is empty")
    source_names: list[str] = []
    for row in frame_rows:
        source = str(row.get("source_image") or "").strip()
        if not source:
            raise ValueError("full-COLMAP frame lacks source_image")
        source_names.append(source)
        declared = str(row.get("full_colmap_split") or "").strip()
        if declared and declared != expected_split:
            raise ValueError(
                f"{view_role} input contains a {declared!r} frame: {source}"
            )
    if len(source_names) != len(set(source_names)):
        raise ValueError("full-COLMAP frames repeat a source image")

    fold = frames.get("full_colmap_fold")
    if not isinstance(fold, Mapping):
        if view_role == HELDOUT_VIEW_ROLE or not legacy_train_allowed:
            raise ValueError(f"{view_role} input requires full_colmap_fold metadata")
        return {
            "fold_contract": "legacy_train_without_fold_metadata",
            "split": "train",
            "state_fit_authorized": True,
            "source_image_count": len(source_names),
        }
    if str(fold.get("schema") or "") != FOLD_SCHEMA:
        raise ValueError("unsupported full-COLMAP fold schema")
    fold_role = str(fold.get("role") or "")
    allowed_roles = {expected_split}
    if view_role == HELDOUT_VIEW_ROLE:
        allowed_roles.add("heldout_reference_only")
    if fold_role not in allowed_roles:
        raise ValueError(
            f"requested {view_role} but frames fold role is {fold.get('role')!r}"
        )
    if fold.get("state_fit_authorized") is not expected_fit:
        raise ValueError(
            f"{view_role} requires state_fit_authorized={str(expected_fit).lower()}"
        )
    if fold_role == "heldout_reference_only" and (
        fold.get("geometry_merge_authorized") is not False
        or fold.get("semantic_mutation_authorized") is not False
    ):
        raise ValueError(
            "heldout_reference_only requires geometry/semantic mutation authorization=false"
        )
    selection = frames.get("selection_contract")
    if isinstance(selection, Mapping):
        declared_split = str(selection.get("split") or "").strip()
        if declared_split and declared_split != expected_split:
            raise ValueError("selection_contract split contradicts full_colmap_fold")
    return {
        "fold_contract": FOLD_SCHEMA,
        "fold_role": fold_role,
        "split": expected_split,
        "state_fit_authorized": expected_fit,
        "source_image_count": len(source_names),
        "physical_timestamp_count": int(fold.get("physical_timestamp_count") or 0),
        "source_plan_sha256": str(fold.get("source_plan_sha256") or ""),
        "source_union_frames_sha256": str(fold.get("source_union_frames_sha256") or ""),
    }


def prepare_plan_for_view_role(
    plan: Mapping[str, Any], frames: Mapping[str, Any], view_role: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a role-scoped plan whose ``selected_views`` cannot cross folds."""

    if str(plan.get("schema") or "") != PLAN_SCHEMA:
        raise ValueError("unsupported full-COLMAP rescue plan")
    frames_contract = validate_frames_view_role(frames, view_role)
    active_split = "heldout" if view_role == HELDOUT_VIEW_ROLE else "train"
    view_field = "heldout_views" if view_role == HELDOUT_VIEW_ROLE else "train_views"
    frame_rows = _rows(frames, "frames")
    frame_by_source = {str(row["source_image"]): row for row in frame_rows}
    expected_names_field = (
        "heldout_names" if active_split == "heldout" else "train_names"
    )
    expected_names = plan.get(expected_names_field)
    if isinstance(expected_names, list):
        normalized_expected = [str(value or "").strip() for value in expected_names]
        if len(normalized_expected) != len(set(normalized_expected)) or any(
            not value for value in normalized_expected
        ):
            raise ValueError(f"plan {expected_names_field} is invalid")
        if set(normalized_expected) != set(frame_by_source):
            raise ValueError(
                f"{view_role} frames do not exactly match plan {expected_names_field}"
            )
    elif view_role == HELDOUT_VIEW_ROLE:
        raise ValueError("heldout-reference requires plan heldout_names")

    scoped = copy.deepcopy(dict(plan))
    objects = scoped.get("objects")
    if not isinstance(objects, list):
        raise ValueError("plan objects must be a JSON list")
    active_names: set[str] = set()
    active_timestamps: set[str] = set()
    opposite_names: set[str] = set()
    opposite_timestamps: set[str] = set()
    plan_view_fields_used: set[str] = set()
    for object_row in objects:
        if not isinstance(object_row, dict):
            raise ValueError("plan objects contains a non-object row")
        object_id = int(object_row.get("object_id"))
        raw_views = object_row.get(view_field)
        if raw_views is None and view_role == TRAIN_VIEW_ROLE:
            raw_views = object_row.get("selected_views")
            view_field_used = "selected_views"
        else:
            view_field_used = view_field
        plan_view_fields_used.add(view_field_used)
        if not isinstance(raw_views, list):
            raise ValueError(f"object {object_id} requires {view_field}")
        views: list[dict[str, Any]] = []
        object_names: set[str] = set()
        object_timestamps: set[str] = set()
        for raw_view in raw_views:
            if not isinstance(raw_view, dict):
                raise ValueError(
                    f"object {object_id} {view_field} has a non-object row"
                )
            view = dict(raw_view)
            name = str(view.get("name") or "").strip()
            timestamp = str(view.get("physical_timestamp") or "").strip()
            declared_split = str(view.get("split") or active_split).strip()
            if not name or not timestamp:
                raise ValueError(
                    f"object {object_id} {view_field} lacks name/physical_timestamp"
                )
            if declared_split != active_split:
                raise ValueError(
                    f"object {object_id} {view_field} contains {declared_split!r} view"
                )
            if name in object_names or timestamp in object_timestamps:
                raise ValueError(
                    f"object {object_id} {view_field} repeats source/physical timestamp"
                )
            frame = frame_by_source.get(name)
            if frame is None:
                raise ValueError(
                    f"object {object_id} {view_field} references frame outside {active_split} fold: {name}"
                )
            frame_split = str(frame.get("full_colmap_split") or active_split).strip()
            if frame_split != active_split:
                raise ValueError(f"frame split contradicts plan view for {name}")
            frame_timestamp = str(frame.get("physical_timestamp") or "").strip()
            if frame_timestamp and frame_timestamp != timestamp:
                raise ValueError(f"physical timestamp mismatch for {name}")
            object_names.add(name)
            object_timestamps.add(timestamp)
            active_names.add(name)
            active_timestamps.add(timestamp)
            view["split"] = active_split
            views.append(view)

        opposite_field = "train_views" if active_split == "heldout" else "heldout_views"
        for opposite in object_row.get(opposite_field) or []:
            if not isinstance(opposite, Mapping):
                raise ValueError(
                    f"object {object_id} {opposite_field} has a non-object row"
                )
            opposite_name = str(opposite.get("name") or "").strip()
            opposite_timestamp = str(opposite.get("physical_timestamp") or "").strip()
            if opposite_name:
                opposite_names.add(opposite_name)
            if opposite_timestamp:
                opposite_timestamps.add(opposite_timestamp)
        object_row["selected_views"] = views
        object_row["selected_count"] = len(views)
        object_row["active_view_field"] = view_field_used

    name_leakage = sorted(active_names.intersection(opposite_names))
    timestamp_leakage = sorted(active_timestamps.intersection(opposite_timestamps))
    if name_leakage or timestamp_leakage:
        raise ValueError(
            "train/heldout leakage in full-COLMAP plan: "
            f"source_images={name_leakage[:8]}, physical_timestamps={timestamp_leakage[:8]}"
        )
    return scoped, {
        **frames_contract,
        "requested_view_role": view_role,
        "active_split": active_split,
        "plan_view_field": (
            next(iter(plan_view_fields_used))
            if len(plan_view_fields_used) == 1
            else "mixed_train_compatibility_fields"
        ),
        "active_source_image_count": len(active_names),
        "active_physical_timestamp_count": len(active_timestamps),
    }


def _resolve_report_frames_path(
    refinement: Mapping[str, Any], refinement_path: Path
) -> Path:
    raw = str(refinement.get("rescue_frames_json") or "").strip()
    if not raw:
        raise ValueError("legacy refinement lacks rescue_frames_json role evidence")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = refinement_path.parent / candidate
    return candidate.resolve(strict=True)


def require_train_fit_refinement(
    refinement: Mapping[str, Any],
    *,
    refinement_path: Path,
    expected_frames_path: Path | None = None,
) -> None:
    """Reject every heldout/evaluation refinement before any state mutation."""

    if (
        str(refinement.get("schema") or "") != REFINEMENT_SCHEMA
        or str(refinement.get("status") or "") != "PASS"
    ):
        raise ValueError("invalid full-COLMAP refinement report")
    role = str(refinement.get("refinement_role") or "").strip()
    if role == HELDOUT_REFINEMENT_ROLE:
        raise ValueError(
            "heldout-reference refinement is evaluation-only and cannot update state"
        )
    if role and role != TRAIN_REFINEMENT_ROLE:
        raise ValueError(f"unsupported full-COLMAP refinement role: {role!r}")
    policy = refinement.get("policy")
    if role == TRAIN_REFINEMENT_ROLE:
        if (
            not isinstance(policy, Mapping)
            or policy.get("state_fit_authorized") is not True
        ):
            raise ValueError("train refinement lacks state_fit_authorized=true")
        view_contract = refinement.get("input_view_contract")
        if not isinstance(view_contract, Mapping):
            raise ValueError("train refinement lacks input_view_contract")
        if (
            str(view_contract.get("active_split") or "") != "train"
            or view_contract.get("state_fit_authorized") is not True
        ):
            raise ValueError(
                "train refinement input_view_contract is not fit-authorized"
            )
    else:
        frames_path = _resolve_report_frames_path(refinement, refinement_path)
        frames = json.loads(frames_path.read_text(encoding="utf-8"))
        if not isinstance(frames, dict):
            raise ValueError(
                "legacy refinement rescue_frames_json must be a JSON object"
            )
        validate_frames_view_role(frames, TRAIN_VIEW_ROLE, legacy_train_allowed=False)

    if expected_frames_path is not None:
        expected = expected_frames_path.expanduser().resolve(strict=True)
        frames = json.loads(expected.read_text(encoding="utf-8"))
        if not isinstance(frames, dict):
            raise ValueError("rescue frames manifest must be a JSON object")
        validate_frames_view_role(frames, TRAIN_VIEW_ROLE, legacy_train_allowed=False)
        raw = str(refinement.get("rescue_frames_json") or "").strip()
        if raw:
            reported = Path(raw).expanduser()
            if not reported.is_absolute():
                reported = refinement_path.parent / reported
            if reported.resolve(strict=True) != expected:
                raise ValueError(
                    "refinement rescue_frames_json differs from merge rescue frames"
                )
