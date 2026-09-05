"""Immutable train/heldout views over one rendered full-COLMAP RGB-D union.

The expensive renderer may produce the union once.  This module publishes two
small ``frames.json`` views whose RGB/depth paths point back to that union.  It
validates the planner contract before writing anything, so a missing frame,
duplicate source image, contradictory split or physical-timestamp leak cannot
silently enter state fitting.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence


PLAN_SCHEMA = "farm.full-colmap-rescue-plan.v1"
FRAMES_SCHEMA = "farm_frames_json_v1"
FOLD_SCHEMA = "farm.full-colmap-rgbd-fold.v1"
MANIFEST_SCHEMA = "farm.full-colmap-rgbd-folds.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _exact_name_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON list")
    result: list[str] = []
    for raw in value:
        name = str(raw or "").strip()
        if not name:
            raise ValueError(f"{field} contains an empty source image")
        result.append(name)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} contains duplicate source images")
    return result


def _assert_exact_count(payload: Mapping[str, Any], field: str, actual: int) -> None:
    if field not in payload:
        raise ValueError(f"plan is missing {field}")
    try:
        claimed = int(payload[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"plan {field} must be an integer") from exc
    if claimed != int(actual):
        raise ValueError(f"plan {field}={claimed}, but exact count is {actual}")


def _relative_artifact(
    raw_path: object,
    *,
    union_root: Path,
    fold_root: Path,
    field: str,
) -> tuple[str, Path]:
    text = str(raw_path or "").strip()
    if not text:
        raise ValueError(f"frame has no {field}")
    relative = Path(text)
    if relative.is_absolute():
        raise ValueError(f"union {field} must be relative: {text}")
    resolved = (union_root / relative).resolve(strict=True)
    try:
        resolved.relative_to(union_root)
    except ValueError as exc:
        raise ValueError(f"union {field} escapes its RGB-D root: {text}") from exc
    if not resolved.is_file():
        raise ValueError(f"union {field} is not a regular file: {resolved}")
    view_path = os.path.relpath(resolved, fold_root).replace(os.sep, "/")
    if Path(view_path).is_absolute():
        raise AssertionError("relative path calculation returned an absolute path")
    if (fold_root / view_path).resolve(strict=True) != resolved:
        raise AssertionError(f"generated {field} path does not resolve to its source")
    return view_path, resolved


def _view_contract(
    plan: Mapping[str, Any],
    *,
    train_names: Sequence[str],
    heldout_names: Sequence[str],
    heldout_reference_only: bool,
) -> tuple[dict[str, str], dict[str, str], dict[int, dict[str, list[str]]]]:
    """Return exact per-image split/timestamp evidence from object view rows."""

    expected_split = {
        **{name: "train" for name in train_names},
        **{name: "heldout" for name in heldout_names},
    }
    timestamp_by_name: dict[str, str] = {}
    split_by_name: dict[str, str] = {}
    object_membership: dict[int, dict[str, list[str]]] = {}
    objects = plan.get("objects")
    if not isinstance(objects, list):
        raise ValueError("plan objects must be a JSON list")
    policy = plan.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("plan policy must be a JSON object")
    minimum_train = (
        0
        if heldout_reference_only
        else max(3, int(policy.get("minimum_train_views_per_object") or 0))
    )
    minimum_heldout = max(2, int(policy.get("minimum_heldout_views_per_object") or 0))

    for raw_object in objects:
        if not isinstance(raw_object, dict):
            raise ValueError("plan objects contains a non-object row")
        object_id = int(raw_object.get("object_id"))
        if object_id in object_membership:
            raise ValueError(f"plan repeats object_id {object_id}")
        rows_by_split: dict[str, list[dict[str, Any]]] = {}
        for split, field in (("train", "train_views"), ("heldout", "heldout_views")):
            rows = raw_object.get(field)
            if not isinstance(rows, list):
                raise ValueError(f"object {object_id} {field} must be a list")
            parsed: list[dict[str, Any]] = []
            for raw_view in rows:
                if not isinstance(raw_view, dict):
                    raise ValueError(f"object {object_id} {field} has a non-object row")
                name = str(raw_view.get("name") or "").strip()
                timestamp = str(raw_view.get("physical_timestamp") or "").strip()
                declared_split = str(raw_view.get("split") or split).strip()
                if not name or not timestamp:
                    raise ValueError(
                        f"object {object_id} {field} lacks name/physical_timestamp"
                    )
                if name not in expected_split:
                    raise ValueError(f"object {object_id} references unselected image {name}")
                if expected_split[name] != split or declared_split != split:
                    raise ValueError(f"object {object_id} contradicts split for {name}")
                if name in split_by_name and (
                    split_by_name[name] != split or timestamp_by_name[name] != timestamp
                ):
                    raise ValueError(f"contradictory split/timestamp evidence for {name}")
                split_by_name[name] = split
                timestamp_by_name[name] = timestamp
                parsed.append(dict(raw_view))
            if len({str(row["name"]) for row in parsed}) != len(parsed):
                raise ValueError(f"object {object_id} repeats a {split} source image")
            if len({str(row["physical_timestamp"]) for row in parsed}) != len(parsed):
                raise ValueError(
                    f"object {object_id} repeats a {split} physical timestamp"
                )
            rows_by_split[split] = parsed

        selected_rows = raw_object.get("selected_views")
        if not isinstance(selected_rows, list):
            raise ValueError(f"object {object_id} selected_views must be a list")
        selected_names = [str(row.get("name") or "") for row in selected_rows]
        object_train_names = [str(row["name"]) for row in rows_by_split["train"]]
        if selected_names != object_train_names:
            raise ValueError(
                f"object {object_id} selected_views must be exactly its train_views"
            )
        if int(raw_object.get("selected_count") or 0) != len(object_train_names):
            raise ValueError(f"object {object_id} selected_count is inconsistent")
        object_heldout_names = [
            str(row["name"]) for row in rows_by_split["heldout"]
        ]
        if int(raw_object.get("heldout_count") or 0) != len(object_heldout_names):
            raise ValueError(f"object {object_id} heldout_count is inconsistent")

        planning_status = str(raw_object.get("planning_status") or "")
        if planning_status == "planned":
            if heldout_reference_only:
                raise ValueError(
                    f"heldout-reference object {object_id} uses train planning status"
                )
            if len(object_train_names) < minimum_train:
                raise ValueError(
                    f"planned object {object_id} has fewer than {minimum_train} train views"
                )
            if len(object_heldout_names) < minimum_heldout:
                raise ValueError(
                    f"planned object {object_id} has fewer than {minimum_heldout} heldout views"
                )
        elif planning_status == "planned_heldout_reference_only":
            if not heldout_reference_only:
                raise ValueError(
                    f"normal plan object {object_id} uses heldout-reference status"
                )
            if object_train_names:
                raise ValueError(
                    f"heldout-reference object {object_id} must not contain train views"
                )
            if len(object_heldout_names) < minimum_heldout:
                raise ValueError(
                    f"heldout-reference object {object_id} has fewer than "
                    f"{minimum_heldout} heldout views"
                )
        elif object_train_names or object_heldout_names:
            raise ValueError(
                f"non-planned object {object_id} must not retain train/heldout views"
            )
        object_membership[object_id] = {
            "train": object_train_names,
            "heldout": object_heldout_names,
        }

    missing_evidence = sorted(set(expected_split).difference(split_by_name))
    extra_evidence = sorted(set(split_by_name).difference(expected_split))
    if missing_evidence or extra_evidence:
        raise ValueError(
            "plan object views do not exactly cover selected source images: "
            f"missing={missing_evidence[:8]}, extra={extra_evidence[:8]}"
        )

    train_timestamps = {
        timestamp_by_name[name] for name in train_names
    }
    heldout_timestamps = {
        timestamp_by_name[name] for name in heldout_names
    }
    overlap = sorted(train_timestamps.intersection(heldout_timestamps))
    if overlap:
        raise ValueError(
            "physical timestamp leakage between train and heldout: "
            + ", ".join(overlap[:8])
        )
    return split_by_name, timestamp_by_name, object_membership


def _validate_plan_provenance(
    plan: Mapping[str, Any], split_by_name: Mapping[str, str]
) -> list[dict[str, Any]]:
    provenance = plan.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("plan provenance must be a JSON object")
    if provenance.get("selected_image_hashes_enabled") is not True:
        raise ValueError("plan source-image SHA-256 provenance is required")
    rows = provenance.get("selected_source_images")
    if not isinstance(rows, list):
        raise ValueError("plan selected_source_images provenance must be a list")
    by_name: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("selected_source_images contains a non-object row")
        name = str(raw.get("name") or "").strip()
        split = str(raw.get("split") or "").strip()
        digest = str(raw.get("sha256") or "").strip().lower()
        if not name or name in by_name:
            raise ValueError("selected_source_images contains an empty/duplicate name")
        if split_by_name.get(name) != split:
            raise ValueError(f"selected source provenance contradicts split for {name}")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"selected source provenance lacks SHA-256 for {name}")
        if int(raw.get("bytes") or 0) <= 0:
            raise ValueError(f"selected source provenance lacks byte size for {name}")
        by_name[name] = dict(raw)
    if set(by_name) != set(split_by_name):
        raise ValueError("selected source provenance does not exactly cover the plan")
    return [by_name[name] for name in sorted(by_name)]


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_full_colmap_fold_views(
    plan_path: Path,
    union_frames_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate and atomically publish train/heldout ``frames.json`` views."""

    plan_path = plan_path.expanduser().resolve(strict=True)
    union_frames_path = union_frames_path.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite fold output: {output_dir}")
    plan = _load_object(plan_path)
    frames_payload = _load_object(union_frames_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"unsupported full-COLMAP plan schema: {plan.get('schema')!r}")
    if frames_payload.get("schema_version") != FRAMES_SCHEMA:
        raise ValueError(f"unsupported frames schema: {frames_payload.get('schema_version')!r}")
    selected_names_role = str(plan.get("selected_names_role") or "")
    heldout_reference_only = selected_names_role == "heldout_reference_only_non_merge"
    if selected_names_role not in {
        "rgbd_render_union_of_train_and_heldout",
        "heldout_reference_only_non_merge",
    }:
        raise ValueError("plan selected_names has an unsupported consumption role")
    policy = plan.get("policy")
    consumption = plan.get("consumption_contract")
    if heldout_reference_only:
        if (
            not isinstance(policy, dict)
            or policy.get("heldout_reference_only") is not True
            or policy.get("state_fit_authorized") is not False
            or policy.get("geometry_merge_authorized") is not False
            or policy.get("semantic_mutation_authorized") is not False
            or not isinstance(consumption, dict)
            or consumption.get("train") != "train_unconsumed"
            or consumption.get("heldout") != "heldout_reference_only"
            or consumption.get("state_fit_authorized") is not False
            or consumption.get("geometry_merge_authorized") is not False
            or consumption.get("semantic_mutation_authorized") is not False
        ):
            raise ValueError(
                "heldout-reference plan lacks the fail-closed non-merge contract"
            )

    selected_names = _exact_name_list(plan.get("selected_names"), field="selected_names")
    train_names = _exact_name_list(plan.get("train_names"), field="train_names")
    heldout_names = _exact_name_list(plan.get("heldout_names"), field="heldout_names")
    if not heldout_names or (not heldout_reference_only and not train_names):
        raise ValueError("both train and heldout folds must be non-empty")
    if heldout_reference_only and train_names:
        raise ValueError("heldout-reference-only plan must have no train names")
    name_overlap = sorted(set(train_names).intersection(heldout_names))
    if name_overlap:
        raise ValueError(f"source image leakage between folds: {name_overlap[:8]}")
    if set(selected_names) != set(train_names).union(heldout_names):
        raise ValueError("train_names + heldout_names do not exactly equal selected_names")
    _assert_exact_count(plan, "unique_rescue_views", len(selected_names))
    _assert_exact_count(plan, "unique_train_views", len(train_names))
    _assert_exact_count(plan, "unique_heldout_views", len(heldout_names))

    split_by_name, timestamp_by_name, object_membership = _view_contract(
        plan,
        train_names=train_names,
        heldout_names=heldout_names,
        heldout_reference_only=heldout_reference_only,
    )
    selected_source_provenance = _validate_plan_provenance(plan, split_by_name)

    frames = frames_payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("union frames JSON has no frames list")
    frame_by_source: dict[str, dict[str, Any]] = {}
    for raw in frames:
        if not isinstance(raw, dict):
            raise ValueError("union frames contains a non-object row")
        source_image = str(raw.get("source_image") or "").strip()
        if not source_image or source_image in frame_by_source:
            raise ValueError("union frames contains an empty/duplicate source_image")
        frame_by_source[source_image] = dict(raw)
    missing = sorted(set(selected_names).difference(frame_by_source))
    extra = sorted(set(frame_by_source).difference(selected_names))
    if missing or extra or len(frames) != len(selected_names):
        raise ValueError(
            "rendered union does not exactly cover selected source images: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    selection = frames_payload.get("selection_contract")
    if not isinstance(selection, dict) or int(selection.get("count") or -1) != len(frames):
        raise ValueError("union selection_contract count is absent or inconsistent")
    if selection.get("exact_order_preserved") is not True:
        raise ValueError("union renderer did not attest exact selected-name order")
    if [str(row.get("source_image") or "") for row in frames] != selected_names:
        raise ValueError("union frame order does not exactly preserve plan selected_names")
    plan_scale = float(plan.get("meters_per_scene_unit") or 0.0)
    frame_scale = float(frames_payload.get("meters_per_scene_unit") or 0.0)
    if plan_scale <= 0.0 or frame_scale <= 0.0 or abs(plan_scale - frame_scale) > 1.0e-12:
        raise ValueError("plan/frames meters_per_scene_unit contract mismatch")

    temporary = output_dir.with_name(f".{output_dir.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        (temporary / "train").mkdir(parents=True)
        (temporary / "heldout").mkdir()
        union_root = union_frames_path.parent.resolve(strict=True)
        artifact_rows: dict[str, list[dict[str, Any]]] = {"train": [], "heldout": []}
        fold_documents: dict[str, dict[str, Any]] = {}
        seen_rgb: set[Path] = set()
        seen_depth: set[Path] = set()
        for split in ("train", "heldout"):
            fold_root = (temporary / split).resolve()
            fold_frames: list[dict[str, Any]] = []
            expected_names = train_names if split == "train" else heldout_names
            expected_set = set(expected_names)
            for source_row in frames:
                source_image = str(source_row["source_image"])
                if source_image not in expected_set:
                    continue
                item = copy.deepcopy(source_row)
                rgb_relative, rgb_path = _relative_artifact(
                    item.get("rgb_path"), union_root=union_root,
                    fold_root=fold_root, field="rgb_path",
                )
                depth_relative, depth_path = _relative_artifact(
                    item.get("depth_path"), union_root=union_root,
                    fold_root=fold_root, field="depth_path",
                )
                if rgb_path in seen_rgb or depth_path in seen_depth:
                    raise ValueError("rendered union reuses an RGB/depth artifact across frames")
                seen_rgb.add(rgb_path)
                seen_depth.add(depth_path)
                item["rgb_path"] = rgb_relative
                item["depth_path"] = depth_relative
                item["physical_timestamp"] = timestamp_by_name[source_image]
                item["full_colmap_split"] = split
                fold_frames.append(item)
                artifact_rows[split].append(
                    {
                        "source_image": source_image,
                        "physical_timestamp": timestamp_by_name[source_image],
                        "rgb": {
                            "path": rgb_relative,
                            "bytes": rgb_path.stat().st_size,
                            "sha256": sha256_file(rgb_path),
                        },
                        "depth": {
                            "path": depth_relative,
                            "bytes": depth_path.stat().st_size,
                            "sha256": sha256_file(depth_path),
                        },
                    }
                )
            if [row["source_image"] for row in fold_frames] != [
                name for name in selected_names if name in expected_set
            ]:
                raise AssertionError(f"{split} fold order is not deterministic")
            document = copy.deepcopy(frames_payload)
            document["cameras"] = [
                camera for camera in list(frames_payload.get("cameras") or [])
                if any(str(row.get("camera") or "") == str(camera) for row in fold_frames)
            ]
            document["frames"] = fold_frames
            document["selection_contract"] = {
                "source": "full_colmap_rescue_plan",
                "split": split,
                "count": len(fold_frames),
                "exact_source_image_coverage": True,
                "exact_union_order_preserved": True,
                "physical_timestamp_globally_disjoint": True,
            }
            document["full_colmap_fold"] = {
                "schema": FOLD_SCHEMA,
                "role": (
                    ("train_unconsumed" if split == "train" else "heldout_reference_only")
                    if heldout_reference_only
                    else split
                ),
                "state_fit_authorized": (
                    split == "train" and not heldout_reference_only
                ),
                "usage": (
                    (
                        "train_unconsumed"
                        if split == "train"
                        else "heldout_reference_only"
                    )
                    if heldout_reference_only
                    else (
                        "mapping_covisibility_sam3_merge_obb_fit"
                        if split == "train"
                        else "frozen_evaluation_only"
                    )
                ),
                "evaluation_status": (
                    "not_applicable"
                    if split == "train"
                    else "reserved_not_consumed"
                ),
                "geometry_merge_authorized": (
                    split == "train" and not heldout_reference_only
                ),
                "semantic_mutation_authorized": (
                    split == "train" and not heldout_reference_only
                ),
                "source_plan_sha256": sha256_file(plan_path),
                "source_union_frames_sha256": sha256_file(union_frames_path),
                "source_image_count": len(fold_frames),
                "physical_timestamp_count": len(
                    {timestamp_by_name[str(row["source_image"])] for row in fold_frames}
                ),
            }
            fold_documents[split] = document
            _atomic_json(temporary / split / "frames.json", document)

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": "PASS",
            "provenance": {
                "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
                "rendered_union_frames": {
                    "path": str(union_frames_path),
                    "sha256": sha256_file(union_frames_path),
                },
                "selected_source_images": selected_source_provenance,
                "rendered_artifacts": artifact_rows,
            },
            "integrity": {
                "exact_selected_source_image_coverage": True,
                "source_image_sets_disjoint": True,
                "physical_timestamp_sets_disjoint": True,
                "rgb_depth_copied": False,
                "relative_paths_resolve_to_union": True,
                "all_rendered_artifacts_sha256_hashed": True,
            },
            "fit_policy": {
                "fit_splits": [] if heldout_reference_only else ["train"],
                "train_usage": (
                    "train_unconsumed"
                    if heldout_reference_only
                    else "mapping_covisibility_sam3_merge_obb_fit"
                ),
                "heldout_usage": (
                    "heldout_reference_only"
                    if heldout_reference_only
                    else "frozen_evaluation_only"
                ),
                "heldout_consumed": False,
                "state_fit_authorized": not heldout_reference_only,
                "geometry_merge_authorized": not heldout_reference_only,
                "semantic_mutation_authorized": not heldout_reference_only,
                "minimum_accepted_views": 3,
                "minimum_independent_physical_timestamps": 2,
            },
            "counts": {
                "union_frames": len(frames),
                "train_frames": len(fold_documents["train"]["frames"]),
                "heldout_frames": len(fold_documents["heldout"]["frames"]),
                "train_physical_timestamps": len(
                    {timestamp_by_name[name] for name in train_names}
                ),
                "heldout_physical_timestamps": len(
                    {timestamp_by_name[name] for name in heldout_names}
                ),
                "objects": len(object_membership),
            },
            "folds": {
                split: {
                    "frames_json": f"{split}/frames.json",
                    "sha256": sha256_file(temporary / split / "frames.json"),
                    "source_images": [
                        str(row["source_image"])
                        for row in fold_documents[split]["frames"]
                    ],
                }
                for split in ("train", "heldout")
            },
            "object_membership": {
                str(object_id): rows
                for object_id, rows in sorted(object_membership.items())
            },
        }
        _atomic_json(temporary / "manifest.json", manifest)
        _atomic_json(
            temporary / "_SUCCESS.json",
            {
                "schema": "farm.full-colmap-rgbd-folds.success.v1",
                "status": "success",
                "manifest": "manifest.json",
                "manifest_sha256": sha256_file(temporary / "manifest.json"),
            },
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return _load_object(output_dir / "manifest.json")
