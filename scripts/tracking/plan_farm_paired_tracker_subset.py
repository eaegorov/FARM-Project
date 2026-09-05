#!/usr/bin/env python3
"""Plan object-anchor-paired tracker episodes from full-COLMAP support tracks.

This is the v2 successor to the scene-balanced v1 tracker subset.  It rebuilds
the exact object-support-to-COLMAP-track eligibility used by an audited rescue
plan, but evaluates all registered images so contiguous video windows are
possible.  OBBs, detector boxes, labels and RGB pixels are never consulted.

The output is planning metadata only: no image is copied, linked or decoded.
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
import math
import os
import platform
import re
import resource
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from farm_runtime.full_colmap_view_planner import (  # noqa: E402
    build_sparse_track_index,
    model_file_manifest,
)
from farm_runtime.paired_tracker_subset_planner import (  # noqa: E402
    PLANNER_REVISION,
    SCHEMA,
    SupportEvidence,
    enumerate_object_visible_windows,
    pair_object_windows,
    select_paired_object_episodes,
    selected_pairs_to_episode_rows,
    validate_paired_episode_rows,
)
from farm_runtime.tracker_subset_planner import (  # noqa: E402
    TrackerFrame,
    compile_identity_pattern,
    metric_world_from_camera,
    parse_image_identity,
)
from scripts.geometry.plan_farm_full_colmap_rescue import (  # noqa: E402
    build_support_track_candidates,
    image_world_to_camera,
    load_state,
    object_support_points,
)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(int(chunk_bytes)):
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path: Path) -> dict[str, object]:
    source = path.expanduser().resolve(strict=True)
    return {
        "path": str(source),
        "bytes": int(source.stat().st_size),
        "sha256": sha256_file(source),
    }


def _current_rss_mib() -> float:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _load_json(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    source = path.expanduser().resolve(strict=True)
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload, raw


def _atomic_json(path: Path, payload: object, *, force: bool) -> None:
    destination = path.expanduser().resolve()
    if destination.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing v2 plan without --force: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _expected_hash(plan: Mapping[str, object], key: str) -> str:
    provenance = plan.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("anchor plan lacks provenance")
    row = provenance.get(key)
    if not isinstance(row, Mapping):
        raise ValueError(f"anchor plan lacks provenance.{key}")
    digest = str(row.get("sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"anchor plan has invalid provenance hash for {key}")
    return digest


def _verify_file_hash(path: Path, expected: str, label: str) -> dict[str, object]:
    row = file_provenance(path)
    if row["sha256"] != expected:
        raise ValueError(
            f"{label} checksum differs from audited full-COLMAP plan: "
            f"{row['sha256']} != {expected}"
        )
    return row


def _verify_colmap_contract(
    model_root: Path, plan: Mapping[str, object]
) -> list[dict[str, object]]:
    provenance = plan.get("provenance")
    expected_rows = (
        provenance.get("colmap_model_files")
        if isinstance(provenance, Mapping)
        else None
    )
    if not isinstance(expected_rows, list) or not expected_rows:
        raise ValueError("anchor plan lacks COLMAP file provenance")
    expected = {
        str(row.get("name")): (int(row.get("bytes", -1)), str(row.get("sha256") or ""))
        for row in expected_rows
        if isinstance(row, Mapping)
    }
    actual = model_file_manifest(model_root)
    actual_by_name = {
        str(row["name"]): (int(row["bytes"]), str(row["sha256"]))
        for row in actual
    }
    if actual_by_name != expected:
        raise ValueError("COLMAP model files differ from audited rescue plan")
    return actual


def _required_policy(plan: Mapping[str, object]) -> dict[str, object]:
    policy = plan.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("anchor plan lacks policy")
    if policy.get("retrieval_mode") != "support_tracks":
        raise ValueError("v2 planner requires an audited support_tracks-only plan")
    if bool(policy.get("default_retrieval_uses_current_obb", True)):
        raise ValueError("anchor plan permits OBB retrieval")
    required = (
        "minimum_object_support_points",
        "maximum_object_support_points",
        "support_trim_quantile",
        "maximum_track_distance_m",
        "minimum_object_track_match_fraction",
        "minimum_track_support_points",
        "minimum_track_support_fraction",
        "minimum_projected_support_points",
        "minimum_in_frame_support_ratio",
        "support_bbox_quantile",
        "min_margin_ratio",
        "min_area_ratio",
        "max_area_ratio",
    )
    missing = [key for key in required if key not in policy]
    if missing:
        raise ValueError("anchor plan lacks required support policy: " + ", ".join(missing))
    return dict(policy)


def _identity_contract(frames: Mapping[str, object]) -> tuple[re.Pattern[str], str, str, str]:
    contract = frames.get("identity_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("frames JSON lacks identity_contract")
    expression = str(contract.get("regex") or "")
    camera_group = str(contract.get("sensor_group") or "camera")
    timestamp_group = str(contract.get("timestamp_group") or "timestamp")
    family_group = str(contract.get("family_group") or "family")
    pattern = compile_identity_pattern(
        expression,
        camera_group=camera_group,
        timestamp_group=timestamp_group,
        view_family_group=family_group,
    )
    return pattern, camera_group, timestamp_group, family_group


def _rescue_anchors_by_object(plan: Mapping[str, object]) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    objects = plan.get("objects")
    if not isinstance(objects, list):
        raise ValueError("anchor plan lacks objects")
    for row in objects:
        if not isinstance(row, Mapping):
            raise ValueError("anchor plan object row must be an object")
        object_id = int(row.get("object_id", -1))
        if object_id < 0 or object_id in result:
            raise ValueError("anchor plan object IDs must be unique and non-negative")
        if row.get("retrieval_mode") != "support_tracks":
            raise ValueError(f"object {object_id} was not routed through support tracks")
        diagnostics = row.get("retrieval_diagnostics")
        if isinstance(diagnostics, Mapping) and bool(diagnostics.get("obb_fallback_used")):
            raise ValueError(f"object {object_id} used forbidden OBB fallback")
        names: set[str] = set()
        for key in ("selected_views", "train_views", "heldout_views"):
            values = row.get(key)
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, Mapping) and value.get("name"):
                    names.add(str(value["name"]))
        result[object_id] = names
    return result


def _safe_image_path(root: Path, name: str) -> Path:
    source_root = root.expanduser().resolve(strict=True)
    candidate = (source_root / name).resolve(strict=True)
    try:
        candidate.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"COLMAP image name escapes image root: {name!r}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--anchor-plan", required=True, type=Path)
    parser.add_argument("--scene-state", required=True, type=Path)
    parser.add_argument("--frames-json", required=True, type=Path)
    parser.add_argument("--colmap-model", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-frames", type=int, default=160)
    parser.add_argument("--minimum-total-frames", type=int, default=120)
    parser.add_argument("--maximum-total-frames", type=int, default=200)
    parser.add_argument("--minimum-episode-frames", type=int, default=8)
    parser.add_argument("--maximum-episode-frames", type=int, default=12)
    parser.add_argument("--preferred-episode-frames", type=int, default=10)
    parser.add_argument("--maximum-gap-multiplier", type=float, default=2.5)
    parser.add_argument("--maximum-translation-m", type=float, default=0.50)
    parser.add_argument("--maximum-rotation-degrees", type=float, default=21.0)
    parser.add_argument("--maximum-windows-per-object", type=int, default=96)
    parser.add_argument("--maximum-pairs-per-object", type=int, default=256)
    parser.add_argument(
        "--qa-object-id",
        action="append",
        default=[],
        type=int,
        help=(
            "Report-only object ID. It never changes eligibility, scores or selection."
        ),
    )
    parser.add_argument(
        "--hash-selected-images",
        action="store_true",
        help="Content-hash selected RGB without decoding or materializing it.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    if not (
        2
        <= int(args.minimum_episode_frames)
        <= int(args.preferred_episode_frames)
        <= int(args.maximum_episode_frames)
    ):
        raise ValueError("episode frame bounds are inconsistent")
    if not (
        1
        <= int(args.minimum_total_frames)
        <= int(args.target_frames)
        <= int(args.maximum_total_frames)
    ):
        raise ValueError("total frame bounds are inconsistent")
    if int(args.minimum_total_frames) < 120 or int(args.maximum_total_frames) > 200:
        raise ValueError("v2 benchmark must stay within the approved 120-200 frame bound")
    qa_ids = sorted(set(int(value) for value in args.qa_object_id))
    if any(value < 0 for value in qa_ids):
        raise ValueError("QA object IDs must be non-negative")

    anchor_path = args.anchor_plan.expanduser().resolve(strict=True)
    state_path = args.scene_state.expanduser().resolve(strict=True)
    frames_path = args.frames_json.expanduser().resolve(strict=True)
    model_root = args.colmap_model.expanduser().resolve(strict=True)
    image_root = args.image_root.expanduser().resolve(strict=True)
    anchor_plan, anchor_raw = _load_json(anchor_path, "anchor plan")
    if anchor_plan.get("schema") != "farm.full-colmap-rescue-plan.v1":
        raise ValueError("unsupported full-COLMAP anchor plan schema")
    policy = _required_policy(anchor_plan)
    state_provenance = _verify_file_hash(
        state_path,
        _expected_hash(anchor_plan, "source_scene_state"),
        "scene state",
    )
    frames_provenance = _verify_file_hash(
        frames_path,
        _expected_hash(anchor_plan, "frames_json"),
        "frames JSON",
    )
    colmap_provenance = _verify_colmap_contract(model_root, anchor_plan)
    frames_payload, _ = _load_json(frames_path, "frames JSON")
    scale = float(frames_payload.get("meters_per_scene_unit") or 0.0)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("frames JSON lacks a positive metric scale")
    if not math.isclose(
        scale,
        float(anchor_plan.get("meters_per_scene_unit") or 0.0),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("metric scale differs between frames JSON and anchor plan")
    pattern, camera_group, timestamp_group, family_group = _identity_contract(
        frames_payload
    )
    anchors_by_object = _rescue_anchors_by_object(anchor_plan)

    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError("pycolmap is required for support-track retrieval") from exc
    reconstruction = pycolmap.Reconstruction(str(model_root))
    sparse_index = build_sparse_track_index(
        reconstruction, meters_per_scene_unit=scale
    )
    registered_frames: list[TrackerFrame] = []
    frame_by_name: dict[str, TrackerFrame] = {}
    for image_id, image in sorted(
        reconstruction.images.items(), key=lambda item: int(item[0])
    ):
        identity = parse_image_identity(
            str(image.name),
            pattern,
            camera_group=camera_group,
            timestamp_group=timestamp_group,
            view_family_group=family_group,
        )
        frame = TrackerFrame(
            name=str(image.name),
            colmap_image_id=int(image_id),
            colmap_camera_id=int(image.camera_id),
            identity=identity,
            world_from_camera_m=metric_world_from_camera(
                image_world_to_camera(image), meters_per_scene_unit=scale
            ),
        )
        if frame.name in frame_by_name:
            raise ValueError(f"duplicate registered image name: {frame.name}")
        frame_by_name[frame.name] = frame
        registered_frames.append(frame)

    state = load_state(state_path)
    raw_object_ids = state.get("object_id")
    if raw_object_ids is None:
        raise ValueError("scene state lacks canonical object IDs")
    if hasattr(raw_object_ids, "detach"):
        raw_object_ids = raw_object_ids.detach().cpu().numpy()
    state_object_ids = np.asarray(raw_object_ids, dtype=np.int64).reshape(-1)
    objects = anchor_plan.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("anchor plan contains no object rows")

    pairs_by_object: dict[int, object] = {}
    object_rows: list[dict[str, object]] = []
    rejected: Counter[str] = Counter()
    for plan_object in objects:
        if not isinstance(plan_object, Mapping):
            raise ValueError("anchor plan object row must be an object")
        object_id = int(plan_object.get("object_id", -1))
        state_index = int(plan_object.get("state_index", -1))
        if (
            object_id < 0
            or state_index < 0
            or state_index >= state_object_ids.size
            or int(state_object_ids[state_index]) != object_id
        ):
            raise ValueError(f"canonical object/state index mismatch for {object_id}")
        support_points, support_diagnostics = object_support_points(
            state,
            state_index,
            trim_quantile=float(policy["support_trim_quantile"]),
            max_points=int(policy["maximum_object_support_points"]),
        )
        if support_points.shape[0] < int(policy["minimum_object_support_points"]):
            rejected["insufficient_metric_object_support"] += 1
            object_rows.append(
                {
                    "object_id": object_id,
                    "state_index": state_index,
                    "status": "rejected",
                    "rejection_reasons": ["insufficient_metric_object_support"],
                    "support": support_diagnostics,
                    "evidence_frames": 0,
                    "candidate_windows": 0,
                    "candidate_pairs": 0,
                }
            )
            pairs_by_object[object_id] = []
            continue
        candidates, retrieval = build_support_track_candidates(
            reconstruction,
            sparse_index,
            support_points,
            identity_pattern=pattern,
            require_identity_match=True,
            # v2 evaluates every registered frame. Excluding the old RGB-D
            # subset would create artificial holes in otherwise valid video.
            existing_names=set(),
            existing_timestamps=set(),
            meters_per_scene_unit=scale,
            maximum_track_distance_m=float(policy["maximum_track_distance_m"]),
            minimum_object_track_match_fraction=float(
                policy["minimum_object_track_match_fraction"]
            ),
            minimum_track_support_points=int(policy["minimum_track_support_points"]),
            minimum_track_support_fraction=float(
                policy["minimum_track_support_fraction"]
            ),
            minimum_projected_support_points=int(
                policy["minimum_projected_support_points"]
            ),
            min_margin_ratio=float(policy["min_margin_ratio"]),
            min_area_ratio=float(policy["min_area_ratio"]),
            max_area_ratio=float(policy["max_area_ratio"]),
            min_in_frame_support_ratio=float(
                policy["minimum_in_frame_support_ratio"]
            ),
            support_bbox_quantile=float(policy["support_bbox_quantile"]),
        )
        evidence: list[SupportEvidence] = []
        for candidate in candidates:
            name = str(candidate.get("name") or "")
            if candidate.get("retrieval_source") != "object_support_colmap_tracks":
                raise ValueError(f"object {object_id} has non-track candidate")
            if candidate.get("bbox_source") != "projected_object_support":
                raise ValueError(f"object {object_id} has forbidden OBB candidate")
            frame = frame_by_name.get(name)
            if frame is None:
                raise ValueError(f"candidate image is not registered: {name}")
            evidence.append(
                SupportEvidence(
                    object_id=object_id,
                    frame=frame,
                    track_support_count=int(candidate["track_support_count"]),
                    track_support_fraction=float(candidate["track_support_fraction"]),
                    projected_support_points=int(candidate["projected_support_points"]),
                    in_frame_support_ratio=float(candidate["in_frame_support_ratio"]),
                    area_ratio=float(candidate["area_ratio"]),
                    margin_ratio=float(candidate["margin_ratio"]),
                    geometric_score=float(candidate["geometric_score"]),
                    view_direction=tuple(float(x) for x in candidate["view_direction"]),
                    rescue_anchor=name in anchors_by_object.get(object_id, set()),
                )
            )
        if evidence:
            windows, window_diagnostics = enumerate_object_visible_windows(
                registered_frames,
                evidence,
                minimum_frames=int(args.minimum_episode_frames),
                maximum_frames=int(args.maximum_episode_frames),
                preferred_frames=int(args.preferred_episode_frames),
                gap_multiplier=float(args.maximum_gap_multiplier),
                maximum_translation_m=float(args.maximum_translation_m),
                maximum_rotation_degrees=float(args.maximum_rotation_degrees),
                maximum_windows=int(args.maximum_windows_per_object),
            )
        else:
            windows = []
            window_diagnostics = {
                "object_id": object_id,
                "registered_frames": len(registered_frames),
                "evidence_frames": 0,
                "eligible_runs": 0,
                "eligible_run_length_max": 0,
                "candidate_windows_before_limit": 0,
                "candidate_windows": 0,
                "rejection_break_counts": {},
            }
        pairs = pair_object_windows(
            windows, maximum_pairs=int(args.maximum_pairs_per_object)
        )
        pairs_by_object[object_id] = pairs
        reasons: list[str] = []
        if not bool(retrieval.get("object_track_match_gate")):
            reasons.append("object_track_match_gate_failed")
        if not windows:
            reasons.append("no_evidence_complete_contiguous_window")
        elif not pairs:
            reasons.append("no_timestamp_disjoint_window_pair")
        for reason in reasons:
            rejected[reason] += 1
        object_rows.append(
            {
                "object_id": object_id,
                "state_index": state_index,
                "status": "pair_candidate" if pairs else "rejected",
                "rejection_reasons": reasons,
                "support": support_diagnostics,
                "retrieval": retrieval,
                "evidence_frames": len(evidence),
                "candidate_windows": len(windows),
                "candidate_pairs": len(pairs),
                "legacy_rescue_anchor_names": len(anchors_by_object.get(object_id, set())),
                "window_diagnostics": window_diagnostics,
            }
        )

    selected, selection = select_paired_object_episodes(
        pairs_by_object,
        target_frames=int(args.target_frames),
        minimum_total_frames=int(args.minimum_total_frames),
        maximum_total_frames=int(args.maximum_total_frames),
    )
    episodes = selected_pairs_to_episode_rows(selected)
    validation = validate_paired_episode_rows(
        episodes,
        minimum_frames=int(args.minimum_episode_frames),
        maximum_frames=int(args.maximum_episode_frames),
    )
    selected_objects = set(selection["selected_objects"])
    for row in object_rows:
        row["selected"] = int(row["object_id"]) in selected_objects
        if row["selected"]:
            row["status"] = "selected_pair"
    selected_names = [
        str(frame["name"])
        for episode in episodes
        for frame in episode["frames"]
    ]
    image_validation: list[dict[str, object]] = []
    total_bytes = 0
    for name in selected_names:
        source = _safe_image_path(image_root, name)
        size = int(source.stat().st_size)
        total_bytes += size
        row: dict[str, object] = {"name": name, "bytes": size}
        if args.hash_selected_images:
            row["sha256"] = sha256_file(source)
        image_validation.append(row)
    qa_rows = []
    object_by_id = {int(row["object_id"]): row for row in object_rows}
    for object_id in qa_ids:
        row = object_by_id.get(object_id)
        qa_rows.append(
            {
                "object_id": object_id,
                "present_in_anchor_plan": row is not None,
                "evidence_frames": int(row.get("evidence_frames", 0)) if row else 0,
                "candidate_windows": int(row.get("candidate_windows", 0)) if row else 0,
                "candidate_pairs": int(row.get("candidate_pairs", 0)) if row else 0,
                "selected": object_id in selected_objects,
                "role": "report_only_no_selection_influence",
            }
        )

    duration = time.perf_counter() - started
    manifest = {
        "schema": SCHEMA,
        "planner_revision": PLANNER_REVISION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "anchor_plan": {
                "path": str(anchor_path),
                "bytes": len(anchor_raw),
                "sha256": hashlib.sha256(anchor_raw).hexdigest(),
                "schema": anchor_plan["schema"],
            },
            "scene_state": state_provenance,
            "frames_json": frames_provenance,
            "colmap_model": str(model_root),
            "colmap_contract_files": colmap_provenance,
            "image_root": str(image_root),
            "registered_images": len(registered_frames),
            "sparse_track_points": int(sparse_index.point_ids.size),
            "meters_per_scene_unit": scale,
        },
        "identity_contract": {
            "match_mode": "fullmatch",
            "regex": pattern.pattern,
            "camera_group": camera_group,
            "timestamp_group": timestamp_group,
            "view_family_group": family_group,
        },
        "policy": {
            "retrieval_authority": "metric object support -> COLMAP tracks -> support projection",
            "obb_read_or_fallback": False,
            "detector_bbox_used": False,
            "semantic_label_used": False,
            "evaluate_all_registered_images": True,
            "legacy_rescue_views_role": "score tie-break bonus only, never eligibility override",
            "qa_object_ids_role": "report only, never eligibility or ranking",
            "every_window_frame_requires_support_track_projection": True,
            "exact_stream": "camera x view_family",
            "minimum_episode_frames": int(args.minimum_episode_frames),
            "preferred_episode_frames": int(args.preferred_episode_frames),
            "maximum_episode_frames": int(args.maximum_episode_frames),
            "maximum_gap_multiplier": float(args.maximum_gap_multiplier),
            "maximum_translation_m": float(args.maximum_translation_m),
            "maximum_rotation_degrees": float(args.maximum_rotation_degrees),
            "target_frames": int(args.target_frames),
            "minimum_total_frames": int(args.minimum_total_frames),
            "maximum_total_frames": int(args.maximum_total_frames),
            "global_physical_timestamp_train_heldout_disjoint": True,
            "duplicate_rgb_forbidden": True,
            "source_support_track_gates": {
                key: policy[key]
                for key in (
                    "minimum_object_support_points",
                    "maximum_object_support_points",
                    "support_trim_quantile",
                    "maximum_track_distance_m",
                    "minimum_object_track_match_fraction",
                    "minimum_track_support_points",
                    "minimum_track_support_fraction",
                    "minimum_projected_support_points",
                    "minimum_in_frame_support_ratio",
                    "support_bbox_quantile",
                    "min_margin_ratio",
                    "min_area_ratio",
                    "max_area_ratio",
                )
            },
        },
        "selection": selection,
        "validation": validation,
        "qa_anchors": qa_rows,
        "objects": object_rows,
        "selected_names": selected_names,
        "selected_image_validation": {
            "files": len(image_validation),
            "total_bytes": total_bytes,
            "content_hashes_included": bool(args.hash_selected_images),
            "entries": image_validation,
        },
        "episodes": episodes,
        "rejections": dict(sorted(rejected.items())),
        "provenance": {
            "planner_sources": [
                file_provenance(Path(__file__)),
                file_provenance(SRC / "farm_runtime" / "paired_tracker_subset_planner.py"),
                file_provenance(SRC / "farm_runtime" / "full_colmap_view_planner.py"),
                file_provenance(ROOT / "scripts/geometry/plan_farm_full_colmap_rescue.py"),
            ],
            "python": sys.version,
            "platform": platform.platform(),
            "duration_seconds": duration,
            "current_cpu_rss_mib": _current_rss_mib(),
            "peak_cpu_rss_mib": _peak_rss_mib(),
            "gpu_used": False,
            "rgb_decoded": False,
            "rgb_copied_or_linked": False,
        },
    }
    _atomic_json(args.output, manifest, force=bool(args.force))
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "registered_images": len(registered_frames),
                "objects_considered": len(object_rows),
                "objects_with_pairs": sum(
                    int(row["candidate_pairs"]) > 0 for row in object_rows
                ),
                "selected_objects": selection["selected_objects"],
                "selected_episodes": len(episodes),
                "selected_frames": selection["selected_frames"],
                "duration_seconds": duration,
                "peak_cpu_rss_mib": _peak_rss_mib(),
                "gpu_used": False,
                "rgb_materialized": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
