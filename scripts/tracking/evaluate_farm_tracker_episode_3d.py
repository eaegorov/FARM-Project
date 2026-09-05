#!/usr/bin/env python3
"""Evaluate one tracker episode as episode-local sparse 3D signatures.

This stage is CPU-only.  It validates every episode/name/camera/shape mapping,
samples COLMAP POINTS2D into 8-bit masks, and writes raw evidence separately
from fail-closed forwarding decisions.  It never merges numeric IDs across
tracker episodes.
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
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from farm_runtime.colmap_pose_reader import read_colmap_cameras_and_poses  # noqa: E402
from farm_runtime.tracker_3d_signature import (  # noqa: E402
    MAX_LOCAL_ID,
    SCHEMA,
    SHADOW_SCHEMA,
    GatePolicy,
    collect_raw_evidence,
    finalize_identity,
    policy_dict,
    read_selected_image_observations,
    read_selected_points3d,
    validate_episode_contract,
    validate_mask_directory,
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    stat = source.stat()
    return {
        "path": str(source),
        "bytes": int(stat.st_size),
        "sha256": sha256_file(source),
    }


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def resolve_metric_scale(
    episode: Mapping[str, Any], explicit_scale: float | None
) -> tuple[float, dict[str, Any]]:
    """Resolve scale from CLI and/or the hash-verified source planner manifest."""

    declared_path = episode.get("source_plan")
    declared_hash = str(episode.get("source_plan_sha256") or "")
    source_plan: dict[str, Any] | None = None
    provenance: dict[str, Any] = {
        "declared_path": declared_path,
        "declared_sha256": declared_hash or None,
        "available": False,
        "hash_verified": False,
        "scale_available": False,
    }
    if declared_path:
        path = Path(str(declared_path)).expanduser()
        if path.is_file():
            path = path.resolve(strict=True)
            actual_hash = sha256_file(path)
            if declared_hash and actual_hash != declared_hash:
                raise ValueError("materialized episode source plan checksum mismatch")
            source_plan = _load_json_object(path, "source plan")
            provenance.update(
                {
                    "available": True,
                    "path": str(path),
                    "bytes": int(path.stat().st_size),
                    "sha256": actual_hash,
                    "hash_verified": bool(declared_hash),
                    "schema": source_plan.get("schema"),
                }
            )
    plan_scale = None
    if source_plan is not None:
        inputs = source_plan.get("inputs")
        if (
            isinstance(inputs, Mapping)
            and inputs.get("meters_per_scene_unit") is not None
        ):
            plan_scale = float(inputs["meters_per_scene_unit"])
            provenance["scale_available"] = True
            provenance["meters_per_scene_unit"] = plan_scale
    if explicit_scale is None and plan_scale is None:
        raise ValueError(
            "metric scale is unavailable: supply --meters-per-scene-unit or keep "
            "the episode's hash-verified source plan accessible"
        )
    scale = float(explicit_scale if explicit_scale is not None else plan_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("meters_per_scene_unit must be finite and positive")
    if explicit_scale is not None and plan_scale is not None:
        if not np.isclose(scale, plan_scale, rtol=1.0e-9, atol=1.0e-12):
            raise ValueError(
                "--meters-per-scene-unit disagrees with the source planner manifest"
            )
    provenance["selected_scale_source"] = (
        "command_line_confirmed_by_source_plan"
        if explicit_scale is not None and plan_scale is not None
        else "command_line" if explicit_scale is not None else "source_plan"
    )
    return scale, provenance


def _load_masks(paths: list[Path]) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    masks: list[np.ndarray] = []
    provenance: list[dict[str, Any]] = []
    for path in paths:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"tracker mask is not PNG: {path}")
            if image.mode not in {"L", "P"}:
                raise ValueError(
                    f"tracker mask must be an 8-bit L/P image, got {image.mode}: {path}"
                )
            values = np.asarray(image)
        if values.dtype != np.uint8 or values.ndim != 2:
            raise ValueError(f"tracker mask is not a 2D uint8 ID map: {path}")
        masks.append(np.ascontiguousarray(values))
        row = file_provenance(path)
        row.update(
            {
                "shape_hw": [int(values.shape[0]), int(values.shape[1])],
                "minimum_local_id": int(values.min(initial=0)),
                "maximum_local_id": int(values.max(initial=0)),
                "unique_local_id_count_including_background": int(
                    np.unique(values).size
                ),
            }
        )
        provenance.append(row)
    return masks, provenance


def _current_rss_mib() -> float:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def _atomic_json(path: Path, payload: Any, *, force: bool) -> None:
    destination = path.expanduser().resolve()
    if destination.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing report without --force: {destination}"
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


def _policy_from_args(args: argparse.Namespace) -> GatePolicy:
    return GatePolicy(
        minimum_frame_support=args.minimum_frame_support,
        minimum_physical_timestamps=args.minimum_physical_timestamps,
        minimum_persistence_ratio=args.minimum_persistence_ratio,
        minimum_median_area_px=args.minimum_median_area_px,
        tiny_area_px=args.tiny_area_px,
        maximum_tiny_frame_fraction=args.maximum_tiny_frame_fraction,
        minimum_median_largest_2d_component_fraction=(
            args.minimum_median_largest_2d_component_fraction
        ),
        minimum_sparse_observations=args.minimum_sparse_observations,
        minimum_unique_sparse_points=args.minimum_unique_sparse_points,
        minimum_point_observations=args.minimum_point_observations,
        minimum_point_label_purity=args.minimum_point_label_purity,
        minimum_qualified_sparse_points=args.minimum_qualified_sparse_points,
        minimum_qualified_sparse_point_fraction=(
            args.minimum_qualified_sparse_point_fraction
        ),
        maximum_ambiguous_sparse_point_fraction=(
            args.maximum_ambiguous_sparse_point_fraction
        ),
        minimum_largest_3d_component_fraction=(
            args.minimum_largest_3d_component_fraction
        ),
        require_points3d=not bool(args.allow_missing_points3d),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-json", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--colmap-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--meters-per-scene-unit", type=float)
    parser.add_argument("--maximum-local-id", type=int, default=MAX_LOCAL_ID)
    parser.add_argument(
        "--maximum-pose-translation-error-m", type=float, default=1.0e-5
    )
    parser.add_argument(
        "--maximum-pose-rotation-error-degrees", type=float, default=1.0e-3
    )
    parser.add_argument("--minimum-frame-support", type=int, default=3)
    parser.add_argument("--minimum-physical-timestamps", type=int, default=3)
    parser.add_argument("--minimum-persistence-ratio", type=float, default=0.25)
    parser.add_argument("--minimum-median-area-px", type=int, default=64)
    parser.add_argument("--tiny-area-px", type=int, default=64)
    parser.add_argument("--maximum-tiny-frame-fraction", type=float, default=0.50)
    parser.add_argument(
        "--minimum-median-largest-2d-component-fraction",
        type=float,
        default=0.60,
    )
    parser.add_argument("--minimum-sparse-observations", type=int, default=12)
    parser.add_argument("--minimum-unique-sparse-points", type=int, default=8)
    parser.add_argument("--minimum-point-observations", type=int, default=2)
    parser.add_argument("--minimum-point-label-purity", type=float, default=0.67)
    parser.add_argument("--minimum-qualified-sparse-points", type=int, default=6)
    parser.add_argument(
        "--minimum-qualified-sparse-point-fraction", type=float, default=0.25
    )
    parser.add_argument(
        "--maximum-ambiguous-sparse-point-fraction", type=float, default=0.35
    )
    parser.add_argument(
        "--minimum-largest-3d-component-fraction", type=float, default=0.50
    )
    parser.add_argument(
        "--allow-missing-points3d",
        action="store_true",
        help="Keep raw 2D/track metrics but do not fail IDs solely for absent xyz.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= int(args.maximum_local_id) <= MAX_LOCAL_ID:
        raise ValueError(f"--maximum-local-id must be in [1, {MAX_LOCAL_ID}]")
    for name, value in (
        ("--maximum-pose-translation-error-m", args.maximum_pose_translation_error_m),
        (
            "--maximum-pose-rotation-error-degrees",
            args.maximum_pose_rotation_error_degrees,
        ),
    ):
        if not np.isfinite(value) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    started = time.perf_counter()
    stages: dict[str, float] = {}
    episode_path = args.episode_json.expanduser().resolve(strict=True)
    mask_dir = args.mask_dir.expanduser().resolve(strict=True)
    model_dir = args.colmap_model.expanduser().resolve(strict=True)
    if not mask_dir.is_dir() or not model_dir.is_dir():
        raise ValueError("--mask-dir and --colmap-model must be directories")
    episode = _load_json_object(episode_path, "episode")
    scale, source_plan_provenance = resolve_metric_scale(
        episode, args.meters_per_scene_unit
    )
    frames = episode.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("episode has no frames")
    image_ids = [int(row.get("colmap_image_id", -1)) for row in frames]
    if min(image_ids) < 0 or len(image_ids) != len(set(image_ids)):
        raise ValueError("episode COLMAP image IDs must be unique and non-negative")

    stage = time.perf_counter()
    source_format, cameras, poses, pose_contract_files = read_colmap_cameras_and_poses(
        model_dir
    )
    poses_by_id = {pose.image_id: pose for pose in poses}
    observations, observation_file = read_selected_image_observations(
        model_dir,
        source_format=source_format,
        selected_image_ids=image_ids,
    )
    validated_frames = validate_episode_contract(
        episode,
        cameras=cameras,
        poses_by_id=poses_by_id,
        observations_by_id=observations,
        meters_per_scene_unit=scale,
        maximum_translation_error_m=args.maximum_pose_translation_error_m,
        maximum_rotation_error_degrees=args.maximum_pose_rotation_error_degrees,
    )
    stages["colmap_and_episode_contract_validation_seconds"] = (
        time.perf_counter() - stage
    )

    stage = time.perf_counter()
    mask_paths = validate_mask_directory(mask_dir, validated_frames)
    masks, mask_provenance = _load_masks(mask_paths)
    policy = _policy_from_args(args)
    policy.validate()
    episode_raw, raw_identities, votes, observation_provenance = collect_raw_evidence(
        episode_id=str(episode["episode_id"]),
        frame_rows=validated_frames,
        masks=masks,
        observations_by_id=observations,
        maximum_local_id=args.maximum_local_id,
    )
    stages["mask_validation_and_sparse_sampling_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    claimed_point_ids = sorted(
        point_id
        for point_id, counts in votes.items()
        if any(int(label) != 0 and int(count) > 0 for label, count in counts.items())
    )
    points3d, points3d_file = read_selected_points3d(
        model_dir,
        source_format=source_format,
        selected_point_ids=claimed_point_ids,
    )
    identities = [
        finalize_identity(
            row,
            votes_by_point=votes,
            observation_provenance_by_point=observation_provenance,
            points3d=points3d,
            points3d_file_available=points3d_file is not None,
            meters_per_scene_unit=scale,
            policy=policy,
        )
        for row in raw_identities
    ]
    stages["points3d_loading_and_signature_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    contract_paths = {path.resolve() for path in pose_contract_files}
    contract_paths.add(observation_file.resolve())
    if points3d_file is not None:
        contract_paths.add(points3d_file.resolve())
    model_provenance = [file_provenance(path) for path in sorted(contract_paths)]
    accepted = [row for row in identities if row["decision"]["passed"]]
    shadow_accepted = [row for row in identities if row["decision_v2_shadow"]["passed"]]
    shadow_recovered = [
        row
        for row in identities
        if row["decision_v2_shadow"]["passed"] and not row["decision"]["passed"]
    ]
    shadow_regressed = [
        row
        for row in identities
        if row["decision"]["passed"] and not row["decision_v2_shadow"]["passed"]
    ]
    shadow_state_totals: dict[str, int] = {}
    for identity in identities:
        for name, count in identity["shadow_v2_evidence"]["counts"].items():
            shadow_state_totals[name] = shadow_state_totals.get(name, 0) + int(count)
    report = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "stage_semantics": (
            "episode-local tracker evidence only; no numeric ID is associated "
            "or merged across episodes"
        ),
        "inputs": {
            "episode": file_provenance(episode_path),
            "episode_id": episode["episode_id"],
            "camera": episode.get("camera"),
            "view_family": episode.get("view_family"),
            "split": episode.get("split"),
            "mask_directory": str(mask_dir),
            "mask_files": mask_provenance,
            "colmap_model": str(model_dir),
            "colmap_format": source_format,
            "colmap_contract_files": model_provenance,
            "source_plan": source_plan_provenance,
            "meters_per_scene_unit": scale,
        },
        "input_contract": {
            "passed": True,
            "frame_count": len(validated_frames),
            "exact_mask_name_set": True,
            "single_colmap_camera": True,
            "native_camera_shape": [
                validated_frames[0]["height"],
                validated_frames[0]["width"],
            ],
            "maximum_local_id": int(args.maximum_local_id),
            "sampling": (
                "nearest integer pixel via round-to-nearest-even after strict "
                "finite/in-bounds POINTS2D validation; background label 0 votes "
                "participate in purity"
            ),
            "shadow_observation_provenance": (
                "each claimed point records source frame, COLMAP image, POINTS2D "
                "coordinate, sampled pixel/label, identity active-span membership, "
                "and identity-present-in-frame state"
            ),
            "maximum_observed_pose_translation_error_m": max(
                row["pose_translation_error_m"] for row in validated_frames
            ),
            "maximum_observed_pose_rotation_error_degrees": max(
                row["pose_rotation_error_degrees"] for row in validated_frames
            ),
        },
        "policy": {
            "name": "farm-tracker-3d-signature-fail-closed-v1",
            "thresholds": policy_dict(policy),
            "raw_metrics_are_threshold_independent": True,
        },
        "raw_episode_metrics": episode_raw,
        "local_identities": identities,
        "v2_shadow": {
            "schema": SHADOW_SCHEMA,
            "mode": "diagnostic-only-no-publish",
            "authoritative_publish_decision_path": "local_identities[].decision",
            "shadow_decision_path": "local_identities[].decision_v2_shadow",
            "automatic_promotion_allowed": False,
            "global_association_authorized": False,
            "policy": {
                "minimum_supported_sparse_points": policy.minimum_qualified_sparse_points,
                "minimum_supported_sparse_point_fraction": (
                    policy.minimum_qualified_sparse_point_fraction
                ),
                "maximum_visible_conflict_fraction_among_supported": (
                    policy.maximum_ambiguous_sparse_point_fraction
                ),
                "minimum_point_observations": policy.minimum_point_observations,
                "minimum_point_label_purity": policy.minimum_point_label_purity,
                "other_v1_fail_closed_checks_preserved": True,
            },
            "summary": {
                "local_identity_count": len(identities),
                "v1_publish_accepted_count": len(accepted),
                "shadow_would_pass_count": len(shadow_accepted),
                "shadow_recovered_from_v1_rejection_count": len(shadow_recovered),
                "shadow_regressed_from_v1_acceptance_count": len(shadow_regressed),
                "shadow_would_pass_identity_keys": [
                    row["identity_key"] for row in shadow_accepted
                ],
                "shadow_recovered_identity_keys": [
                    row["identity_key"] for row in shadow_recovered
                ],
                "point_state_totals": shadow_state_totals,
            },
        },
        "summary": {
            "local_identity_count": len(identities),
            "accepted_for_cross_episode_association_count": len(accepted),
            "rejected_count": len(identities) - len(accepted),
            "accepted_identity_keys": [row["identity_key"] for row in accepted],
            "claimed_foreground_sparse_point_count": len(claimed_point_ids),
            "claimed_points3d_coordinates_found": len(points3d),
            "points3d_file_available": points3d_file is not None,
        },
        "measurement": {
            "cpu_only": True,
            "gpu_used": False,
            "wall_seconds": time.perf_counter() - started,
            "peak_cpu_rss_mib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            ),
            "current_cpu_rss_mib": _current_rss_mib(),
            "stages_seconds": stages,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "implementation_files": [
                file_provenance(Path(__file__).resolve()),
                file_provenance(
                    ROOT / "src" / "farm_runtime" / "tracker_3d_signature.py"
                ),
            ],
        },
        "limitations": [
            "A passing local ID is only eligible for a later 3D association stage.",
            "Numeric local IDs from different episode_id namespaces are never equivalent here.",
            "Sparse COLMAP points sample masks only at reconstructed feature coordinates; dense Gaussian lifting still requires depth/multiview exclusivity and heldout QC.",
            "The v2-shadow decision is an ablation only and cannot replace the authoritative v1 publish decision or authorize GG training.",
            "COLMAP POINTS2D establishes sparse feature visibility, but no dense depth-based occlusion test is available in this stage.",
        ],
    }
    stages["provenance_hashing_and_report_seconds"] = time.perf_counter() - stage
    report["measurement"]["wall_seconds"] = time.perf_counter() - started
    report["measurement"]["stages_seconds"] = stages
    _atomic_json(args.output, report, force=args.force)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "episode_id": episode["episode_id"],
                "output": str(args.output.expanduser().resolve()),
                **report["summary"],
                "measurement": report["measurement"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
