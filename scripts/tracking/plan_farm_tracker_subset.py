#!/usr/bin/env python3
"""Plan a bounded, continuous full-COLMAP subset for tracker A/B tests.

This command reads only COLMAP cameras and image poses.  It never decodes or
copies RGB, never loads points3D, and never invokes a GPU model.  The single
JSON output is an auditable input contract for either SAM3 tracking or the
Gaussian Grouping SAM+identity branch.
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
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from farm_runtime.colmap_pose_reader import (  # noqa: E402
    ColmapCamera,
    read_colmap_cameras_and_poses,
)
from farm_runtime.tracker_subset_planner import (  # noqa: E402
    SCHEMA,
    SPLIT_POLICY,
    TrackerFrame,
    assign_global_timestamp_folds,
    build_continuity_regions,
    compile_identity_pattern,
    episodes_to_manifest_rows,
    load_anchor_names,
    metric_world_from_camera,
    parse_image_identity,
    select_balanced_episodes,
    validate_episode_rows,
)

DEFAULT_IDENTITY_EXAMPLE = (
    r"(?P<camera>cam[0-9]+)_(?P<timestamp>[0-9]+)_"
    r"(?P<family>center|yaw_left|yaw_right|pitch_up|pitch_down)\.(?:png|jpg|jpeg)"
)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(int(chunk_bytes))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path: Path) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    return {
        "path": str(source),
        "bytes": int(stat.st_size),
        "sha256": sha256_file(source),
    }


def safe_image_path(image_root: Path, name: str) -> Path:
    root = image_root.expanduser().resolve()
    candidate = (root / str(name)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"COLMAP image name escapes image root: {name!r}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _scale_from_anchor(path: Path | None) -> float | None:
    if path is None:
        return None
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("meters_per_scene_unit")
    if value is None and isinstance(payload.get("policy"), Mapping):
        value = payload["policy"].get("meters_per_scene_unit")
    if value is None:
        return None
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("anchor plan has invalid meters_per_scene_unit")
    return scale


def _camera_row(camera: ColmapCamera) -> dict[str, object]:
    return {
        "colmap_camera_id": camera.camera_id,
        "model": camera.model,
        "width": camera.width,
        "height": camera.height,
        "params": list(camera.params),
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


def _atomic_json(path: Path, payload: object, *, force: bool) -> None:
    destination = path.expanduser().resolve()
    if destination.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing plan without --force: {destination}"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one CPU-only JSON manifest of continuous camera/view-family "
            "episodes for a fair SAM3 versus SAM+identity tracker comparison."
        )
    )
    parser.add_argument("--colmap-model", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--identity-regex",
        required=True,
        help=(
            "Full-match regex with named camera, timestamp and family groups; "
            f"Factory example: {DEFAULT_IDENTITY_EXAMPLE}"
        ),
    )
    parser.add_argument("--camera-group", default="camera")
    parser.add_argument("--timestamp-group", default="timestamp")
    parser.add_argument("--view-family-group", default="family")
    parser.add_argument(
        "--meters-per-scene-unit",
        type=float,
        help=(
            "Metric COLMAP scale. May be omitted only when --anchor-plan carries "
            "a positive meters_per_scene_unit."
        ),
    )
    parser.add_argument(
        "--anchor-plan",
        type=Path,
        help="Optional V11/V10 rescue plan; selected views are soft preferences only.",
    )
    parser.add_argument("--target-frames", type=int, default=200)
    parser.add_argument("--minimum-total-frames", type=int, default=160)
    parser.add_argument("--maximum-total-frames", type=int, default=240)
    parser.add_argument("--minimum-episode-frames", type=int, default=8)
    parser.add_argument("--maximum-episode-frames", type=int, default=32)
    parser.add_argument("--maximum-gap-multiplier", type=float, default=2.5)
    parser.add_argument("--maximum-translation-m", type=float, default=0.50)
    parser.add_argument("--maximum-rotation-degrees", type=float, default=21.0)
    parser.add_argument(
        "--fold-block-size",
        type=int,
        default=32,
        help="Number of globally ordered physical timestamps in one fold block.",
    )
    parser.add_argument("--heldout-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", default="farm-tracker-ab-v1")
    parser.add_argument(
        "--allow-unmatched-images",
        action="store_true",
        help="Explicitly skip registered names that do not match the identity regex.",
    )
    parser.add_argument(
        "--hash-selected-images",
        action="store_true",
        help="Also content-hash the selected RGB files (they are never decoded).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace an existing output manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    model_root = args.colmap_model.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    if not model_root.is_dir():
        raise FileNotFoundError(model_root)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    anchor_path = args.anchor_plan.expanduser().resolve() if args.anchor_plan else None
    anchor_names, anchor_provenance = load_anchor_names(anchor_path)
    scale = args.meters_per_scene_unit
    scale_source = "command_line"
    if scale is None:
        scale = _scale_from_anchor(anchor_path)
        scale_source = "anchor_plan"
    if scale is None:
        raise ValueError(
            "metric pose gates require --meters-per-scene-unit, or an anchor plan "
            "with meters_per_scene_unit"
        )
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("meters_per_scene_unit must be finite and positive")
    pattern = compile_identity_pattern(
        args.identity_regex,
        camera_group=args.camera_group,
        timestamp_group=args.timestamp_group,
        view_family_group=args.view_family_group,
    )
    source_format, cameras, images, model_files = read_colmap_cameras_and_poses(
        model_root
    )
    matched: list[TrackerFrame] = []
    unmatched: list[str] = []
    for image in images:
        try:
            identity = parse_image_identity(
                image.name,
                pattern,
                camera_group=args.camera_group,
                timestamp_group=args.timestamp_group,
                view_family_group=args.view_family_group,
            )
        except ValueError:
            unmatched.append(image.name)
            continue
        matched.append(
            TrackerFrame(
                name=image.name,
                colmap_image_id=image.image_id,
                colmap_camera_id=image.camera_id,
                identity=identity,
                world_from_camera_m=metric_world_from_camera(
                    image.camera_from_world, meters_per_scene_unit=scale
                ),
            )
        )
    if unmatched and not args.allow_unmatched_images:
        examples = ", ".join(repr(name) for name in unmatched[:5])
        raise ValueError(
            f"{len(unmatched)} registered image(s) violate the identity contract; "
            f"examples: {examples}. Use --allow-unmatched-images only deliberately."
        )
    if not matched:
        raise ValueError("identity contract matched no registered COLMAP image")
    folds, fold_blocks = assign_global_timestamp_folds(
        (frame.identity.timestamp_value for frame in matched),
        heldout_fraction=args.heldout_fraction,
        block_size=args.fold_block_size,
        split_seed=args.split_seed,
    )
    if set(folds.values()) != {"train", "heldout"}:
        raise ValueError(
            "global timeline did not produce both train and heldout; reduce "
            "--fold-block-size for this scene"
        )
    regions, continuity = build_continuity_regions(
        matched,
        folds,
        gap_multiplier=args.maximum_gap_multiplier,
        maximum_translation_m=args.maximum_translation_m,
        maximum_rotation_degrees=args.maximum_rotation_degrees,
    )
    selected, selection = select_balanced_episodes(
        regions,
        anchor_names=anchor_names,
        target_frames=args.target_frames,
        minimum_total_frames=args.minimum_total_frames,
        maximum_total_frames=args.maximum_total_frames,
        minimum_episode_frames=args.minimum_episode_frames,
        maximum_episode_frames=args.maximum_episode_frames,
    )
    if set(selection["frames_by_split"]) != {"train", "heldout"}:
        raise ValueError(
            "continuous coverage cannot form minimum-length episodes in both folds"
        )
    episode_rows = episodes_to_manifest_rows(selected, anchor_names)
    validate_episode_rows(
        episode_rows,
        minimum_episode_frames=args.minimum_episode_frames,
        maximum_episode_frames=args.maximum_episode_frames,
        maximum_translation_m=args.maximum_translation_m,
        maximum_rotation_degrees=args.maximum_rotation_degrees,
    )
    selected_names = [
        str(frame["name"]) for episode in episode_rows for frame in episode["frames"]
    ]
    image_rows: list[dict[str, object]] = []
    selected_bytes = 0
    for name in selected_names:
        source = safe_image_path(image_root, name)
        stat = source.stat()
        row: dict[str, object] = {
            "name": name,
            "bytes": int(stat.st_size),
        }
        selected_bytes += int(stat.st_size)
        if args.hash_selected_images:
            row["sha256"] = sha256_file(source)
        image_rows.append(row)
    registered_names = {image.name for image in images}
    matched_anchors = sorted(anchor_names.intersection(registered_names))
    selected_anchors = sorted(anchor_names.intersection(selected_names))
    camera_ids = sorted(
        {
            int(frame["colmap_camera_id"])
            for episode in episode_rows
            for frame in episode["frames"]
        }
    )
    duration = time.perf_counter() - started
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "colmap_model": str(model_root),
            "colmap_format": source_format,
            "colmap_contract_files": [file_provenance(path) for path in model_files],
            "image_root": str(image_root),
            "registered_images": len(images),
            "matched_images": len(matched),
            "unmatched_images": len(unmatched),
            "unmatched_examples": unmatched[:20],
            "meters_per_scene_unit": scale,
            "metric_scale_source": scale_source,
            "anchor_plan": {
                **anchor_provenance,
                "registered_names": len(matched_anchors),
                "selected_names": len(selected_anchors),
                "missing_examples": sorted(anchor_names.difference(registered_names))[
                    :20
                ],
            },
        },
        "identity_contract": {
            "match_mode": "fullmatch",
            "regex": args.identity_regex,
            "camera_group": args.camera_group,
            "timestamp_group": args.timestamp_group,
            "view_family_group": args.view_family_group,
            "unmatched_policy": (
                "skip_explicitly_allowed"
                if args.allow_unmatched_images
                else "fail_closed"
            ),
        },
        "policy": {
            "split_policy": SPLIT_POLICY,
            "heldout_fraction": args.heldout_fraction,
            "fold_block_size": args.fold_block_size,
            "split_seed": args.split_seed,
            "maximum_gap_multiplier": args.maximum_gap_multiplier,
            "adaptive_gap_statistic": "per-exact-stream lower quartile of positive gaps",
            "maximum_translation_m": args.maximum_translation_m,
            "maximum_rotation_degrees": args.maximum_rotation_degrees,
            "minimum_episode_frames": args.minimum_episode_frames,
            "maximum_episode_frames": args.maximum_episode_frames,
            "target_frames": args.target_frames,
            "minimum_total_frames": args.minimum_total_frames,
            "maximum_total_frames": args.maximum_total_frames,
            "rgb_materialization": "none",
            "anchor_role": "soft preference only; never an eligibility override",
        },
        "folds": {
            "global_physical_timestamps": len(folds),
            "blocks": fold_blocks,
        },
        "continuity": continuity,
        "selection": selection,
        "cameras": [_camera_row(cameras[camera_id]) for camera_id in camera_ids],
        "selected_names": selected_names,
        "selected_image_validation": {
            "files": len(image_rows),
            "total_bytes": selected_bytes,
            "content_hashes_included": bool(args.hash_selected_images),
            "entries": image_rows,
        },
        "episodes": episode_rows,
        "provenance": {
            "planner_sources": [
                file_provenance(Path(__file__)),
                file_provenance(SRC / "farm_runtime" / "tracker_subset_planner.py"),
                file_provenance(SRC / "farm_runtime" / "colmap_pose_reader.py"),
            ],
            "python": sys.version,
            "platform": platform.platform(),
            "duration_seconds": duration,
            "current_cpu_rss_mib": _current_rss_mib(),
            "peak_rss_measurement": "use the standard external measured-stage wrapper",
            "gpu_used": False,
            "rgb_decoded": False,
            "rgb_copied_or_linked": False,
        },
    }
    _atomic_json(args.output, manifest, force=args.force)
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "registered_images": len(images),
                "matched_images": len(matched),
                "selected_frames": selection["selected_frames"],
                "selected_episodes": selection["selected_episodes"],
                "frames_by_split": selection["frames_by_split"],
                "duration_seconds": duration,
                "current_cpu_rss_mib": _current_rss_mib(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
