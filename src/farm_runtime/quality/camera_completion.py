"""Select a bounded camera recovery batch from COLMAP metadata before RGB decode.

Frustum coverage is not visibility. Register/render the selected batch, then use
surface-evidence to reject occluded views before running segmentation models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import numpy as np

from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.surface_evidence import SurfaceInputs
from farm_runtime.quality_baseline import describe_file, write_json


def camera_candidate(
    points,
    priority,
    K,
    world_to_camera,
    shape,
    *,
    min_short_pixels=32,
    min_long_pixels=64,
):
    points = np.asarray(points, float)
    priority = np.asarray(priority, bool)
    K = np.asarray(K, float)
    pose = np.asarray(world_to_camera, float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or priority.shape != (len(points),)
        or K.shape != (3, 3)
        or pose.shape != (3, 4)
        or not all(np.isfinite(x).all() for x in (points, K, pose))
    ):
        raise ValueError(
            "finite metric points, camera and aligned priority mask required"
        )
    if len(points) < 20:
        return None
    h, w = shape
    if min(h, w) <= 0 or not 0 < min_short_pixels <= min_long_pixels:
        raise ValueError(
            "positive image dimensions and ordered pixel thresholds required"
        )
    if not priority.any():
        priority = np.ones(len(points), bool)
    camera_points = points @ pose[:, :3].T + pose[:, 3]
    z = camera_points[:, 2]
    valid = z > 0.05
    uv = np.full((len(points), 2), np.nan)
    xy = camera_points[valid] @ K.T
    uv[valid] = xy[:, :2] / xy[:, 2, None]
    inside = (
        valid
        & (uv[:, 0] > 0.03 * w)
        & (uv[:, 0] < 0.97 * w)
        & (uv[:, 1] > 0.03 * h)
        & (uv[:, 1] < 0.97 * h)
    )
    if inside.mean() < 0.9 or inside[priority].mean() < 0.9:
        return None
    box = np.quantile(uv[inside], [0.02, 0.98], axis=0)
    extent = box[1] - box[0]
    if min(extent) < min_short_pixels or max(extent) < min_long_pixels:
        return None
    area = float(np.prod(extent / [w, h]))
    center = -pose[:, :3].T @ pose[:, 3]
    return dict(
        frustum_fraction=float(inside.mean()),
        unconfirmed_frustum_fraction=float(inside[priority].mean()),
        projected_area_fraction=area,
        projected_extent_pixels=extent.tolist(),
        center_m=center.tolist(),
        object_distance_m=float(np.linalg.norm(np.median(points, axis=0) - center)),
        score=float(
            2 * inside[priority].mean() + inside.mean() + min(0.5, np.sqrt(area))
        ),
    )


def independent_views(candidates, budget, min_center_distance_m=0.15):
    if (
        not 1 <= budget <= 8
        or not np.isfinite(min_center_distance_m)
        or min_center_distance_m < 0
    ):
        raise ValueError("budget 1..8 and nonnegative camera separation required")
    chosen = []
    for row in sorted(candidates, key=lambda r: (-r["score"], r["name"])):
        if row["timestamp"] in {r["timestamp"] for r in chosen}:
            continue
        if any(
            np.linalg.norm(np.asarray(row["center_m"]) - r["center_m"])
            < min_center_distance_m
            for r in chosen
        ):
            continue
        chosen.append(row)
        if len(chosen) == budget:
            break
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "split-packet", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--group-id", type=int, required=True)
    parser.add_argument("--timestamp-budget", type=int, default=2)
    parser.add_argument("--min-short-pixels", type=int, default=32)
    parser.add_argument("--min-long-pixels", type=int, default=64)
    args = parser.parse_args(argv)
    if (
        args.output.exists()
        or not 1 <= args.timestamp_budget <= 8
        or not 0 < args.min_short_pixels <= args.min_long_pixels
    ):
        raise ValueError(
            "new output, timestamp budget 1..8 and ordered pixel thresholds required"
        )
    started = time.monotonic()
    audit = json.loads(args.audit.read_text())
    if audit.get("closed_test_opened") is not False:
        raise ValueError("development point audit required")
    packet = json.loads(args.split_packet.read_text())
    allowed = {
        str(ts)
        for ts, role in packet["split_by_timestamp"].items()
        if role in ("train", "dev")
    }
    if not allowed:
        raise ValueError("explicit nonempty train/dev timestamp allowlist required")
    inputs = SurfaceInputs(checked_file(audit["source_geometry"]))
    groups = {g["group_id"]: g for g in audit["groups"]}
    if args.group_id not in groups:
        raise ValueError("group absent from surface audit")
    group = groups[args.group_id]
    prefix = f"group_{args.group_id:04d}"
    with np.load(checked_file(audit["point_evidence"]), allow_pickle=False) as archive:
        points = archive[prefix + "_points"]
        priority = ~archive[prefix + "_corroborated"]
    existing = set(group["source_timestamps"])
    if not existing <= allowed:
        raise ValueError("source timestamps outside development allowlist")
    prep = json.loads(
        checked_file(inputs.geometry["inputs"]["prep_summary"]).read_text()
    )
    scale = prep["metric_scale"]["meters_per_scene_unit"]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("positive finite metric scene scale required")
    identity = prep["identity_grouping"]
    pattern = re.compile(identity["regex"])
    model = Path(prep["inputs"]["colmap_model"])
    import pycolmap
    from scripts.prepare_colmap_3dgs_rgbd import camera_model_name

    reconstruction = pycolmap.Reconstruction(str(model))
    views, candidates, excluded = {}, [], 0
    for image in reconstruction.images.values():
        match = pattern.fullmatch(image.name)
        if match is None:
            continue
        timestamp = match[identity["timestamp_group"]]
        if timestamp not in allowed:
            excluded += 1
            continue
        row = dict(
            name=image.name,
            timestamp=timestamp,
            sensor=match[identity["sensor_group"]],
            family=match[identity["family_group"]],
        )
        views[image.name] = row
        if timestamp in existing:
            continue
        camera = reconstruction.cameras[image.camera_id]
        if camera_model_name(camera) not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise ValueError("undistorted pinhole COLMAP cameras required")
        pose = np.asarray(image.cam_from_world.matrix()).copy()
        pose[:, 3] *= scale
        metrics = camera_candidate(
            points,
            priority,
            camera.calibration_matrix(),
            pose,
            (camera.height, camera.width),
            min_short_pixels=args.min_short_pixels,
            min_long_pixels=args.min_long_pixels,
        )
        if metrics is not None:
            candidates.append(dict(**row, **metrics))
    chosen = independent_views(candidates, args.timestamp_budget)
    selected = []
    for row in chosen:
        selected.extend(
            sorted(
                v["name"]
                for v in views.values()
                if v["timestamp"] == row["timestamp"] and v["family"] == row["family"]
            )
        )
    anchors = [
        name for name, row in inputs.frames.items() if str(row["frame_id"]) in existing
    ]
    names = list(dict.fromkeys(anchors + selected))
    args.output.mkdir(parents=True)
    for filename, values in [
        ("selected_names.txt", selected),
        ("preparation_names.txt", names),
    ]:
        (args.output / filename).write_text(
            "\n".join(values) + ("\n" if values else "")
        )
    camera_files = [model / name for name in ("cameras.bin", "images.bin")]
    if not all(p.is_file() for p in camera_files):
        camera_files = [model / name for name in ("cameras.txt", "images.txt")]
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.camera-completion-selection.v1",
            source_audit=describe_file(args.audit),
            source_packet=describe_file(args.split_packet),
            source_geometry=describe_file(inputs.geometry_path),
            source_colmap=[describe_file(p) for p in camera_files],
            meters_per_scene_unit=scale,
            group_id=args.group_id,
            timestamp_budget=args.timestamp_budget,
            minimum_extent_pixels=[args.min_short_pixels, args.min_long_pixels],
            cameras_total=len(reconstruction.images),
            nondevelopment_cameras_excluded_before_rgb=excluded,
            eligible_candidates=len(candidates),
            selected=chosen,
            selected_names=selected,
            existing_anchor_views=anchors,
            preparation_names=names,
            top_candidates=sorted(candidates, key=lambda r: (-r["score"], r["name"]))[
                :40
            ],
            seconds=time.monotonic() - started,
            selection_uses_only_geometry_and_split_metadata=True,
            rgb_pixels_decoded=0,
            depth_visibility_verified=False,
            registration_verified=False,
            closed_test_opened=False,
            release_eligible=False,
        ),
    )
    print(
        json.dumps(dict(selected_names=selected, candidates=len(candidates))),
        flush=True,
    )
    return 0
