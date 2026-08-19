#!/usr/bin/env python3
"""Build scene-generic ShapeR PKLs from the verified Gaussian CSR bank.

Membership is read once from ``verified_instance_bank.npz``.  Held-out masks
may be used here because Gaussian membership is already frozen and this stage
never writes labels back to the lift.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))

from tools.farm_shaper_bridge.common import (  # noqa: E402
    diversity_metrics,
    load_mask_pair,
    load_run,
    open_graphdeco_ply,
    project_metric,
    resolve_mask_path,
    sha256_file,
    utc_now,
    verify_run_ply_fingerprint,
)
from tools.farm_shaper_bridge.shaper_contracts import (  # noqa: E402
    SHAPER_INPUT_SCHEMA,
    artifact_record,
    atomic_json,
    atomic_pickle,
    bind_verified_lift,
    load_shaper_config,
    require_current_sources_match_authority,
    resolve_farm_source_authority,
    safe_name,
)


SOURCE_FILES = (
    "tools/farm_shaper_bridge/build_shaper_inputs.py",
    "tools/farm_shaper_bridge/shaper_contracts.py",
    "tools/farm_shaper_bridge/common.py",
    "configs/shaper_bridge.v1.yaml",
)


class ViewQualityError(ValueError):
    """A valid object lacks enough view evidence for useful ShapeR input."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Immutable successful FARM run")
    parser.add_argument("--ply", type=Path, required=True, help="Exact immutable source 3DGS PLY")
    parser.add_argument("--lift", type=Path, required=True, help="Completed Gaussian lift directory")
    parser.add_argument("--output", type=Path, required=True, help="New/empty ShapeR input directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "shaper_bridge.v1.yaml",
    )
    parser.add_argument("--allow-nonrelease-lift", action="store_true")
    parser.add_argument("--allow-legacy-run", action="store_true")
    parser.add_argument("--allow-unsigned-source", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def _spatial_sample(
    points: np.ndarray,
    confidence: np.ndarray,
    timestamp_support: np.ndarray,
    maximum: int,
) -> np.ndarray:
    """Deterministic voxel-coverage sampling with evidence-aware tie breaks."""

    count = len(points)
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    evidence = np.asarray(confidence, dtype=np.float64) * (
        1.0 + 0.08 * np.log1p(np.asarray(timestamp_support, dtype=np.float64))
    )
    order = np.argsort(-evidence, kind="stable")
    origin = np.min(points, axis=0)
    extent = float(np.max(np.ptp(points, axis=0)))
    low = max(extent / 1024.0, 1.0e-5)
    high = max(extent / 2.0, 2.0e-5)
    best = order[:maximum]
    for _ in range(24):
        cell = 0.5 * (low + high)
        keys = np.floor((points[order] - origin) / cell).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        chosen = order[np.sort(first)]
        best = chosen
        if len(chosen) > maximum:
            low = cell
        else:
            high = cell
    if len(best) > maximum:
        best = best[:maximum]
    elif len(best) < maximum:
        used = np.zeros(count, dtype=bool)
        used[best] = True
        best = np.concatenate((best, order[~used[order]][: maximum - len(best)]))
    return np.sort(best.astype(np.int64))


def object_transforms(center_m: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return world->model and model->world transforms for row-vector points."""

    center = np.asarray(center_m, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    world_to_model = np.eye(4, dtype=np.float64)
    world_to_model[:3, :3] = rotation.T
    world_to_model[:3, 3] = -rotation.T @ center
    model_to_world = np.linalg.inv(world_to_model)
    return world_to_model, model_to_world


def _angular_separation(left: np.ndarray, right: np.ndarray) -> float:
    left = left / max(float(np.linalg.norm(left)), 1.0e-12)
    right = right / max(float(np.linalg.norm(right)), 1.0e-12)
    return float(math.acos(float(np.clip(np.dot(left, right), -1.0, 1.0))))


def select_diverse_views(candidates: Sequence[Mapping[str, Any]], maximum: int) -> list[Mapping[str, Any]]:
    """Greedily keep strong views while rewarding new time/sensor/family/baseline."""

    remaining = sorted(
        candidates,
        key=lambda row: (-int(row["visible_count"]), int(row["image_id"])),
    )
    selected: list[Mapping[str, Any]] = []
    maximum = max(0, int(maximum))
    while remaining and len(selected) < maximum:
        if not selected:
            winner = remaining[0]
        else:
            seen_frames = {str(row["timestamp_ns"]) for row in selected}
            seen_sensors = {str(row["sensor"]) for row in selected}
            seen_families = {str(row["family"]) for row in selected}
            maximum_visible = max(int(row["visible_count"]) for row in remaining)

            def score(row: Mapping[str, Any]) -> tuple[float, int]:
                quality = math.log1p(int(row["visible_count"])) / math.log1p(maximum_visible)
                direction = np.asarray(row["view_direction"], dtype=np.float64)
                angle = min(
                    _angular_separation(direction, np.asarray(other["view_direction"], dtype=np.float64))
                    for other in selected
                ) / math.pi
                novelty = (
                    0.32 * (str(row["timestamp_ns"]) not in seen_frames)
                    + 0.12 * (str(row["sensor"]) not in seen_sensors)
                    + 0.10 * (str(row["family"]) not in seen_families)
                )
                return 0.58 * quality + 0.42 * angle + novelty, -int(row["image_id"])

            winner = max(remaining, key=score)
        selected.append(winner)
        remaining.remove(winner)
    return selected


def _validate_rgb(path: Path, expected_shape: tuple[int, int]) -> bytes:
    payload = path.read_bytes()
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.shape[:2] != tuple(expected_shape):
        raise ValueError(f"RGB dimensions differ from the native FARM frame: {path}")
    return payload


def _build_views(
    *,
    run,
    obj,
    points_world: np.ndarray,
    points_model: np.ndarray,
    point_radius_m: np.ndarray,
    model_to_world: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    view_config = config["views"]
    candidates_by_image: dict[int, dict[str, Any]] = {}
    rejection = Counter()
    mask_digests: list[dict[str, Any]] = []
    absolute_tolerance = float(view_config["depth_absolute_tolerance_m"])
    relative_tolerance = float(view_config["depth_relative_tolerance"])
    radius_multiplier = float(view_config["radius_tolerance_multiplier"])
    minimum_visible = int(view_config["minimum_visible_points"])

    for observation in obj.observations:
        frame = run.frame(observation.image_id)
        mask_path = resolve_mask_path(run, observation)
        raw_mask, _, mask_sha = load_mask_pair(mask_path, frame.depth_size)
        mask_digests.append({
            "image_id": frame.image_id,
            "path": mask_path.name,
            "sha256": mask_sha,
        })
        depth = np.load(frame.depth_path, mmap_mode="r")
        if depth.shape != frame.depth_size or depth.dtype != np.dtype("float32"):
            raise ValueError(f"native FARM depth contract mismatch: {frame.depth_path}")
        u, v, z = project_metric(points_world, frame.T_world_cam, frame.K)
        ui, vi = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
        inside = (
            np.isfinite(z) & (z > 0)
            & (ui >= 0) & (vi >= 0)
            & (ui < frame.depth_size[1]) & (vi < frame.depth_size[0])
        )
        selected = np.flatnonzero(inside)
        if not len(selected):
            rejection["outside_frame"] += 1
            continue
        rendered = np.asarray(depth[vi[selected], ui[selected]], dtype=np.float32)
        tolerance = np.maximum(
            absolute_tolerance + relative_tolerance * rendered,
            radius_multiplier * point_radius_m[selected],
        )
        surface = (
            np.isfinite(rendered) & (rendered > 0)
            & (np.abs(z[selected] - rendered) <= tolerance)
        )
        selected = selected[surface]
        if not len(selected):
            rejection["depth_inconsistent"] += 1
            continue
        selected = selected[raw_mask[vi[selected], ui[selected]]]
        if len(selected) < minimum_visible:
            rejection["insufficient_mask_points"] += 1
            continue
        uv = np.column_stack((u[selected], v[selected])).astype(np.float32)
        camera_from_world = np.linalg.inv(frame.T_world_cam)
        camera_from_model = camera_from_world @ model_to_world
        reconstructed = (
            points_model[selected] @ model_to_world[:3, :3].T + model_to_world[:3, 3]
        )
        check_u, check_v, _ = project_metric(reconstructed, frame.T_world_cam, frame.K)
        residual = float(np.max(np.linalg.norm(
            np.column_stack((check_u, check_v)) - uv, axis=1
        )))
        if residual > float(view_config["maximum_projection_residual_px"]):
            raise RuntimeError(f"world/model projection round-trip failed for image {frame.image_id}")
        x0, y0 = np.min(uv, axis=0)
        x1, y1 = np.max(uv, axis=0)
        occupancy = float(
            max(0.0, x1 - x0) * max(0.0, y1 - y0)
            / (frame.depth_size[0] * frame.depth_size[1])
        )
        center = np.asarray(frame.T_world_cam[:3, 3], dtype=np.float64)
        candidate = {
            "image_id": frame.image_id,
            "frame_id": frame.frame_id,
            "timestamp_ns": frame.timestamp_ns,
            "sensor": frame.sensor,
            "camera": frame.camera,
            "family": frame.family,
            "visible_count": int(len(selected)),
            "occupancy": occupancy,
            "projection_residual_px": residual,
            "camera_center_m": center,
            "view_direction": np.asarray(obj.center_m, dtype=np.float64) - center,
            "uv": uv,
            "visible_points_model": points_model[selected].astype(np.float32),
            "K": np.asarray(frame.K, dtype=np.float32),
            "T_camera_model": camera_from_model.astype(np.float32),
            "rgb_path": frame.rgb_path,
        }
        previous = candidates_by_image.get(frame.image_id)
        if previous is None or candidate["visible_count"] > previous["visible_count"]:
            candidates_by_image[frame.image_id] = candidate

    selected = select_diverse_views(
        list(candidates_by_image.values()), int(view_config["maximum_views"])
    )
    selected.sort(key=lambda row: int(row["image_id"]))
    centres = [np.asarray(row["camera_center_m"]) for row in selected]
    diversity = diversity_metrics(centres, obj.center_m) if centres else {
        "baseline_span_m": 0.0, "angular_span_degrees": 0.0
    }
    distinct_timestamps = len({str(row["timestamp_ns"]) for row in selected})
    if len(selected) < int(view_config["minimum_valid_views"]):
        raise ViewQualityError("fewer_than_minimum_valid_views")
    if distinct_timestamps < int(view_config["minimum_distinct_timestamps"]):
        raise ViewQualityError("fewer_than_minimum_distinct_timestamps")
    if diversity["baseline_span_m"] < float(view_config["minimum_baseline_m"]):
        raise ViewQualityError("insufficient_camera_baseline")
    if diversity["angular_span_degrees"] < float(view_config["minimum_angular_span_degrees"]):
        raise ViewQualityError("insufficient_camera_angular_span")

    encoded_cache: dict[Path, bytes] = {}
    payload_views: list[dict[str, Any]] = []
    for row in selected:
        rgb_path = Path(row["rgb_path"])
        if rgb_path not in encoded_cache:
            encoded_cache[rgb_path] = _validate_rgb(
                rgb_path, run.frame(int(row["image_id"])).depth_size
            )
        image_data = encoded_cache[rgb_path]
        K4 = np.eye(4, dtype=np.float32)
        K4[:3, :3] = row["K"]
        payload_views.append({
            **row,
            "image_data": image_data,
            "K4": K4,
        })
    audit = {
        "observations_evaluated": len(obj.observations),
        "unique_valid_images": len(candidates_by_image),
        "selected_views": len(payload_views),
        "distinct_timestamps": distinct_timestamps,
        "sensors": sorted({str(row["sensor"]) for row in payload_views}),
        "families": sorted({str(row["family"]) for row in payload_views}),
        "diversity": diversity,
        "rejections": dict(sorted(rejection.items())),
        "mask_artifacts": mask_digests,
    }
    return payload_views, audit


def _build_object(
    *,
    run,
    table,
    obj,
    source_indices: np.ndarray,
    confidence: np.ndarray,
    timestamp_support: np.ndarray,
    lift_sha256: str,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    shape_config = config["shape"]
    report: dict[str, Any] = {
        "object_id": obj.object_id,
        "category": obj.category,
        "verified_source_gaussians": int(len(source_indices)),
    }
    if len(source_indices) < int(shape_config["minimum_verified_gaussians"]):
        report.update(status="SKIP", reason="insufficient_verified_gaussians")
        return None, report

    points_world_all = np.column_stack((
        table.data["x"][source_indices],
        table.data["y"][source_indices],
        table.data["z"][source_indices],
    )).astype(np.float64)
    points_world_all *= run.meters_per_scene_unit
    radii_all = np.exp(np.column_stack((
        table.data["scale_0"][source_indices],
        table.data["scale_1"][source_indices],
        table.data["scale_2"][source_indices],
    )).astype(np.float32)).max(axis=1) * run.meters_per_scene_unit
    world_to_model, model_to_world = object_transforms(obj.center_m, obj.rotation)
    points_model_all = (
        points_world_all @ world_to_model[:3, :3].T + world_to_model[:3, 3]
    )
    chosen = _spatial_sample(
        points_model_all,
        confidence,
        timestamp_support,
        int(shape_config["maximum_points"]),
    )
    points_world = points_world_all[chosen]
    points_model = points_model_all[chosen]
    radius_m = radii_all[chosen].astype(np.float32)
    if not np.isfinite(points_model).all() or len(points_model) < int(shape_config["minimum_verified_gaussians"]):
        report.update(status="SKIP", reason="invalid_sampled_points")
        return None, report
    robust_quantile = float(shape_config["bounds_absolute_quantile"])
    bounds = np.maximum(
        np.quantile(np.abs(points_model), robust_quantile, axis=0)
        * float(shape_config["bounds_padding"]),
        float(shape_config["minimum_bound_m"]),
    )
    roundtrip = points_model @ model_to_world[:3, :3].T + model_to_world[:3, 3]
    transform_error = float(np.max(np.abs(roundtrip - points_world)))
    if transform_error > float(shape_config["maximum_transform_error_m"]):
        report.update(status="SKIP", reason="world_model_transform_mismatch")
        return None, report
    try:
        views, view_audit = _build_views(
            run=run,
            obj=obj,
            points_world=points_world,
            points_model=points_model,
            point_radius_m=radius_m,
            model_to_world=model_to_world,
            config=config,
        )
    except ViewQualityError as exc:
        report.update(status="SKIP", reason=str(exc))
        return None, report

    category = str(obj.category or "object").strip()
    description = str(obj.description or category).strip()
    caption = f"{category}. {description}" if description.lower() != category.lower() else category
    camera_model = torch.from_numpy(np.stack([row["T_camera_model"] for row in views]))
    camera_params = torch.from_numpy(np.stack([row["K4"] for row in views]))
    projections = [torch.from_numpy(row["uv"]) for row in views]
    visible_points = [torch.from_numpy(row["visible_points_model"]) for row in views]
    image_data = [row["image_data"] for row in views]
    payload: dict[str, Any] = {
        "schema_version": "farm-shaper-bridge.pkl.v2",
        "already_rectified_pinhole": True,
        "is_ariagen2": False,
        "scene_id": run.scene_id,
        "category": category,
        "caption": caption,
        "points_model": torch.from_numpy(points_model.astype(np.float32)),
        "bounds": torch.from_numpy(bounds.astype(np.float32)),
        "inv_dist_std": torch.zeros(len(points_model), dtype=torch.float32),
        "dist_std": torch.zeros(len(points_model), dtype=torch.float32),
        "T_model_world": torch.from_numpy(world_to_model.astype(np.float32)),
        "T_zup_obj": torch.from_numpy(world_to_model.astype(np.float32)),
        "Ts_camera_model": camera_model,
        "Ts_rgbCamera_model": camera_model.clone(),
        "camera_params": camera_params,
        "rgb_camera_params": camera_params.clone(),
        "object_point_projections": projections,
        "rgb_object_point_projections": [value.clone() for value in projections],
        "visible_points_model": visible_points,
        "rgb_visible_points_model": [value.clone() for value in visible_points],
        "image_data": image_data,
        "rgb_image_data": list(image_data),
        "farm_object_id": obj.object_id,
        "farm_source_gaussian_indices": torch.from_numpy(source_indices[chosen].astype(np.int64)),
        "farm_point_confidence": torch.from_numpy(confidence[chosen].astype(np.float32)),
        "farm_timestamp_support": torch.from_numpy(timestamp_support[chosen].astype(np.int32)),
        "farm_verified_instance_bank_sha256": lift_sha256,
        "farm_selected_view_ids": [int(row["image_id"]) for row in views],
        "coordinate_units": "metres",
    }
    normalized = points_model * (0.9 / float(np.max(bounds)))
    report.update(
        status="PASS",
        sampled_points=int(len(points_model)),
        bounds_m=bounds.tolist(),
        normalized_point_retention=float(np.mean(np.all(np.abs(normalized) <= 1.0, axis=1))),
        transform_roundtrip_max_error_m=transform_error,
        confidence={
            "minimum": float(np.min(confidence[chosen])),
            "median": float(np.median(confidence[chosen])),
            "maximum": float(np.max(confidence[chosen])),
        },
        timestamp_support={
            "minimum": int(np.min(timestamp_support[chosen])),
            "median": float(np.median(timestamp_support[chosen])),
            "maximum": int(np.max(timestamp_support[chosen])),
        },
        views=view_audit,
        visible_points_per_view={
            "minimum": min(int(row["visible_count"]) for row in views),
            "median": float(np.median([row["visible_count"] for row in views])),
            "maximum": max(int(row["visible_count"]) for row in views),
        },
        maximum_projection_residual_px=max(float(row["projection_residual_px"]) for row in views),
    )
    return payload, report


def execute(args: argparse.Namespace) -> dict[str, Any] | None:
    started = time.perf_counter()
    project_root = Path(__file__).resolve().parents[2]
    run = load_run(args.run, allow_legacy=bool(args.allow_legacy_run))
    authority = resolve_farm_source_authority(
        run.run_dir,
        project_root,
        required_files=SOURCE_FILES,
        allow_unsigned=bool(args.allow_unsigned_source),
        allow_incomplete_snapshot_fallback=bool(
            run.legacy and args.allow_legacy_run and args.allow_unsigned_source
        ),
    )
    require_current_sources_match_authority(authority, project_root, SOURCE_FILES)
    if authority.fallback is not None:
        print(
            "WARNING: using current versioned FARM source for an explicitly "
            "unsigned/nonrelease legacy run because its signed snapshot predates the bridge",
            file=sys.stderr,
            flush=True,
        )
    requested_config = args.config.expanduser().resolve(strict=True)
    authoritative_config = authority.project_root / "configs" / "shaper_bridge.v1.yaml"
    if sha256_file(requested_config) != sha256_file(authoritative_config):
        raise ValueError("--config differs from the selected FARM source authority")
    config, config_sha = load_shaper_config(authoritative_config)
    table = open_graphdeco_ply(args.ply)
    preflight = verify_run_ply_fingerprint(run, table.path)
    source_sha = sha256_file(table.path)
    lift = bind_verified_lift(
        args.lift,
        run=run,
        source_ply_path=table.path,
        source_gaussian_count=table.count,
        source_ply_sha256=source_sha,
        allow_nonrelease=bool(args.allow_nonrelease_lift),
    )
    object_by_id = {obj.object_id: obj for obj in run.objects}
    missing = sorted(set(lift.bank.object_ids.tolist()) - set(object_by_id))
    if missing:
        raise ValueError(f"verified lift IDs absent from final FARM presentation objects: {missing}")

    plan = {
        "schema_version": "farm.shaper-inputs.plan.v1",
        "scene_id": run.scene_id,
        "source_gaussians": table.count,
        "verified_objects": len(lift.bank.object_ids),
        "verified_gaussians": len(lift.bank.indices),
        "source_ply_sha256": source_sha,
        "lift_result_sha256": lift.result_sha256,
        "verified_instance_bank_sha256": lift.bank.sha256,
        "signed_farm_source": authority.signed,
        "farm_source_tree_sha256": authority.tree_sha256,
        "source_authority_fallback": authority.fallback,
        "config_sha256": config_sha,
    }
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return None

    output = args.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing non-empty ShapeR input output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    scene_slug = safe_name(run.scene_id)
    for bank_row in range(len(lift.bank.object_ids)):
        object_id, indices, confidence, support = lift.bank.object_slice(bank_row)
        obj = object_by_id[object_id]
        payload, report = _build_object(
            run=run,
            table=table,
            obj=obj,
            source_indices=indices,
            confidence=confidence,
            timestamp_support=support,
            lift_sha256=lift.bank.sha256,
            config=config,
        )
        if payload is not None:
            filename = f"{scene_slug}__{object_id:06d}__{safe_name(obj.category, fallback='object')}.pkl"
            path = output / filename
            atomic_pickle(path, payload)
            report["artifact"] = artifact_record(path, root=output)
        rows.append(report)
        print(
            f"[{bank_row + 1:03d}/{len(lift.bank.object_ids):03d}] "
            f"#{object_id:06d} {obj.category}: {report['status']} "
            f"points={report.get('sampled_points', 0)}",
            flush=True,
        )

    passed = sum(row["status"] == "PASS" for row in rows)
    if passed == 0:
        atomic_json(output / "_FAILED.json", {
            "schema_version": "farm.shaper-inputs.failure.v1",
            "status": "failed",
            "scene_id": run.scene_id,
            "reason": "no_verified_object_passed_shaper_input_gates",
            "rows": rows,
        })
        raise RuntimeError("no verified object passed ShapeR input gates")
    release_eligible = bool(authority.signed and lift.release_eligible and not run.legacy)
    result = {
        "schema_version": SHAPER_INPUT_SCHEMA,
        "status": "PASS",
        "quality_status": "PASS" if passed == len(rows) else "WARN",
        "release_eligible": release_eligible,
        "scene_id": run.scene_id,
        "created_utc": utc_now(),
        "inputs": {
            "farm_run": str(run.run_dir),
            "farm_success_sha256": run.success_sha256,
            "farm_acceptance_sha256": run.acceptance_sha256,
            "source_ply": str(table.path),
            "source_ply_sha256": source_sha,
            "source_ply_count": table.count,
            "source_ply_preflight": preflight,
            "gaussian_lift": str(lift.directory),
            "gaussian_lift_result_sha256": lift.result_sha256,
            "verified_instance_bank_sha256": lift.bank.sha256,
            "config_sha256": config_sha,
        },
        "source_authority": {
            "signed": authority.signed,
            "tree_sha256": authority.tree_sha256,
            "project_root": str(authority.project_root),
            "fallback": authority.fallback,
        },
        "contracts": {
            "verified_csr_only": True,
            "dense_label_scan": False,
            "membership_frozen_before_all_view_use": True,
            "source_gaussian_order_preserved": True,
            "source_ply_mutated": False,
            "gaussian_lift_mutated": False,
        },
        "objects": {
            "verified_in_bank": len(rows),
            "prepared": passed,
            "skipped": len(rows) - passed,
            "rows": rows,
        },
        "timing_seconds": {"total": time.perf_counter() - started},
        "limitations": [
            "ShapeR inputs condition a generative model; they are not ground-truth meshes.",
            "Objects without enough verified Gaussians or camera diversity are explicitly skipped.",
        ],
    }
    atomic_json(output / "result.json", result)
    marker_name = "_SUCCESS.json" if release_eligible else "_NONRELEASE_SUCCESS.json"
    atomic_json(output / marker_name, {
        "schema_version": "farm.shaper-inputs.success.v1",
        "status": "success",
        "release_eligible": release_eligible,
        "scene_id": run.scene_id,
        "result": "result.json",
        "result_sha256": sha256_file(output / "result.json"),
        "verified_instance_bank_sha256": lift.bank.sha256,
        "source_ply_sha256": source_sha,
    })
    print(json.dumps({
        "status": result["status"],
        "quality_status": result["quality_status"],
        "release_eligible": release_eligible,
        "prepared_objects": passed,
        "skipped_objects": len(rows) - passed,
        "seconds": result["timing_seconds"]["total"],
    }, indent=2))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        execute(parse_args(argv))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
