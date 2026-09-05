#!/usr/bin/env python3
"""Prepare an F1 pilot from original COLMAP RGB; no predicted masks become gold."""
from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import resource
import time

import numpy as np

from farm_runtime.colmap_pose_reader import read_colmap_cameras_and_poses
from farm_runtime.quality_baseline import describe_file, json_digest, write_json
from farm_runtime.quality_benchmark import (
    DIRECTIONS, annotation_template, assign_splits, exposure_ledger, image_identity, verify_packet,
)


def select_timestamps(rows: list[dict], count: int) -> list[dict]:
    """Greedy spatially diverse navigation views; uses camera geometry only."""
    selected = []
    pool = list(rows)
    while pool and len(selected) < count:
        def score(r):
            if not selected:
                return r["navigation_score"]
            c = np.asarray(r["camera_center_m"])
            separation = min(float(np.linalg.norm(c - np.asarray(x["camera_center_m"]))) for x in selected)
            return r["navigation_score"] * (0.2 + min(separation, 2.0))
        best = max(pool, key=lambda r: (score(r), r["name"]))
        selected.append(best)
        pool = [r for r in pool if r["physical_timestamp"] != best["physical_timestamp"]]
    return selected


def build(args) -> dict:
    started = time.monotonic()
    if args.output.exists():
        raise ValueError("output must be a new directory")
    if not math.isfinite(args.meters_per_scene_unit) or args.meters_per_scene_unit <= 0:
        raise ValueError("explicit calibrated meters_per_scene_unit required")
    source_format, cameras, poses, model_files = read_colmap_cameras_and_poses(args.colmap_model)
    catalog = json.loads(args.catalog.read_text())
    catalog = catalog if isinstance(catalog, list) else catalog["objects"]
    objects = {int(r["id"]): r for r in catalog}
    requested = [int(x) for x in args.object_ids.split(",")]
    if len(set(requested)) != len(requested) or set(requested) - objects.keys():
        raise ValueError("pilot IDs must be unique and present in the navigation catalog")
    ledger = exposure_ledger(args.history_roots, args.output)
    image_groups = defaultdict(list)
    for pose in poses:
        identity = image_identity(pose.name)
        image_groups[identity["timestamp"]].append((pose, identity))
    splits = assign_splits(sorted(image_groups), set(ledger["observed_timestamps"]), seed=args.seed, test_fraction=args.test_fraction)
    if not all(role in splits.values() for role in ("train", "dev", "test")):
        raise ValueError("no unexposed test pool: use a new scene; never recycle viewed test images")
    support_by_object = {}
    track_counts = {}
    support_diagnostics = {}
    if args.support_state:
        import pycolmap
        from scripts.geometry.plan_farm_full_colmap_rescue import load_state, object_support_points
        from farm_runtime.full_colmap_view_planner import build_sparse_track_index, match_support_to_sparse_tracks, project_metric_support
        state = load_state(args.support_state)
        state_ids = np.asarray(state["object_id"], dtype=np.int64).reshape(-1)
        reconstruction = pycolmap.Reconstruction(str(args.colmap_model))
        sparse = build_sparse_track_index(reconstruction, meters_per_scene_unit=args.meters_per_scene_unit)
        for obj in requested:
            indexes = np.flatnonzero(state_ids == obj)
            if len(indexes) != 1:
                raise ValueError(f"missing unique voxel support for {obj}")
            support, diagnostics = object_support_points(state, int(indexes[0]), trim_quantile=0.995, max_points=2000)
            if len(support) < 4:
                raise ValueError(f"missing voxel support for {obj}; no OBB fallback")
            counts, match = match_support_to_sparse_tracks(support, sparse, maximum_distance_m=0.08)
            support_by_object[obj] = support
            track_counts[obj] = counts
            support_diagnostics[obj] = {"support": diagnostics, "match": match}
    observations = []
    navigation = []
    deficits = []
    image_descriptors = {}
    for obj in requested:
        center = np.asarray(objects[obj]["center_m"], dtype=float)
        dimensions = np.asarray(objects[obj]["dimensions_m"], dtype=float)
        candidates = defaultdict(list)
        for ts, members in sorted(image_groups.items()):
            best = []
            for pose, ident in members:
                cam = cameras[pose.camera_id]
                if cam.model != "PINHOLE":
                    raise ValueError("pilot requires undistorted PINHOLE source images")
                rotation = pose.camera_from_world[:, :3]
                translation = pose.camera_from_world[:, 3] * args.meters_per_scene_unit
                point = rotation @ center + translation
                if point[2] <= 0.10:
                    continue
                fx, fy, cx, cy = cam.params
                u, v = fx * point[0] / point[2] + cx, fy * point[1] / point[2] + cy
                if not (0.05 * cam.width < u < 0.95 * cam.width and 0.05 * cam.height < v < 0.95 * cam.height):
                    continue
                diameter_px = fx * float(np.linalg.norm(dimensions)) / point[2]
                if not 48 <= diameter_px <= 2 * max(cam.width, cam.height):
                    continue
                camera_center = -rotation.T @ translation
                nav_score = min(diameter_px, 1200) / 1200
                evidence = {}
                if args.support_state:
                    count = track_counts[obj].get(pose.image_id, 0)
                    if count < 2:
                        continue
                    projected = project_metric_support(
                        support_by_object[obj], world_to_camera=pose.camera_from_world,
                        intrinsics=np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                        width=cam.width, height=cam.height, meters_per_scene_unit=args.meters_per_scene_unit,
                        min_margin_ratio=0.015, min_area_ratio=0.0005, max_area_ratio=0.70,
                        min_in_frame_ratio=0.80, bbox_quantile=0.01)
                    if projected is None:
                        continue
                    x0, y0, x1, y1 = projected["bbox_xyxy"]
                    u, v = (x0+x1)/2, (y0+y1)/2
                    diameter_px = max(x1-x0, y1-y0)
                    nav_score = projected["geometric_score"] * min(1.0, count/10)
                    evidence = {"support_bbox_xyxy": projected["bbox_xyxy"], "track_count": count,
                                "in_frame_support_ratio": projected["in_frame_support_ratio"]}
                best.append({"name": pose.name, "physical_timestamp": ts, "object_id": obj,
                             "center_px": [u, v], "diameter_px": diameter_px,
                             "camera_center_m": camera_center.tolist(),
                             "navigation_score": nav_score, **evidence})
            if best:
                candidates[splits[ts]].append(max(best, key=lambda r: (r["navigation_score"], r["name"])))
        chosen = []
        for split in ("train", "dev", "test"):
            picked = select_timestamps(candidates[split], args.timestamps_per_split)
            if len(picked) < args.timestamps_per_split:
                deficits.append({"object_id": obj, "split": split, "available": len(picked),
                                 "required": args.timestamps_per_split})
            chosen.extend(picked)
        for nav in chosen:
            ts = nav["physical_timestamp"]
            navigation.append({**nav, "split": splits[ts], "is_ground_truth": False})
            # Include every registered lens and direction, even when the center
            # does not project into it. A human decides visibility/absence.
            for pose, ident in sorted(image_groups[ts], key=lambda p: p[0].name):
                cam = cameras[pose.camera_id]
                if pose.name not in image_descriptors:
                    image_descriptors[pose.name] = describe_file(args.image_root / pose.name)
                fx, fy, cx, cy = cam.params
                observations.append({
                    "observation_id": f"object_{obj:06d}__{pose.name}", "object_id": obj,
                    "image_name": pose.name, "source_image": image_descriptors[pose.name],
                    "physical_timestamp": ts, "camera": ident["camera"], "direction": ident["direction"],
                    "split": splits[ts], "shape_hw": [cam.height, cam.width],
                    "colmap_image_id": pose.image_id, "colmap_camera_id": pose.camera_id,
                    "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                    "camera_from_world_scene_units": pose.camera_from_world.tolist(),
                    "annotation_status": "UNANNOTATED", "negative_evidence": False,
                })
    packet = {
        "schema": "farm.manual-gold-packet.v1", "scene_id": args.scene_id,
        "status": "PENDING_TWO_INDEPENDENT_HUMAN_REVIEWS", "gold": False,
        "objects": [{"object_id": i, "navigation_center_m": objects[i]["center_m"],
                     "navigation_dimensions_m": objects[i]["dimensions_m"],
                     "navigation_is_ground_truth": False} for i in requested],
        "observations": observations, "split_by_timestamp": splits,
        "historically_observed_timestamps": ledger["observed_timestamps"],
        "exposure_ledger_sha256": json_digest(ledger),
        "test_status": "RESERVED_NOT_DECODED_BY_PACKET_BUILDER",
        "test_independence_limitations": ledger["limitation"],
        "selection": {"seed": args.seed, "timestamps_per_split": args.timestamps_per_split, "test_fraction": args.test_fraction,
                      "policy": "voxel_support_tracks_navigation_all_lenses_and_directions" if args.support_state else "geometry_only_navigation",
                      "support_state": describe_file(args.support_state) if args.support_state else None,
                      "support_diagnostics": support_diagnostics,
                      "visibility_requires_human_review": True, "deficits": deficits},
        "source_ply": describe_file(args.source_ply),
        "colmap_model": [describe_file(p) for p in sorted(args.colmap_model.iterdir()) if p.is_file()],
        "navigation_catalog": describe_file(args.catalog),
        "meters_per_scene_unit": args.meters_per_scene_unit,
        "scale_evidence": args.scale_evidence,
        "coordinate_contract": "COLMAP world-to-camera R,t; K/original pixels; geometry in scene units; metric scale explicit",
    }
    errors = verify_packet(packet)
    if errors:
        raise ValueError(errors)
    args.output.mkdir(parents=True)
    write_json(args.output / "packet.json", packet)
    write_json(args.output / "exposure_ledger.json", ledger)
    write_json(args.output / "navigation.json", {"is_ground_truth": False, "views": navigation})
    write_json(args.output / "cohort_expansion.json", {
        "status": "NEEDS_INDEPENDENT_SCENE_WIDE_DISCOVERY_ANNOTATION", "gold": False,
        "limitation": "Existing IDs alone cannot measure recall of missed objects. Add objects absent from this catalog.",
        "objects": [{"object_id": i, "pilot": i in requested, "strata": [], "human_reviewed": False,
                     "navigation_center_m": objects[i]["center_m"]} for i in sorted(objects)],
    })
    for split in ("train", "dev", "test"):
        write_json(args.output / "annotation_templates" / f"{split}.json", annotation_template(packet, split))
    report = {"schema": "farm.gold-preparation-report.v1", "packet_sha256": json_digest(packet),
              "status": packet["status"], "f1_gate": "BLOCKED", "algorithm_changes_allowed": False,
              "objects": len(requested), "observations": len(observations),
              "unique_source_images": len(image_descriptors),
              "physical_timestamps_by_split": dict(Counter(splits.values())),
              "observations_by_split": dict(Counter(r["split"] for r in observations)),
              "direction_counts": dict(Counter(r["direction"] for r in observations)),
              "deficits": deficits, "wall_seconds": time.monotonic() - started,
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "gpu_used": False, "decoded_test_pixels": 0}
    write_json(args.output / "preparation_report.json", report)
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-id", required=True)
    p.add_argument("--colmap-model", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--source-ply", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--history-roots", nargs="+", type=Path, required=True)
    p.add_argument("--meters-per-scene-unit", type=float, required=True)
    p.add_argument("--scale-evidence", required=True)
    p.add_argument("--support-state", type=Path)
    p.add_argument("--test-fraction", type=float, default=0.20)
    p.add_argument("--object-ids", default="37,27,2,108,0,47")
    p.add_argument("--timestamps-per-split", type=int, default=2)
    p.add_argument("--seed", default="farm-v12-manual-gold-2026-09-04-v1")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.timestamps_per_split < 1:
        p.error("timestamps-per-split must be positive")
    print(json.dumps(build(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
