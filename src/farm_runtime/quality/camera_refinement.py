"""Refine frozen development cameras against 3DGS while preserving the rig."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image

from farm_runtime.camera_alignment import match_render_to_source, refine_camera_rig
from farm_runtime.colmap_pose_reader import read_colmap_cameras_and_poses
from farm_runtime.quality_baseline import describe_file, write_json
from tools.farm_shaper_bridge import gaussian_lift as lift
from tools.farm_shaper_bridge.common import load_run, open_graphdeco_ply


def apply_camera_refinement(run, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["run_success_sha256"] != run.success_sha256
        or manifest["meters_per_scene_unit"] != run.meters_per_scene_unit
    ):
        raise ValueError(
            "camera refinement belongs to another source run or metric scale"
        )
    if manifest.get("schema") != "farm.camera-refinement.v1":
        raise ValueError("unsupported camera refinement schema")
    rows = {r["source_image"]: r for r in manifest["frames"]}
    if len(rows) != len(manifest["frames"]):
        raise ValueError("duplicate refined camera")
    frames = []
    for frame in run.frames:
        row = rows.get(frame.source_image)
        if row is None:
            frames.append(frame)
            continue
        if not np.allclose(
            frame.T_world_cam, row["original_T_world_cam"], rtol=0, atol=1e-10
        ):
            raise ValueError("camera refinement initial pose changed")
        pose = np.asarray(row["T_world_cam"], float)
        if (
            pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-10)
            or not np.allclose(
                pose[:3, :3].T @ pose[:3, :3], np.eye(3), rtol=0, atol=1e-6
            )
            or np.linalg.det(pose[:3, :3]) < 0.999
        ):
            raise ValueError("refined camera must be a proper rigid pose")
        depth = manifest_path.parent / row["depth"]["path"]
        if describe_file(depth)["sha256"] != row["depth"]["sha256"]:
            raise ValueError("refined depth changed")
        frames.append(
            replace(
                frame,
                T_world_cam=pose,
                depth_path=depth,
            )
        )
    # Failed registration is unknown evidence, not a usable object observation.
    rejected = {r["timestamp"] for r in manifest["corrections"] if not r["accepted"]}
    objects = tuple(
        replace(
            obj,
            observations=tuple(
                o
                for o in obj.observations
                if run.frame(o.image_id).frame_id not in rejected
            ),
        )
        for obj in run.objects
    )
    return replace(run, frames=tuple(frames), objects=objects)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "ply", "split", "packet", "colmap", "images", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument(
        "--reuse",
        type=Path,
        help="Reuse frozen corrections for identical source inputs",
    )
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("output must be new")
    run = load_run(args.run, allow_legacy=True)
    split = json.loads(args.split.read_text())
    packet = json.loads(args.packet.read_text())
    objects = {int(r["object_id"]) for r in split["objects"]}
    timestamps = set(split["build_timestamps"]) | set(split["heldout_timestamps"])
    target_ids = {
        o.image_id
        for obj in run.objects
        if obj.object_id in objects
        for o in obj.observations
        if run.frame(o.image_id).physical_timestamp in timestamps
    }
    by_timestamp = {}
    for i in sorted(target_ids):
        f = run.frame(i)
        if packet["split_by_timestamp"].get(f.frame_id) not in ("train", "dev"):
            raise ValueError("reserved test or unknown timestamp")
        by_timestamp.setdefault(f.frame_id, []).append(f)
    _, cameras, poses, _ = read_colmap_cameras_and_poses(args.colmap)
    contract = run.frames_doc["identity_contract"]
    regex = re.compile(contract["regex"])
    centers = {}
    for pose in poses:
        match = regex.search(pose.name)
        if match and match.group(contract["family_group"]) == "center":
            centers.setdefault(match.group(contract["timestamp_group"]), []).append(
                pose
            )
    table = open_graphdeco_ply(args.ply)
    if describe_file(args.ply)["sha256"] != packet["source_ply"]["sha256"]:
        raise ValueError("source PLY changed")
    reused = {}
    reused_frames = {}
    if args.reuse:
        old = json.loads(args.reuse.read_text())
        if (
            old["run_success_sha256"] != run.success_sha256
            or old["source_ply"]["sha256"] != packet["source_ply"]["sha256"]
        ):
            raise ValueError("reused corrections belong to different inputs")
        reused = {r["timestamp"]: r for r in old["corrections"]}
        reused_frames = {r["source_image"]: r for r in old["frames"]}
    gs = lift.load_gaussians(table, run.meters_per_scene_unit)
    import torch
    from gsplat import rasterization

    dc = np.column_stack([table.data[f"f_dc_{i}"] for i in range(3)]).astype(np.float32)
    rest = (
        np.column_stack([table.data[f"f_rest_{i}"] for i in range(45)])
        .astype(np.float32)
        .reshape(table.count, 3, 15)
        .transpose(0, 2, 1)
    )
    colors = torch.from_numpy(np.concatenate((dc[:, None, :], rest), axis=1)).cuda()
    del dc, rest
    args.output.mkdir(parents=True)
    (args.output / "depth").mkdir()
    corrections, frame_rows = [], []
    started = time.monotonic()

    def render(frame):
        view, K, h, w = lift.camera_tensors(run, frame)
        with torch.inference_mode():
            rgb, alpha, _ = rasterization(
                **lift.raster_kwargs(gs, run, view, K, h, w),
                colors=colors,
                sh_degree=3,
                render_mode="RGB+ED",
            )
        return (
            np.clip(rgb[0, ..., :3].cpu().numpy() * 255, 0, 255).astype(np.uint8),
            rgb[0, ..., 3].cpu().numpy() * run.meters_per_scene_unit,
            alpha[0, ..., 0].cpu().numpy(),
        )

    for rank, (timestamp, targets) in enumerate(sorted(by_timestamp.items())):
        observations = []
        sources = []
        for pose in sorted(centers.get(timestamp, []), key=lambda p: p.name):
            camera = cameras[pose.camera_id]
            if camera.model != "PINHOLE":
                raise ValueError("requires rectified PINHOLE inputs")
            h, w = targets[0].depth_size
            fx, fy, cx, cy = camera.params
            K = np.array(
                [
                    [fx * w / camera.width, 0, cx * w / camera.width],
                    [0, fy * h / camera.height, cy * h / camera.height],
                    [0, 0, 1.0],
                ]
            )
            w2c = np.eye(4)
            w2c[:3] = pose.camera_from_world
            c2w = np.linalg.inv(w2c)
            c2w[:3, 3] *= run.meters_per_scene_unit
            temporary = replace(
                targets[0], K=K, T_world_cam=c2w, source_image=pose.name
            )
            source_path = args.images / pose.name
            source = np.array(
                Image.open(source_path)
                .convert("RGB")
                .resize((w, h), Image.Resampling.BILINEAR)
            )
            if timestamp in reused:
                observations.append(dict(T_world_cam=c2w))
                continue
            rendered, depth, alpha = render(temporary)
            points, pixels, audit = match_render_to_source(
                rendered, source, depth, alpha, K, c2w
            )
            observations.append(
                dict(
                    points_world_m=points,
                    pixels=pixels,
                    K=K,
                    T_world_cam=c2w,
                    shape_hw=(h, w),
                )
            )
            sources.append({"source_image": describe_file(source_path), **audit})
        if not observations:
            corrections.append(
                {
                    "timestamp": timestamp,
                    "accepted": False,
                    "reason": "no_center_cameras",
                }
            )
            continue
        if timestamp in reused:
            audit = dict(reused[timestamp])
            sources = audit.pop("sources", [])
            # Revalidate RGB bindings before trusting cached photometric evidence.
            for source in sources:
                old_source = source["source_image"]
                current = describe_file(args.images / Path(old_source["path"]).name)
                if current["sha256"] != old_source["sha256"]:
                    raise ValueError("reused registration RGB changed")
            delta_anchor = np.asarray(audit.get("delta_anchor", np.eye(4)))
            refined = [observations[0]["T_world_cam"] @ delta_anchor]
            audit["reused"] = True
        else:
            refined, audit = refine_camera_rig(observations)
        if audit["accepted"]:
            delta = refined[0] @ np.linalg.inv(observations[0]["T_world_cam"])
            for frame in targets:
                if frame.source_image in reused_frames:
                    original_row = reused_frames[frame.source_image]
                    old_depth = args.reuse.parent / original_row["depth"]["path"]
                    if (
                        describe_file(old_depth)["sha256"]
                        != original_row["depth"]["sha256"]
                    ):
                        raise ValueError("reused depth changed")
                    link = args.output / "depth" / old_depth.name
                    import os

                    link.symlink_to(os.path.relpath(old_depth, link.parent))
                    row = dict(original_row)
                    row["depth"] = {
                        **original_row["depth"],
                        "path": str(link.relative_to(args.output)),
                    }
                    frame_rows.append(row)
                    continue
                corrected = replace(frame, T_world_cam=delta @ frame.T_world_cam)
                _, depth, alpha = render(corrected)
                valid = (
                    np.isfinite(depth)
                    & (depth >= run.rgbd_config["depth_min_m"])
                    & (depth <= run.rgbd_config["depth_max_m"])
                    & (alpha >= run.rgbd_config["alpha_min"])
                )
                depth = np.where(valid, depth, 0.0).astype(np.float32)
                path = args.output / "depth" / (Path(frame.source_image).stem + ".npy")
                np.save(path, depth, allow_pickle=False)
                descriptor = describe_file(path)
                descriptor["path"] = str(path.relative_to(args.output))
                frame_rows.append(
                    {
                        "source_image": frame.source_image,
                        "original_T_world_cam": frame.T_world_cam.tolist(),
                        "T_world_cam": corrected.T_world_cam.tolist(),
                        "depth": descriptor,
                    }
                )
        corrections.append({"timestamp": timestamp, "sources": sources, **audit})
        print(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "rank": rank + 1,
                    "total": len(by_timestamp),
                    "accepted": audit["accepted"],
                    "before_px": audit.get("validation_before_px"),
                    "after_px": audit.get("validation_after_px"),
                }
            ),
            flush=True,
        )
    write_json(
        args.output / "manifest.json",
        {
            "schema": "farm.camera-refinement.v1",
            "run_success_sha256": run.success_sha256,
            "meters_per_scene_unit": run.meters_per_scene_unit,
            "source_ply": describe_file(args.ply),
            "split": describe_file(args.split),
            "packet": describe_file(args.packet),
            "corrections": corrections,
            "frames": frame_rows,
            "seconds": time.monotonic() - started,
            "rig_baseline_preserved": True,
            "object_masks_used_for_pose_fit": False,
            "reserved_test_opened": False,
            "scope": "development mask observations; frozen input COLMAP and PLY unchanged",
            "release_eligible": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
