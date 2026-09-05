"""Evaluate official Boxer OBB proposals on fixed registered FARM observations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from farm_runtime.obb_proposals import (
    z_up_rotation,
    depth_world_points,
    project_world,
    rotate_pinhole_camera,
)
from farm_runtime.angular_discovery import rotate_image
from farm_runtime.proposal_geometry import stable_depth
from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.proposal_geometry import read_masks, read_observations
from farm_runtime.quality_baseline import describe_file, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scope-evidence", "boxer-repo", "model-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--world-up", type=float, nargs=3, required=True)
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument(
        "--upright",
        action="store_true",
        help="Apply declared image quarter-turn with its exact camera transform",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("output must be new")
    started = time.monotonic()
    scope = json.loads(args.scope_evidence.read_text())
    if scope.get("reserved_test_opened") is not False:
        raise ValueError("development-only scope evidence required")
    geometry = json.loads(checked_file(scope["source_groups"]).read_text())
    if geometry.get("test_opened") is not False:
        raise ValueError("development-only geometry required")
    proposals_path = checked_file(geometry["inputs"]["proposals"])
    _, observations = read_observations(proposals_path)
    transient_path = checked_file(geometry["inputs"]["transients"])
    _, transients = read_observations(transient_path)
    observations = {r["name"]: r for r in observations}
    transients = {r["name"]: r for r in transients}
    frames_path = checked_file(geometry["inputs"]["frames"])
    frames = json.loads(frames_path.read_text())
    if (
        frames.get("depth_units") != "metres"
        or frames.get("pose_translation_units") != "metres"
    ):
        raise ValueError("metric registered RGBD required")
    trusted = {
        name
        for g in frames["camera_registration"]["groups"]
        if g["trusted"]
        for name in g["source_images"]
    }
    frames = {f["source_image"]: f for f in frames["frames"]}
    depth_descriptors = {f["source"]: f["depth_artifact"] for f in geometry["frames"]}
    selected = {n: g["id"] for g in scope["groups"] for n in g["members"]}
    by_frame = defaultdict(list)
    for node in geometry["nodes"]:
        if node["id"] in selected:
            by_frame[node["frame"]].append(node)
    names = sorted(by_frame)
    if args.limit_frames:
        names = names[: args.limit_frames]
    if not names or any(name not in trusted for name in names):
        raise ValueError("trusted registered camera required for every query")
    G = z_up_rotation(args.world_up)
    # External source remains a separate, pinned research dependency. Do not
    # vendor its CC-BY-NC code into FARM or import GUI / OWL / dataset loaders.
    sys.path.insert(0, str(args.boxer_repo.resolve()))
    import torch
    import boxernet.dinov3_wrapper as dino
    from boxernet.boxernet import BoxerNet, smart_load
    from loaders.base_loader import BaseLoader
    from utils.tw.pose import PoseTW

    checkpoint = args.model_dir / "boxernet_hw960in2x6d768-c88128f8.ckpt"
    dino.CKPT_PATH = str(args.model_dir)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    before = time.monotonic()
    model = BoxerNet(raw["cfg"]["model"])
    state = smart_load(model.state_dict(), raw["model"])
    missing = set(model.state_dict()) - set(state)
    if any(not key.startswith("dino.") for key in missing):
        raise ValueError("Boxer head weights did not load completely")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError("unexpected Boxer checkpoint tensors")
    model.eval().cuda()
    model.hw = raw["cfg"]["dataset"]["image_hw"]
    model.device = "cuda"
    del raw, state
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - before
    args.output.mkdir(parents=True)
    (args.output / "visuals").mkdir()
    image_size = int(model.hw) if isinstance(model.hw, int) else int(model.hw[0])
    if image_size % 16:
        raise ValueError("Boxer input must be a multiple of its patch size")
    torch.cuda.reset_peak_memory_stats()
    rows, view_rows = [], []
    for name in names:
        tick = time.monotonic()
        obs, frame = observations[name], frames[name]
        masks = read_masks(obs, proposals_path.parent)
        depth_path = checked_file(depth_descriptors[name])
        depth = np.load(depth_path, allow_pickle=False)
        K = np.asarray(frame["K"], float)
        pose = np.asarray(frame["T_world_cam"], float)
        with Image.open(checked_file(obs["source_image"])) as im:
            image = im.convert("RGB").resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )
        if obs["shape_hw"][0] != obs["shape_hw"][1] or depth.shape[0] != depth.shape[1]:
            raise ValueError(
                "current Boxer pilot adapter supports square source cameras"
            )
        tm = read_masks(transients[name], transient_path.parent)
        exclusion = (
            np.logical_or.reduce(tm) if tm else np.zeros(obs["grid_shape_hw"], bool)
        )
        exclusion = np.asarray(
            Image.fromarray(exclusion).resize(
                (depth.shape[1], depth.shape[0]), Image.Resampling.NEAREST
            )
        )
        exclusion = binary_dilation(exclusion, iterations=2)
        points = depth_world_points(depth, K, pose, stable_depth(depth) & ~exclusion)
        if not len(points):
            raise ValueError("no stable static depth for OBB proposal")
        model_K = K.copy()
        model_K[0] *= image_size / depth.shape[1]
        model_K[1] *= image_size / depth.shape[0]
        turns = obs["applied_quarter_turns"] if args.upright else 0
        if turns:
            image = Image.fromarray(rotate_image(np.asarray(image), turns))
            model_K, pose, _ = rotate_pinhole_camera(
                model_K, pose, (image_size, image_size), turns
            )
        boxer_pose = pose.copy()
        boxer_pose[:3] = G @ pose[:3]
        boxer_points = points @ G.T
        uv, _ = project_world(points, model_K, pose)
        uv_transformed, _ = project_world(boxer_points, model_K, boxer_pose)
        transform_error = float(np.max(np.abs(uv - uv_transformed)))
        if transform_error > 1e-4:
            raise ValueError("world-up conversion changed camera projection")
        boxes = []
        for node in by_frame[name]:
            mask = rotate_image(masks[node["representative_detection"]], turns)
            yy, xx = np.where(mask)
            boxes.append([xx.min(), xx.max(), yy.min(), yy.max()])
        boxes = np.asarray(boxes, np.float32)
        boxes[:, :2] *= image_size / obs["grid_shape_hw"][1]
        boxes[:, 2:] *= image_size / obs["grid_shape_hw"][0]
        cam = BaseLoader.pinhole_from_K(
            image_size,
            image_size,
            model_K[0, 0],
            model_K[1, 1],
            model_K[0, 2],
            model_K[1, 2],
        )
        datum = dict(
            img0=torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()
            / 255,
            cam0=cam,
            T_world_rig0=PoseTW.from_Rt(
                torch.tensor(boxer_pose[:3, :3], dtype=torch.float32),
                torch.tensor(boxer_pose[:3, 3], dtype=torch.float32),
            ),
            sdp_w=torch.tensor(boxer_points, dtype=torch.float32),
            bb2d=torch.from_numpy(boxes),
        )
        torch.cuda.synchronize()
        encode_start = time.monotonic()
        with torch.inference_mode():
            inputs = model.prepare_inputs(datum)
            encoded = model.encode(inputs)
            predicted = model.query(inputs, encoded)
        torch.cuda.synchronize()
        inference_seconds = time.monotonic() - encode_start
        obbs = predicted["obbs_pr_w"][0].cpu()
        centers = obbs.bb3_center_world.numpy() @ G
        rotations = G.T @ obbs.T_world_object.R.numpy()
        dimensions = obbs.bb3_diagonal.numpy()
        corners = obbs.bb3corners_world.numpy() @ G
        for j, node in enumerate(by_frame[name]):
            row = dict(
                group_id=selected[node["id"]],
                node_id=node["id"],
                source_image=name,
                timestamp=node["timestamp"],
                center_m=centers[j].tolist(),
                rotation=rotations[j].tolist(),
                dimensions_m=dimensions[j].tolist(),
                corners_m=corners[j].tolist(),
                model_score=float(obbs.prob[j].item()),
                score_kind="inverse_one_plus_predicted_variance_not_calibrated_accuracy",
                native_gaussian_ownership_assigned=False,
                physical_extent_validated=False,
            )
            if not np.isfinite(
                np.r_[centers[j], rotations[j].ravel(), dimensions[j]]
            ).all():
                raise ValueError("non-finite OBB")
            rows.append(row)
            canvas = image.copy()
            draw = ImageDraw.Draw(canvas)
            uvc, z = project_world(corners[j], model_K, pose)
            # Boxer corner ordering is not assumed: connect nearest vertices in
            # local box coordinates by pairs differing in exactly one axis.
            local = (corners[j] - centers[j]) @ rotations[j]
            for a in range(8):
                for b in range(a + 1, 8):
                    delta = np.abs(local[a] - local[b])
                    if np.count_nonzero(delta > 1e-4) == 1 and z[a] > 0 and z[b] > 0:
                        draw.line(
                            [tuple(uvc[a]), tuple(uvc[b])], fill="#22ff88", width=3
                        )
            x0, x1, y0, y1 = boxes[j]
            draw.rectangle([x0, y0, x1, y1], outline="#ffd84d", width=2)
            draw.text(
                (8, 8),
                f"group {row['group_id']} | Boxer OBB hypothesis | {row['model_score']:.3f}",
                fill="white",
            )
            visual = (
                args.output
                / "visuals"
                / f"group_{row['group_id']:04d}_node_{node['id']:04d}.jpg"
            )
            canvas.save(visual, quality=94)
            row["visual"] = describe_file(visual)
        view_rows.append(
            dict(
                source_image=name,
                applied_quarter_turns=turns,
                queries=len(boxes),
                static_depth_points=len(points),
                world_transform_projection_error_pixels=transform_error,
                inference_seconds=inference_seconds,
                total_seconds=time.monotonic() - tick,
                positive_depth_patch_fraction=float(
                    (encoded["sdp_patch0"] > 0).float().mean()
                ),
            )
        )
        print(json.dumps(view_rows[-1]), flush=True)
    groups = []
    for group_id in sorted({r["group_id"] for r in rows}):
        member = [r for r in rows if r["group_id"] == group_id]
        centers = np.asarray([r["center_m"] for r in member])
        dims = np.asarray([r["dimensions_m"] for r in member])
        dims[:, :2] = np.sort(
            dims[:, :2], axis=1
        )  # Report yaw-90-equivalent sizes consistently.
        spread = np.ptp(dims, axis=0) / np.maximum(np.median(dims, axis=0), 1e-8)
        groups.append(
            dict(
                group_id=group_id,
                observations=len(member),
                independent_timestamps=len({r["timestamp"] for r in member}),
                maximum_center_disagreement_m=float(
                    np.linalg.norm(centers[:, None] - centers[None], axis=-1).max()
                ),
                median_dimensions_horizontal_sorted_height_m=np.median(
                    dims, axis=0
                ).tolist(),
                relative_dimension_spread=spread.tolist(),
                fused_box_emitted=False,
            )
        )
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.boxer-obb-pilot.v1",
            source_evidence=describe_file(args.scope_evidence),
            source_frames=describe_file(frames_path),
            model_checkpoint=describe_file(checkpoint),
            backbone=describe_file(
                args.model_dir / "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
            ),
            boxer_commit=subprocess.check_output(
                ["git", "-C", str(args.boxer_repo), "rev-parse", "HEAD"], text=True
            ).strip(),
            external_source_license="CC-BY-NC-4.0; research dependency, not vendored into FARM",
            source_world_up=args.world_up,
            source_to_boxer_world_rotation=G.tolist(),
            model_input_size=image_size,
            upright_images=args.upright,
            model_dtype="float32",
            model_load_seconds=load_seconds,
            total_seconds=time.monotonic() - started,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            model_dimension_bounds_m=[model.head.min_dim, model.head.bbox_max],
            observations=rows,
            views=view_rows,
            groups=groups,
            closed_test_opened=False,
            release_eligible=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
