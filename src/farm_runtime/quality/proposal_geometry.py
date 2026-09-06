"""Audit fixed proposals against registered static depth and other views."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from farm_runtime.angular_discovery import rotate_image
from farm_runtime.proposal_geometry import (
    GeometryPolicy,
    associate,
    equivalent_masks,
    stable_depth,
    surface_points,
)
from farm_runtime.quality_baseline import describe_file, write_json


def read_observations(path):
    manifest = json.loads(path.read_text())
    if manifest.get("test_opened") is not False:
        raise ValueError("explicit development-only proposal manifest required")
    observations = manifest["observations"]
    if len({r["name"] for r in observations}) != len(observations):
        raise ValueError("unique source images required")
    return manifest, observations


def read_masks(row, directory):
    path = directory / "masks" / Path(row["mask_artifact"]["path"]).name
    if describe_file(path)["sha256"] != row["mask_artifact"]["sha256"]:
        raise ValueError("proposal mask changed")
    masks = []
    with np.load(path, allow_pickle=False) as archive:
        for detection in row["detections"]:
            x0, y0, x1, y1 = detection["grid_window_xyxy"]
            h, w = row["grid_shape_hw"]
            if not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h:
                raise ValueError("invalid mask window")
            tile = archive[detection["logit_key"]]
            if tile.shape != (y1 - y0, x1 - x0) or not np.isfinite(tile).all():
                raise ValueError("invalid mask tile")
            mask = np.zeros((h, w), bool)
            mask[y0:y1, x0:x1] = tile > 0
            masks.append(mask)
    return masks


def review_groups(groups, nodes, observations, frames, output):
    """Show every multi-timestamp group with crops of the original RGB pixels."""
    paths, rgb_cache = [], {}
    for group in groups:
        if group["independent_timestamps"] < 2:
            continue
        ids = group["members"]
        ids = [
            ids[i]
            for i in np.unique(
                np.linspace(0, len(ids) - 1, min(6, len(ids))).round().astype(int)
            )
        ]
        sheet = Image.new("RGB", (480 * len(ids), 510), "#111827")
        draw = ImageDraw.Draw(sheet)
        for col, i in enumerate(ids):
            node = nodes[i]
            obs = observations[node["frame"]]
            turns = obs["applied_quarter_turns"]
            if node["frame"] not in rgb_cache:
                with Image.open(obs["source_image"]["path"]) as im:
                    rgb_cache[node["frame"]] = rotate_image(
                        np.asarray(im.convert("RGB")), turns
                    )
            source = rgb_cache[node["frame"]]
            mask = rotate_image(node["mask"], turns)
            excluded = rotate_image(
                node["mask"] & frames[node["frame"]]["excluded"], turns
            )
            yy, xx = np.where(mask)
            pad = max(12, round(0.15 * max(np.ptp(xx), np.ptp(yy))))
            h, w = mask.shape
            x0, x1 = max(0, int(xx.min()) - pad), min(w, int(xx.max()) + pad + 1)
            y0, y1 = max(0, int(yy.min()) - pad), min(h, int(yy.max()) + pad + 1)
            fh, fw = source.shape[:2]
            rgb = source[
                round(y0 * fh / h) : round(y1 * fh / h),
                round(x0 * fw / w) : round(x1 * fw / w),
            ].copy()
            shape = (rgb.shape[1], rgb.shape[0])
            foreground = np.asarray(
                Image.fromarray(mask[y0:y1, x0:x1]).resize(
                    shape, Image.Resampling.NEAREST
                )
            )
            transient = np.asarray(
                Image.fromarray(excluded[y0:y1, x0:x1]).resize(
                    shape, Image.Resampling.NEAREST
                )
            )
            rgb[foreground] = (
                rgb[foreground] * 0.6 + np.array([30, 210, 255]) * 0.4
            ).astype(np.uint8)
            rgb[transient] = (
                rgb[transient] * 0.4 + np.array([255, 45, 40]) * 0.6
            ).astype(np.uint8)
            tile = Image.fromarray(rgb)
            scale = min(470 / tile.width, 420 / tile.height)
            tile = tile.resize(
                (max(1, round(tile.width * scale)), max(1, round(tile.height * scale))),
                Image.Resampling.LANCZOS,
            )
            sheet.paste(
                tile,
                (col * 480 + (470 - tile.width) // 2, 80 + (420 - tile.height) // 2),
            )
            draw.text(
                (col * 480 + 6, 8),
                f"group {group['id']} | node {i} | {node['timestamp']}",
                fill="white",
            )
            draw.text((col * 480 + 6, 27), node["frame"], fill="white")
            labels = ", ".join(node["labels"])
            draw.text((col * 480 + 6, 46), labels[:70], fill="#fbbf24")
            draw.text((col * 480 + 6, 61), labels[70:140], fill="#fbbf24")
        path = output / "visuals" / f"group_{group['id']:04d}.jpg"
        sheet.save(path, quality=93)
        paths.append(describe_file(path))
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("proposals", "rgbd", "transients", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--views-from", type=Path)
    parser.add_argument(
        "--partial-view-association",
        action="store_true",
        help="Experimental: evaluate border-clipped matches within their observed field of view.",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("output directory must be new")
    started = time.monotonic()
    policy = GeometryPolicy(partial_view_association=args.partial_view_association)
    _, observations = read_observations(args.proposals)
    if args.views_from:
        _, selection = read_observations(args.views_from)
        by_name = {r["name"]: r for r in observations}
        observations = [by_name[r["name"]] for r in selection]
    _, transient_rows = read_observations(args.transients)
    transient_rows = {r["name"]: r for r in transient_rows}
    index = json.loads((args.rgbd / "frames.json").read_text())
    summary = json.loads((args.rgbd / "prep_summary.json").read_text())
    if (
        summary.get("status") != "complete"
        or index.get("depth_units") != "metres"
        or index.get("pose_translation_units") != "metres"
    ):
        raise ValueError("completed registered metric RGBD input required")
    registered = {f["source_image"]: f for f in index["frames"]}
    trusted = {
        name
        for g in index["camera_registration"]["groups"]
        if g["trusted"]
        for name in g["source_images"]
    }
    args.output.mkdir(parents=True)
    (args.output / "visuals").mkdir()
    frames, node_rows, nodes, rejected, frame_rows = {}, [], [], [], []
    cloud_arrays = {}
    for obs in observations:
        name = obs["name"]
        if name not in registered or name not in trusted:
            rejected.append({"source": name, "reason": "untrusted_registration"})
            continue
        transient = transient_rows[name]
        if (
            transient["source_image"] != obs["source_image"]
            or transient["grid_shape_hw"] != obs["grid_shape_hw"]
        ):
            raise ValueError("transient/source pixel grids differ")
        if (
            describe_file(Path(obs["source_image"]["path"]))["sha256"]
            != obs["source_image"]["sha256"]
        ):
            raise ValueError("source RGB changed")
        f = registered[name]
        if str(f["frame_id"]) != obs["timestamp"]:
            raise ValueError("physical timestamp mismatch")
        depth_path = args.rgbd / f["depth_path"]
        source_depth = np.load(depth_path, allow_pickle=False)
        h, w = obs["grid_shape_hw"]
        depth = np.asarray(
            Image.fromarray(source_depth).resize((w, h), Image.Resampling.NEAREST)
        )
        K = np.asarray(f["K"]).copy()
        K[0] *= w / source_depth.shape[1]
        K[1] *= h / source_depth.shape[0]
        transient_masks = read_masks(transient, args.transients.parent)
        excluded = (
            np.logical_or.reduce(transient_masks)
            if transient_masks
            else np.zeros((h, w), bool)
        )
        excluded = binary_dilation(excluded, iterations=2)
        frames[name] = dict(
            depth=depth,
            K=K,
            T_world_cam=np.asarray(f["T_world_cam"]),
            excluded=excluded,
        )
        masks = read_masks(obs, args.proposals.parent)
        equivalents = equivalent_masks(
            masks,
            [d["score"] for d in obs["detections"]],
            policy.duplicate_iou,
            priorities=[d.get("source_priority", 0) for d in obs["detections"]],
        )
        valid = stable_depth(depth, policy)
        frame_rows.append(
            dict(
                source=name,
                proposals=len(masks),
                distinct_scopes=len(equivalents),
                transient_pixels=int(excluded.sum()),
                depth_artifact=describe_file(depth_path),
            )
        )
        for members in equivalents:
            representative = members[0]
            mask = masks[representative]
            points, metrics = surface_points(
                mask, depth, K, f["T_world_cam"], excluded, policy, valid_depth=valid
            )
            labels = sorted({obs["detections"][j]["label"] for j in members})
            row = dict(
                frame=name,
                timestamp=obs["timestamp"],
                detection_indices=members,
                representative_detection=representative,
                labels=labels,
                **metrics,
            )
            if (
                len(points) < policy.min_points
                or metrics["excluded_pixels"] / max(1, metrics["mask_pixels"]) > 0.5
            ):
                rejected.append(dict(**row, reason="insufficient_static_surface"))
                continue
            detection = obs["detections"][representative]
            if "source_priority" in detection:
                row["source_priority"] = detection["source_priority"]
            row["id"] = len(nodes)
            node_rows.append(row)
            nodes.append(dict(**row, mask=mask, points=points))
            cloud_arrays[f"node_{row['id']:04d}"] = points
    build_seconds = time.monotonic() - started
    before = time.monotonic()
    components, evidence = associate(nodes, frames, policy)
    association_seconds = time.monotonic() - before
    group_rows = []
    for group_id, members in enumerate(components):
        timestamps = sorted({nodes[i]["timestamp"] for i in members})
        labels = sorted({label for i in members for label in nodes[i]["labels"]})
        group_rows.append(
            dict(
                id=group_id,
                members=members,
                timestamps=timestamps,
                independent_timestamps=len(timestamps),
                candidate_labels=labels,
                status=(
                    "multiview_candidate"
                    if len(timestamps) >= 2
                    else "single_timestamp_unconfirmed"
                ),
            )
        )
    np.savez_compressed(args.output / "surface_points.npz", **cloud_arrays)
    sheets = review_groups(
        group_rows, nodes, {r["name"]: r for r in observations}, frames, args.output
    )
    write_json(args.output / "evidence.json", evidence)
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.proposal-surface-association.v1",
            policy=asdict(policy),
            inputs={
                "proposals": describe_file(args.proposals),
                "transients": describe_file(args.transients),
                "frames": describe_file(args.rgbd / "frames.json"),
                "prep_summary": describe_file(args.rgbd / "prep_summary.json"),
                "views_from": (
                    describe_file(args.views_from) if args.views_from else None
                ),
            },
            frames=frame_rows,
            nodes=node_rows,
            groups=group_rows,
            rejected=rejected,
            sheets=sheets,
            counters={
                "input_observations": len(observations),
                "raw_proposals": sum(len(r["detections"]) for r in observations),
                "surface_nodes": len(nodes),
                "components": len(components),
                "multiview_components": sum(
                    g["independent_timestamps"] >= 2 for g in group_rows
                ),
                "rejection_reasons": dict(Counter(r["reason"] for r in rejected)),
                "scope_alternative_pairs": len(evidence["scope_alternatives"]),
                "mutual_edges": len(evidence["mutual_edges"]),
                "ambiguous_directions": evidence["ambiguous_direction_count"],
            },
            timing={
                "build_seconds": build_seconds,
                "association_seconds": association_seconds,
                "total_seconds": time.monotonic() - started,
            },
            evidence_artifact=describe_file(args.output / "evidence.json"),
            surface_artifact=describe_file(args.output / "surface_points.npz"),
            negative_evidence="Only reconstructed surfaces visible at compatible depth outside a candidate mask; occluded/excluded/missing pixels unknown.",
            labels_are_hypotheses=True,
            native_gaussian_ownership_assigned=False,
            counts_are_not_recall=True,
            test_opened=False,
            release_eligible=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
