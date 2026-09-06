"""Prepare upright RGB and original masks for semantic review of spatial groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from farm_runtime.angular_discovery import rotate_image
from farm_runtime.object_scope import aggregate_scope_containment
from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.proposal_geometry import read_observations, read_masks
from farm_runtime.quality_baseline import describe_file, write_json
from farm_runtime.semantic_refinement import select_review_views


def native_evidence(path, geometry_path, observations):
    """Bind retained object views by source filename, never local frame numbers."""
    from farm_runtime.quality.native_observations import load_prepared

    run, _, doc = load_prepared(path)
    record = doc.get("source_geometry")
    if not record or record["sha256"] != describe_file(geometry_path)["sha256"]:
        raise ValueError("native masks belong to a different geometry namespace")
    checked_file(record)
    frames = {r["image_id"]: r for r in doc["frames"]}
    result = {}
    for obj in doc["objects"]:
        masks = {}
        for row in obj["masks"]:
            name = row["source_name"]
            if name not in observations:
                observations[name] = dict(
                    name=name,
                    source_image=frames[row["image_id"]]["source"],
                    applied_quarter_turns=frames[row["image_id"]][
                        "applied_quarter_turns"
                    ],
                )
            if (
                frames[row["image_id"]]["source"]["sha256"]
                != observations[name]["source_image"]["sha256"]
            ):
                raise ValueError("native and semantic RGB differ")
            frame = run.frame(row["image_id"])
            masks[name] = dict(
                row, native_shape_hw=list(frame.depth_size), timestamp=frame.frame_id
            )
        result[obj["object_id"]] = masks
    return result


def semantic_mask(original, native_row):
    """Keep original full-resolution masks unless a crop was actually replaced."""
    if original is not None and (
        native_row is None or not native_row.get("parent_mask")
    ):
        return original, "geometry_original"
    from tools.farm_shaper_bridge.common import load_mask_pair

    mask = load_mask_pair(
        checked_file(native_row["mask"]), tuple(native_row["native_shape_hw"])
    )[0]
    if mask.ndim != 2 or not mask.any():
        raise ValueError("nonempty refined native mask required for appearance")
    return mask, (
        "refined_native_raw"
        if native_row.get("parent_mask")
        else "additional_native_raw"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--group-id", type=int, action="append", default=[])
    parser.add_argument("--views", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-input", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("output must be new")
    started = time.monotonic()
    geometry = json.loads(args.groups.read_text())
    if geometry.get("test_opened") is not False:
        raise ValueError("development-only geometry required")
    proposals_path = checked_file(geometry["inputs"]["proposals"])
    _, observations = read_observations(proposals_path)
    evidence = json.loads(checked_file(geometry["evidence_artifact"]).read_text())
    relations = aggregate_scope_containment(
        geometry["groups"], geometry["nodes"], evidence["scope_alternatives"]
    )
    selected = [g for g in geometry["groups"] if g["independent_timestamps"] >= 2]
    if args.group_id:
        requested = set(args.group_id)
        selected = [g for g in selected if g["id"] in requested]
        if {g["id"] for g in selected} != requested:
            raise ValueError(
                "all requested groups must have independent multiview evidence"
            )
    if not selected:
        raise ValueError("no multiview groups selected")
    nodes = {n["id"]: n for n in geometry["nodes"]}
    observations = {r["name"]: r for r in observations}
    native = (
        native_evidence(args.native_input, args.groups, observations)
        if args.native_input
        else None
    )
    if native is not None and not {g["id"] for g in selected} <= native.keys():
        raise ValueError("semantic group absent from native input")
    frame_ids = {name: i for i, name in enumerate(sorted(observations))}
    mask_cache, rgb_cache = {}, {}

    def mask_for(node, oid):
        obs = observations[node["frame"]]
        original = None
        if node["representative_detection"] is not None:
            if node["frame"] not in mask_cache:
                mask_cache[node["frame"]] = read_masks(obs, proposals_path.parent)
            original = mask_cache[node["frame"]][node["representative_detection"]]
        row = native[oid][node["frame"]] if native is not None else None
        return semantic_mask(original, row)

    args.output.mkdir(parents=True)
    for sub in ("crops", "masks"):
        (args.output / sub).mkdir()
    rows, group_rows = [], []
    for group in selected:
        candidates = []
        group_nodes = {nodes[i]["frame"]: nodes[i] for i in group["members"]}
        if native is not None:
            for name, row in native[group["id"]].items():
                group_nodes.setdefault(
                    name,
                    dict(
                        id=None,
                        frame=name,
                        timestamp=row["timestamp"],
                        representative_detection=None,
                    ),
                )
        for node in group_nodes.values():
            node_id = node["id"]
            if native is not None and node["frame"] not in native[group["id"]]:
                continue  # Quarantined object/view is unknown, not appearance evidence.
            obs = observations[node["frame"]]
            mask, mask_source = mask_for(node, group["id"])
            clipped = bool(
                mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
            )
            candidates.append(
                dict(
                    object_id=group["id"],
                    node_id=node_id,
                    source_name=node["frame"],
                    image_id=frame_ids[node["frame"]],
                    timestamp=node["timestamp"],
                    source_image=obs["source_image"],
                    foreground_pixels=int(mask.sum()),
                    foreground_fraction=float(mask.mean()),
                    mask_source=mask_source,
                    native_mask=(
                        native[group["id"]][node["frame"]]["mask"]
                        if native is not None
                        else None
                    ),
                    clipped=clipped,
                )
            )
        if len({r["timestamp"] for r in candidates}) < 2:
            raise ValueError("two retained independent semantic views required")
        chosen = select_review_views(candidates, args.views)
        if len(chosen) < args.views:
            # A clipped independent view can explain a part that is occluded in
            # the best view. Keep its clipped flag; never claim invisible extent.
            other = [
                r
                for r in candidates
                if r["timestamp"] not in {c["timestamp"] for c in chosen}
            ]
            chosen += select_review_views(other, args.views - len(chosen))
        for row in chosen:
            node = group_nodes[row["source_name"]]
            obs = observations[node["frame"]]
            mask, _ = mask_for(node, group["id"])
            if node["frame"] not in rgb_cache:
                with Image.open(checked_file(obs["source_image"])) as im:
                    rgb_cache[node["frame"]] = np.asarray(im.convert("RGB"))
            rgb = rgb_cache[node["frame"]]
            yy, xx = np.where(mask)
            pad = max(12, round(0.25 * max(np.ptp(xx), np.ptp(yy))))
            h, w = mask.shape
            box = [
                max(0, int(xx.min()) - pad),
                max(0, int(yy.min()) - pad),
                min(w, int(xx.max()) + pad + 1),
                min(h, int(yy.max()) + pad + 1),
            ]
            x0, y0, x1, y1 = box
            fh, fw = rgb.shape[:2]
            source_box = [
                round(x0 * fw / w),
                round(y0 * fh / h),
                round(x1 * fw / w),
                round(y1 * fh / h),
            ]
            sx0, sy0, sx1, sy1 = source_box
            crop = rgb[sy0:sy1, sx0:sx1]
            mask_crop = np.asarray(
                Image.fromarray(mask[y0:y1, x0:x1]).resize(
                    (crop.shape[1], crop.shape[0]), Image.Resampling.NEAREST
                )
            )
            turns = obs["applied_quarter_turns"]
            crop, mask_crop = rotate_image(crop, turns), rotate_image(mask_crop, turns)
            stem = f"{group['id']:06d}_{row['image_id']:06d}"
            crop_path = args.output / "crops" / (stem + ".jpg")
            mask_path = args.output / "masks" / (stem + ".npz")
            Image.fromarray(crop).save(crop_path, quality=95)
            np.savez_compressed(mask_path, baseline=mask_crop)
            rows.append(
                dict(
                    **row,
                    crop=describe_file(crop_path),
                    masks=describe_file(mask_path),
                    source_grid_xyxy=box,
                    crop_source_xyxy=source_box,
                    turns=turns,
                )
            )
        group_rows.append(
            dict(
                **group,
                retained_independent_timestamps=len(
                    {r["timestamp"] for r in candidates}
                ),
                review_image_ids=[r["image_id"] for r in chosen],
            )
        )
    write_json(
        args.output / "relations.json",
        dict(
            schema="farm.observed-scope-relations.v1",
            relations=relations,
            physical_ownership_assigned=False,
            native_gaussian_ownership_assigned=False,
        ),
    )
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.object-scope-evidence.v1",
            observations=rows,
            groups=group_rows,
            source_groups=describe_file(args.groups),
            source_native_input=(
                describe_file(args.native_input) if args.native_input else None
            ),
            source_proposals=describe_file(proposals_path),
            relations=describe_file(args.output / "relations.json"),
            detector_labels_are_hypotheses=True,
            detector_labels_in_images=False,
            selection="explicit_group_ids" if args.group_id else "all_multiview_groups",
            total_seconds=time.monotonic() - started,
            reserved_test_opened=False,
            release_eligible=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
