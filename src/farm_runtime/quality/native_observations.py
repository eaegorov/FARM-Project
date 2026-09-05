"""Adapt validated development observations to FARM's existing exact native lift.

The adapter creates no production success/acceptance marker. It preserves the
source PLY row order, registered native depth and existing contribution gates.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import re
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion

from farm_runtime.angular_discovery import rotate_image
from farm_runtime.native_object_geometry import fit_native_obb
from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.proposal_geometry import read_masks, read_observations
from farm_runtime.quality.surface_evidence import SurfaceInputs
from farm_runtime.quality_baseline import describe_file, write_json
from tools.farm_shaper_bridge import gaussian_lift as lift
from tools.farm_shaper_bridge.common import (
    Frame,
    FarmObject,
    MaskObservation,
    RunData,
    build_verified_csr,
    open_graphdeco_ply,
)
from tools.farm_shaper_bridge.lift_refinement import (
    refine_connected_claims,
    apply_provisional_geometry_gate,
)


def write_native_mask(path, mask, shape, erosion_pixels):
    """Encode the same bit-packed raw/inlier contract used by FARM masks.

    Nearest resampling preserves the observed binary proposal; it does not
    claim new native-resolution segmentation detail. Inlier is a 2D interior.
    """
    h, w = shape
    raw = np.asarray(
        Image.fromarray(np.asarray(mask, bool)).resize((w, h), Image.Resampling.NEAREST)
    )
    inlier = (
        binary_erosion(raw, iterations=erosion_pixels) if erosion_pixels else raw.copy()
    )
    arrays = {"image_shape": np.asarray(shape, np.int32)}
    for name, values in [("raw", raw), ("inlier", inlier)]:
        arrays[name + "_shape"] = np.asarray(shape, np.int32)
        arrays[name + "_bbox_xyxy"] = np.asarray([0, 0, w, h], np.int32)
        arrays[name + "_bits"] = np.packbits(values.ravel(), bitorder="little")
    np.savez_compressed(path, **arrays)
    return describe_file(path)


def prepare(validation_path, config, output):
    validation = json.loads(validation_path.read_text())
    if (
        validation.get("closed_test_opened") is not False
        or validation.get("release_eligible") is not False
    ):
        raise ValueError("explicit development-only validation required")
    audit_path = checked_file(validation["source_audit"])
    audit = json.loads(audit_path.read_text())
    inputs = SurfaceInputs(checked_file(audit["source_geometry"]))
    prep_path = inputs.frames_path.parent / "run_manifest.json"
    prep = json.loads(prep_path.read_text())
    summary = json.loads(
        checked_file(inputs.geometry["inputs"]["prep_summary"]).read_text()
    )
    if (
        prep.get("status") != "complete"
        or not prep.get("qa_passed")
        or prep["fingerprint"] != summary["fingerprint"]
    ):
        raise ValueError("matching completed RGBD manifest required")
    extra_path = checked_file(validation["source_proposals"])
    _, extras = read_observations(extra_path)
    extra = {r["name"]: r for r in extras}
    masks = {name: read_masks(row, extra_path.parent) for name, row in extra.items()}
    for descriptor in validation.get("supplements", []):
        path = checked_file(descriptor)
        _, rows = read_observations(path)
        for row in rows:
            name = row["name"]
            base = extra[name]
            if any(
                row[k] != base[k]
                for k in (
                    "timestamp",
                    "source_image",
                    "grid_shape_hw",
                    "applied_quarter_turns",
                )
            ):
                raise ValueError("supplement frame/grid mismatch")
            extra[name] = dict(base, detections=base["detections"] + row["detections"])
            masks[name] += read_masks(row, path.parent)
    groups = {g["id"]: g for g in inputs.geometry["groups"]}
    observations = {}
    source_rows = {}
    for row in validation["groups"]:
        oid = row["group_id"]
        group = groups[oid]
        for node_id in group["members"]:
            node = inputs.nodes[node_id]
            name = node["frame"]
            observations[oid, name] = dict(
                mask=inputs.mask(node), source_kind="original_geometry", node_id=node_id
            )
            source_rows[name] = inputs.observations[name]
        for match in row["extra_matches"]:
            if match["selected_detection"] is None:
                continue
            if match["decision"] != "matched_static_surface":
                raise ValueError("unaccepted extra mask")
            name = match["name"]
            index = match["selected_detection"]
            if (oid, name) in observations:
                raise ValueError("duplicate group/frame observation")
            observations[oid, name] = dict(
                mask=masks[name][index],
                source_kind="validated_additional_view",
                detection_index=index,
            )
            source_rows[name] = extra[name]
    output.mkdir(parents=True)
    (output / "masks").mkdir()
    (output / "exclusions").mkdir()
    frames = []
    exclusions = {}
    frame_rows = []
    for image_id, name in enumerate(sorted(source_rows)):
        row = inputs.frames[name]
        source = source_rows[name]
        f = inputs.frame(name)
        if name not in inputs.trusted or str(row["frame_id"]) != source["timestamp"]:
            raise ValueError("untrusted registration or timestamp mismatch")
        checked_file(source["source_image"])
        excluded = f["excluded"].copy()
        if name in extra:
            if not any(q["prompt"] == "person" for q in extra[name]["queries"]):
                raise ValueError("missing transient exclusion query")
            people = [
                m
                for m, d in zip(masks[name], extra[name]["detections"])
                if d["label"] == "person"
            ]
            if people:
                excluded |= binary_dilation(np.logical_or.reduce(people), iterations=2)
        shape = tuple(row["depth_size"])
        native_excluded = np.asarray(
            Image.fromarray(excluded).resize(
                (shape[1], shape[0]), Image.Resampling.NEAREST
            )
        )
        path = output / "exclusions" / f"{image_id:06d}.npy"
        np.save(path, native_excluded, allow_pickle=False)
        exclusions[image_id] = describe_file(path)
        frame = Frame(
            image_id,
            str(row["frame_id"]),
            int(row["timestamp_ns"]),
            row["camera"],
            source["sensor"],
            source["family"],
            name,
            shape,
            np.asarray(row["K"], float),
            np.asarray(row["T_world_cam"], float),
            inputs.frames_path.parent / row["rgb_path"],
            inputs.frames_path.parent / row["depth_path"],
        )
        frames.append(frame)
        frame_rows.append(
            dict(
                image_id=image_id,
                name=name,
                physical_timestamp_ns=frame.timestamp_ns,
                source=source["source_image"],
                depth=describe_file(frame.depth_path),
                exclusion=exclusions[image_id],
                applied_quarter_turns=source["applied_quarter_turns"],
            )
        )
    by_name = {f.source_image: f for f in frames}
    objects = []
    overrides = {}
    object_rows = []
    split_rows = []
    for row in validation["groups"]:
        oid = row["group_id"]
        group = groups[oid]
        object_observations = []
        mask_rows = []
        for (object_id, name), evidence in observations.items():
            if object_id != oid:
                continue
            frame = by_name[name]
            path = output / "masks" / f"{oid:06d}_{frame.image_id:06d}.npz"
            descriptor = write_native_mask(
                path,
                evidence["mask"],
                frame.depth_size,
                int(config["mask"]["erosion_pixels"]),
            )
            overrides[oid, frame.image_id] = path
            object_observations.append(
                MaskObservation(oid, frame.image_id, path.name, {"path": str(path)})
            )
            mask_rows.append(
                dict(
                    image_id=frame.image_id,
                    source_name=name,
                    physical_timestamp_ns=frame.timestamp_ns,
                    mask=descriptor,
                    source_grid_hw=list(evidence["mask"].shape),
                    **{k: v for k, v in evidence.items() if k != "mask"},
                )
            )
        # Use original observed geometry. Extra-view point pruning is not a
        # shortcut to ownership, and no compact-box gate pre-deletes points.
        box = row["boxes"]["all_observed"]
        objects.append(
            FarmObject(
                oid,
                " / ".join(group["candidate_labels"]),
                "",
                np.asarray(box["center_m"]),
                np.asarray(box["dimensions_m"]),
                np.asarray(box["rotation_world_from_box"]),
                tuple(object_observations),
            )
        )
        timestamps = sorted(
            {str(by_name[m["source_name"]].timestamp_ns) for m in mask_rows}, key=int
        )
        object_rows.append(
            dict(
                object_id=oid,
                candidate_labels=group["candidate_labels"],
                geometry=box,
                masks=mask_rows,
            )
        )
        split_rows.append(
            dict(
                object_id=oid,
                build_timestamps=timestamps,
                heldout_timestamps=[],
                strict_eligible=False,
                eligibility="development_build_only",
            )
        )
    index = json.loads(inputs.frames_path.read_text())
    scale = float(index["meters_per_scene_unit"])
    rgbd = prep["fingerprint_payload"]["config"]
    if scale != float(rgbd["meters_per_scene_unit"]):
        raise ValueError("scene scale mismatch")
    split = dict(
        build_timestamps=sorted({f.physical_timestamp for f in frames}, key=int),
        heldout_timestamps=[],
        objects=split_rows,
    )
    manifest = dict(
        schema="farm.native-observation-input.v1",
        source_validation=describe_file(validation_path),
        source_geometry=describe_file(inputs.geometry_path),
        source_frames=describe_file(inputs.frames_path),
        source_prep_manifest=describe_file(prep_path),
        source_ply=prep["fingerprint_payload"]["inputs"]["ply"],
        object_id_namespace="source geometry group IDs; distinct from legacy FARM IDs",
        timestamp_authority="Registered frames.json; nominal timestamps are local to this preparation. Source frame ID is the cross-run identity.",
        frames=frame_rows,
        objects=object_rows,
        split=split,
        meters_per_scene_unit=scale,
        mask_resampling="nearest binary 640-grid to native registered depth grid; no new detail inferred",
        inlier_semantics="2D eroded mask interior; not independent 3D inlier ground truth",
        closed_test_opened=False,
        release_eligible=False,
    )
    write_json(output / "manifest.json", manifest)
    run = RunData(
        output,
        index["scene_id"],
        index,
        tuple(frames),
        tuple(objects),
        scale,
        rgbd,
        "",
        None,
        {},
        {},
        True,
        mask_overrides=overrides,
        observation_exclusions=exclusions,
    )
    return run, split, manifest


def load_prepared(path):
    """Reload a frozen development input without rerunning segmentation."""
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema") != "farm.native-observation-input.v1"
        or manifest.get("closed_test_opened") is not False
    ):
        raise ValueError("development native-observation input required")
    index = json.loads(checked_file(manifest["source_frames"]).read_text())
    prep = json.loads(checked_file(manifest["source_prep_manifest"]).read_text())
    config = prep["fingerprint_payload"]["config"]
    registered = {f["source_image"]: f for f in index["frames"]}
    trusted = {
        name
        for group in index["camera_registration"]["groups"]
        if group["trusted"]
        for name in group["source_images"]
    }
    frames, exclusions = [], {}
    for record in manifest["frames"]:
        name = record["name"]
        if record["image_id"] != len(frames) or name not in trusted:
            raise ValueError(
                "contiguous local frame IDs and trusted registration required"
            )
        row = registered[name]
        if int(row["timestamp_ns"]) != record["physical_timestamp_ns"]:
            raise ValueError("registered timestamp mapping changed")
        match = re.fullmatch(config["identity_regex"], name)
        if match is None:
            raise ValueError("source frame identity cannot be decoded")
        source_frames = checked_file(manifest["source_frames"]).parent
        depth = checked_file(record["depth"])
        if depth.resolve() != (source_frames / row["depth_path"]).resolve():
            raise ValueError("native depth identity changed")
        checked_file(record["source"])
        checked_file(record["exclusion"])
        frame = Frame(
            len(frames),
            str(row["frame_id"]),
            int(row["timestamp_ns"]),
            row["camera"],
            match[config["sensor_group"]],
            match[config["family_group"]],
            name,
            tuple(row["depth_size"]),
            np.asarray(row["K"], float),
            np.asarray(row["T_world_cam"], float),
            source_frames / row["rgb_path"],
            depth,
        )
        frames.append(frame)
        exclusions[frame.image_id] = record["exclusion"]
    overrides, objects = {}, []
    for row in manifest["objects"]:
        oid = row["object_id"]
        observations = []
        for item in row["masks"]:
            frame = frames[item["image_id"]]
            if (
                frame.source_image != item["source_name"]
                or frame.timestamp_ns != item["physical_timestamp_ns"]
            ):
                raise ValueError("mask/frame identity mismatch")
            key = oid, frame.image_id
            if key in overrides:
                raise ValueError("duplicate object/frame mask")
            mask = checked_file(item["mask"])
            overrides[key] = mask
            observations.append(
                MaskObservation(oid, frame.image_id, mask.name, {"path": str(mask)})
            )
        box = row["geometry"]
        objects.append(
            FarmObject(
                oid,
                " / ".join(row["candidate_labels"]),
                "",
                np.asarray(box["center_m"]),
                np.asarray(box["dimensions_m"]),
                np.asarray(box["rotation_world_from_box"]),
                tuple(observations),
            )
        )
    scale = float(manifest["meters_per_scene_unit"])
    if scale != float(index["meters_per_scene_unit"]) or scale != float(
        config["meters_per_scene_unit"]
    ):
        raise ValueError("registered metric scale changed")
    run = RunData(
        path.parent,
        index["scene_id"],
        index,
        tuple(frames),
        tuple(objects),
        scale,
        config,
        "",
        None,
        {},
        {},
        True,
        mask_overrides=overrides,
        observation_exclusions=exclusions,
    )
    return run, manifest["split"], manifest


def reverse_review(gaussians, run, bank, config, input_manifest, output):
    by_id = {
        int(oid): bank["indices"][bank["indptr"][i] : bank["indptr"][i + 1]]
        for i, oid in enumerate(bank["object_ids"])
    }
    frames = {r["image_id"]: r for r in input_manifest["frames"]}
    rows = []
    for obj in run.objects:
        if obj.object_id not in by_id:
            continue
        selected = []
        timestamps = set()
        for obs in obj.observations:
            frame = run.frame(obs.image_id)
            if frame.physical_timestamp not in timestamps:
                selected.append(obs)
                timestamps.add(frame.physical_timestamp)
        for obs in selected:
            frame = run.frame(obs.image_id)
            decoded, _, _, _ = lift._load_view_masks(
                run, frame, {obj.object_id: [obs]}, config
            )
            mass = lift.reverse_render(gaussians, run, frame, [obj.object_id], by_id)[
                0
            ][..., 0]
            raw = decoded[obj.object_id]["raw"]
            predicted = mass >= config["heldout"]["alpha_threshold"]
            with Image.open(checked_file(frames[frame.image_id]["source"])) as im:
                rgb = np.asarray(im.convert("RGB").resize((raw.shape[1], raw.shape[0])))
            layers = []
            for mask, color in [
                (None, None),
                (raw, [30, 205, 255]),
                (predicted, [240, 170, 35]),
            ]:
                panel = rgb.copy()
                if mask is not None:
                    panel[mask] = (panel[mask] * 0.5 + np.asarray(color) * 0.5).astype(
                        np.uint8
                    )
                panel = Image.fromarray(
                    rotate_image(panel, frames[frame.image_id]["applied_quarter_turns"])
                )
                panel.thumbnail((640, 640))
                layers.append(panel)
            sheet = Image.new("RGB", (1920, 680), "#111827")
            draw = ImageDraw.Draw(sheet)
            draw.text(
                (8, 8),
                f"group {obj.object_id} | {frame.source_image} | BUILD consistency, not heldout accuracy",
                fill="white",
            )
            for i, (panel, label) in enumerate(
                zip(layers, ["RGB", "2D source mask", "Native Gaussian reverse render"])
            ):
                draw.text((i * 640 + 8, 27), label, fill="white")
                sheet.paste(panel, (i * 640, 40))
            path = output / f"group_{obj.object_id:04d}_view_{frame.image_id:04d}.jpg"
            sheet.save(path, quality=94)
            rows.append(describe_file(path))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--validation", type=Path)
    source.add_argument("--input", type=Path, help="Reuse a frozen prepared input")
    for name in ("ply", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("new output required")
    started = time.monotonic()
    config, _ = lift.load_config(args.config)
    args.output.mkdir(parents=True)
    if args.input:
        run, split, input_manifest = load_prepared(args.input)
        input_path = args.input
    else:
        run, split, input_manifest = prepare(
            args.validation, config, args.output / "input"
        )
        input_path = args.output / "input" / "manifest.json"
    source = describe_file(args.ply)
    if source["sha256"] != input_manifest["source_ply"]["sha256"]:
        raise ValueError("source PLY changed")
    write_json(args.output / "config.json", config)
    if args.prepare_only:
        print(
            json.dumps(
                dict(
                    frames=len(run.frames),
                    objects=len(run.objects),
                    seconds=time.monotonic() - started,
                )
            ),
            flush=True,
        )
        return 0
    table = open_graphdeco_ply(args.ply)
    gaussians = lift.load_gaussians(table, run.meters_per_scene_unit)
    alignment = lift.alignment_guard(gaussians, run, config)
    import torch

    reports = []
    for mode in ("exclusions_off", "exclusions_on"):
        dest = args.output / mode
        dest.mkdir()
        (dest / "visuals").mkdir()
        variant = (
            replace(run, observation_exclusions=None)
            if mode == "exclusions_off"
            else run
        )
        tick = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        evidence, candidates = lift.build_candidates(variant, gaussians, split, config)
        build_rows, timing = lift.accumulate_build_evidence(
            gaussians, variant, split, evidence, config
        )
        owner, confidence, support, object_rows, conflicts, blocked = lift.make_claims(
            evidence, table.count, config
        )
        owner, confidence, support, refinement = refine_connected_claims(
            gaussians.means_m,
            gaussians.radius_m,
            evidence,
            owner,
            confidence,
            support,
            blocked,
            config,
        )
        owner, confidence, support, geometry = apply_provisional_geometry_gate(
            owner, confidence, support, object_rows, table.count, config
        )
        bank = build_verified_csr(owner, confidence, support)
        np.savez_compressed(dest / "proposal_bank.npz", **bank)
        native_geometry = []
        for obj in run.objects:
            indices = np.flatnonzero(owner == obj.object_id)
            if not len(indices):
                continue
            weights = gaussians.opacity_cpu[indices] * confidence[indices]
            box = fit_native_obb(gaussians.means_m[indices], weights, obj.rotation)
            native_geometry.append(
                dict(
                    object_id=obj.object_id,
                    native_gaussians=len(indices),
                    geometry=box,
                    physical_extent_validated=False,
                )
            )
        elapsed = time.monotonic() - tick
        visuals = reverse_review(
            gaussians, variant, bank, config, input_manifest, dest / "visuals"
        )
        report = dict(
            mode=mode,
            native_build_seconds=elapsed,
            total_with_visuals_seconds=time.monotonic() - tick,
            timing=timing,
            source_gaussian_count=table.count,
            source_row_order_preserved=True,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            candidates=candidates,
            objects=object_rows,
            conflicts=conflicts,
            refinement=refinement,
            geometry_gate=geometry,
            native_geometry=native_geometry,
            bank=describe_file(dest / "proposal_bank.npz"),
            visuals=visuals,
            closed_test_opened=False,
            release_eligible=False,
        )
        write_json(dest / "evidence_sources.json", dict(build=build_rows))
        write_json(dest / "report.json", report)
        reports.append(describe_file(dest / "report.json"))
        print(
            json.dumps(
                dict(
                    mode=mode,
                    native_build_seconds=elapsed,
                    counts={
                        r["object_id"]: r["native_gaussians"] for r in native_geometry
                    },
                )
            ),
            flush=True,
        )
        del evidence, owner, confidence, support, bank
        torch.cuda.empty_cache()
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.native-observation-pilot.v1",
            input=describe_file(input_path),
            config=describe_file(args.output / "config.json"),
            source_ply=source,
            reports=reports,
            alignment=alignment,
            total_seconds=time.monotonic() - started,
            interpretation="Same-input transient exclusion ablation with existing exact FARM lift. Reverse images use build observations, not heldout evaluation. Production bank unchanged.",
            closed_test_opened=False,
            release_eligible=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
