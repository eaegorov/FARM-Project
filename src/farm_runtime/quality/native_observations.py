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


def validation_observations(validation_path, geometry_path=None):
    """Read accepted observations with the original geometry ID namespace."""
    validation = json.loads(validation_path.read_text())
    if (
        validation.get("closed_test_opened") is not False
        or validation.get("release_eligible") is not False
    ):
        raise ValueError("explicit development-only validation required")
    audit_path = checked_file(validation["source_audit"])
    audit = json.loads(audit_path.read_text())
    source_geometry = checked_file(audit["source_geometry"])
    if geometry_path is not None and (
        describe_file(geometry_path)["sha256"] != audit["source_geometry"]["sha256"]
    ):
        raise ValueError("recovery belongs to a different geometry namespace")
    inputs = SurfaceInputs(source_geometry)
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
    selected_ids = [row["group_id"] for row in validation["groups"]]
    if (
        len(set(selected_ids)) != len(selected_ids)
        or not set(selected_ids) <= groups.keys()
    ):
        raise ValueError("unique known recovery group IDs required")
    for row in validation["groups"]:
        oid = row["group_id"]
        group = groups[oid]
        for node_id in group["members"]:
            node = inputs.nodes[node_id]
            name = node["frame"]
            if (oid, name) in observations:
                raise ValueError("duplicate group/frame observation")
            observations[oid, name] = dict(
                mask=inputs.mask(node), source_kind="original_geometry", node_id=node_id
            )
            source_rows[name] = inputs.observations[name]
        for match in row["extra_matches"]:
            if match["selected_detection"] is None:
                continue
            name = match["name"]
            index = match["selected_detection"]
            eligible = {
                c["detection_index"]
                for c in match["candidates"]
                if c["eligible"] is True
            }
            if (
                match["decision"] != "matched_static_surface"
                or type(index) is not int
                or index not in eligible
                or name not in extra
                or not 0 <= index < len(masks[name])
                or extra[name]["detections"][index]["label"] == "person"
            ):
                raise ValueError("geometrically accepted extra object mask required")
            if (
                name not in inputs.trusted
                or match["timestamp"] != extra[name]["timestamp"]
                or extra[name]["timestamp"] != str(inputs.frames[name]["frame_id"])
                or not any(q["prompt"] == "person" for q in extra[name]["queries"])
            ):
                raise ValueError(
                    "registered recovery timestamp and person query required"
                )
            if (oid, name) in observations:
                raise ValueError("duplicate group/frame observation")
            observations[oid, name] = dict(
                mask=masks[name][index],
                source_kind="validated_additional_view",
                detection_index=index,
            )
            source_rows[name] = extra[name]
    return inputs, validation, observations, source_rows, extra, masks


def confirmed_recovery_groups(geometry_path, validation_path):
    """Add a bounded recovery cohort without displacing the original native budget."""
    inputs, validation, observations, _, _, _ = validation_observations(
        validation_path, geometry_path
    )
    if len(validation["groups"]) > 16:
        raise ValueError("recovery cohort must contain at most 16 groups")
    result = []
    for row in validation["groups"]:
        oid = row["group_id"]
        evidence = [
            (name, obs) for (gid, name), obs in observations.items() if gid == oid
        ]
        timestamps = {inputs.frames[name]["timestamp_ns"] for name, _ in evidence}
        if len(timestamps) >= 2 and any(
            obs["source_kind"] == "validated_additional_view" for _, obs in evidence
        ):
            result.append(oid)
    return sorted(result)


def prepare(validation_path, config, output):
    inputs, validation, observations, source_rows, extra, masks = (
        validation_observations(validation_path)
    )
    return _prepare_native(
        inputs,
        validation["groups"],
        observations,
        source_rows,
        extra,
        masks,
        config,
        output,
        source_validation=describe_file(validation_path),
    )


def prepare_geometry(
    geometry_path, config, output, world_up, group_ids=None, *, recovery_validation=None
):
    """Freeze observed multi-timestamp groups without inventing validation data.

    Geometric association is an identity hypothesis, not verified object scope.
    The resulting masks and boxes remain development-only.
    """
    from farm_runtime.obb_proposals import fit_surface_envelope, z_up_rotation
    from scripts.geometry.refine_farm_object_geometry import _fit_robust_obb

    z_up_rotation(world_up)  # Validate before using gravity in the existing fitter.
    recovery, extra, extra_masks = {}, {}, {}
    if recovery_validation:
        inputs, validation, recovery, _, extra, extra_masks = validation_observations(
            recovery_validation, geometry_path
        )
        if len(validation["groups"]) > 16:
            raise ValueError("recovery cohort must contain at most 16 groups")
    else:
        inputs = SurfaceInputs(geometry_path)
    if (
        inputs.geometry.get("schema") != "farm.proposal-surface-association.v1"
        or inputs.geometry.get("release_eligible") is not False
    ):
        raise ValueError("development surface association required")
    groups = {g["id"]: g for g in inputs.geometry["groups"]}
    selected = (
        sorted(
            gid
            for gid, g in groups.items()
            if g["independent_timestamps"] >= 2
            or len(
                {
                    inputs.frames[name]["timestamp_ns"]
                    for (oid, name) in recovery
                    if oid == gid
                }
            )
            >= 2
        )
        if group_ids is None
        else list(dict.fromkeys(group_ids))
    )
    if not selected or not set(selected) <= groups.keys():
        raise ValueError("nonempty known group IDs required")
    selection, observations, source_rows = [], {}, {}
    for oid in selected:
        group = groups[oid]
        members, points, timestamps, _ = inputs.support(group)
        actual_timestamps = {n["timestamp"] for n in members}
        added = {
            name: obs
            for (gid, name), obs in recovery.items()
            if gid == oid and obs["source_kind"] == "validated_additional_view"
        }
        names = {n["frame"] for n in members} | added.keys()
        if not names <= inputs.trusted:
            raise ValueError("trusted registration required")
        effective_timestamps = {inputs.frames[name]["timestamp_ns"] for name in names}
        if (
            len(effective_timestamps) < 2
            or len(actual_timestamps) != group["independent_timestamps"]
            or len({n["frame"] for n in members}) != len(members)
        ):
            raise ValueError(
                "two independent timestamps and unique group/frame required"
            )
        if not np.isfinite(points).all():
            raise ValueError("finite registered object support required")
        rotation = _fit_robust_obb(
            points,
            0.005,
            orientation_mode="gravity_yaw",
            up_vector=world_up,
        )["rotation_matrix"]
        weights = np.zeros(len(points))
        for timestamp in np.unique(timestamps):
            selected_points = timestamps == timestamp
            weights[selected_points] = 1 / selected_points.sum()
        box = fit_surface_envelope(
            points,
            weights,
            rotation,
            "FARM gravity-PCA; original observed support",
        )
        selection.append(dict(group_id=oid, boxes={"all_observed": box}))
        for node in members:
            name = node["frame"]
            transient = inputs.transients.get(name)
            if (
                transient is None
                or not any(q["prompt"] == "person" for q in transient["queries"])
                or any(d["label"] != "person" for d in transient["detections"])
            ):
                raise ValueError("explicit person-only transient evidence required")
            observations[oid, name] = dict(
                mask=inputs.mask(node),
                source_kind="original_geometry",
                node_id=node["id"],
            )
            source_rows[name] = inputs.observations[name]
        for name, obs in added.items():
            if (oid, name) in observations:
                raise ValueError("duplicate group/frame observation")
            observations[oid, name] = obs
            source_rows[name] = extra[name]
    return _prepare_native(
        inputs,
        selection,
        observations,
        source_rows,
        extra,
        extra_masks,
        config,
        output,
        source_validation=(
            describe_file(recovery_validation) if recovery_validation else None
        ),
    )


def freeze_source_ply(record):
    """Upgrade a matching preparation stat fingerprint to a content binding.

    Older RGBD preparations used size/mtime. Checking those before hashing
    preserves their original guarantee; the new SHA does not retroactively
    claim that the old preparation had a content hash.
    """
    path = Path(record["path"])
    stat = path.stat()
    if "sha256" not in record:
        if (
            record.get("size") != stat.st_size
            or record.get("mtime_ns") != stat.st_mtime_ns
        ):
            raise ValueError(
                "source PLY no longer matches preparation stat fingerprint"
            )
    current = describe_file(path)
    if "sha256" in record and current["sha256"] != record["sha256"]:
        raise ValueError("source PLY changed since preparation")
    return {
        **record,
        **current,
        "preparation_binding": (
            "sha256" if "sha256" in record else "size_mtime_then_sha256"
        ),
    }


def registered_identity(row, config):
    """Decode the identity using the frozen RGBD preparation contract."""
    match = re.fullmatch(config["identity_regex"], row["source_image"])
    if match is None:
        raise ValueError("source frame identity cannot be decoded")
    if match[config.get("timestamp_group", "timestamp")] != str(row["frame_id"]):
        raise ValueError("registered timestamp identity mismatch")
    return match[config["sensor_group"]], match[config["family_group"]]


def _prepare_native(
    inputs,
    selected_groups,
    observations,
    source_rows,
    extra,
    masks,
    config,
    output,
    *,
    source_validation=None,
):
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
    rgbd = prep["fingerprint_payload"]["config"]
    groups = {g["id"]: g for g in inputs.geometry["groups"]}
    source_ply = freeze_source_ply(prep["fingerprint_payload"]["inputs"]["ply"])
    output.mkdir(parents=True)
    (output / "masks").mkdir()
    (output / "exclusions").mkdir()
    frames = []
    exclusions = {}
    frame_rows = []
    for image_id, name in enumerate(sorted(source_rows)):
        if name not in inputs.trusted:
            raise ValueError("untrusted registration")
        row = inputs.frames[name]
        source = source_rows[name]
        sensor, family = registered_identity(row, rgbd)
        if (
            source.get("sensor", sensor) != sensor
            or source.get("family", family) != family
        ):
            raise ValueError(
                "proposal metadata conflicts with registered sensor/family"
            )
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
            sensor,
            family,
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
    for row in selected_groups:
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
        source_validation=source_validation,
        source_geometry=describe_file(inputs.geometry_path),
        observation_origin=(
            "validated_additional_views"
            if source_validation
            else "geometric_association_only"
        ),
        source_frames=describe_file(inputs.frames_path),
        source_prep_manifest=describe_file(prep_path),
        source_ply=source_ply,
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
        sensor, family = registered_identity(row, config)
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
            sensor,
            family,
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
    """Render bounded batches of objects sharing a frame; retain output order."""
    by_id = {
        int(oid): bank["indices"][bank["indptr"][i] : bank["indptr"][i + 1]]
        for i, oid in enumerate(bank["object_ids"])
    }
    frames = {r["image_id"]: r for r in input_manifest["frames"]}
    jobs = {}
    order = []
    for obj in run.objects:
        if obj.object_id not in by_id:
            continue
        timestamps = set()
        for obs in obj.observations:
            frame = run.frame(obs.image_id)
            if frame.physical_timestamp in timestamps:
                continue
            timestamps.add(frame.physical_timestamp)
            jobs.setdefault(obs.image_id, []).append((obj.object_id, obs))
            order.append((obj.object_id, obs.image_id))
    rows = {}
    for image_id, items in jobs.items():
        frame = run.frame(image_id)
        with Image.open(checked_file(frames[image_id]["source"])) as im:
            rgb = np.asarray(
                im.convert("RGB").resize((frame.depth_size[1], frame.depth_size[0]))
            )
        # Eight objects use17 render channels, within lift's32-channel contract.
        # Keep memory bounded independently of the scene's total object count.
        for start in range(0, len(items), 8):
            batch = items[start : start + 8]
            object_ids = [oid for oid, _ in batch]
            decoded, _, _, _ = lift._load_view_masks(
                run, frame, {oid: [obs] for oid, obs in batch}, config
            )
            masses = lift.reverse_render(gaussians, run, frame, object_ids, by_id)[0]
            for column, (oid, obs) in enumerate(batch):
                raw = decoded[oid]["raw"]
                predicted = masses[..., column] >= config["heldout"]["alpha_threshold"]
                layers = []
                for mask, color in [
                    (None, None),
                    (raw, [30, 205, 255]),
                    (predicted, [240, 170, 35]),
                ]:
                    panel = rgb.copy()
                    if mask is not None:
                        panel[mask] = (
                            panel[mask] * 0.5 + np.asarray(color) * 0.5
                        ).astype(np.uint8)
                    panel = Image.fromarray(
                        rotate_image(panel, frames[image_id]["applied_quarter_turns"])
                    )
                    panel.thumbnail((640, 640))
                    layers.append(panel)
                sheet = Image.new("RGB", (1920, 680), "#111827")
                draw = ImageDraw.Draw(sheet)
                draw.text(
                    (8, 8),
                    f"group {oid} | {frame.source_image} | BUILD consistency, not heldout accuracy",
                    fill="white",
                )
                for i, (panel, label) in enumerate(
                    zip(
                        layers,
                        ["RGB", "2D source mask", "Native Gaussian reverse render"],
                    )
                ):
                    draw.text((i * 640 + 8, 27), label, fill="white")
                    sheet.paste(panel, (i * 640, 40))
                path = output / f"group_{oid:04d}_view_{image_id:04d}.jpg"
                sheet.save(path, quality=94)
                rows[oid, image_id] = describe_file(path)
    return [rows[key] for key in order]


def retained_timestamp_counts(path, geometry_path):
    """Count effective native views after recovery or quarantine."""
    run, _, doc = load_prepared(path)
    record = doc.get("source_geometry")
    if not record or record["sha256"] != describe_file(geometry_path)["sha256"]:
        raise ValueError("native masks belong to a different geometry namespace")
    checked_file(record)
    counts = {
        obj.object_id: len(
            {run.frame(obs.image_id).physical_timestamp for obs in obj.observations}
        )
        for obj in run.objects
    }
    if len(counts) != len(run.objects):
        raise ValueError("unique native object IDs required")
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--validation", type=Path)
    source.add_argument(
        "--geometry", type=Path, help="Freeze geometric groups directly"
    )
    parser.add_argument("--group-id", type=int, action="append")
    parser.add_argument("--world-up", type=float, nargs=3)
    parser.add_argument("--recovery-validation", type=Path)
    source.add_argument("--input", type=Path, help="Reuse a frozen prepared input")
    for name in ("ply", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--mode", choices=("both", "exclusions_on", "exclusions_off"), default="both"
    )
    parser.add_argument(
        "--scope-alternative-budget",
        type=int,
        default=0,
        help="Preserve up to 16 nested scope hypotheses from cached VJP",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("new output required")
    if bool(args.geometry) != bool(args.world_up) or (
        args.group_id and not args.geometry
    ):
        raise ValueError(
            "--world-up is required only with --geometry; --group-id requires --geometry"
        )
    if args.recovery_validation and not args.geometry:
        raise ValueError("--recovery-validation requires --geometry")
    if not 0 <= args.scope_alternative_budget <= 16:
        raise ValueError("scope alternative budget must be from 0 to 16")
    started = time.monotonic()
    config, _ = lift.load_config(args.config)
    args.output.mkdir(parents=True)
    if args.input:
        run, split, input_manifest = load_prepared(args.input)
        input_path = args.input
    elif args.geometry:
        run, split, input_manifest = prepare_geometry(
            args.geometry,
            config,
            args.output / "input",
            args.world_up,
            args.group_id,
            recovery_validation=args.recovery_validation,
        )
        input_path = args.output / "input" / "manifest.json"
    else:
        run, split, input_manifest = prepare(
            args.validation, config, args.output / "input"
        )
        input_path = args.output / "input" / "manifest.json"
    if args.scope_alternative_budget and not input_manifest.get("source_geometry"):
        raise ValueError("scope alternatives require frozen source geometry")
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
    modes = ("exclusions_off", "exclusions_on") if args.mode == "both" else (args.mode,)
    for mode in modes:
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
        contributor_recovery = None
        if config["candidate"].get("rendered_recovery_budget", 0):
            from farm_runtime.quality.contributor_recovery import (
                recover_contributor_candidates,
            )

            contributor_recovery = recover_contributor_candidates(
                variant, gaussians, split, evidence, config, object_rows
            )
            if contributor_recovery["updated_object_ids"]:
                for candidate in candidates:
                    oid = candidate["object_id"]
                    if oid in contributor_recovery["updated_object_ids"]:
                        candidate["initial_spatial_candidate_gaussians"] = candidate[
                            "candidate_gaussians"
                        ]
                        candidate["candidate_gaussians"] = len(evidence[oid].indices)
                        candidate["search"]["rendered_recovery"] = True
                owner, confidence, support, object_rows, conflicts, blocked = (
                    lift.make_claims(evidence, table.count, config)
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
        alternatives = None
        if args.scope_alternative_budget:
            from farm_runtime.quality.scope_ownership import preserve_scope_alternatives

            alternatives = preserve_scope_alternatives(
                input_manifest,
                evidence,
                gaussians,
                config,
                bank,
                object_rows,
                dest / "scope_alternatives",
                args.scope_alternative_budget,
                run=variant,
                split=split,
            )
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
            scope_alternatives=alternatives,
            contributor_recovery=contributor_recovery,
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
            modes=list(modes),
            interpretation="Existing exact FARM lift with explicit transient-exclusion modes. Reverse images use build observations, not heldout evaluation. Production bank unchanged.",
            closed_test_opened=False,
            release_eligible=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
