"""Apply immutable build-only mask overrides to an existing FARM run."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from farm_runtime.quality_baseline import describe_file


def checked_file(description):
    path = Path(description["path"])
    if describe_file(path)["sha256"] != description["sha256"]:
        raise ValueError("refinement input hash mismatch")
    return path


def apply_mask_refinement(run, manifest_path, split_path, camera_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("schema") != "farm.build-mask-refinement.v1":
        raise ValueError("unsupported build mask refinement")
    if (
        manifest["run_success_sha256"] != run.success_sha256
        or manifest["meters_per_scene_unit"] != run.meters_per_scene_unit
    ):
        raise ValueError("mask refinement belongs to another run or scale")
    for field, path in (("split", split_path), ("cameras", camera_path)):
        if (
            path is None
            or describe_file(Path(path))["sha256"] != manifest[field]["sha256"]
        ):
            raise ValueError("mask refinement split/cameras mismatch")
    split = json.loads(Path(split_path).read_text())
    build, heldout = set(split["build_timestamps"]), set(split["heldout_timestamps"])
    if build & heldout:
        raise ValueError("overlapping build/heldout timestamps")
    existing = {
        (o.object_id, obs.image_id) for o in run.objects for obs in o.observations
    }
    overrides = {}
    for row in manifest["masks"]:
        key = (row["object_id"], row["image_id"])
        if key in overrides or key not in existing:
            raise ValueError("duplicate or absent mask observation")
        frame = run.frame(row["image_id"])
        if frame.physical_timestamp not in build or frame.physical_timestamp in heldout:
            raise ValueError("mask refinement may only replace build observations")
        if frame.physical_timestamp != str(row["physical_timestamp_ns"]):
            raise ValueError("mask timestamp mismatch")
        overrides[key] = checked_file(row["mask"])
    # Several old detections can contribute to one object/frame. Replace their
    # union once, without loading the same new mask once per old detection.
    objects = []
    for obj in run.objects:
        seen, observations = set(), []
        for observation in obj.observations:
            key = (obj.object_id, observation.image_id)
            if key in overrides and key in seen:
                continue
            seen.add(key)
            observations.append(observation)
        objects.append(replace(obj, observations=tuple(observations)))
    return replace(run, objects=tuple(objects), mask_overrides=overrides)
