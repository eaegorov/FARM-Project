"""Persist bounded registered views without changing the geometry ID namespace."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality_baseline import describe_file, write_json


def extension_kwargs(document):
    record = document.get("registered_extension")
    return {"registered_extension": checked_file(record)} if record else {}


def trusted_names(index):
    return {
        name
        for group in index["camera_registration"]["groups"]
        if group["trusted"]
        for name in group["source_images"]
    }


def merge_indices(original, additional, original_root, additional_root, names):
    """Preserve original poses/timestamps; remap only added physical timestamps."""
    for key in ("depth_units", "pose_translation_units"):
        if original.get(key) != "metres" or additional.get(key) != "metres":
            raise ValueError("metric registered frames required")
    if original.get("identity_contract") != additional.get("identity_contract"):
        raise ValueError("registered camera identity contract changed")
    scale = original["meters_per_scene_unit"]
    if (
        not np.isfinite(scale)
        or scale <= 0
        or scale != additional["meters_per_scene_unit"]
    ):
        raise ValueError("registered scene scale changed")
    old = {r["source_image"]: r for r in original["frames"]}
    extra = {r["source_image"]: r for r in additional["frames"]}
    if len(old) != len(original["frames"]) or len(extra) != len(additional["frames"]):
        raise ValueError("unique registered source names required")
    names = list(names)
    if (
        not names
        or len(set(names)) != len(names)
        or not set(names) <= extra.keys() - old.keys()
    ):
        raise ValueError("unique new registered source names required")
    if not set(names) <= trusted_names(additional):
        raise ValueError("all added views must have trusted registration")
    anchors = old.keys() & extra.keys()
    if not anchors or not anchors <= trusted_names(original) & trusted_names(
        additional
    ):
        raise ValueError("trusted common registration anchors required")
    for name in anchors:
        for key in ("frame_id", "camera", "depth_size", "K", "T_world_cam"):
            if old[name][key] != extra[name][key]:
                raise ValueError("original registration anchor changed")
    timestamps = {}
    for row in original["frames"]:
        if type(row["timestamp_ns"]) is not int:
            raise ValueError("integer physical timestamps required")
        identity, value = str(row["frame_id"]), row["timestamp_ns"]
        if identity in timestamps and timestamps[identity] != value:
            raise ValueError("one timestamp per physical frame ID required")
        timestamps[identity] = value
    if len(set(timestamps.values())) != len(timestamps):
        raise ValueError("different physical timestamps must not collide")
    next_ns = max(timestamps.values())
    for identity in sorted(
        {str(extra[n]["frame_id"]) for n in names} - timestamps.keys()
    ):
        next_ns += 100_000_000
        timestamps[identity] = next_ns

    def relocated(row, directory):
        result = copy.deepcopy(row)
        for key in ("rgb_path", "depth_path"):
            result[key] = str((Path(directory) / row[key]).resolve())
        return result

    merged = copy.deepcopy(original)
    merged["frames"] = [relocated(r, original_root) for r in original["frames"]]
    for name in sorted(names):
        row = relocated(extra[name], additional_root)
        row["timestamp_ns"] = timestamps[str(row["frame_id"])]
        K, pose = np.asarray(row["K"], float), np.asarray(row["T_world_cam"], float)
        if (
            K.shape != (3, 3)
            or pose.shape != (4, 4)
            or not np.isfinite(K).all()
            or not np.isfinite(pose).all()
        ):
            raise ValueError("finite registered camera matrices required")
        merged["frames"].append(row)
    for group in additional["camera_registration"]["groups"]:
        selected = [n for n in group["source_images"] if n in names]
        if selected:
            merged["camera_registration"]["groups"].append(
                dict(group, source_images=selected)
            )
    merged["cameras"] = list(
        dict.fromkeys(
            [*original.get("cameras", []), *(r["camera"] for r in merged["frames"])]
        )
    )
    merged["camera_registration"]["retained_views"] = len(trusted_names(merged))
    merged["selection_contract"] = dict(
        original=original.get("selection_contract"),
        extension_names=sorted(names),
        interpretation="Original observations preserved; bounded trusted development extension",
    )
    return merged, sorted(anchors)


def verify_scene_sources(original, additional):
    """Compare COLMAP/PLY metadata and independently bind their current contents."""

    def records(doc):
        inputs = doc["fingerprint_payload"]["inputs"]
        rows = [*inputs["colmap"], inputs["ply"]]
        result = {r["path"]: r for r in rows}
        if len(result) != len(rows):
            raise ValueError("unique scene source files required")
        return result

    a, b = records(original), records(additional)
    if a.keys() != b.keys():
        raise ValueError("registered extension belongs to another scene")
    verified = []
    for name, prior in a.items():
        current = b[name]
        for key in ("size", "mtime_ns"):
            if prior[key] != current[key]:
                raise ValueError("registered scene source fingerprint changed")
        path = Path(name)
        stat = path.stat()
        if stat.st_size != prior["size"] or stat.st_mtime_ns != prior["mtime_ns"]:
            raise ValueError("scene source changed after preparation")
        record = describe_file(path)
        if any(
            r.get("sha256", record["sha256"]) != record["sha256"]
            for r in (prior, current)
        ):
            raise ValueError("registered scene source content changed")
        verified.append(record)
    return verified


def load_extension(path, geometry_path, original_frames=None):
    doc = json.loads(Path(path).read_text())
    if (
        doc.get("schema") != "farm.registered-frame-extension.v1"
        or doc.get("closed_test_opened") is not False
        or doc.get("release_eligible") is not False
    ):
        raise ValueError("development registered frame extension required")
    geometry = checked_file(doc["source_geometry"])
    if describe_file(geometry_path)["sha256"] != describe_file(geometry)["sha256"]:
        raise ValueError("registered extension geometry namespace changed")
    source = checked_file(doc["source_frames"])
    if (
        original_frames is not None
        and source.resolve() != Path(original_frames).resolve()
    ):
        raise ValueError("registered extension source frame index changed")
    for key in (
        "source_prep_manifest",
        "additional_frames",
        "additional_prep_manifest",
        "source_plan",
        "source_packet",
    ):
        checked_file(doc[key])
    for record in doc["new_depths"]:
        checked_file(record)
    checked_file(doc["merged_frames"])
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("geometry", "preparation", "plan", "split-packet", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--view-budget", type=int, default=16)
    args = parser.parse_args(argv)
    if args.output.exists() or not 1 <= args.view_budget <= 24:
        raise ValueError("new output and registered view budget 1..24 required")
    geometry = json.loads(args.geometry.read_text())
    if geometry.get("test_opened") is not False:
        raise ValueError("development geometry required")
    original_path = checked_file(geometry["inputs"]["frames"])
    original = json.loads(original_path.read_text())
    additional_path = args.preparation / "frames.json"
    additional = json.loads(additional_path.read_text())
    old_prep_path, new_prep_path = (
        original_path.parent / "run_manifest.json",
        args.preparation / "run_manifest.json",
    )
    old_prep, new_prep = [
        json.loads(p.read_text()) for p in (old_prep_path, new_prep_path)
    ]
    if any(
        d.get("status") != "complete" or d.get("qa_passed") is not True
        for d in (old_prep, new_prep)
    ):
        raise ValueError("completed registered preparations required")
    if (
        old_prep["fingerprint"]
        != json.loads(checked_file(geometry["inputs"]["prep_summary"]).read_text())[
            "fingerprint"
        ]
    ):
        raise ValueError("original geometry preparation changed")
    plan = json.loads(args.plan.read_text())
    packet = json.loads(args.split_packet.read_text())
    if plan.get("test_opened") is not False:
        raise ValueError("development extension plan required")
    allowed = {
        str(t)
        for t, split in packet["split_by_timestamp"].items()
        if split in ("train", "dev")
    }
    old_names = {r["source_image"] for r in original["frames"]}
    names = [
        r["source_image"]
        for r in additional["frames"]
        if r["source_image"] in plan["sources"] and r["source_image"] not in old_names
    ]
    if not 1 <= len(names) <= args.view_budget:
        raise ValueError("registered extension exceeds view budget or is empty")
    for row in additional["frames"]:
        if row["source_image"] in names:
            source = plan["sources"][row["source_image"]]
            if str(row["frame_id"]) not in allowed or source["timestamp"] != str(
                row["frame_id"]
            ):
                raise ValueError("new registered view outside development allowlist")
            checked_file(source["source_image"])
    merged, anchors = merge_indices(
        original, additional, original_path.parent, additional_path.parent, names
    )
    old_rows = {r["source_image"]: r for r in original["frames"]}
    new_rows = {r["source_image"]: r for r in additional["frames"]}
    for name in anchors:
        a = np.load(
            original_path.parent / old_rows[name]["depth_path"], allow_pickle=False
        )
        b = np.load(
            additional_path.parent / new_rows[name]["depth_path"], allow_pickle=False
        )
        if not np.array_equal(a, b):
            raise ValueError("original anchor depth changed")
    verified = verify_scene_sources(old_prep, new_prep)
    args.output.mkdir(parents=True)
    write_json(args.output / "frames.json", merged)
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.registered-frame-extension.v1",
            source_geometry=describe_file(args.geometry),
            source_frames=describe_file(original_path),
            source_prep_manifest=describe_file(old_prep_path),
            additional_frames=describe_file(additional_path),
            additional_prep_manifest=describe_file(new_prep_path),
            source_plan=describe_file(args.plan),
            source_packet=describe_file(args.split_packet),
            merged_frames=describe_file(args.output / "frames.json"),
            new_names=sorted(names),
            anchor_names=anchors,
            anchor_camera_and_depth_exact=True,
            new_depths=[
                describe_file(additional_path.parent / new_rows[n]["depth_path"])
                for n in sorted(names)
            ],
            verified_scene_sources=verified,
            timestamp_mapping="Original values preserved; new physical frame IDs allocated after original maximum",
            closed_test_opened=False,
            release_eligible=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
