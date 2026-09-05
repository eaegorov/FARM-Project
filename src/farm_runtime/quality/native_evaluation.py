"""Compare frozen native banks on new, geometry-associated development views."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.native_observations import load_prepared, write_native_mask
from farm_runtime.quality.proposal_geometry import read_masks, read_observations
from farm_runtime.quality.surface_evidence import SurfaceInputs
from farm_runtime.quality.surface_validation import select_observation
from farm_runtime.quality_baseline import describe_file, write_json
from tools.farm_shaper_bridge import gaussian_lift as lift
from tools.farm_shaper_bridge.common import open_graphdeco_ply


def require_new_timestamps(names, sources, excluded):
    """Use source timestamp IDs: nominal prep nanoseconds are bundle-local."""
    if len(names) != len(set(names)):
        raise ValueError("duplicate control source view")
    for name in names:
        if name not in sources or str(sources[name]["timestamp"]) in excluded:
            raise ValueError("control view overlaps earlier discovery/build timestamps")


def prepare_control(plan_path, proposals_path, output):
    plan = json.loads(plan_path.read_text())
    if (
        plan.get("schema") != "farm.native-control-plan.v1"
        or plan.get("test_opened") is not False
    ):
        raise ValueError("frozen development control plan required")
    build_path = checked_file(plan["frozen_input"])
    build_run, _, build = load_prepared(build_path)
    inputs = SurfaceInputs(checked_file(plan["source_geometry"]))
    if build["source_geometry"]["sha256"] != plan["source_geometry"]["sha256"]:
        raise ValueError("control association geometry differs from native build")
    inputs.frames_path = checked_file(plan["frames"])
    index = json.loads(inputs.frames_path.read_text())
    prep = json.loads(checked_file(plan["prep"]).read_text())
    summary = json.loads(checked_file(plan["prep_summary"]).read_text())
    # Preparation may use its documented size/mtime fingerprint. Bind it to
    # the full frozen source hash now and retain the original metadata check.
    ply_record = prep["fingerprint_payload"]["inputs"]["ply"]
    frozen_ply = checked_file(build["source_ply"])
    stat = frozen_ply.stat()
    if (
        Path(ply_record["path"]).resolve() != frozen_ply.resolve()
        or ply_record["size"] != stat.st_size
        or ply_record["mtime_ns"] != stat.st_mtime_ns
        or (
            "sha256" in ply_record
            and ply_record["sha256"] != build["source_ply"]["sha256"]
        )
    ):
        raise ValueError("control preparation source changed")
    if (
        prep.get("status") != "complete"
        or not prep.get("qa_passed")
        or not summary["alignment_qa"]["passed"]
        or prep["fingerprint"] != summary["fingerprint"]
        or index["meters_per_scene_unit"] != build_run.meters_per_scene_unit
    ):
        raise ValueError("control preparation/PLY/scale mismatch")
    trusted = {
        n
        for g in index["camera_registration"]["groups"]
        if g["trusted"]
        for n in g["source_images"]
    }
    inputs.frames = {
        f["source_image"]: f for f in index["frames"] if f["source_image"] in trusted
    }
    inputs.recorded_depths = plan["depths"]
    inputs._frames = {}
    source_manifest, observations = read_observations(proposals_path)
    if source_manifest["plan"]["sha256"] != describe_file(plan_path)["sha256"]:
        raise ValueError("control proposals do not belong to frozen plan")
    by_name = {r["name"]: r for r in observations}
    names = list(
        dict.fromkeys(
            v["name"] for g in plan["selected_groups"] for v in g["selected_views"]
        )
    )
    if set(by_name) != set(names) or len(by_name) != len(observations):
        raise ValueError("control observation set differs from selected views")
    excluded = set(plan["excluded_previous_timestamps"]) | {
        f.frame_id for f in build_run.frames
    }
    require_new_timestamps(names, plan["sources"], excluded)
    validation = json.loads(checked_file(plan["validation"]).read_text())
    with np.load(checked_file(validation["point_evidence"]), allow_pickle=False) as a:
        point_data = {k: a[k] for k in a.files}
    output.mkdir(parents=True)
    (output / "masks").mkdir()
    (output / "exclusions").mkdir()
    frame_rows, masks, geometric_frames = [], {}, {}
    for name in names:
        obs = by_name[name]
        if (
            name not in trusted
            or obs["source_image"] != plan["sources"][name]["source_image"]
        ):
            raise ValueError("control source RGB/registration mismatch")
        checked_file(obs["source_image"])
        row = inputs.frames[name]
        if str(row["frame_id"]) != str(obs["timestamp"]):
            raise ValueError("control source timestamp mismatch")
        frame = dict(inputs.frame(name))
        if tuple(obs["grid_shape_hw"]) != frame["depth"].shape:
            raise ValueError("control grid mismatch")
        if not any(q["prompt"] == "person" for q in obs["queries"]):
            raise ValueError("explicit transient query required")
        decoded = read_masks(obs, proposals_path.parent)
        people = [
            m for m, d in zip(decoded, obs["detections"]) if d["label"] == "person"
        ]
        unknown = np.zeros(frame["depth"].shape, bool)
        if people:
            unknown = binary_dilation(np.logical_or.reduce(people), iterations=2)
        frame["excluded"] = unknown
        masks[name], geometric_frames[name] = decoded, frame
        fid = len(frame_rows)
        h, w = row["depth_size"]
        exclusion = output / "exclusions" / f"{fid:06d}.npy"
        np.save(
            exclusion,
            np.asarray(
                Image.fromarray(unknown).resize((w, h), Image.Resampling.NEAREST)
            ),
        )
        frame_rows.append(
            dict(
                image_id=fid,
                name=name,
                physical_timestamp_ns=int(row["timestamp_ns"]),
                source=obs["source_image"],
                depth=describe_file(inputs.frames_path.parent / row["depth_path"]),
                exclusion=describe_file(exclusion),
                applied_quarter_turns=obs["applied_quarter_turns"],
            )
        )
    frame_by_name = {r["name"]: r for r in frame_rows}
    groups = {g["id"]: g for g in inputs.geometry["groups"]}
    objects = {r["object_id"]: deepcopy(r) for r in build["objects"]}
    decisions = []
    for selection in plan["selected_groups"]:
        oid = selection["group_id"]
        obj = objects[oid]
        obj["masks"] = []
        _, points, _, radii = inputs.support(groups[oid])
        prefix = f"group_{oid:04d}"
        if not np.array_equal(points, point_data[prefix + "_points"]):
            raise ValueError("control association point order changed")
        core = point_data[prefix + "_corroborated"]
        for selected in selection["selected_views"]:
            name = selected["name"]
            obs = by_name[name]
            chosen, candidates, decision = select_observation(
                points,
                core,
                masks[name],
                obs["detections"],
                geometric_frames[name],
                None,
                max(0.04, float(np.median(radii))),
            )
            decisions.append(
                dict(
                    object_id=oid,
                    name=name,
                    selected_detection=chosen,
                    decision=decision,
                    candidates=candidates,
                )
            )
            if chosen is None:
                continue
            record = frame_by_name[name]
            path = output / "masks" / f"{oid:06d}_{record['image_id']:06d}.npz"
            obj["masks"].append(
                dict(
                    image_id=record["image_id"],
                    source_name=name,
                    physical_timestamp_ns=record["physical_timestamp_ns"],
                    mask=write_native_mask(
                        path, masks[name][chosen], inputs.frames[name]["depth_size"], 2
                    ),
                    source_grid_hw=obs["grid_shape_hw"],
                    source_kind="automatic_development_control",
                    detection_index=chosen,
                )
            )
    manifest = deepcopy(build)
    manifest.update(
        source_frames=plan["frames"],
        source_prep_manifest=plan["prep"],
        frames=frame_rows,
        objects=list(objects.values()),
        source_control_plan=describe_file(plan_path),
        source_control_proposals=describe_file(proposals_path),
        frozen_build_input=plan["frozen_input"],
        observation_role="automatic_development_control",
        source_timestamp_namespace="frame_id from source image; timestamp_ns is local to preparation",
        split=dict(
            build_timestamps=[],
            heldout_timestamps=sorted(
                {str(r["physical_timestamp_ns"]) for r in frame_rows}, key=int
            ),
            objects=[
                dict(
                    object_id=oid,
                    strict_eligible=False,
                    eligibility="automatic_development_control",
                )
                for oid in objects
            ],
        ),
    )
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "association.json",
        dict(decisions=decisions, unknown_is_not_background=True),
    )
    return load_prepared(output / "manifest.json"), decisions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "proposals", "ply", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--variant", action="append", help="Frozen variant name; default: all"
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("new output required")
    started = time.monotonic()
    plan = json.loads(args.plan.read_text())
    # Check every frozen artifact before opening evaluation masks.
    frozen = {}
    for row in plan["frozen_variants"]:
        parent = json.loads(checked_file(row["manifest"]).read_text())
        report_path = checked_file(row["report"])
        report = json.loads(report_path.read_text())
        if (
            parent["input"]["sha256"] != plan["frozen_input"]["sha256"]
            or row["report"] not in parent["reports"]
        ):
            raise ValueError("frozen bank belongs to a different build input")
        frozen[row["name"]] = (
            checked_file(report["bank"]),
            checked_file(parent["config"]),
            row,
        )
    selected = args.variant or list(frozen)
    if len(selected) != len(set(selected)) or not set(selected) <= frozen.keys():
        raise ValueError("unknown or repeated frozen variant")
    (run, split, manifest), decisions = prepare_control(
        args.plan, args.proposals, args.output / "input"
    )
    source = describe_file(args.ply)
    if source["sha256"] != manifest["source_ply"]["sha256"]:
        raise ValueError("source PLY changed")
    table = open_graphdeco_ply(args.ply)
    gs = lift.load_gaussians(table, run.meters_per_scene_unit)
    reports = []
    for name in selected:
        bank_path, config_path, provenance = frozen[name]
        config, _ = lift.load_config(config_path)
        alignment = lift.alignment_guard(gs, run, config)
        owner = np.full(table.count, -1, np.int32)
        with np.load(bank_path, allow_pickle=False) as bank:
            if (
                (bank["indices"] < 0).any()
                or (bank["indices"] >= table.count).any()
                or len(np.unique(bank["indices"])) != len(bank["indices"])
            ):
                raise ValueError("invalid or conflicting frozen Gaussian IDs")
            for i, oid in enumerate(bank["object_ids"]):
                owner[bank["indices"][bank["indptr"][i] : bank["indptr"][i + 1]]] = oid
        rows, accepted, timing, audit = lift.heldout_qc(gs, run, split, owner, config)
        if accepted:
            raise ValueError("automatic control must never accept a production bank")
        result = dict(
            variant=name,
            frozen=provenance,
            objects=rows,
            timing=timing,
            views=audit,
            alignment=alignment,
            release_eligible=False,
            interpretation="Automatic new-view consistency, not independently annotated accuracy",
        )
        dest = args.output / (name + ".json")
        write_json(dest, result)
        reports.append(describe_file(dest))
        print(
            json.dumps(
                dict(
                    variant=name,
                    objects=[
                        dict(
                            object_id=r["object_id"],
                            views=len(r["views"]),
                            summaries=r["summaries"],
                        )
                        for r in rows
                    ],
                )
            ),
            flush=True,
        )
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.native-development-control.v1",
            plan=describe_file(args.plan),
            proposals=describe_file(args.proposals),
            source_ply=source,
            input=describe_file(args.output / "input/manifest.json"),
            association=describe_file(args.output / "input/association.json"),
            reports=reports,
            total_seconds=time.monotonic() - started,
            closed_test_opened=False,
            release_eligible=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
