"""Schedule bounded crop refinement from other-timestamp native contradictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from farm_runtime.angular_discovery import rotate_image, upright_quarter_turns
from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.native_observations import load_prepared
from farm_runtime.quality.native_refinement import (
    POLICY,
    collect_timestamp_votes,
    other_timestamp_memberships,
    proposal_metrics,
)
from farm_runtime.quality_baseline import describe_file, write_json
from tools.farm_shaper_bridge import gaussian_lift as lift
from tools.farm_shaper_bridge.common import open_graphdeco_ply


def select_observations(rows, budget, per_object=2):
    """Prioritize actual contradictions, with object diversity and one vote per TS."""
    if type(budget) is not int or not 1 <= budget <= 32:
        raise ValueError("crop budget must be in 1..32")
    if type(per_object) is not int or not 1 <= per_object <= 3:
        raise ValueError("per-object crop budget must be in 1..3")
    seen = set()
    eligible = defaultdict(list)
    for original in rows:
        row = dict(original)
        key = row["object_id"], row["image_id"]
        if key in seen:
            raise ValueError("duplicate object observation")
        seen.add(key)
        m = row["metrics"]
        if m["foreground_mass"] < POLICY["minimum_foreground_mass"]:
            continue
        recall, background = m["foreground_recall"], m["background_fraction"]
        if recall is None or background is None:
            continue
        if not all(np.isfinite(x) and 0 <= x <= 1 for x in (recall, background)):
            raise ValueError("invalid source agreement metrics")
        score = m["score"]
        if score is None or not np.isfinite(score):
            raise ValueError("invalid source agreement score")
        # Even perfect recall can hide removable background. Use the same score
        # gain required by candidate selection, rather than a separate 90/10 gate.
        gain_bound = max(0.0, 1.0 - score)
        if gain_bound < POLICY["minimum_score_gain"]:
            continue
        row["priority"] = gain_bound
        eligible[row["object_id"]].append(row)
    queues = {}
    for oid, candidates in eligible.items():
        candidates.sort(key=lambda r: (-r["priority"], r["image_id"]))
        timestamps, queue = set(), []
        for row in candidates:
            if row["timestamp"] not in timestamps:
                queue.append(row)
                timestamps.add(row["timestamp"])
        queues[oid] = queue[:per_object]
    selected = []
    for rank in range(per_object):
        layer = [queue[rank] for queue in queues.values() if len(queue) > rank]
        layer.sort(key=lambda r: (-r["priority"], r["object_id"], r["image_id"]))
        selected.extend(layer[: budget - len(selected)])
        if len(selected) == budget:
            break
    return selected


def crop_bounds(region):
    yy, xx = np.where(region)
    if not len(xx):
        raise ValueError("nonempty evidence region required")
    h, w = region.shape
    margin = max(20, round(0.25 * max(np.ptp(xx), np.ptp(yy))))
    return [
        max(0, int(xx.min()) - margin),
        max(0, int(yy.min()) - margin),
        min(w, int(xx.max()) + margin + 1),
        min(h, int(yy.max()) + margin + 1),
    ]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--world-up", type=float, nargs=3, required=True)
    p.add_argument("--crop-budget", type=int, default=12)
    p.add_argument("--crops-per-object", type=int, default=2)
    args = p.parse_args(argv)
    select_observations([], args.crop_budget, args.crops_per_object)
    up = np.asarray(args.world_up, float)
    if not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8:
        raise ValueError("finite nonzero world-up required")
    if args.output.exists():
        raise ValueError("new output required")
    started = time.monotonic()
    native = json.loads(args.native.read_text())
    if native.get("closed_test_opened") is not False:
        raise ValueError("development native input required")
    input_path = checked_file(native["input"])
    run, split, manifest = load_prepared(input_path)
    if (
        manifest.get("observation_role", "build") != "build"
        or split["heldout_timestamps"]
    ):
        raise ValueError("build observations only")
    config_path = checked_file(native["config"])
    config, _ = lift.load_config(config_path)
    source_path = checked_file(native["source_ply"])
    if native["source_ply"]["sha256"] != manifest["source_ply"]["sha256"]:
        raise ValueError("source PLY mismatch")
    reports = [
        (checked_file(r), json.loads(checked_file(r).read_text()))
        for r in native["reports"]
    ]
    reports = [(path, r) for path, r in reports if r["mode"] == "exclusions_on"]
    if len(reports) != 1:
        raise ValueError("one exclusion-aware native report required")
    report_path, report = reports[0]
    bank_path = checked_file(report["bank"])
    with np.load(bank_path, allow_pickle=False) as b:
        members = {
            int(oid): b["indices"][b["indptr"][i] : b["indptr"][i + 1]]
            for i, oid in enumerate(b["object_ids"])
        }
    # Two OTHER timestamps must support a core; a two-view object cannot supply it.
    wanted = {
        obj.object_id
        for obj in run.objects
        if len({run.frame(obs.image_id).physical_timestamp for obs in obj.observations})
        >= 3
    }
    args.output.mkdir(parents=True)
    for name in ("crops", "seeds", "evidence"):
        (args.output / name).mkdir()
    diagnostics, maps = [], {}
    accumulation_seconds = 0.0
    if wanted:
        gs = lift.load_gaussians(
            open_graphdeco_ply(source_path), run.meters_per_scene_unit
        )
        lift.alignment_guard(gs, run, config)
        candidates, votes, accumulation_seconds = collect_timestamp_votes(
            gs, run, split, config, wanted
        )
        for obj in run.objects:
            oid = obj.object_id
            if oid not in wanted:
                continue
            for obs in obj.observations:
                frame = run.frame(obs.image_id)
                raw = lift._load_view_masks(run, frame, {oid: [obs]}, config)[0][oid][
                    "raw"
                ]
                fg, bg = other_timestamp_memberships(
                    votes[oid],
                    frame.physical_timestamp,
                    POLICY["minimum_other_positive_timestamps"],
                    POLICY["minimum_other_negative_timestamps"],
                )
                ids = candidates[oid].indices
                mass = lift.reverse_render(
                    gs, run, frame, [0, 1], {0: ids[fg], 1: ids[bg]}
                )[0]
                frame_record = next(
                    r for r in manifest["frames"] if r["image_id"] == frame.image_id
                )
                excluded = np.load(
                    checked_file(frame_record["exclusion"]), allow_pickle=False
                )
                mass[excluded] = 0
                factor = (
                    float(config["render"]["evidence_reference_long_side"])
                    / max(frame.depth_size)
                ) ** 2
                metrics = proposal_metrics(
                    raw, mass[..., 0] * factor, mass[..., 1] * factor
                )
                row = dict(
                    object_id=oid,
                    image_id=frame.image_id,
                    timestamp=frame.frame_id,
                    physical_timestamp_ns=frame.timestamp_ns,
                    metrics=metrics,
                    other_positive_native_count=int(fg.sum()),
                    other_negative_native_count=int(bg.sum()),
                )
                diagnostics.append(row)
                maps[oid, frame.image_id] = (raw, mass, frame_record, excluded)
    selected = select_observations(diagnostics, args.crop_budget, args.crops_per_object)
    rows = []
    for item in selected:
        oid, image_id = item["object_id"], item["image_id"]
        frame = run.frame(image_id)
        raw, mass, frame_record, excluded = maps[oid, image_id]
        probability = lift.reverse_render(
            gs, run, frame, [oid], {oid: members.get(oid, np.zeros(0, np.int64))}
        )[0][..., 0]
        probability = np.maximum(probability, mass[..., 0])
        probability[excluded] = 0
        x0, y0, x1, y1 = crop_bounds(raw | (probability >= 0.1))
        h, w = raw.shape
        source = checked_file(frame_record["source"])
        with Image.open(source) as im:
            im = im.convert("RGB")
            source_box = [
                round(x0 * im.width / w),
                round(y0 * im.height / h),
                round(x1 * im.width / w),
                round(y1 * im.height / h),
            ]
            crop = im.crop(source_box)
        yy, xx = np.where(raw | (probability >= 0.1))
        turns = upright_quarter_turns(
            frame.T_world_cam[:3, :3].T,
            args.world_up,
            image_point=[float(np.median(xx)), float(np.median(yy))],
            intrinsics=frame.K,
        )["applied_quarter_turns_ccw"]
        prob = np.asarray(
            Image.fromarray(probability[y0:y1, x0:x1]).resize(
                crop.size, Image.Resampling.BILINEAR
            )
        )
        local = np.asarray(
            Image.fromarray(raw[y0:y1, x0:x1]).resize(
                crop.size, Image.Resampling.NEAREST
            )
        )
        name = f"{oid:06d}_{image_id:06d}"
        crop_path = args.output / "crops" / f"{name}.jpg"
        seed_path = args.output / "seeds" / f"{name}.npz"
        evidence_path = args.output / "evidence" / f"{name}.npz"
        Image.fromarray(rotate_image(np.asarray(crop), turns)).save(
            crop_path, quality=95
        )
        np.savez_compressed(
            seed_path,
            probability=rotate_image(prob, turns).astype(np.float16),
            legacy=rotate_image(local, turns),
        )
        np.savez_compressed(
            evidence_path,
            foreground=mass[..., 0].astype(np.float16),
            background=mass[..., 1].astype(np.float16),
        )
        rows.append(
            dict(
                item,
                source_image=describe_file(source),
                crop_grid_xyxy=[x0, y0, x1, y1],
                crop_source_xyxy=source_box,
                grid_shape_hw=[h, w],
                turns=turns,
                crop=describe_file(crop_path),
                seeds=describe_file(seed_path),
                other_timestamp_maps=describe_file(evidence_path),
                foreground_pixels=int((prob >= 0.1).sum()),
                clipped=bool(x0 == 0 or y0 == 0 or x1 == w or y1 == h),
            )
        )
    objects = {r["object_id"]: r for r in manifest["objects"]}
    concepts = {}
    for oid in sorted({row["object_id"] for row in rows}):
        labels = objects[oid].get("candidate_labels", [])
        labels = list(
            dict.fromkeys(t.strip() for t in labels if isinstance(t, str) and t.strip())
        )
        if labels:
            concepts[str(oid)] = sorted(labels, key=lambda t: (-len(t.split()), t))[:3]
    write_json(args.output / "concepts.json", concepts)
    write_json(args.output / "split.json", split)
    write_json(
        args.output / "schedule.json",
        dict(
            policy=POLICY,
            crop_budget=args.crop_budget,
            crops_per_object=args.crops_per_object,
            eligible_object_ids=sorted(wanted),
            diagnostics=diagnostics,
            selected=selected,
            accumulation_seconds=accumulation_seconds,
        ),
    )
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.object-refinement-evidence.v1",
            native_input=describe_file(input_path),
            native_report=describe_file(report_path),
            native_stage=describe_file(args.native),
            config=describe_file(config_path),
            bank=describe_file(bank_path),
            source_ply=manifest["source_ply"],
            cameras=manifest["source_frames"],
            split=describe_file(args.output / "split.json"),
            observations=rows,
            world_up=args.world_up,
            schedule=describe_file(args.output / "schedule.json"),
            concepts=describe_file(args.output / "concepts.json"),
            concept_policy="Up to three existing geometric candidate labels; queries, not verified identity.",
            seconds=time.monotonic() - started,
            seed_policy="native membership union other-timestamp foreground; current transient pixels unknown",
            legacy_heldout_used=False,
            reserved_test_opened=False,
            automatic_materialization_authorized=False,
            interpretation="Crop scheduling is not a new foreground vote or physical scope decision.",
        ),
    )
    print(
        json.dumps(
            dict(
                eligible_objects=len(wanted),
                inspected_observations=len(diagnostics),
                selected_crops=len(rows),
                seconds=time.monotonic() - started,
            )
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    main()
