"""Compare crop proposals using native evidence from other physical timestamps.

This development tool does not establish semantic part ownership. It reuses
FARM's exact contribution votes and never treats missing observations as BG.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from farm_runtime.angular_discovery import rotate_image
from farm_runtime.segmentation_refinement import restore_crop_mask
from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.native_observations import load_prepared, write_native_mask
from farm_runtime.quality_baseline import describe_file, write_json
from tools.farm_shaper_bridge import gaussian_lift as lift
from tools.farm_shaper_bridge.common import open_graphdeco_ply

POLICY = dict(
    minimum_other_positive_timestamps=2,
    minimum_other_negative_timestamps=2,
    minimum_foreground_mass=20.0,
    minimum_foreground_recall=0.9,
    minimum_source_relative_recall=0.8,
    maximum_background_fraction=0.1,
    minimum_score_gain=0.02,
    ambiguity_score_margin=0.02,
    ambiguity_mask_iou=0.85,
    quarantine_background_fraction=0.3,
    minimum_retained_object_timestamps=2,
    expansion_boundary_tolerance_pixels=2,
    expansion_foreground_threshold=0.02,
    maximum_unexplained_expansion_fraction=0.5,
)


def other_timestamp_memberships(
    votes, excluded, minimum_positive=2, minimum_negative=2
):
    """Unsigned counters are widened before summing; the entire TS is excluded."""
    if excluded not in votes:
        raise ValueError("excluded timestamp absent from evidence")
    shape = votes[excluded][0].shape
    positive = np.zeros(shape, np.uint32)
    negative = np.zeros(shape, np.uint32)
    for timestamp, (p, n) in votes.items():
        if p.shape != shape or n.shape != shape:
            raise ValueError("candidate vote shapes differ")
        if timestamp != excluded:
            positive += p.astype(np.uint32)
            negative += n.astype(np.uint32)
    # Contradictory support is unknown, never resolved by one marginal vote.
    return (positive >= minimum_positive) & (negative == 0), (
        negative >= minimum_negative
    ) & (positive == 0)


def proposal_metrics(mask, foreground, background):
    if (
        mask.dtype != bool
        or mask.shape != foreground.shape
        or mask.shape != background.shape
    ):
        raise ValueError("boolean proposal and matching evidence maps required")
    if not np.isfinite(foreground).all() or not np.isfinite(background).all():
        raise ValueError("finite evidence required")
    if np.any(foreground < 0) or np.any(background < 0):
        raise ValueError("non-negative contribution required")
    total = float(foreground.sum(dtype=np.float64))
    captured = float(foreground[mask].sum(dtype=np.float64))
    leaked = float(background[mask].sum(dtype=np.float64))
    return dict(
        foreground_mass=total,
        foreground_recall=captured / total if total else None,
        background_mass_in_mask=leaked,
        background_fraction=leaked / (captured + leaked) if captured + leaked else None,
        score=(captured - leaked) / total if total else None,
        mask_pixels=int(mask.sum()),
    )


def expansion_evidence(mask, source, foreground, policy=POLICY):
    """Unknown pixels cannot justify substantial growth beyond the source edge."""
    from scipy.ndimage import binary_dilation

    allowed = binary_dilation(
        source, iterations=policy["expansion_boundary_tolerance_pixels"]
    )
    extension = mask & ~allowed
    count = int(extension.sum())
    unexplained = int(
        (extension & (foreground <= policy["expansion_foreground_threshold"])).sum()
    )
    fraction = unexplained / count if count else 0.0
    return dict(
        beyond_source_boundary_pixels=count,
        unexplained_pixels=unexplained,
        unexplained_fraction=fraction,
        supported=fraction <= policy["maximum_unexplained_expansion_fraction"],
    )


def choose_proposal(masks, foreground, background, policy=POLICY):
    """Return a candidate recommendation or explicit abstention, with baseline."""
    metrics = {
        key: proposal_metrics(mask, foreground, background)
        for key, mask in masks.items()
    }
    baseline = metrics["source"]
    if baseline["foreground_mass"] < policy["minimum_foreground_mass"]:
        return dict(
            decision="insufficient_other_view_evidence", selected=None, metrics=metrics
        )
    # Rendered native support can include RGB/GS mismatch or pre-existing
    # mask leakage. Below the absolute target, allow a proposal only if it
    # preserves at least the source coverage, with a separate identity floor.
    required_recall = max(
        policy["minimum_source_relative_recall"],
        min(policy["minimum_foreground_recall"], baseline["foreground_recall"]),
    )
    eligible = []
    for key, row in metrics.items():
        row["eligible"] = bool(
            row["foreground_recall"] >= required_recall
            and row["background_fraction"] is not None
            and row["background_fraction"] <= policy["maximum_background_fraction"]
        )
        if key != "source" and row["eligible"]:
            eligible.append(key)
    eligible.sort(key=lambda key: (-metrics[key]["score"], key))
    if not eligible:
        return dict(decision="no_consistent_candidate", selected=None, metrics=metrics)
    best = eligible[0]
    if metrics[best]["score"] - baseline["score"] < policy["minimum_score_gain"]:
        return dict(
            decision="no_material_gain", selected=None, best=best, metrics=metrics
        )
    for key in eligible[1:]:
        if (
            metrics[best]["score"] - metrics[key]["score"]
            > policy["ambiguity_score_margin"]
        ):
            break
        union = np.count_nonzero(masks[best] | masks[key])
        iou = np.count_nonzero(masks[best] & masks[key]) / max(1, union)
        if iou < policy["ambiguity_mask_iou"]:
            return dict(
                decision="ambiguous_scope",
                selected=None,
                alternatives=[best, key],
                metrics=metrics,
            )
    extension = expansion_evidence(masks[best], masks["source"], foreground, policy)
    metrics[best]["expansion"] = extension
    if not extension["supported"]:
        return dict(
            decision="unsupported_expansion", selected=None, best=best, metrics=metrics
        )
    return dict(
        decision="other_view_consistent_improvement", selected=best, metrics=metrics
    )


def quarantine_inconsistent_observations(manifest, decisions, policy=POLICY):
    """Omit a rejected object observation entirely: absent means unknown.

    Keep source masks in the audit, and retain at least two physical timestamps
    per object. Never remove a whole camera frame or another object's mask.
    """
    objects = {row["object_id"]: row for row in manifest["objects"]}
    quarantined = []
    candidates = [r for r in decisions if r["decision"] == "no_consistent_candidate"]
    candidates.sort(key=lambda r: -(r["metrics"]["source"]["background_fraction"] or 0))
    for row in candidates:
        source = row["metrics"]["source"]
        if (
            source["foreground_mass"] < policy["minimum_foreground_mass"]
            or source["foreground_recall"] >= policy["minimum_source_relative_recall"]
            or (source["background_fraction"] or 0)
            < policy["quarantine_background_fraction"]
        ):
            continue
        obj = objects[row["object_id"]]
        remaining = [m for m in obj["masks"] if m["image_id"] != row["image_id"]]
        if (
            len({m["physical_timestamp_ns"] for m in remaining})
            < policy["minimum_retained_object_timestamps"]
        ):
            row["quarantine"] = "insufficient_remaining_timestamps"
            continue
        omitted = [m for m in obj["masks"] if m["image_id"] == row["image_id"]]
        if len(omitted) != 1:
            raise ValueError("quarantine observation is not unique")
        obj["masks"] = remaining
        row["quarantine"] = "object_observation_unknown"
        quarantined.append(
            dict(
                object_id=row["object_id"],
                observation=omitted[0],
                reason="strong_other_timestamp_contradiction_without_consistent_replacement",
            )
        )
    manifest["quarantined_observations"] = quarantined
    if quarantined:
        for row in manifest["split"]["objects"]:
            row["build_timestamps"] = sorted(
                {
                    str(m["physical_timestamp_ns"])
                    for m in objects[row["object_id"]]["masks"]
                },
                key=int,
            )
        manifest["split"]["build_timestamps"] = sorted(
            {
                t
                for row in manifest["split"]["objects"]
                for t in row["build_timestamps"]
            },
            key=int,
        )
    return quarantined


def _zero_evidence(item):
    return replace(
        item,
        **{
            name: np.zeros_like(getattr(item, name))
            for name in (
                "positive_weight",
                "negative_weight",
                "visible_weight",
                "positive_timestamps",
                "negative_timestamps",
            )
        },
    )


def collect_timestamp_votes(gs, run, split, config, wanted):
    """Accumulate the unchanged source cohort once per physical timestamp."""
    candidates, _ = lift.build_candidates(run, gs, split, config)
    if not set(wanted) <= candidates.keys():
        raise ValueError("unknown candidate requested for timestamp evidence")
    votes = {oid: {} for oid in wanted}
    started = time.monotonic()
    for timestamp in split["build_timestamps"]:
        fresh = {oid: _zero_evidence(item) for oid, item in candidates.items()}
        lift.accumulate_build_evidence(
            gs, run, dict(split, build_timestamps=[timestamp]), fresh, config
        )
        for oid in wanted:
            item = fresh[oid]
            votes[oid][timestamp] = item.positive_timestamps, item.negative_timestamps
        del fresh
    return candidates, votes, time.monotonic() - started


def _read_proposals(paths, input_path, run):
    rows, sources = {}, []
    native = json.loads(input_path.read_text())
    native_frames = {row["image_id"]: row for row in native["frames"]}
    for path in paths:
        doc = json.loads(path.read_text())
        if (
            doc.get("schema") != "farm.sam-refinement-proposals.v1"
            or doc.get("release_eligible") is not False
        ):
            raise ValueError("development SAM proposals required")
        evidence = json.loads(checked_file(doc["evidence"]).read_text())
        if checked_file(evidence["native_input"]).resolve() != input_path.resolve():
            raise ValueError("proposal native input mismatch")
        if evidence["cameras"]["sha256"] != native["source_frames"]["sha256"]:
            raise ValueError("proposal camera source mismatch")
        if (
            evidence.get("reserved_test_opened") is not False
            or evidence.get("legacy_heldout_used") is not False
        ):
            raise ValueError("build-only proposals required")
        evidence_rows = {
            (r["object_id"], r["image_id"]): r for r in evidence["observations"]
        }
        source_id = len(sources)
        sources.append(describe_file(path))
        seen = set()
        for row in doc["observations"]:
            key = int(row["object_id"]), int(row["image_id"])
            if key in seen:
                raise ValueError("duplicate proposal object/frame")
            seen.add(key)
            if key not in evidence_rows:
                raise ValueError("proposal absent from frozen crop evidence")
            for field in (
                "timestamp",
                "physical_timestamp_ns",
                "crop",
                "crop_grid_xyxy",
                "crop_source_xyxy",
                "grid_shape_hw",
                "turns",
                "source_image",
            ):
                if row[field] != evidence_rows[key][field]:
                    raise ValueError("proposal changed frozen crop evidence")
            if row["source_image"] != native_frames[key[1]]["source"]:
                raise ValueError("proposal RGB differs from native observation")
            frame = run.frame(key[1])
            if (
                row["timestamp"] != frame.frame_id
                or row["physical_timestamp_ns"] != frame.timestamp_ns
            ):
                raise ValueError("proposal source timestamp mismatch")
            if list(frame.depth_size) != row["grid_shape_hw"]:
                raise ValueError("proposal native grid mismatch")
            checked_file(row["crop"])
            checked_file(row["source_image"])
            if key in rows:
                prior = rows[key]["row"]
                for field in (
                    "crop",
                    "crop_grid_xyxy",
                    "crop_source_xyxy",
                    "grid_shape_hw",
                    "turns",
                    "source_image",
                ):
                    if row[field] != prior[field]:
                        raise ValueError("proposal crop provenance mismatch")
            else:
                rows[key] = dict(row=row, candidates=[])
            with np.load(checked_file(row["masks"]), allow_pickle=False) as arrays:
                for candidate in row["candidates"]:
                    mask = arrays[candidate["key"]]
                    box, local = restore_crop_mask(mask, row)
                    rows[key]["candidates"].append(
                        dict(
                            key=f"source_{source_id}:{candidate['key']}",
                            backend=doc["backend"],
                            proposal=source_id,
                            candidate=candidate,
                            box=box,
                            local=local,
                        )
                    )
    return rows, sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "config", "ply", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--proposals", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("new output required")
    started = time.monotonic()
    run, split, manifest = load_prepared(args.input)
    if (
        manifest.get("observation_role", "build") != "build"
        or split["heldout_timestamps"]
    ):
        raise ValueError("build observations only")
    proposals, sources = _read_proposals(args.proposals, args.input, run)
    valid_pairs = {
        (obj.object_id, obs.image_id) for obj in run.objects for obs in obj.observations
    }
    if not set(proposals).issubset(valid_pairs):
        raise ValueError("proposal has no source object observation")
    config, _ = lift.load_config(args.config)
    if describe_file(args.ply)["sha256"] != manifest["source_ply"]["sha256"]:
        raise ValueError("source PLY changed")
    args.output.mkdir(parents=True)
    for name in ("evidence", "visuals", "input/masks"):
        (args.output / name).mkdir(parents=True)
    write_json(args.output / "policy.json", POLICY)
    gs = lift.load_gaussians(open_graphdeco_ply(args.ply), run.meters_per_scene_unit)
    lift.alignment_guard(gs, run, config)
    wanted = {oid for oid, _ in proposals}
    candidates, votes, accumulation_seconds = collect_timestamp_votes(
        gs, run, split, config, wanted
    )
    objects = {obj.object_id: obj for obj in run.objects}
    output_manifest = json.loads(json.dumps(manifest))
    output_objects = {row["object_id"]: row for row in output_manifest["objects"]}
    decisions = []
    for (oid, image_id), item in sorted(proposals.items()):
        frame = run.frame(image_id)
        obj = objects[oid]
        obs = next(obs for obs in obj.observations if obs.image_id == image_id)
        decoded = lift._load_view_masks(run, frame, {oid: [obs]}, config)[0][oid]
        raw = decoded["raw"]
        fg, bg = other_timestamp_memberships(
            votes[oid],
            frame.physical_timestamp,
            POLICY["minimum_other_positive_timestamps"],
            POLICY["minimum_other_negative_timestamps"],
        )
        indices = candidates[oid].indices
        mass = lift.reverse_render(
            gs, run, frame, [0, 1], {0: indices[fg], 1: indices[bg]}
        )[0]
        excluded = np.load(
            checked_file(manifest["frames"][image_id]["exclusion"]), allow_pickle=False
        )
        mass[excluded] = 0
        # Normalize only for minimum-mass eligibility; ratios are scale invariant.
        mass *= (
            float(config["render"]["evidence_reference_long_side"])
            / max(frame.depth_size)
        ) ** 2
        masks = {"source": raw}
        candidate_metadata = []
        for candidate in item["candidates"]:
            mask = raw.copy()
            x0, y0, x1, y1 = candidate["box"]
            mask[y0:y1, x0:x1] = candidate["local"]
            masks[candidate["key"]] = mask
            candidate_metadata.append(
                {k: v for k, v in candidate.items() if k != "local"}
            )
        result = choose_proposal(masks, mass[..., 0], mass[..., 1])
        selected = result["selected"]
        evidence_path = args.output / "evidence" / f"{oid:06d}_{image_id:06d}.npz"
        np.savez_compressed(
            evidence_path,
            foreground_mass=mass[..., 0].astype(np.float16),
            background_mass=mass[..., 1].astype(np.float16),
            foreground_indices=indices[fg],
            background_indices=indices[bg],
        )
        if selected:
            path = args.output / "input/masks" / f"{oid:06d}_{image_id:06d}.npz"
            descriptor = write_native_mask(
                path,
                masks[selected],
                frame.depth_size,
                int(config["mask"]["erosion_pixels"]),
            )
            target = next(
                m for m in output_objects[oid]["masks"] if m["image_id"] == image_id
            )
            target["parent_mask"] = target["mask"]
            target["mask"] = descriptor
            target["source_grid_hw"] = list(frame.depth_size)
            target["source_kind"] = "development_crop_refinement_other_timestamp_check"
            target["refinement_candidate"] = selected
        row = item["row"]
        x0, y0, x1, y1 = row["crop_grid_xyxy"]
        with Image.open(checked_file(row["crop"])) as image:
            rgb = np.asarray(image.convert("RGB"))
        chosen_key = selected or result.get("best", "source")
        tiles = []
        layers = [
            (None, "RGB"),
            (raw, "Source mask"),
            (mass[..., 0] > 0.02, "Other TS: foreground"),
            (mass[..., 1] > 0.02, "Other TS: background"),
            (masks[chosen_key], chosen_key),
        ]
        for index, (mask, title) in enumerate(layers):
            panel = rgb.copy()
            if mask is not None:
                local = rotate_image(mask[y0:y1, x0:x1], row["turns"])
                local = np.asarray(
                    Image.fromarray(local).resize(
                        (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST
                    )
                )
                color = np.array([250, 75, 70] if index == 3 else [20, 200, 255])
                panel[local] = (0.5 * panel[local] + 0.5 * color).astype(np.uint8)
            tile = Image.fromarray(panel)
            tile.thumbnail((400, 480))
            tiles.append((tile, title))
        sheet = Image.new("RGB", (2000, 550), "#111827")
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (8, 8),
            f"{oid}/{image_id} | {result['decision']} | BUILD diagnostic, no independent accuracy",
            fill="white",
        )
        for i, (tile, title) in enumerate(tiles):
            draw.text((400 * i + 8, 30), title, fill="white")
            sheet.paste(tile, (400 * i + (400 - tile.width) // 2, 60))
        visual = args.output / "visuals" / f"{oid:06d}_{image_id:06d}.jpg"
        sheet.save(visual, quality=95)
        decisions.append(
            dict(
                object_id=oid,
                image_id=image_id,
                excluded_physical_timestamp=frame.physical_timestamp,
                excluded_source_timestamp=frame.frame_id,
                evidence=describe_file(evidence_path),
                candidates=candidate_metadata,
                visual=describe_file(visual),
                **result,
            )
        )
        print(
            json.dumps(
                dict(
                    object_id=oid,
                    image_id=image_id,
                    decision=result["decision"],
                    selected=selected,
                )
            ),
            flush=True,
        )
    quarantined = quarantine_inconsistent_observations(output_manifest, decisions)
    output_manifest.update(
        parent_input=describe_file(args.input),
        refinement_policy=describe_file(args.output / "policy.json"),
        refinement_proposals=sources,
        release_eligible=False,
    )
    output_manifest["mask_resampling"] = (
        "Selected crop proposals restored with inverse quarter-turn and nearest resize; original mask outside crop preserved."
    )
    write_json(args.output / "input/manifest.json", output_manifest)
    write_json(
        args.output / "report.json",
        dict(
            schema="farm.native-crop-refinement.v1",
            input=describe_file(args.input),
            config=describe_file(args.config),
            proposals=sources,
            output_input=describe_file(args.output / "input/manifest.json"),
            policy=POLICY,
            observations=decisions,
            quarantined_observations=quarantined,
            accumulation_seconds=accumulation_seconds,
            total_seconds=time.monotonic() - started,
            interpretation="Development recommendations, not semantic ownership proof. Candidate universe and crop prompts use the frozen full build input; only scoring votes exclude the tested physical timestamp and all sibling views. Unknown or conflicting native points contribute neither class. No closed test or production promotion.",
            closed_test_opened=False,
            release_eligible=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
