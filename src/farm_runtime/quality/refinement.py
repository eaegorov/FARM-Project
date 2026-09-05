"""Prepare registered evidence and compare cached SAM refinement proposals."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from farm_runtime.angular_discovery import rotate_image, upright_quarter_turns
from farm_runtime.quality_baseline import describe_file, write_json


def prepare(args):
    from tools.farm_shaper_bridge import gaussian_lift as lift
    from tools.farm_shaper_bridge.common import load_run, open_graphdeco_ply
    from farm_runtime.quality.camera_refinement import apply_camera_refinement
    from farm_runtime.quality.lift_review import read_bank

    run = apply_camera_refinement(load_run(args.run, allow_legacy=True), args.cameras)
    split = json.loads(args.split.read_text())
    packet = json.loads(args.packet.read_text())
    config, _ = lift.load_config(args.config)
    bank = read_bank(args.bank)
    source = open_graphdeco_ply(args.ply)
    if describe_file(args.ply)["sha256"] != packet["source_ply"]["sha256"]:
        raise ValueError("wrong source PLY")
    gs = lift.load_gaussians(source, run.meters_per_scene_unit)
    args.output.mkdir(parents=True)
    (args.output / "crops").mkdir()
    (args.output / "seeds").mkdir()
    started = time.monotonic()
    rows = []
    for obj in run.objects:
        if obj.object_id not in bank:
            continue
        observations = defaultdict(list)
        for obs in obj.observations:
            frame = run.frame(obs.image_id)
            if frame.physical_timestamp in split["build_timestamps"]:
                if packet["split_by_timestamp"].get(frame.frame_id) not in (
                    "train",
                    "dev",
                ):
                    raise ValueError("reserved timestamp requested")
                observations[obs.image_id].append(obs)
        for image_id, obs in sorted(observations.items()):
            frame = run.frame(image_id)
            decoded, _, _, _ = lift._load_view_masks(
                run, frame, {obj.object_id: obs}, config
            )
            raw = decoded[obj.object_id]["raw"]
            probability = lift.reverse_render(
                gs, run, frame, [obj.object_id], {obj.object_id: bank[obj.object_id]}
            )[0][..., 0]
            yy, xx = np.where(probability >= 0.1)
            if len(xx) < 8:
                continue
            h, w = raw.shape
            margin = max(20, int(0.25 * max(xx.max() - xx.min(), yy.max() - yy.min())))
            x0, x1 = max(0, int(xx.min()) - margin), min(w, int(xx.max()) + margin + 1)
            y0, y1 = max(0, int(yy.min()) - margin), min(h, int(yy.max()) + margin + 1)
            path = args.images / frame.source_image
            with Image.open(path) as im:
                im = im.convert("RGB")
                source_box = [
                    round(x0 * im.width / w),
                    round(y0 * im.height / h),
                    round(x1 * im.width / w),
                    round(y1 * im.height / h),
                ]
                crop = im.crop(source_box)
            turns = upright_quarter_turns(
                frame.T_world_cam[:3, :3].T,
                args.world_up,
                image_point=[float(np.median(xx)), float(np.median(yy))],
                intrinsics=frame.K,
            )["applied_quarter_turns_ccw"]
            prob_crop = np.asarray(
                Image.fromarray(probability[y0:y1, x0:x1]).resize(
                    crop.size, Image.Resampling.BILINEAR
                )
            )
            raw_crop = np.asarray(
                Image.fromarray(raw[y0:y1, x0:x1]).resize(
                    crop.size, Image.Resampling.NEAREST
                )
            )
            rgb = rotate_image(np.asarray(crop), turns)
            prob_crop, raw_crop = rotate_image(prob_crop, turns), rotate_image(
                raw_crop, turns
            )
            name = f"{obj.object_id:06d}_{image_id:06d}"
            rgb_path = args.output / "crops" / (name + ".jpg")
            seed_path = args.output / "seeds" / (name + ".npz")
            Image.fromarray(rgb).save(rgb_path, quality=95)
            np.savez_compressed(
                seed_path, probability=prob_crop.astype(np.float16), legacy=raw_crop
            )
            rows.append(
                dict(
                    object_id=obj.object_id,
                    image_id=image_id,
                    timestamp=frame.frame_id,
                    physical_timestamp_ns=int(frame.physical_timestamp),
                    source_image=describe_file(path),
                    crop_grid_xyxy=[x0, y0, x1, y1],
                    crop_source_xyxy=source_box,
                    grid_shape_hw=[h, w],
                    turns=turns,
                    crop=describe_file(rgb_path),
                    seeds=describe_file(seed_path),
                    foreground_pixels=len(xx),
                    clipped=bool(
                        xx.min() == 0
                        or yy.min() == 0
                        or xx.max() == w - 1
                        or yy.max() == h - 1
                    ),
                )
            )
        print(
            json.dumps(
                dict(
                    object_id=obj.object_id,
                    prepared=sum(r["object_id"] == obj.object_id for r in rows),
                )
            ),
            flush=True,
        )
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.object-refinement-evidence.v1",
            observations=rows,
            run=str(args.run),
            bank=describe_file(args.bank),
            split=describe_file(args.split),
            cameras=describe_file(args.cameras),
            source_ply=describe_file(args.ply),
            source_commit=args.source_commit,
            seconds=time.monotonic() - started,
            legacy_heldout_used=False,
            reserved_test_opened=False,
        ),
    )


def sam(args):
    from farm_runtime.segmentation_refinement import CachedSAMRefiner, prompt_variants
    import torch

    evidence = json.loads(args.evidence.read_text())
    args.output.mkdir(parents=True)
    (args.output / "masks").mkdir()
    (args.output / "visuals").mkdir()
    started = time.monotonic()
    model = CachedSAMRefiner(args.model)
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - started
    rows, skipped = [], []
    chosen = evidence["observations"]
    if args.limit:
        chosen = sorted(chosen, key=lambda r: (-r["foreground_pixels"], r["image_id"]))[
            : args.limit
        ]
    for index, row in enumerate(chosen):
        image = Image.open(row["crop"]["path"]).convert("RGB")
        with np.load(row["seeds"]["path"], allow_pickle=False) as seeds:
            probability = seeds["probability"].astype(np.float32)
        variants = prompt_variants(probability)
        if not variants:
            skipped.append(
                dict(
                    object_id=row["object_id"],
                    image_id=row["image_id"],
                    reason="insufficient_foreground_support",
                )
            )
            continue
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        candidates = model.predict(image, variants)
        torch.cuda.synchronize()
        name = Path(row["crop"]["path"]).stem
        arrays = {"baseline": probability >= 0.1}
        proposals = []
        core = probability >= max(0.2, 0.65 * float(probability.max()))
        for rank, candidate in enumerate(candidates):
            mask = candidate.pop("logits") > 0
            coverage = float((mask & core).sum() / max(1, core.sum()))
            ratio = float(mask.sum() / max(1, arrays["baseline"].sum()))
            candidate.update(
                key=f"candidate_{rank}",
                core_coverage=coverage,
                area_ratio=ratio,
                geometry_eligible=bool(coverage >= 0.9 and 0.3 <= ratio <= 3.0),
            )
            arrays[candidate["key"]] = mask
            proposals.append(candidate)
        path = args.output / "masks" / (name + ".npz")
        np.savez_compressed(path, **arrays)
        shown = [dict(key="baseline", variant="baseline", predicted_iou=0.0)]
        for variant in variants:
            candidates_v = [p for p in proposals if p["variant"] == variant["name"]]
            shown.append(
                max(
                    candidates_v,
                    key=lambda c: (c["geometry_eligible"], c["predicted_iou"]),
                )
            )
        canvas = Image.new("RGB", (1600, 500), "#111827")
        draw = ImageDraw.Draw(canvas)
        for col, candidate in enumerate(shown):
            mask = arrays[candidate["key"]]
            rgb = np.asarray(image).copy()
            rgb[mask] = (rgb[mask] * 0.6 + np.array([40, 210, 255]) * 0.4).astype(
                np.uint8
            )
            tile = Image.fromarray(rgb)
            tile.thumbnail((390, 440))
            canvas.paste(
                tile,
                (col * 400 + (390 - tile.width) // 2, 40 + (440 - tile.height) // 2),
            )
            draw.text(
                (col * 400 + 8, 12), f"{col}: {candidate['variant']}", fill="white"
            )
        visual = args.output / "visuals" / (name + ".jpg")
        canvas.save(visual, quality=92)
        rows.append(
            dict(
                **row,
                candidates=proposals,
                display_candidates=shown,
                masks=describe_file(path),
                seconds=time.monotonic() - tick,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                encoder_calls=1,
            )
        )
        if (index + 1) % 10 == 0:
            print(json.dumps(dict(done=index + 1, total=len(chosen))), flush=True)
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.sam-refinement-proposals.v1",
            observations=rows,
            skipped=skipped,
            evidence=describe_file(args.evidence),
            model_config=describe_file(args.model / "config.json"),
            model_load_seconds=load_seconds,
            total_seconds=time.monotonic() - started,
            release_eligible=False,
        ),
    )


def materialize(args):
    from tools.farm_shaper_bridge.common import (
        load_run,
        resolve_mask_path,
        load_mask_pair,
    )
    from farm_runtime.quality.camera_refinement import apply_camera_refinement
    from farm_runtime.quality.mask_refinement import checked_file
    from farm_runtime.segmentation_refinement import (
        select_mask_proposal,
        restore_crop_mask,
    )

    proposals = json.loads(args.proposals.read_text())
    evidence = json.loads(checked_file(proposals["evidence"]).read_text())
    split = json.loads(checked_file(evidence["split"]).read_text())
    run = apply_camera_refinement(
        load_run(Path(evidence["run"]), allow_legacy=True),
        checked_file(evidence["cameras"]),
    )
    review = json.loads(args.review.read_text()) if args.review else {"objects": []}
    if args.review and review.get("schema") != "farm.local-object-review.v1":
        raise ValueError(
            "mask choices require a mask-review manifest, not semantic roles"
        )
    if (
        args.review
        and review["proposals"]["sha256"] != describe_file(args.proposals)["sha256"]
    ):
        raise ValueError("review belongs to different proposals")
    decisions = {}
    for obj in review["objects"]:
        if obj["parsed"] is not None:
            for row in obj["parsed"]["views"]:
                decisions[(obj["object_id"], row["image_id"])] = row["choice"]
    observations = defaultdict(list)
    for obj in run.objects:
        for observation in obj.observations:
            observations[(obj.object_id, observation.image_id)].append(observation)
    args.output.mkdir(parents=True)
    (args.output / "masks").mkdir()
    masks, audit = [], []
    for row in proposals["observations"]:
        key = (row["object_id"], row["image_id"])
        frame = run.frame(row["image_id"])
        if (
            frame.physical_timestamp not in split["build_timestamps"]
            or frame.physical_timestamp in split["heldout_timestamps"]
        ):
            raise ValueError("only build observations may be refined")
        with np.load(checked_file(row["masks"]), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        selected, reason = select_mask_proposal(
            row, arrays, decisions.get(key), key in decisions
        )
        audit.append(
            dict(object_id=key[0], image_id=key[1], selected=selected, reason=reason)
        )
        if selected is None:
            continue
        raw, inlier = np.zeros(frame.depth_size, bool), np.zeros(frame.depth_size, bool)
        sources = []
        for obs in observations[key]:
            source_path = resolve_mask_path(run, obs)
            child_raw, child_inlier, _ = load_mask_pair(source_path, frame.depth_size)
            raw |= child_raw
            inlier |= child_inlier
            sources.append(describe_file(source_path))
        box, local = restore_crop_mask(arrays[selected], row)
        x0, y0, x1, y1 = box
        raw[y0:y1, x0:x1] = local
        inlier &= raw
        h, w = raw.shape
        path = args.output / "masks" / f"{key[0]:06d}_{key[1]:06d}.npz"
        np.savez_compressed(
            path,
            image_shape=np.array([h, w]),
            raw_bits=np.packbits(raw, bitorder="little"),
            raw_shape=np.array([h, w]),
            raw_bbox_xyxy=np.array([0, 0, w, h]),
            inlier_bits=np.packbits(inlier, bitorder="little"),
            inlier_shape=np.array([h, w]),
            inlier_bbox_xyxy=np.array([0, 0, w, h]),
        )
        masks.append(
            dict(
                object_id=key[0],
                image_id=key[1],
                physical_timestamp_ns=frame.timestamp_ns,
                mask=describe_file(path),
                original_masks=sources,
                selected=selected,
                reason=reason,
            )
        )
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.build-mask-refinement.v1",
            run_success_sha256=run.success_sha256,
            meters_per_scene_unit=run.meters_per_scene_unit,
            masks=masks,
            audit=audit,
            split=evidence["split"],
            cameras=evidence["cameras"],
            proposals=describe_file(args.proposals),
            review=describe_file(args.review) if args.review else None,
            heldout_masks_unchanged=True,
            reserved_test_opened=False,
            release_eligible=False,
        ),
    )
    print(json.dumps(dict(selected=len(masks), total=len(audit))))


def orient(args):
    """Reorient cached build evidence without rerendering the scene."""
    from tools.farm_shaper_bridge.common import load_run
    from farm_runtime.quality.camera_refinement import apply_camera_refinement

    source = json.loads(args.evidence.read_text())
    run = apply_camera_refinement(
        load_run(Path(source["run"]), allow_legacy=True),
        Path(source["cameras"]["path"]),
    )
    if (
        describe_file(Path(source["cameras"]["path"]))["sha256"]
        != source["cameras"]["sha256"]
    ):
        raise ValueError("source cameras changed")
    args.output.mkdir(parents=True)
    (args.output / "crops").mkdir()
    (args.output / "seeds").mkdir()
    rows = []
    for row in source["observations"]:
        for field in ("crop", "seeds", "source_image"):
            if (
                describe_file(Path(row[field]["path"]))["sha256"]
                != row[field]["sha256"]
            ):
                raise ValueError("evidence changed")
        frame = run.frame(row["image_id"])
        with np.load(row["seeds"]["path"], allow_pickle=False) as seed:
            probability = rotate_image(seed["probability"], -row["turns"])
            legacy = rotate_image(seed["legacy"], -row["turns"])
        y, x = np.where(probability >= 0.1)
        x0, y0, x1, y1 = row["crop_grid_xyxy"]
        pixel = [
            x0 + float(np.median(x)) * (x1 - x0) / probability.shape[1],
            y0 + float(np.median(y)) * (y1 - y0) / probability.shape[0],
        ]
        orientation = upright_quarter_turns(
            frame.T_world_cam[:3, :3].T,
            args.world_up,
            image_point=pixel,
            intrinsics=frame.K,
        )
        turns = orientation["applied_quarter_turns_ccw"]
        with Image.open(row["source_image"]["path"]) as im:
            rgb = np.asarray(im.convert("RGB").crop(row["crop_source_xyxy"]))
        crop = args.output / "crops" / Path(row["crop"]["path"]).name
        seeds = args.output / "seeds" / Path(row["seeds"]["path"]).name
        Image.fromarray(rotate_image(rgb, turns)).save(crop, quality=95)
        np.savez_compressed(
            seeds,
            probability=rotate_image(probability, turns),
            legacy=rotate_image(legacy, turns),
        )
        rows.append(
            dict(
                row,
                turns=turns,
                orientation=orientation,
                crop=describe_file(crop),
                seeds=describe_file(seeds),
            )
        )
    write_json(
        args.output / "manifest.json",
        dict(
            source,
            observations=rows,
            reoriented_from=describe_file(args.evidence),
            world_up=args.world_up,
        ),
    )


def vlm(args):
    import torch
    from farm_runtime.semantic_refinement import (
        LocalObjectReviewer,
        select_review_views,
        REVIEW_PROMPT,
        SEMANTIC_PROMPT,
    )

    proposals = json.loads(args.proposals.read_text())
    args.output.mkdir(parents=True)
    (args.output / "visuals").mkdir()
    (args.output / "prompt.txt").write_text(
        SEMANTIC_PROMPT if args.semantic_only else REVIEW_PROMPT
    )
    started = time.monotonic()
    model = LocalObjectReviewer(args.model)
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - started
    groups = defaultdict(list)
    for row in proposals["observations"]:
        groups[row["object_id"]].append(row)
    results = []
    for object_id, observations in sorted(groups.items()):
        chosen = select_review_views(observations, args.views)
        images, sheets = [], []
        for row in chosen:
            image = Image.open(row["crop"]["path"]).convert("RGB")
            with np.load(row["masks"]["path"], allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            canvas = Image.new("RGB", (1750, 400), "#111827")
            draw = ImageDraw.Draw(canvas)
            for col in range(5):
                rgb = np.asarray(image).copy()
                if col:
                    mask = arrays[row["display_candidates"][col - 1]["key"]]
                    rgb[mask] = (
                        rgb[mask] * 0.6 + np.array([40, 210, 255]) * 0.4
                    ).astype(np.uint8)
                tile = Image.fromarray(rgb)
                tile.thumbnail((342, 360))
                canvas.paste(
                    tile,
                    (
                        col * 350 + (342 - tile.width) // 2,
                        30 + (360 - tile.height) // 2,
                    ),
                )
                draw.text(
                    (col * 350 + 8, 8),
                    "PHOTO" if not col else str(col - 1),
                    fill="white",
                )
            path = (
                args.output / "visuals" / f"{object_id:06d}_{row['image_id']:06d}.jpg"
            )
            if args.semantic_only:
                canvas = image.copy()
                draw = ImageDraw.Draw(canvas)
                yy, xx = np.where(arrays["baseline"])
                if len(xx):
                    draw.rectangle(
                        (int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())),
                        outline="#ffd84d",
                        width=max(1, round(max(image.size) / 350)),
                    )
            canvas.save(path, quality=94)
            images.append(canvas)
            sheets.append(describe_file(path))
        torch.cuda.reset_peak_memory_stats()
        response = model.review(
            images, [r["image_id"] for r in chosen], semantic_only=args.semantic_only
        )
        result = dict(
            object_id=object_id,
            **response,
            sheets=sheets,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        )
        results.append(result)
        write_json(args.output / f"object_{object_id:06d}.json", result)
        print(json.dumps(dict(object_id=object_id, response=response)), flush=True)
    write_json(
        args.output / "manifest.json",
        dict(
            schema=(
                "farm.local-object-semantics.v1"
                if args.semantic_only
                else "farm.local-object-review.v1"
            ),
            objects=results,
            proposals=describe_file(args.proposals),
            model_config=describe_file(args.model / "config.json"),
            prompt=describe_file(args.output / "prompt.txt"),
            model_load_seconds=load_seconds,
            total_seconds=time.monotonic() - started,
            human_labels_in_prompt=False,
            reserved_test_opened=False,
            release_eligible=False,
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    p = sub.add_parser("prepare")
    for name in (
        "run",
        "cameras",
        "split",
        "packet",
        "bank",
        "config",
        "images",
        "ply",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--world-up", type=float, nargs=3, required=True)
    p.add_argument("--source-commit", required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser("sam")
    for name in ("evidence", "model", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=sam)
    p = sub.add_parser("materialize")
    for name in ("proposals", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--review", type=Path)
    p.set_defaults(func=materialize)
    p = sub.add_parser("orient")
    for name in ("evidence", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--world-up", type=float, nargs=3, required=True)
    p.set_defaults(func=orient)
    p = sub.add_parser("vlm")
    for name in ("proposals", "model", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--views", type=int, choices=(1, 2, 3), default=2)
    p.add_argument("--semantic-only", action="store_true")
    p.set_defaults(func=vlm)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("output must be new")
    return args.func(args)


if __name__ == "__main__":
    main()
