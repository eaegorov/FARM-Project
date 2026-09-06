"""Propose local RGB masks from registered surface prompts, without object labels.

Projections are prompts, never independent foreground observations. The result
must pass surface-validation before it can be used by native-observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from farm_runtime.angular_discovery import rotate_image
from farm_runtime.proposal_geometry import project_evidence
from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.proposal_geometry import read_masks, read_observations
from farm_runtime.quality.surface_evidence import SurfaceInputs
from farm_runtime.quality_baseline import describe_file, write_json
from farm_runtime.segmentation_refinement import CachedSAMRefiner, prompt_variants


def projected_crop(points, frame, image, turns):
    evidence = project_evidence(
        points,
        np.ones(frame["depth"].shape, bool),
        **frame,
        mask_is_dilated=True,
        include_point_indices=True,
    )
    xy = np.rint(evidence["surface_pixels_xy"]).astype(int)
    h, w = frame["depth"].shape
    xy = xy[(xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)]
    if len(xy) < 20:
        return None
    support = np.zeros((h, w), bool)
    support[xy[:, 1], xy[:, 0]] = True
    support = binary_dilation(support, iterations=1)
    y, x = np.where(support)
    padding = max(12, round(0.5 * max(np.ptp(x), np.ptp(y))))
    x0, x1 = max(0, int(x.min()) - padding), min(w, int(x.max()) + padding + 1)
    y0, y1 = max(0, int(y.min()) - padding), min(h, int(y.max()) + padding + 1)
    box = [
        round(x0 * image.width / w),
        round(y0 * image.height / h),
        round(x1 * image.width / w),
        round(y1 * image.height / h),
    ]
    crop = image.crop(box)
    seed = np.asarray(
        Image.fromarray(support[y0:y1, x0:x1]).resize(
            crop.size, Image.Resampling.NEAREST
        )
    )
    return dict(
        crop=Image.fromarray(rotate_image(np.asarray(crop), turns)),
        seed=rotate_image(seed, turns).astype(np.float32),
        crop_grid_xyxy=[x0, y0, x1, y1],
        crop_source_xyxy=box,
        turns=turns,
        visible_surface_points=len(xy),
        crop_reaches_image_edge=bool(x0 == 0 or y0 == 0 or x1 == w or y1 == h),
    )


def restore_crop_logits(logits, turns, window):
    logits = np.asarray(logits, np.float32)
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError("finite two-dimensional tracker logits required")
    x0, y0, x1, y1 = window
    if not 0 <= x0 < x1 or not 0 <= y0 < y1:
        raise ValueError("positive crop window required")
    unrotated = rotate_image(logits, -turns)
    probability = 1 / (1 + np.exp(-np.clip(unrotated, -30, 30)))
    probability = np.asarray(
        Image.fromarray(probability).resize(
            (x1 - x0, y1 - y0), Image.Resampling.BILINEAR
        )
    )
    eps = np.finfo(np.float16).eps
    p = np.clip(probability, eps, 1 - eps)
    return np.log(p / (1 - p)).astype(np.float16)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "proposals", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--crop-budget", type=int, default=12)
    parser.add_argument(
        "--validation",
        type=Path,
        help="Limit crops to unresolved matches in this validation",
    )
    args = parser.parse_args(argv)
    if args.output.exists() or not 1 <= args.crop_budget <= 32:
        raise ValueError("new output and crop budget in 1..32 required")
    audit = json.loads(args.audit.read_text())
    if audit.get("closed_test_opened") is not False:
        raise ValueError("development-only surface audit required")
    base_doc, base_rows = read_observations(args.proposals)
    if base_doc["plan"]["sha256"] != audit["adaptive_plan"]["sha256"]:
        raise ValueError("proposals must use the selected development plan")
    plan = json.loads(checked_file(audit["adaptive_plan"]).read_text())
    base = {r["name"]: r for r in base_rows}
    unresolved = None
    if args.validation:
        validation = json.loads(args.validation.read_text())
        if (
            validation.get("closed_test_opened") is not False
            or checked_file(validation["source_audit"]).resolve()
            != args.audit.resolve()
            or checked_file(validation["source_proposals"]).resolve()
            != args.proposals.resolve()
        ):
            raise ValueError(
                "validation must bind the same development audit and proposals"
            )
        unresolved = {
            (g["group_id"], m["name"])
            for g in validation["groups"]
            for m in g["extra_matches"]
            if m["selected_detection"] is None
        }
    inputs = SurfaceInputs(checked_file(audit["source_geometry"]))
    with np.load(checked_file(audit["point_evidence"]), allow_pickle=False) as z:
        clouds = {k: z[k] for k in z.files}
    requests = [
        (g["group_id"], s["name"])
        for g in audit["groups"]
        for s in g["selected_views"]
        if unresolved is None or (g["group_id"], s["name"]) in unresolved
    ]
    if len(set(requests)) != len(requests):
        raise ValueError("unique group/view requests required")
    prepared = {}
    for _, name in requests:
        if name in prepared:
            continue
        obs = base[name]
        if (
            name not in inputs.trusted
            or obs["source_image"] != plan["sources"][name]["source_image"]
            or obs["timestamp"] != str(inputs.frames[name]["frame_id"])
            or not any(q["prompt"] == "person" for q in obs["queries"])
        ):
            raise ValueError(
                "registered source identity and explicit person query required"
            )
        frame = dict(inputs.frame(name))
        if tuple(obs["grid_shape_hw"]) != frame["depth"].shape:
            raise ValueError("registered observation grid mismatch")
        masks = read_masks(obs, args.proposals.parent)
        people = [m for m, d in zip(masks, obs["detections"]) if d["label"] == "person"]
        if people:
            frame["excluded"] = frame["excluded"] | binary_dilation(
                np.logical_or.reduce(people), iterations=2
            )
        prepared[name] = frame
    args.output.mkdir(parents=True)
    for name in ("masks", "crops", "visuals"):
        (args.output / name).mkdir()
    tick = time.monotonic()
    model, load_seconds = None, 0.0
    records, skipped, packed = [], [], {}
    for gid, name in requests[: args.crop_budget]:
        prefix = f"group_{gid:04d}"
        with Image.open(checked_file(base[name]["source_image"])) as image:
            packet = projected_crop(
                clouds[prefix + "_points"][clouds[prefix + "_core"]],
                prepared[name],
                image.convert("RGB"),
                base[name]["applied_quarter_turns"],
            )
        if packet is None:
            skipped.append(
                dict(group_id=gid, name=name, reason="insufficient_visible_surface")
            )
            continue
        crop, seed = packet.pop("crop"), packet.pop("seed")
        # No negative ring: unobserved surface is unknown, not background.
        variants = prompt_variants(seed)[:2]
        if not variants:
            skipped.append(
                dict(group_id=gid, name=name, reason="insufficient_foreground_support")
            )
            continue
        import torch

        if model is None:
            start = time.monotonic()
            model = CachedSAMRefiner(args.model)
            torch.cuda.synchronize()
            load_seconds = time.monotonic() - start
        stem = f"{gid:04d}_{Path(name).stem}"
        crop_path = args.output / "crops" / f"{stem}.png"
        crop.save(crop_path)
        start = time.monotonic()
        candidates = model.predict(crop, variants)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        arrays, detections = packed.setdefault(name, ({}, []))
        canvas = Image.new("RGB", (360 * (len(candidates) + 1), 420), "#111827")
        draw = ImageDraw.Draw(canvas)
        shown = [
            dict(variant="projected support", logits=seed - 0.5, predicted_iou=0),
            *candidates,
        ]
        for col, candidate in enumerate(shown):
            rgb = np.asarray(crop).copy()
            mask = candidate["logits"] > 0
            rgb[mask] = (rgb[mask] * 0.6 + np.array([30, 210, 255]) * 0.4).astype(
                np.uint8
            )
            tile = Image.fromarray(rgb)
            tile.thumbnail((350, 360))
            canvas.paste(
                tile,
                (col * 360 + (350 - tile.width) // 2, 50 + (360 - tile.height) // 2),
            )
            draw.text(
                (col * 360 + 5, 8),
                f"{candidate['variant']} score {candidate['predicted_iou']:.3f}",
                fill="white",
            )
        visual = args.output / "visuals" / f"{stem}.jpg"
        canvas.save(visual, quality=92)
        for candidate in candidates:
            tile = restore_crop_logits(
                candidate["logits"], packet["turns"], packet["crop_grid_xyxy"]
            )
            key = f"group_{gid:04d}_candidate_{len(detections):03d}"
            arrays[key] = tile
            detections.append(
                dict(
                    label="object",
                    score=candidate["predicted_iou"],
                    score_kind="tracker_predicted_iou",
                    logit_key=key,
                    grid_window_xyxy=packet["crop_grid_xyxy"],
                    positive_grid_pixels=int((tile > 0).sum()),
                    source_group_id=gid,
                    variant=candidate["variant"],
                )
            )
        records.append(
            dict(
                group_id=gid,
                name=name,
                source_image=base[name]["source_image"],
                **packet,
                crop=describe_file(crop_path),
                visual=describe_file(visual),
                prompts=variants,
                candidates=len(candidates),
                seconds=elapsed,
                encoder_calls=1,
            )
        )
    rows = []
    for name, (arrays, detections) in packed.items():
        path = args.output / "masks" / (Path(name).stem + ".npz")
        np.savez_compressed(path, **arrays)
        rows.append(
            dict(
                base[name],
                detections=detections,
                queries=[],
                mask_artifact=describe_file(path),
                logit_semantics="SAM tracker probability resized on original RGB grid; >0 foreground",
            )
        )
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.projected-surface-tracker.v1",
            observations=rows,
            source_audit=describe_file(args.audit),
            source_proposals=describe_file(args.proposals),
            source_validation=(
                describe_file(args.validation) if args.validation else None
            ),
            model_weights=(
                describe_file(args.model / "model.safetensors") if model else None
            ),
            model_load_seconds=load_seconds,
            total_seconds=time.monotonic() - tick,
            crops=records,
            skipped=skipped,
            crop_budget=args.crop_budget,
            deferred_requests=[
                dict(group_id=g, name=n) for g, n in requests[args.crop_budget :]
            ],
            source_image_encoder_calls=len(records),
            test_opened=False,
            release_eligible=False,
            labels_are_semantic=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
