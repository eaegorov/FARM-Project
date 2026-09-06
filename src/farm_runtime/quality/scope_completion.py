"""Propose visible missing object parts from validated RGB crops.

VLM coordinates only prompt SAM. The resulting masks remain proposals and must
pass surface-validation before native lifting; confidence is not geometry.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from farm_runtime.angular_discovery import rotate_image
from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.proposal_geometry import read_masks, read_observations
from farm_runtime.quality.surface_tracker import restore_crop_logits
from farm_runtime.quality_baseline import describe_file, write_json
from farm_runtime.segmentation_refinement import interior_points

PROMPT = """Inspect the physical completeness of a segmentation. Image0 is the original photograph. Image1 is an aligned binary diagram of the current mask (white selected, black unselected); it does not show material or color. The target is the physical object covered by the white mask.
Does the mask omit a visible integral part of this same object? Distinguish object parts from neighboring objects, external cables and background. Objects can have integral sections of different materials or colors. Do not include external connecting cables. Do not invent hidden shape. If a visible integral part is clearly missing, give a bounding box around the entire visible object and 1 to 3 points well inside the missing integral parts. Optionally give up to 3 background points inside visible external connecting cables or adjacent surfaces. Otherwise abstain from new prompts.
All coordinates use the ORIGINAL Image0 normalized to 0..1000, x from left and y from top. Return ONLY JSON: {"scope_complete": boolean, "box": [x0,y0,x1,y1] or null, "add_foreground": [[x,y],...], "background": [[x,y],...], "confidence": "high|medium|low", "reason": "brief visual explanation"}."""


def validate_response(raw, image_ids=None):
    result = json.loads(raw)
    if (
        not isinstance(result, dict)
        or set(result)
        != {
            "scope_complete",
            "box",
            "add_foreground",
            "background",
            "confidence",
            "reason",
        }
        or type(result["scope_complete"]) is not bool
    ):
        raise ValueError("exact completeness response fields required")
    if result["confidence"] not in ("high", "medium", "low") or not isinstance(
        result["reason"], str
    ):
        raise ValueError("invalid completeness confidence/reason")

    def coordinates(value, count):
        return (
            isinstance(value, list)
            and len(value) == count
            and all(
                type(v) in (int, float) and np.isfinite(v) and 0 <= v <= 1000
                for v in value
            )
        )

    for key in ("add_foreground", "background"):
        points = result[key]
        if (
            not isinstance(points, list)
            or len(points) > 3
            or any(not coordinates(p, 2) for p in points)
        ):
            raise ValueError("at most three finite normalized points required")
    box = result["box"]
    if box is not None and (
        not coordinates(box, 4) or box[0] >= box[2] or box[1] >= box[3]
    ):
        raise ValueError("positive normalized box required")
    return result


def completion_prompts(result, current):
    """Keep the existing extent when VLM returns a box around only a missing part."""
    current = np.asarray(current, bool)
    if current.ndim != 2 or not current.any():
        raise ValueError("nonempty two-dimensional current mask required")
    if (
        result is None
        or result["scope_complete"]
        or result["confidence"] != "high"
        or result["box"] is None
    ):
        return []
    h, w = current.shape

    def outside(points):
        scaled = [
            [min(w - 1, x * w / 1000), min(h - 1, y * h / 1000)] for x, y in points
        ]
        # A VLM background point may never contradict the existing object core.
        return [p for p in scaled if not current[round(p[1]), round(p[0])]]

    positive, negative = outside(result["add_foreground"]), outside(
        result["background"]
    )
    if not positive:
        return []
    ys, xs = np.where(current)
    box = [result["box"][i] * (w if i % 2 == 0 else h) / 1000 for i in range(4)]
    box = [
        min(box[0], float(xs.min())),
        min(box[1], float(ys.min())),
        max(min(w - 1, box[2]), float(xs.max())),
        max(min(h - 1, box[3]), float(ys.max())),
    ]
    core = interior_points(current, 2)
    return [
        dict(name="vlm_box", box=box, points=[], labels=[]),
        dict(
            name="vlm_missing_parts",
            box=box,
            points=core + positive + negative,
            labels=[1] * (len(core) + len(positive)) + [0] * len(negative),
        ),
    ]


def validated_requests(tracker_path, validation_path):
    """Resolve flattened detection indices, including multiple objects in one view."""
    tracker, rows = read_observations(tracker_path)
    validation = json.loads(validation_path.read_text())
    if (
        tracker.get("schema") != "farm.projected-surface-tracker.v1"
        or tracker.get("release_eligible") is not False
    ):
        raise ValueError("development surface tracker required")
    if (
        validation.get("schema") != "farm.surface-validation.v1"
        or validation.get("closed_test_opened") is not False
        or validation.get("release_eligible") is not False
    ):
        raise ValueError("development surface validation required")
    for key in ("source_audit", "source_proposals"):
        if (
            checked_file(tracker[key]).resolve()
            != checked_file(validation[key]).resolve()
        ):
            raise ValueError("tracker and validation source mismatch")
    sources = [checked_file(validation["source_proposals"])] + [
        checked_file(d) for d in validation.get("supplements", [])
    ]
    source_ids = [
        i for i, path in enumerate(sources) if path.resolve() == tracker_path.resolve()
    ]
    if len(source_ids) != 1:
        raise ValueError("tracker must appear exactly once in validation supplements")
    source_id = source_ids[0]
    offsets = {}
    for path in sources[:source_id]:
        _, previous = read_observations(path)
        for row in previous:
            offsets[row["name"]] = offsets.get(row["name"], 0) + len(row["detections"])
    by_name = {row["name"]: row for row in rows}
    crops = {(row["group_id"], row["name"]): row for row in tracker["crops"]}
    if len(by_name) != len(rows) or len(crops) != len(tracker["crops"]):
        raise ValueError("unique source observations and group/view crops required")
    requests, skipped, seen = [], [], set()
    for group in validation["groups"]:
        for match in group["extra_matches"]:
            key = group["group_id"], match["name"]
            if key in seen:
                raise ValueError("duplicate validation group/view")
            seen.add(key)
            if key not in crops:
                continue
            selected = match["selected_detection"]
            if match["decision"] != "matched_static_surface" or selected is None:
                skipped.append(
                    dict(group_id=key[0], name=key[1], reason="unresolved_identity")
                )
                continue
            local = selected - offsets.get(key[1], 0)
            obs, record = by_name[key[1]], crops[key]
            if obs["timestamp"] != match["timestamp"]:
                raise ValueError("selected observation timestamp mismatch")
            if not 0 <= local < len(obs["detections"]):
                skipped.append(
                    dict(group_id=key[0], name=key[1], reason="selected_other_source")
                )
                continue
            detection = obs["detections"][local]
            eligible = next(
                (c for c in match["candidates"] if c["detection_index"] == selected),
                None,
            )
            if (
                eligible is None
                or eligible["eligible"] is not True
                or detection["source_group_id"] != key[0]
            ):
                raise ValueError("accepted tracker detection/group mismatch")
            if (
                detection["grid_window_xyxy"] != record["crop_grid_xyxy"]
                or obs["source_image"] != record["source_image"]
                or obs["applied_quarter_turns"] != record["turns"]
            ):
                raise ValueError("selected crop identity/window mismatch")
            requests.append(
                dict(
                    record=record,
                    observation=obs,
                    detection_index=local,
                    selected_detection=selected,
                    timestamp=match["timestamp"],
                )
            )
    return requests, skipped


def bounded_requests(requests, budget, per_object):
    groups = {}
    for request in requests:
        gid = request["record"]["group_id"]
        queue = groups.setdefault(gid, [])
        if request["timestamp"] not in {r["timestamp"] for r in queue}:
            queue.append(request)
    chosen = [
        queue[i]
        for i in range(per_object)
        for queue in groups.values()
        if i < len(queue)
    ][:budget]
    return chosen


def review_canvas(crop, current, prompts, candidates):
    views = [
        ("Original RGB", crop),
        ("Current mask (extent only)", Image.fromarray(current).convert("RGB")),
    ]
    marked = crop.copy()
    draw = ImageDraw.Draw(marked)
    if prompts:
        draw.rectangle(prompts[-1]["box"], outline="yellow", width=3)
        for (x, y), label in zip(prompts[-1]["points"], prompts[-1]["labels"]):
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="lime" if label else "red")
        views.append(("VLM prompts (not evidence)", marked))
    for i, candidate in enumerate(candidates):
        rgb = np.asarray(crop).copy()
        mask = candidate["logits"] > 0
        rgb[mask] = (rgb[mask] * 0.6 + np.array([30, 210, 255]) * 0.4).astype(np.uint8)
        views.append((f'{i}: {candidate["variant"]}', Image.fromarray(rgb)))
    cols = min(3, len(views))
    canvas = Image.new(
        "RGB", (360 * cols, 400 * ((len(views) + cols - 1) // cols)), "#111827"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(views):
        x, y = index % cols * 360, index // cols * 400
        image = ImageOps.contain(image, (350, 350))
        canvas.paste(
            image, (x + (360 - image.width) // 2, y + 40 + (350 - image.height) // 2)
        )
        draw.text((x + 5, y + 8), title, fill="white")
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tracker", "validation", "model", "sam-model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--crop-budget", type=int, default=4)
    parser.add_argument("--crops-per-object", type=int, default=2)
    args = parser.parse_args(argv)
    if (
        args.output.exists()
        or not 1 <= args.crop_budget <= 16
        or not 1 <= args.crops_per_object <= 3
    ):
        raise ValueError(
            "new output, crop budget 1..16 and per-object budget 1..3 required"
        )
    requests, skipped = validated_requests(args.tracker, args.validation)
    selected = bounded_requests(requests, args.crop_budget, args.crops_per_object)
    args.output.mkdir(parents=True)
    for folder in ("masks", "visuals", "diagrams"):
        (args.output / folder).mkdir()
    (args.output / "prompt.txt").write_text(PROMPT)
    tick = time.monotonic()
    reviewer, refiner, packets, mask_cache = None, None, [], {}
    for request in selected:
        record, obs = request["record"], request["observation"]
        name = record["name"]
        if name not in mask_cache:
            mask_cache[name] = read_masks(obs, args.tracker.parent)
        mask = mask_cache[name][request["detection_index"]]
        x0, y0, x1, y1 = record["crop_grid_xyxy"]
        with Image.open(checked_file(record["crop"])) as image:
            crop = image.convert("RGB")
        current = np.asarray(
            Image.fromarray(rotate_image(mask[y0:y1, x0:x1], record["turns"])).resize(
                crop.size, Image.Resampling.NEAREST
            ),
            dtype=bool,
        )
        if not current.any():
            raise ValueError("selected crop mask is empty")
        diagram = Image.fromarray(current).convert("RGB")
        stem = f'{record["group_id"]:04d}_{Path(name).stem}'
        diagram_path = args.output / "diagrams" / (stem + ".png")
        diagram.save(diagram_path)
        if reviewer is None:
            from farm_runtime.semantic_refinement import LocalObjectReviewer

            reviewer = LocalObjectReviewer(args.model)
        response = reviewer.ask(
            PROMPT, [crop, diagram], [0, 1], validate_response, max_new_tokens=320
        )
        if response.get("validation_error") is None:
            parsed = validate_response(response["raw"])
            if parsed != response["parsed"]:
                raise ValueError("parsed completeness response differs from raw")
        else:
            parsed = None
        prompts = completion_prompts(parsed, current)
        packets.append((request, crop, current, response, prompts, diagram_path))
    used_reviewer = reviewer is not None
    if used_reviewer:
        del reviewer
        gc.collect()
        import torch

        torch.cuda.empty_cache()
    packed, outputs = {}, []
    sam_seconds = 0.0
    for request, crop, current, response, prompts, diagram_path in packets:
        record, obs = request["record"], request["observation"]
        name = record["name"]
        candidates = []
        if prompts:
            if refiner is None:
                from farm_runtime.segmentation_refinement import CachedSAMRefiner

                refiner = CachedSAMRefiner(args.sam_model)
            start = time.monotonic()
            candidates = refiner.predict(crop, prompts)
            sam_seconds += time.monotonic() - start
        arrays, detections, _ = packed.setdefault(name, ({}, [], obs))
        for candidate in candidates:
            tile = restore_crop_logits(
                candidate["logits"], record["turns"], record["crop_grid_xyxy"]
            )
            key = f"mask_{len(detections):04d}"
            arrays[key] = tile
            detections.append(
                dict(
                    label="object",
                    score=candidate["predicted_iou"],
                    score_kind="tracker_predicted_iou",
                    logit_key=key,
                    grid_window_xyxy=record["crop_grid_xyxy"],
                    positive_grid_pixels=int((tile > 0).sum()),
                    source_group_id=record["group_id"],
                    variant=candidate["variant"],
                )
            )
        stem = f'{record["group_id"]:04d}_{Path(name).stem}'
        visual = args.output / "visuals" / (stem + ".jpg")
        review_canvas(crop, current, prompts, candidates).save(visual, quality=92)
        outputs.append(
            dict(
                group_id=record["group_id"],
                name=name,
                source_crop=record["crop"],
                source_image=record["source_image"],
                source_mask=describe_file(diagram_path),
                selected_detection=request["selected_detection"],
                local_detection_index=request["detection_index"],
                response=response,
                prompts=prompts,
                candidates=len(candidates),
                visual=describe_file(visual),
            )
        )
    observations = []
    for name, (arrays, detections, obs) in packed.items():
        path = args.output / "masks" / (Path(name).stem + ".npz")
        np.savez_compressed(path, **arrays)
        observations.append(
            dict(
                obs,
                detections=detections,
                queries=[],
                mask_artifact=describe_file(path),
            )
        )
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.visible-scope-completion.v1",
            observations=observations,
            source_tracker=describe_file(args.tracker),
            source_validation=describe_file(args.validation),
            prompt=describe_file(args.output / "prompt.txt"),
            model_config=(
                describe_file(args.model / "config.json") if used_reviewer else None
            ),
            sam_weights=(
                describe_file(args.sam_model / "model.safetensors") if refiner else None
            ),
            outputs=outputs,
            skipped=skipped,
            eligible_requests=len(requests),
            crop_budget=args.crop_budget,
            crops_per_object=args.crops_per_object,
            deferred_requests=[
                dict(group_id=r["record"]["group_id"], name=r["record"]["name"])
                for r in requests
                if r not in selected
            ],
            vlm_calls=len(packets),
            sam_encoder_calls=sum(bool(p[4]) for p in packets),
            sam_prediction_seconds=sam_seconds,
            total_seconds=time.monotonic() - tick,
            preserve_core_box=True,
            test_opened=False,
            closed_test_opened=False,
            release_eligible=False,
            native_selection_applied=False,
            interpretation="VLM prompts propose visible integral parts, never evidence of ownership. Validate masks geometrically before native lifting.",
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
