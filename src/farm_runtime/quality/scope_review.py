"""Bounded cross-view VLM arbitration of geometrically eligible object scopes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from farm_runtime.angular_discovery import rotate_image
from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.proposal_geometry import read_masks, read_observations
from farm_runtime.quality.surface_evidence import SurfaceInputs
from farm_runtime.quality_baseline import describe_file, write_json

PROMPT = """Choose the segmentation candidate preserving the SAME physical object scope across two registered photographs. Sheet0 shows REFERENCE PHOTO and its selected MASK: white means selected pixels, black means unselected pixels. Sheet1 shows the NEW PHOTO and numbered candidate MASK DIAGRAMS. The diagrams describe extent only, never color or material.
Choose the candidate covering the visible complete object represented in the reference, including integral parts, while excluding neighboring objects and external infrastructure. Do not substitute a frame, a single panel, or a disconnected patch for the reference object. Do not fill genuine openings revealing unrelated background. All candidates passed geometric identity checks; geometry alone did not resolve physical scope. Compare actual photos and mask boundaries. Abstain if a unique acceptable mask cannot be established. No category labels or model scores are supplied.
Return ONLY JSON with exactly these fields: {"candidate_id": integer identifier from Sheet1 or null, "confidence": "high|medium|low", "reason": "brief visual reason, at most 40 words"}."""


def validate_response(raw, eligible):
    result = json.loads(raw)
    if not isinstance(result, dict) or set(result) != {
        "candidate_id",
        "confidence",
        "reason",
    }:
        raise ValueError("exact scope response fields required")
    choice = result["candidate_id"]
    if choice is not None and (type(choice) is not int or choice not in eligible):
        raise ValueError("scope choice must be a geometrically eligible candidate")
    if result["confidence"] not in ("high", "medium", "low") or not isinstance(
        result["reason"], str
    ):
        raise ValueError("invalid scope confidence/reason")
    return result


def consensus_choice(responses, eligible):
    """Two distinct display orders must agree; abstention never changes geometry."""
    if len(responses) != 2:
        return None
    choices, orders = [], []
    for response in responses:
        order = response["order"]
        if sorted(order) != sorted(eligible) or len(order) != len(set(order)):
            raise ValueError("scope display order/candidate mismatch")
        orders.append(order)
        if response.get("validation_error") is not None:
            return None
        parsed = validate_response(response["raw"], eligible)
        if parsed != response["parsed"]:
            raise ValueError("scope parsed response differs from raw output")
        if parsed["confidence"] != "high" or parsed["candidate_id"] is None:
            return None
        choices.append(parsed["candidate_id"])
    return choices[0] if orders[0] != orders[1] and choices[0] == choices[1] else None


def load_scope_choices(path, audit, proposals, supplements, partial_view):
    """Bind arbitration to the exact validation inputs and candidate evidence."""
    review = json.loads(Path(path).read_text())
    if (
        review.get("schema") != "farm.cross-view-scope-review.v1"
        or review.get("closed_test_opened") is not False
        or review.get("release_eligible") is not False
    ):
        raise ValueError("development scope review required")
    baseline = json.loads(checked_file(review["validation"]).read_text())
    if (
        baseline.get("closed_test_opened") is not False
        or baseline.get("release_eligible") is not False
        or baseline.get("scope_review") is not None
        or baseline.get("partial_view_association", False) != partial_view
    ):
        raise ValueError(
            "scope review requires the same unarbitrated validation policy"
        )
    for descriptor, current in [
        (baseline["source_audit"], audit),
        (baseline["source_proposals"], proposals),
    ]:
        if checked_file(descriptor).resolve() != Path(current).resolve():
            raise ValueError("scope review validation source mismatch")
    bound = [checked_file(d).resolve() for d in baseline.get("supplements", [])]
    if bound != [Path(p).resolve() for p in supplements]:
        raise ValueError("scope review supplement order mismatch")
    checked_file(review["prompt"])
    if review.get("model_config") is not None:
        checked_file(review["model_config"])
    matches = {
        (g["group_id"], m["name"]): m
        for g in baseline["groups"]
        for m in g["extra_matches"]
    }
    result = {}
    for row in review["reviews"]:
        key = row["group_id"], row["name"]
        if key in result or key not in matches:
            raise ValueError("duplicate or absent scope review match")
        match = matches[key]
        eligible = sorted(
            c["detection_index"] for c in match["candidates"] if c["eligible"]
        )
        if (
            match["decision"] != "ambiguous_competing_scope"
            or row["eligible_candidates"] != eligible
        ):
            raise ValueError("scope review candidates differ from geometric validation")
        checked_file(row["reference_image"])
        checked_file(row["reference_canvas"])
        for response in row["responses"]:
            checked_file(response["canvas"])
        result[key] = dict(
            choice=consensus_choice(row["responses"], eligible),
            candidates=match["candidates"],
        )
    return result


def crop_packet(obs, masks):
    """Original RGB plus aligned diagrams, including neighboring context."""
    union = np.logical_or.reduce(masks)
    y, x = np.where(union)
    if not len(x):
        raise ValueError("nonempty candidate extent required")
    h, w = union.shape
    pad = max(12, round(0.25 * max(np.ptp(x), np.ptp(y))))
    x0, x1 = max(0, int(x.min()) - pad), min(w, int(x.max()) + pad + 1)
    y0, y1 = max(0, int(y.min()) - pad), min(h, int(y.max()) + pad + 1)
    with Image.open(checked_file(obs["source_image"])) as image:
        image = image.convert("RGB")
        crop = image.crop(
            (
                round(x0 * image.width / w),
                round(y0 * image.height / h),
                round(x1 * image.width / w),
                round(y1 * image.height / h),
            )
        )
    turns = obs["applied_quarter_turns"]
    return Image.fromarray(rotate_image(np.asarray(crop), turns)), [
        Image.fromarray(rotate_image(m[y0:y1, x0:x1], turns)).convert("RGB")
        for m in masks
    ]


def paste(canvas, image, box):
    x, y, w, h = box
    tile = ImageOps.contain(image, (w, h))
    canvas.paste(tile, (x + (w - tile.width) // 2, y + (h - tile.height) // 2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("validation", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--object-budget", type=int, default=4)
    args = parser.parse_args(argv)
    if args.output.exists() or not 1 <= args.object_budget <= 16:
        raise ValueError("new output and object budget in 1..16 required")
    started = time.monotonic()
    validation = json.loads(args.validation.read_text())
    if (
        validation.get("closed_test_opened") is not False
        or validation.get("release_eligible") is not False
        or validation.get("scope_review") is not None
    ):
        raise ValueError("unarbitrated development validation required")
    audit = json.loads(checked_file(validation["source_audit"]).read_text())
    inputs = SurfaceInputs(checked_file(audit["source_geometry"]))
    extra, masks = {}, {}
    for descriptor in [
        validation["source_proposals"],
        *validation.get("supplements", []),
    ]:
        path = checked_file(descriptor)
        _, rows = read_observations(path)
        for obs in rows:
            name = obs["name"]
            if name in extra:
                if any(
                    obs[k] != extra[name][k]
                    for k in (
                        "timestamp",
                        "source_image",
                        "grid_shape_hw",
                        "applied_quarter_turns",
                    )
                ):
                    raise ValueError("scope supplement identity mismatch")
                masks[name] += read_masks(obs, path.parent)
            else:
                extra[name] = obs
                masks[name] = read_masks(obs, path.parent)
    groups = {g["id"]: g for g in inputs.geometry["groups"]}
    args.output.mkdir(parents=True)
    (args.output / "prompt.txt").write_text(PROMPT)
    reviews, deferred, reviewer = [], [], None
    model_load_seconds = 0.0
    for group in validation["groups"]:
        for match in group["extra_matches"]:
            if match["decision"] != "ambiguous_competing_scope":
                continue
            gid, name = group["group_id"], match["name"]
            eligible = sorted(
                c["detection_index"] for c in match["candidates"] if c["eligible"]
            )
            references = []
            for node_id in groups[gid]["members"]:
                node = inputs.nodes[node_id]
                mask = inputs.mask(node)
                if (
                    node["timestamp"] != match["timestamp"]
                    and mask.any()
                    and not (
                        mask[0].any()
                        or mask[-1].any()
                        or mask[:, 0].any()
                        or mask[:, -1].any()
                    )
                ):
                    references.append((int(mask.sum()), node, mask))
            reason = (
                "budget"
                if len(reviews) >= args.object_budget
                else (
                    "candidate_count"
                    if not 2 <= len(eligible) <= 4
                    else (
                        "no_unclipped_independent_reference" if not references else None
                    )
                )
            )
            if reason:
                deferred.append(dict(group_id=gid, name=name, reason=reason))
                continue
            _, node, mask = max(references, key=lambda r: (r[0], -r[1]["id"]))
            source = inputs.observations[node["frame"]]
            photo, diagrams = crop_packet(source, [mask])
            reference = Image.new("RGB", (960, 520), "#111827")
            draw = ImageDraw.Draw(reference)
            draw.text((8, 8), "REFERENCE PHOTO | selected object scope", fill="white")
            draw.text((488, 8), "REFERENCE MASK: white = selected", fill="white")
            paste(reference, photo, (0, 40, 470, 470))
            paste(reference, diagrams[0], (480, 40, 470, 470))
            prefix = f"{len(reviews):04d}"
            ref_path = args.output / f"{prefix}_reference.jpg"
            reference.save(ref_path, quality=94)
            photo, diagrams = crop_packet(
                extra[name], [masks[name][i] for i in eligible]
            )
            responses = []
            if reviewer is None:
                from farm_runtime.semantic_refinement import LocalObjectReviewer
                import torch

                before = time.monotonic()
                reviewer = LocalObjectReviewer(args.model)
                torch.cuda.synchronize()
                model_load_seconds = time.monotonic() - before
            for n, order in enumerate([eligible, list(reversed(eligible))]):
                canvas = Image.new("RGB", (1280, 680), "#111827")
                draw = ImageDraw.Draw(canvas)
                draw.text((8, 8), "NEW PHOTO", fill="white")
                paste(canvas, photo, (0, 35, 620, 635))
                for j, index in enumerate(order):
                    x, y = 640 + (j % 2) * 320, (j // 2) * 340
                    draw.text((x + 8, y + 8), f"Candidate {index}", fill="white")
                    paste(
                        canvas, diagrams[eligible.index(index)], (x, y + 35, 310, 300)
                    )
                path = args.output / f"{prefix}_candidates_{n}.jpg"
                canvas.save(path, quality=94)
                response = reviewer.ask(
                    PROMPT,
                    [reference, canvas],
                    [0, 1],
                    lambda raw, ids: validate_response(raw, eligible),
                    max_new_tokens=128,
                )
                responses.append(
                    dict(order=order, canvas=describe_file(path), **response)
                )
                print(json.dumps(dict(group_id=gid, name=name, **response)), flush=True)
            reviews.append(
                dict(
                    group_id=gid,
                    name=name,
                    eligible_candidates=eligible,
                    reference_node=node["id"],
                    reference_image=source["source_image"],
                    reference_canvas=describe_file(ref_path),
                    responses=responses,
                )
            )
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.cross-view-scope-review.v1",
            validation=describe_file(args.validation),
            prompt=describe_file(args.output / "prompt.txt"),
            model_config=(
                describe_file(args.model / "config.json") if reviewer else None
            ),
            object_budget=args.object_budget,
            reviews=reviews,
            deferred=deferred,
            model_load_seconds=model_load_seconds,
            total_seconds=time.monotonic() - started,
            closed_test_opened=False,
            release_eligible=False,
            automatic_selection_applied=False,
        ),
    )
    return 0
