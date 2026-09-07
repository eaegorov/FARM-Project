"""Review physical whole/part families from two independent source photographs."""

from __future__ import annotations

import argparse
from functools import lru_cache
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from farm_runtime.angular_discovery import rotate_image
from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.native_observations import load_prepared
from farm_runtime.quality_baseline import describe_file, write_json
from tools.farm_shaper_bridge.common import (
    load_mask_pair,
    load_observation_exclusion,
    resolve_mask_path,
)

SCHEMA = "farm.scope-family-review.v1"
PARENT_SCOPES = {
    "single_whole_object", "multiple_independent_objects", "structural_surface",
    "object_part", "unclear",
}
RELATIONS = {"integral_part", "separate_object", "unclear"}
CONFIDENCES = {"high", "medium", "low"}
PROMPT = """Assess physical object scope using TWO photographs of the same reconstruction taken at different physical times. Each sheet shows FULL VIEW with the parent outlined in orange, PARENT CONTEXT with the selected parent lightly shaded, and precise numbered CHILD CLOSE-UPS with coloured outlines and light shading. Matching numbered markers locate children in the parent context. These colours and markers are artificial annotations, never object material or colour. The surrounding unshaded photograph supplies context, not additional selected pixels. No detector labels are supplied.
Judge the PARENT from its highlighted extent in the FULL VIEW and PARENT CONTEXT, never from the close-up boundary. A selected door/front panel is an object_part even when its close-up looks like a complete cabinet. Do not assume that adjacent unselected surfaces are included in the parent. Detached selected islands or a boundary spanning several machines require particular care.
First decide whether the PARENT selects one whole physical object, several independent objects, a structural surface, only part of an object, or an unclear scope. A cabinet with its doors can be one object; a row of separate cabinets/machines is not automatically one object. An integrated conveyor may be one assembly if its visible structure supports this. Do not join adjacent machines because they touch or have similar colours.
For EACH child, decide whether it is an integral part of that SAME parent object, a separate physical object, or unclear. Enclosure doors and built-in controls can be integral parts. A poster/sign mounted on a wall or machine, a fire extinguisher, a loose box, and an independent device are separate objects despite mask containment. External pipes, wires and neighbouring equipment are not automatically integral. Containment alone does not establish physical ownership. Use BOTH actual photographs and identify the selected child visually in its close-up. Your reason must refer to what the marked pixels actually show and how they attach to the parent; do not guess a wire, handle or panel from a silhouette. If an item cannot be located confidently, mark it unclear. Use medium/low confidence when attachment or the whole-object boundary is not visible in both views.
Return ONLY JSON with exactly these fields:
{"parent_scope":"single_whole_object|multiple_independent_objects|structural_surface|object_part|unclear","confidence":"high|medium|low","reason":"brief visual reason","children":[{"child_id":integer printed on the sheets,"relation":"integral_part|separate_object|unclear","confidence":"high|medium|low","reason":"brief visual reason"}],"label":string or null,"caption":string or null}.
Include each shown child exactly once. Label/caption are optional visual proposals for the parent, separate from the physical decision. Do not guess equipment function or specifications. Keep each reason under 40 words and caption under 40 words."""


def validate_response(raw, child_ids):
    """Strict IDs and enums prevent model text from authorizing extra objects."""
    result = json.loads(raw)
    if not isinstance(result, dict) or set(result) != {
        "parent_scope", "confidence", "reason", "children", "label", "caption"
    }:
        raise ValueError("exact family response fields required")
    if result["parent_scope"] not in PARENT_SCOPES or result["confidence"] not in CONFIDENCES:
        raise ValueError("invalid parent scope/confidence")
    if not isinstance(result["reason"], str) or not result["reason"].strip():
        raise ValueError("visual parent reason required")
    for field in ("label", "caption"):
        if result[field] is not None and not isinstance(result[field], str):
            raise ValueError("appearance proposals must be strings or null")
    if not isinstance(result["children"], list):
        raise ValueError("child decisions must be a list")
    ids = []
    for child in result["children"]:
        if not isinstance(child, dict) or set(child) != {
            "child_id", "relation", "confidence", "reason"
        }:
            raise ValueError("exact child response fields required")
        oid = child["child_id"]
        if type(oid) is not int or child["relation"] not in RELATIONS:
            raise ValueError("invalid child ID/relation")
        if child["confidence"] not in CONFIDENCES or not isinstance(child["reason"], str) or not child["reason"].strip():
            raise ValueError("visual child reason and confidence required")
        ids.append(oid)
    if sorted(ids) != sorted(child_ids) or len(ids) != len(set(ids)):
        raise ValueError("every shown child must occur exactly once")
    return result


def consensus_response(responses, child_ids, frame_ids):
    """Agreement under changed child AND camera order, never majority voting."""
    unknown = dict(parent_scope="unclear", parent_accepted=False,
                   integral_ids=[], separate_ids=[], unclear_ids=sorted(child_ids))
    if len(responses) != 2:
        return unknown
    parsed, orders = [], []
    for response in responses:
        order, views = response["child_order"], response["frame_order"]
        if sorted(order) != sorted(child_ids) or len(order) != len(set(order)):
            raise ValueError("child display order differs from eligible children")
        if sorted(views) != sorted(frame_ids) or len(views) != 2 or len(set(views)) != 2:
            raise ValueError("two distinct registered frame IDs required")
        orders.append((tuple(order), tuple(views)))
        if response.get("validation_error") is not None:
            return unknown
        result = validate_response(response["raw"], child_ids)
        if result != response["parsed"]:
            raise ValueError("parsed family response differs from raw output")
        parsed.append(result)
    if orders[0] == orders[1]:
        return unknown
    if any(r["confidence"] != "high" for r in parsed) or parsed[0]["parent_scope"] != parsed[1]["parent_scope"]:
        return unknown
    scope = parsed[0]["parent_scope"]
    buckets = {name: [] for name in RELATIONS}
    maps = [{r["child_id"]: r for r in answer["children"]} for answer in parsed]
    for oid in sorted(child_ids):
        a, b = maps[0][oid], maps[1][oid]
        relation = (a["relation"] if a["confidence"] == b["confidence"] == "high"
                    and a["relation"] == b["relation"] else "unclear")
        if relation == "integral_part" and scope != "single_whole_object":
            relation = "unclear"
        buckets[relation].append(oid)
    return dict(parent_scope=scope, parent_accepted=scope == "single_whole_object",
                integral_ids=buckets["integral_part"], separate_ids=buckets["separate_object"],
                unclear_ids=buckets["unclear"])


def combine_packets(parent_id, children, packets):
    """Direct children only; packet disagreement cannot construct a family."""
    scopes = {p["consensus"]["parent_scope"] for p in packets}
    accepted = bool(packets) and scopes == {"single_whole_object"}
    votes = {oid: set() for oid in children}
    for packet in packets:
        for field in ("integral_ids", "separate_ids", "unclear_ids"):
            for oid in packet["consensus"][field]:
                votes[oid].add(field)
    integral, separate, unclear = [], [], []
    for oid, value in sorted(votes.items()):
        if accepted and value == {"integral_ids"}:
            integral.append(oid)
        elif value == {"separate_ids"}:
            separate.append(oid)
        else:
            unclear.append(oid)
    proposals = [r["parsed"] for packet in packets for r in packet["responses"] if r.get("parsed")]
    labels = {r["label"] for r in proposals}
    captions = {r["caption"] for r in proposals}
    return dict(parent_id=parent_id, parent_scope=next(iter(scopes)) if len(scopes) == 1 else "unclear",
                accepted=accepted, parent_accepted=accepted, integral_ids=integral, separate_ids=separate,
                uncertain_ids=unclear, unclear_ids=unclear, packets=packets,
                packet_ids=list(range(len(packets))),
                label=next(iter(labels)) if len(labels) == 1 else None,
                caption=next(iter(captions)) if len(captions) == 1 else None,
                reason="Two display orders agree on one whole object in every packet." if accepted else
                       "Whole-object scope is rejected or lacks consistent high-confidence agreement.",
                appearance_proposals=[dict(label=r["parsed"]["label"], caption=r["parsed"]["caption"])
                                      for p in packets for r in p["responses"] if r.get("parsed")],
                physical_family_auto_applied=False)


def mark_ambiguous_parents(decisions):
    """Do not connect whole objects through a child selected by both of them."""
    owners = {}
    for row in decisions:
        if row["parent_accepted"]:
            for oid in row["integral_ids"]:
                owners.setdefault(oid, []).append(row["parent_id"])
    ambiguous = {oid: sorted(parents) for oid, parents in owners.items() if len(parents) > 1}
    for row in decisions:
        row["ambiguous_integral_ids"] = sorted(set(row["integral_ids"]) & ambiguous.keys())
        row["unambiguous_integral_ids"] = sorted(set(row["integral_ids"]) - ambiguous.keys())
    return [dict(child_id=oid, competing_parent_ids=parents) for oid, parents in sorted(ambiguous.items())]


def plan_packets(run, split, parent_id, children, mask):
    """Cover each eligible relation in two actual build timestamps per packet."""
    objects = {obj.object_id: obj for obj in run.objects}
    allowed = set(split["build_timestamps"])
    if allowed & set(split["heldout_timestamps"]):
        raise ValueError("build and heldout timestamps overlap")
    object_build = {row["object_id"]: set(row["build_timestamps"]) for row in split["objects"]}
    by_object = {oid: {obs.image_id for obs in objects[oid].observations
                      if run.frame(obs.image_id).physical_timestamp in allowed & object_build[oid]}
                 for oid in [parent_id, *children]}
    observations = {}
    pair_evidence = {oid: [] for oid in children}
    for image_id in sorted(by_object[parent_id]):
        parent, parent_source = mask(parent_id, image_id)
        if not parent.any():
            continue
        for child in children:
            if image_id not in by_object[child]:
                continue
            child_mask, child_source = mask(child, image_id)
            area = int(child_mask.sum())
            if area == 0:
                continue
            overlap = int(np.count_nonzero(parent & child_mask))
            child_coverage, parent_coverage = overlap / area, overlap / int(parent.sum())
            if child_coverage < 0.85 or parent_coverage >= 0.80:
                continue
            evidence = dict(image_id=image_id, physical_timestamp=run.frame(image_id).physical_timestamp,
                            child_coverage=child_coverage, parent_coverage=parent_coverage,
                            child_pixels=area, parent_pixels=int(parent.sum()),
                            parent_mask=parent_source, child_mask=child_source)
            pair_evidence[child].append(evidence)
            observations.setdefault(image_id, {})[child] = evidence
    remaining = {oid for oid, rows in pair_evidence.items()
                 if len({r["physical_timestamp"] for r in rows}) >= 2}
    deferred = sorted(set(children) - remaining)
    packets = []
    while remaining:
        choices = []
        for a, b in combinations(sorted(observations), 2):
            if run.frame(a).physical_timestamp == run.frame(b).physical_timestamp:
                continue
            shared = remaining & observations[a].keys() & observations[b].keys()
            # Up to six children remain individually legible on each sheet.
            picked = sorted(shared, key=lambda oid: (-min(observations[a][oid]["child_pixels"], observations[b][oid]["child_pixels"]), oid))[:6]
            if picked:
                pixels = sum(min(observations[a][oid]["child_pixels"], observations[b][oid]["child_pixels"]) for oid in picked)
                choices.append((len(picked), pixels, -a, -b, picked))
        if not choices:
            raise AssertionError("eligible children lack an independent camera pair")
        _, _, na, nb, picked = max(choices)
        frame_ids = [-na, -nb]
        packets.append(dict(child_ids=sorted(picked), frame_ids=frame_ids,
                            timestamps=[run.frame(fid).physical_timestamp for fid in frame_ids],
                            evidence=[dict(child_id=oid, views=[observations[fid][oid] for fid in frame_ids]) for oid in sorted(picked)]))
        remaining -= set(picked)
    return packets, deferred


def _mask_crop(mask, padding):
    yy, xx = np.where(mask)
    if not len(xx):
        raise ValueError("nonempty grounded review mask required")
    h, w = mask.shape
    pad = max(12, round(padding * max(np.ptp(xx), np.ptp(yy))))
    return [max(0, int(xx.min()) - pad), max(0, int(yy.min()) - pad),
            min(w, int(xx.max()) + pad + 1), min(h, int(yy.max()) + pad + 1)]


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _overlay_crop(photo, mask, crop, size, colour, *, fill=0.12, markers=()):
    """Crop original RGB, then align masks exactly on the displayed pixel grid."""
    x0, y0, x1, y1 = crop
    h, w = mask.shape
    rgb = photo.crop((round(x0 * photo.width / w), round(y0 * photo.height / h),
                      round(x1 * photo.width / w), round(y1 * photo.height / h)))
    rgb = ImageOps.contain(rgb, size)
    selected = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8)).resize(rgb.size, Image.Resampling.NEAREST)
    pixels = np.asarray(rgb).copy()
    target = np.asarray(selected) > 0
    pixels[target] = np.rint((1 - fill) * pixels[target] + fill * np.asarray(colour)).astype(np.uint8)
    # Draw after resizing: boundaries stay visible without changing mask extent.
    border = np.asarray(selected.filter(ImageFilter.MaxFilter(3))) != np.asarray(selected.filter(ImageFilter.MinFilter(3)))
    pixels[border] = colour
    result = Image.fromarray(pixels)
    draw, font = ImageDraw.Draw(result), _font(17)
    occupied = []
    for oid, child, child_colour in markers:
        yy, xx = np.where(child[y0:y1, x0:x1])
        if not len(xx):
            continue
        x = round(float(np.median(xx)) * result.width / (x1 - x0))
        y = round(float(np.median(yy)) * result.height / (y1 - y0))
        label = str(oid)
        box = draw.textbbox((0, 0), label, font=font)
        tw, th = box[2] - box[0] + 8, box[3] - box[1] + 8
        anchor = (x, y)
        offsets = [(0, 0)] + [(dx, dy) for distance in (28, 56, 84)
                              for dx, dy in ((0, -distance), (distance, 0),
                                             (0, distance), (-distance, 0))]
        for dx, dy in offsets:
            x = min(max(0, anchor[0] + dx), max(0, result.width - tw))
            y = min(max(0, anchor[1] + dy), max(0, result.height - th))
            if not any(x < bx1 + 3 and x + tw + 3 > bx0 and y < by1 + 3 and y + th + 3 > by0
                       for bx0, by0, bx1, by1 in occupied):
                break
        occupied.append((x, y, x + tw, y + th))
        draw.line((anchor, (x + tw // 2, y + th // 2)), fill=child_colour, width=1)
        draw.rounded_rectangle((x, y, x + tw, y + th), radius=3, fill=(15, 23, 42), outline=child_colour, width=2)
        draw.text((x + 4, y + 2 - box[1]), label, fill=child_colour, font=font)
    return result


def make_sheet(frame, metadata, parent_id, child_order, mask):
    """A full-view anchor prevents tightly cropped parts from looking whole."""
    source = checked_file(metadata["source"])
    with Image.open(source) as raw:
        original = raw.convert("RGB")
    turns = metadata["applied_quarter_turns"]
    photo = Image.fromarray(rotate_image(np.asarray(original), turns))
    masks = {oid: rotate_image(mask(oid, frame.image_id)[0], turns)
             for oid in [parent_id, *child_order]}
    parent = masks[parent_id]
    h, w = parent.shape
    context_crop = _mask_crop(parent, 0.60)
    palette = [(56, 189, 248), (167, 139, 250), (74, 222, 128),
               (251, 113, 133), (250, 204, 21), (45, 212, 191)]
    # Colours follow IDs, so display-order changes do not alter the annotation.
    colours = {oid: palette[i] for i, oid in enumerate(sorted(child_order))}
    markers = [(oid, masks[oid], colours[oid]) for oid in child_order]
    parent_colour = (251, 146, 60)
    images = [
        _overlay_crop(photo, parent, [0, 0, w, h], (580, 380), parent_colour, fill=0.04),
        _overlay_crop(photo, parent, context_crop, (580, 380), parent_colour, fill=0.10, markers=markers),
    ]
    child_crops = {}
    for oid in child_order:
        child_crops[oid] = _mask_crop(masks[oid], 0.35)
        images.append(_overlay_crop(photo, masks[oid], child_crops[oid], (380, 216), colours[oid], fill=0.14))
    rows = (len(child_order) + 2) // 3
    canvas = Image.new("RGB", (1200, 430 + 266 * rows), "#111827")
    draw, font = ImageDraw.Draw(canvas), _font(20)
    cells = [(0, 0, 600, 430, "FULL VIEW - parent outlined", (255, 255, 255)),
             (600, 0, 600, 430, f"PARENT {parent_id} - selected extent", parent_colour)]
    cells += [(400 * (i % 3), 430 + 266 * (i // 3), 400, 266,
               f"CHILD {oid} - selected pixels", colours[oid]) for i, oid in enumerate(child_order)]
    for image, (x, y, cw, ch, title, colour) in zip(images, cells):
        draw.text((x + 12, y + 8), title, fill=colour, font=font)
        canvas.paste(image, (x + (cw - image.width) // 2, y + 36 + (ch - 44 - image.height) // 2))
    return canvas, dict(source=describe_file(source), crop_xyxy=context_crop,
                        full_view_xyxy=[0, 0, w, h], child_crop_xyxy=child_crops,
                        mask_grid_hw=[h, w], applied_quarter_turns=turns,
                        crop_coordinate_system="oriented_mask_grid",
                        visual_encoding="full_context_plus_aligned_rgb_mask_overlays")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--parent-budget", type=int, default=24)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() or not 1 <= args.parent_budget <= 128:
        raise ValueError("new output and parent budget in 1..128 required")
    started = time.monotonic()
    source_catalog = describe_file(args.catalog)
    catalog = json.loads(checked_file(source_catalog).read_text())
    if catalog.get("schema") != "farm.quality-scene-catalog.v1" or catalog.get("closed_test_opened") is not False:
        raise ValueError("development quality catalog required")
    native_path = checked_file(catalog["native_output"])
    native = json.loads(native_path.read_text())
    input_path = checked_file(native["input"])
    run, split, manifest = load_prepared(input_path)
    if native["source_ply"]["sha256"] != catalog["source_ply"]["sha256"]:
        raise ValueError("catalog/native source mismatch")
    if manifest["source_geometry"]["sha256"] != catalog["object_id_namespace"]["geometry"]["sha256"]:
        raise ValueError("catalog/native object namespace mismatch")
    objects = {o.object_id: o for o in run.objects}
    observations = {(o.object_id, obs.image_id): obs for o in run.objects for obs in o.observations}
    frames = {r["image_id"]: r for r in manifest["frames"]}
    parents = {}
    for relation in catalog["scope_relations"]:
        if relation["status"] == "multiview_containment":
            parent, child = relation["outer_group"], relation["inner_group"]
            if parent in objects and child in objects and parent != child:
                parents.setdefault(parent, set()).add(child)

    @lru_cache(maxsize=96)
    def mask(oid, image_id):
        frame = run.frame(image_id)
        path = resolve_mask_path(run, observations[oid, image_id])
        raw, _, digest = load_mask_pair(path, frame.depth_size)
        excluded, exclusion_source = load_observation_exclusion(run, frame)
        if excluded is not None:
            raw &= ~excluded
        return raw, dict(path=str(path), sha256=digest, exclusion=exclusion_source)

    args.output.mkdir(parents=True)
    (args.output / "images").mkdir()
    (args.output / "prompt.txt").write_text(PROMPT)
    decisions, deferred, reviewer = [], [], None
    model_seconds = 0.0
    # Prioritize parents with more independently observed relations; IDs break ties only.
    schedule = sorted(parents, key=lambda oid: (-len(parents[oid]), oid))
    for parent in schedule[:args.parent_budget]:
        children = sorted(parents[parent])
        planned, missing = plan_packets(run, split, parent, children, mask)
        deferred += [dict(parent_id=parent, child_id=oid, reason="fewer_than_two_current_build_timestamps") for oid in missing]
        packets = []
        for rank, packet in enumerate(planned):
            responses = []
            for order_id in range(2):
                child_order = packet["child_ids"] if order_id == 0 else list(reversed(packet["child_ids"]))
                frame_order = packet["frame_ids"] if order_id == 0 else list(reversed(packet["frame_ids"]))
                sheets, sources = [], []
                for fid in frame_order:
                    canvas, source = make_sheet(run.frame(fid), frames[fid], parent, child_order, mask)
                    path = args.output / "images" / f"parent_{parent:06d}_packet_{rank:02d}_order_{order_id}_frame_{fid:06d}.jpg"
                    canvas.save(path, quality=93)
                    sheets.append(canvas)
                    sources.append(dict(frame_id=fid, physical_timestamp=run.frame(fid).physical_timestamp,
                                        image=describe_file(path), **source))
                if args.prepare_only:
                    response = dict(raw=None, parsed=None, validation_error="prepare_only")
                else:
                    if reviewer is None:
                        from farm_runtime.semantic_refinement import LocalObjectReviewer
                        tick = time.monotonic()
                        reviewer = LocalObjectReviewer(args.model)
                        model_seconds = time.monotonic() - tick
                    response = reviewer.ask(PROMPT, sheets, frame_order,
                                            lambda raw, ids: validate_response(raw, packet["child_ids"]),
                                            max_new_tokens=768)
                responses.append(dict(child_order=child_order, frame_order=frame_order,
                                      sheets=sources, **response))
            packets.append(dict(**packet, responses=responses,
                                consensus=consensus_response(responses, packet["child_ids"], packet["frame_ids"])))
        decision = combine_packets(parent, children, packets)
        decisions.append(decision)
        write_json(args.output / f"parent_{parent:06d}.json", decision)
        print(json.dumps(dict(parent_id=parent, parent_scope=decision["parent_scope"], integral_ids=decision["integral_ids"], packets=len(packets))), flush=True)
    deferred += [dict(parent_id=parent, reason="parent_budget") for parent in schedule[args.parent_budget:]]
    ambiguous = mark_ambiguous_parents(decisions)
    for decision in decisions:
        write_json(args.output / f"parent_{decision['parent_id']:06d}.json", decision)
    up_vectors = np.asarray([obj.rotation[:, 2] for obj in run.objects], dtype=float)
    up = np.median(up_vectors, axis=0) if len(up_vectors) else np.zeros(3)
    up_norm = float(np.linalg.norm(up))
    source_world_up = (up / up_norm).tolist() if np.isfinite(up_norm) and up_norm > 1e-8 else None
    checked_file(source_catalog)
    write_json(args.output / "manifest.json", dict(
        schema=SCHEMA, source_catalog=source_catalog, source_native=describe_file(native_path),
        source_input=describe_file(input_path), prompt=describe_file(args.output / "prompt.txt"),
        model_path=str(args.model), model_config=describe_file(args.model / "config.json") if reviewer else None,
        parent_budget=args.parent_budget, eligible_parent_ids=schedule,
        decisions=decisions, ambiguous_children=ambiguous,
        ambiguous_child_ids=[row["child_id"] for row in ambiguous], deferred=deferred,
        source_world_up=source_world_up,
        source_world_up_method="median of prepared object OBB up axes, normalized",
        prepare_only=args.prepare_only, model_load_seconds=model_seconds,
        total_seconds=time.monotonic() - started, detector_labels_used_as_truth=False,
        physical_family_auto_applied=False, transitive_assembly_applied=False,
        closed_test_opened=False, release_eligible=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
