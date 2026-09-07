"""Review current multiview masks for meaningful whole-object scope."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from farm_runtime.angular_discovery import rotate_image

from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.native_observations import load_prepared
from farm_runtime.quality.object_admission import (
    PROMPT as VIEW_PROMPT, aggregate_admission, validate_admission_response,
)
from farm_runtime.quality.scope_family_review import make_sheet, _font
from farm_runtime.quality_baseline import describe_file, write_json
from tools.farm_shaper_bridge.common import (
    load_mask_pair, load_observation_exclusion, resolve_mask_path,
)

SCHEMA = "farm.proposal-admission-review.v1"
PROMPT = VIEW_PROMPT + """
You receive two sheets showing the SAME candidate at DIFFERENT physical times.
Each sheet contains a full scene and a close-up of the selected pixels. Judge
the highlighted pixels, not an attractive neighbouring object in the context.
Describe each sheet independently. Do not infer that unhighlighted equipment
belongs to the selection. A floor mask around a cart is the floor, not the cart.
The orange annotation is artificial and is not an object colour.
Return ONLY {"views":[{"image_id":the sheet ID,"scope":...,"mask_fit":...,
"confidence":...,"reason":"short visual explanation"}, ...]}.
Include each supplied image_id exactly once. Use no detector labels, model
names, object numbers or expected scene inventory as evidence.
"""


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate response field")
        result[key] = value
    return result


def model_fingerprint(path):
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files or not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("complete local model weights required")
    return [dict(name=str(p.relative_to(path)), **describe_file(p)) for p in files]


def review_prompt(protocol, image_style="context"):
    if image_style not in ("context", "isolated"):
        raise ValueError("unknown admission image style")
    suffix = "" if image_style == "context" else """
The right tile shows ONLY the selected source pixels. Its neutral gray matte is
artificial, not scene material or missing object geometry. The surrounding scene
on the left is context, not selected pixels. Use the left view for attachment and
whole-object context; judge the selected target from the isolated pixels.
"""
    if protocol == "scope_v1":
        return PROMPT + suffix
    if protocol != "witness_v2":
        raise ValueError("unknown admission protocol")
    from farm_runtime.quality.object_admission import PROMPT_V2
    instructions = PROMPT_V2.split("Return ONLY JSON", 1)[0]
    return instructions + """
You receive TWO sheets of the same candidate at different physical times.
Describe each sheet independently, using the highlighted pixels only. The scene
and close-up on one sheet show the SAME selection. Orange is an annotation.
Return ONLY {"views":[{"image_id":integer supplied with the sheet,
"target_kind":"independent_item|intrinsic_component|building_structure|collection|unclear",
"mask_fit":"fits_target|partial_target|crosses_objects|unclear",
"confidence":"high|medium|low","reason":"short visual reason",
"parent_witness":{"kind":"bounded_manufactured_object|building_support|none|unclear",
"relation":"integral_component|mounted_separate|none|unclear",
"visual_evidence":"what larger item and attachment are actually visible"}}, ...]}.
Include each image_id exactly once. Do not copy decisions between sheets.
""" + suffix



def _isolated_selected_crop(photo, mask, size=(580, 380)):
    """Preserve mask holes and use only selected RGB in interpolation support."""
    values = np.asarray(mask)
    if values.ndim != 2 or values.dtype != np.bool_ or not values.any():
        raise ValueError("nonempty two-dimensional boolean mask required")
    yy, xx = np.where(values)
    padding = max(1, int(np.ceil(0.12 * max(np.ptp(xx) + 1, np.ptp(yy) + 1))))
    h, w = values.shape
    crop = [max(0, int(xx.min()) - padding), max(0, int(yy.min()) - padding),
            min(w, int(xx.max()) + 1 + padding), min(h, int(yy.max()) + 1 + padding)]
    x0, y0, x1, y1 = crop
    source_crop = [round(x0 * photo.width / w), round(y0 * photo.height / h),
                   round(x1 * photo.width / w), round(y1 * photo.height / h)]
    if source_crop[0] >= source_crop[2] or source_crop[1] >= source_crop[3]:
        raise ValueError("selected crop has no source image pixels")
    selected = Image.fromarray((values * 255).astype(np.uint8)).resize(
        photo.size, Image.Resampling.NEAREST,
    ).crop(source_crop)
    rgba = photo.convert("RGBA").crop(source_crop)
    rgba.putalpha(selected)
    # Pillow resizes RGBA through premultiplied alpha: unselected original RGB
    # cannot bleed into selected pixels along boundaries or holes.
    resized = ImageOps.contain(rgba, size, method=Image.Resampling.BICUBIC)
    target = np.asarray(selected.resize(resized.size, Image.Resampling.NEAREST)) > 0
    pixels = np.full((resized.height, resized.width, 3), 128, dtype=np.uint8)
    pixels[target] = np.asarray(resized.convert("RGB"))[target]
    return Image.fromarray(pixels), dict(
        crop_xyxy=crop, source_rgb_crop_xyxy=source_crop,
        crop_padding_fraction=0.12, matte_rgb=[128, 128, 128],
        rgb_resampling="bicubic_premultiplied_alpha", mask_resampling="nearest",
        selected_display_pixels=int(target.sum()), mask_modified=False,
    )


def make_admission_sheet(frame, metadata, oid, mask, *, image_style="context"):
    """Two aligned views of one authoritative selection, without hierarchy cues.

    Context style exactly preserves the existing neutral-heading v2 image.
    Isolated style changes only the right cell; its gray pixels are a display
    matte, never an alteration of the source mask or a completion of its holes.
    """
    if image_style not in ("context", "isolated"):
        raise ValueError("unknown admission image style")
    sheet, source = make_sheet(frame, metadata, oid, [], mask)
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width, 35), fill="#111827")
    draw.text((12, 8), "FULL VIEW - selected region", fill="white", font=_font(20))
    draw.text((612, 8), "SELECTED PIXELS - same region", fill=(251, 146, 60), font=_font(20))
    source.update(neutral_selection_headings=True, image_style=image_style)
    if image_style == "context":
        return sheet, source
    with Image.open(checked_file(metadata["source"])) as original:
        photo = Image.fromarray(rotate_image(np.asarray(original.convert("RGB")),
                                            metadata["applied_quarter_turns"]))
    selected = rotate_image(mask(oid, frame.image_id)[0], metadata["applied_quarter_turns"])
    isolated, isolation = _isolated_selected_crop(photo, selected)
    draw.rectangle((600, 0, sheet.width, sheet.height), fill="#111827")
    draw.text((612, 8), "ONLY SELECTED PIXELS", fill="white", font=_font(20))
    x = 600 + (600 - isolated.width) // 2
    y = 36 + (430 - 44 - isolated.height) // 2
    sheet.paste(isolated, (x, y))
    isolation["display_xyxy"] = [x, y, x + isolated.width, y + isolated.height]
    source.update(crop_xyxy=isolation["crop_xyxy"], isolated_view=isolation,
                  visual_encoding="full_context_plus_isolated_selected_pixels")
    return sheet, source


def validate_views(raw, image_ids, protocol="scope_v1"):
    validator = validate_admission_response
    if protocol == "witness_v2":
        from farm_runtime.quality.object_admission import validate_parent_witness_response
        validator = validate_parent_witness_response
    elif protocol != "scope_v1":
        raise ValueError("unknown admission protocol")
    value = json.loads(raw, object_pairs_hook=_unique_fields)
    if not isinstance(value, dict) or set(value) != {"views"}:
        raise ValueError("exact views response required")
    rows = value["views"]
    if not isinstance(rows, list) or len(rows) != len(image_ids):
        raise ValueError("one response per supplied view required")
    seen = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("source view response required")
        view_id = row.get("image_id")
        if type(view_id) is str and view_id in {str(fid) for fid in image_ids}:
            view_id = int(view_id)
            row["image_id"] = view_id
        if type(view_id) is not int:
            raise ValueError("source image_id must exactly match a supplied identifier")
        seen.append(view_id)
        validator({key: item for key, item in row.items() if key != "image_id"})
    if len(set(seen)) != len(seen) or set(seen) != set(image_ids):
        raise ValueError("response view IDs differ from source views")
    return value


def select_views(run, split, obj, mask):
    """Largest visible source mask per physical time, then two distinct times."""
    allowed = set(split["build_timestamps"])
    if allowed & set(split["heldout_timestamps"]):
        raise ValueError("build/heldout timestamps overlap")
    rows = {row["object_id"]: row for row in split["objects"]}
    allowed &= set(rows[obj.object_id]["build_timestamps"])
    by_time = {}
    for obs in obj.observations:
        frame = run.frame(obs.image_id)
        timestamp = frame.physical_timestamp
        if timestamp not in allowed:
            continue
        raw, _ = mask(obj.object_id, obs.image_id)
        pixels = int(raw.sum())
        if not pixels:
            continue
        candidate = (pixels, -obs.image_id, obs.image_id)
        if candidate > by_time.get(timestamp, (-1, 0, 0)):
            by_time[timestamp] = candidate
    ordered = sorted(by_time.items(), key=lambda item: (-item[1][0], item[0]))
    return [item[1][2] for item in ordered[:2]]


def run_review(catalog_path, model_path, output, *, limit=0, prepare_only=False,
               resume=False, protocol="scope_v1", reviewer=None, image_style="context"):
    if reviewer is not None:
        loaded_model = getattr(reviewer, "model_path", None)
        if loaded_model is None or Path(loaded_model).resolve() != model_path.resolve():
            raise ValueError("injected reviewer model path differs from declared model")
    prompt = review_prompt(protocol, image_style)
    aggregate = aggregate_admission
    if protocol == "witness_v2":
        from farm_runtime.quality.object_admission import aggregate_witness_admission
        aggregate = aggregate_witness_admission
    validator = lambda raw, ids: validate_views(raw, ids, protocol)
    if limit < 0 or limit > 512:
        raise ValueError("review limit must be 0 (all) or 1..512")
    if output.exists() and not resume:
        raise ValueError("new output required unless --resume is explicit")
    started = time.monotonic()
    source_catalog = describe_file(catalog_path)
    catalog = json.loads(checked_file(source_catalog).read_text())
    if (catalog.get("schema") != "farm.quality-scene-catalog.v1"
            or catalog.get("closed_test_opened") is not False):
        raise ValueError("development source catalog required")
    native_path = checked_file(catalog["native_output"])
    native = json.loads(native_path.read_text())
    input_path = checked_file(native["input"])
    run, split, inputs = load_prepared(input_path)
    if inputs["source_geometry"]["sha256"] != catalog["object_id_namespace"]["geometry"]["sha256"]:
        raise ValueError("catalog/native geometry namespace differs")
    if native["source_ply"]["sha256"] != catalog["source_ply"]["sha256"]:
        raise ValueError("catalog/native source scene differs")
    catalog_rows = {row["object_id"]: row for row in catalog["objects"]}
    objects = {obj.object_id: obj for obj in run.objects}
    if set(catalog_rows) != set(objects):
        raise ValueError("source catalog and native object cohorts differ")
    observations = {
        (obj.object_id, obs.image_id): obs
        for obj in run.objects for obs in obj.observations
    }
    frames = {row["image_id"]: row for row in inputs["frames"]}

    @lru_cache(maxsize=64)
    def mask(oid, fid):
        frame = run.frame(fid)
        path = resolve_mask_path(run, observations[oid, fid])
        raw, _, digest = load_mask_pair(path, frame.depth_size)
        exclusion, exclusion_source = load_observation_exclusion(run, frame)
        if exclusion is not None:
            raw &= ~exclusion
        return raw, dict(path=str(path), sha256=digest, exclusion=exclusion_source)

    output.mkdir(parents=True, exist_ok=True)
    (output / "inputs").mkdir(exist_ok=True)
    prompt_path = output / "prompt.txt"
    if prompt_path.exists() and prompt_path.read_text() != prompt:
        raise ValueError("resume prompt changed")
    prompt_path.write_text(prompt)
    model_config = describe_file(model_path / "config.json")
    binding = dict(
        source_catalog=source_catalog, source_native=describe_file(native_path),
        source_input=describe_file(input_path), prompt=describe_file(prompt_path),
        model_config=model_config, model_files=model_fingerprint(model_path),
        protocol=protocol, image_style=image_style,
        image_policy=(dict(right="context", crop_padding_fraction=0.60)
                      if image_style == "context" else
                      dict(right="isolated_selected_pixels", crop_padding_fraction=0.12,
                           matte_rgb=[128, 128, 128], rgb_resampling="bicubic_premultiplied_alpha",
                           mask_resampling="nearest", mask_modified=False)),
        inference_policy=dict(max_new_tokens=600, do_sample=False,
                              dtype="bfloat16", attn_implementation="sdpa"),
        reviewer_code=describe_file(Path(__file__).parents[1] / "semantic_refinement.py"),
        runner_code=describe_file(Path(__file__)),
        admission_code=describe_file(Path(__file__).parent / "object_admission.py"),
        image_code=describe_file(Path(__file__).parent / "scope_family_review.py"),
    )
    plan_path = output / "plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if old["binding"] != binding:
            raise ValueError("resume input/model/prompt binding changed")
    schedule = sorted(objects)
    write_json(plan_path, dict(
        schema=SCHEMA, binding=binding, group_ids=schedule,
        selector="all native candidates; two largest source masks at independent build times",
        detector_labels_in_prompt=False, closed_test_opened=False,
    ))
    load_seconds, records = 0.0, []
    selected = schedule[:limit] if limit else schedule
    for oid in selected:
        obj = objects[oid]
        fids = select_views(run, split, obj, mask)
        images, sources = [], []
        for fid in fids:
            if protocol == "scope_v1" and image_style == "context":
                # Preserve historical scope-v1 display pixels for its default.
                sheet, source = make_sheet(run.frame(fid), frames[fid], oid, [], mask)
                source["image_style"] = image_style
            else:
                sheet, source = make_admission_sheet(
                    run.frame(fid), frames[fid], oid, mask, image_style=image_style,
                )
            image_path = output / "inputs" / f"object_{oid:06d}_view_{fid:06d}.jpg"
            buffer = io.BytesIO()
            sheet.save(buffer, format="JPEG", quality=93)
            encoded = buffer.getvalue()
            if image_path.exists() and hashlib.sha256(image_path.read_bytes()).digest() != hashlib.sha256(encoded).digest():
                raise ValueError("cached admission image changed; existing evidence preserved")
            if not image_path.exists():
                image_path.write_bytes(encoded)
            # Feed exactly the RGB decoded from the frozen evidence artifact.
            # A pre-compression sheet is not pixel-identical to its JPEG.
            with Image.open(io.BytesIO(encoded)) as stored_image:
                images.append(stored_image.convert("RGB"))
            sources.append(dict(
                image_id=fid, physical_timestamp=run.frame(fid).physical_timestamp,
                image=describe_file(image_path), mask=mask(oid, fid)[1], **source,
            ))
        target = output / f"object_{oid:06d}.json"
        response, reused = None, False
        if resume and target.exists():
            previous = json.loads(target.read_text())
            if previous["binding"] != binding or previous["sources"] != sources:
                raise ValueError("cached admission evidence changed")
            candidate = previous.get("response")
            if candidate and candidate.get("parsed") is not None and candidate.get("validation_error") is None:
                if validator(candidate["raw"], fids) != candidate["parsed"]:
                    raise ValueError("cached response differs from parsed source")
                response, reused = candidate, True
        if response is None and len(fids) == 2 and not prepare_only:
            if reviewer is None:
                from farm_runtime.semantic_refinement import LocalObjectReviewer
                tick = time.monotonic()
                reviewer = LocalObjectReviewer(model_path)
                load_seconds = time.monotonic() - tick
            response = reviewer.ask(prompt, images, fids, validator,
                                    max_new_tokens=600)
        parsed = {row["image_id"]: row for row in response["parsed"]["views"]} if response and response.get("parsed") else {}
        observations_for_decision = []
        for source in sources:
            fid = source["image_id"]
            answer = parsed.get(fid)
            observations_for_decision.append(dict(
                view_id=fid, physical_timestamp=source["physical_timestamp"],
                response={k: v for k, v in answer.items() if k != "image_id"} if answer else None,
                dependency_valid=True, fallback_used=False,
                validation_error=response.get("validation_error") if response else "no_model_review",
            ))
        decision = aggregate(observations_for_decision)
        record = dict(
            object_id=oid, binding=binding, sources=sources, response=response,
            decision=decision, response_reused=reused,
            native_gaussians=catalog_rows[oid]["native_gaussians"],
            native_mask_quality_assessed=False,
        )
        records.append(record)
        write_json(target, record)
        print(json.dumps(dict(
            scene=run.scene_id, object_id=oid, decision=decision["decision"],
            views=len(fids), reviewed=len(records), total=len(selected),
            seconds=response.get("seconds") if response else None,
            reused=reused,
        )), flush=True)
    checked_file(source_catalog)
    result = dict(
        schema=SCHEMA, **binding, records=records,
        pending_object_ids=sorted(set(schedule) - set(selected)),
        counts=dict(Counter(row["decision"]["decision"] for row in records)),
        input_mask_scope_only=True, native_masks_modified=False,
        model_review_is_ground_truth=False, closed_test_opened=False,
        release_eligible=False, prepare_only=prepare_only,
        model_load_seconds=load_seconds, total_seconds=time.monotonic() - started,
    )
    write_json(output / "manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--protocol", choices=("scope_v1", "witness_v2"), default="scope_v1")
    parser.add_argument("--image-style", choices=("context", "isolated"), default="context")
    args = parser.parse_args(argv)
    run_review(args.catalog, args.model, args.output, limit=args.limit,
               prepare_only=args.prepare_only, resume=args.resume, protocol=args.protocol,
               image_style=args.image_style)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
