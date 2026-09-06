"""Bounded, local visual review of object identity and mask alternatives."""

from __future__ import annotations

import json
import math
import time
import numpy as np

REVIEW_PROMPT = """Review one physical object seen in multiple photographs. In each sheet, the left tile is the unmodified photo; numbered tiles 0..3 show alternative cyan masks. Tile 0 is an approximate target cue, not ground truth. All sheets refer to the same object.
Identify the WHOLE target object at the most specific level directly supported by visible evidence. Do not guess its hidden function or exact subtype. Panels, ventilation slots, and cables alone do not establish device function. Prefer a broad physical category when different functions remain possible, and state this in uncertainty. Mention only parts actually visible; do not add a lid, gauge, mounting hardware, or nozzle merely because that class often has one. Separate attached functional parts from neighboring objects, unrelated pipes, cables, supports and background. A container is the whole container, not just its contents. A sign is the entire sign, not individual symbols. Occluded parts need not be invented. A partial object at the image edge remains a valid visible instance.
For each sheet choose the best numbered mask covering the visible target, including its body and integral parts, while excluding unrelated objects. Choose -1 if none is acceptable. Do not trust a mask solely because it resembles tile 0. External wiring and building pipes are separate infrastructure even when connected to the object. Integral handles and short built-in fittings belong to the object. Thin attachments may be optional when their ownership is unclear. Verify whether the mask actually follows the visible outer boundary; do not invent an exclusion that the mask does not show.
Return only JSON with this schema:
{"label": "conservative English category", "caption": "one factual English sentence about visible appearance", "include": ["visible integral parts"], "exclude": ["visible neighboring items"], "uncertainty": "what cannot be determined, or empty string", "views": [{"image_id": 123, "choice": 0, "reason": "brief visual reason"}]}.
No dimensions, brand guesses, or text not legible in the photo. Image IDs below are identifiers, not category hints.
"""


SEMANTIC_PROMPT = """Describe the SAME physical target in these photographs. A thin yellow rectangle is only an approximate location cue. Inspect the original RGB appearance; no segmentation candidates are shown. The target is the main coherent object inside the cue, not every item touched by the rectangle.
Return a conservative observable category and one factual appearance sentence. Do not infer hidden function, exact subtype, brand or mounting merely from common associations. Put a functional label only when distinct visible structure or clearly legible text establishes it; otherwise use null and state the uncertainty. A cabinet-shaped object with cables may have several possible functions.
Describe ownership explicitly. Parts fixed inside the main enclosure, handles, doors and integral fittings belong to integral_parts. External building pipes and installation wiring belong to external_connections even when connected. A surface, shelf or pallet beneath the object is a supporting_object, not an integral part. A separate object inside a container belongs to contained_objects; the container remains the whole container. Do not list one item in two ownership groups. Only mention visible parts. Do not invent hidden or absent details. If grouping is uncertain, say so rather than forcing a merge.
Return only this JSON schema:
{"label": "observable English object category", "caption": "one factual English sentence", "functional_identity": {"label": null, "direct_visual_evidence": ""}, "integral_parts": ["part"], "external_connections": ["connection"], "supporting_objects": ["support"], "contained_objects": ["content"], "uncertainty": ["what is unresolved"], "observed_image_ids": [123, 456]}.
Image IDs are identifiers, not category hints.
"""


SCOPE_PROMPT = """Inspect one spatially associated target candidate across views. Each sheet has two panels: PHOTO shows the original RGB with an approximate yellow location rectangle; MASK DIAGRAM is a separate black-and-white segmentation diagram, where white marks the candidate pixels. The diagram is not a photograph: its colors and solid shape never establish material, wrapping, paint or physical completeness. Describe appearance only from PHOTO. A mask is a hypothesis, not truth. No detector category is provided. Check both RGB appearance and what the mask covers. The visible candidate may be a whole object, an integral part, a collection of separate objects, a covering, or a patch of a larger surface. Do not upgrade a door, handle, panel, wrapping sheet or covered contents into the full parent machine. A partial view at the image border is still valid evidence; do not invent unseen extent. If the views appear to refer to different objects, state that explicitly.
Return a conservative observable label for the ACTUAL candidate scope and one factual appearance sentence. Hidden device function, brand and dimensions must not be guessed. Distinguish integral parts, external connections, supporting objects and contained objects. Report only visible components; no component may occur in two ownership lists. Wrapped objects can be distinct from their removable wrapping. Geometry containment alone does not establish ownership.
Return only JSON:
{"label":"observable English candidate category","caption":"one factual English sentence","functional_identity":{"label":null,"direct_visual_evidence":""},"integral_parts":[],"external_connections":[],"supporting_objects":[],"contained_objects":[],"uncertainty":[],"observed_image_ids":[123,456],"same_physical_target":true,"scope":{"kind":"whole_object|integral_part|object_collection|covering|surface_region|unclear","parent_description":null,"mask_coverage_issues":[]}}.
Choose exactly one scope kind. same_physical_target is true, false, or null if identity is uncertain. parent_description is a visible possible larger parent or null; it does not authorize merging. mask_coverage_issues lists visible omissions or unrelated inclusions, or is empty. A valid JSON response is not proof of mask accuracy.
"""


def validate_scope(text, image_ids):
    result = validate_semantics(text, image_ids)
    if result["label"].strip() in {
        "whole_object",
        "integral_part",
        "object_collection",
        "surface_region",
        "unclear",
    }:
        raise ValueError("scope enum is not an observable object label")
    if "same_physical_target" not in result or (
        result["same_physical_target"] is not None
        and type(result["same_physical_target"]) is not bool
    ):
        raise ValueError("explicit target identity verdict required")
    scope = result.get("scope")
    if not isinstance(scope, dict) or scope.get("kind") not in {
        "whole_object",
        "integral_part",
        "object_collection",
        "covering",
        "surface_region",
        "unclear",
    }:
        raise ValueError("invalid physical scope kind")
    if "parent_description" not in scope or (
        scope["parent_description"] is not None
        and not isinstance(scope["parent_description"], str)
    ):
        raise ValueError("parent description must be a string or null")
    issues = scope.get("mask_coverage_issues")
    if not isinstance(issues, list) or not all(isinstance(x, str) for x in issues):
        raise ValueError("mask coverage issues must be strings")
    return result


def validate_semantics(text, image_ids):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(text)
    for field in ("label", "caption"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise ValueError("nonempty label/caption required")
        if " ".join(result[field].lower().split()) in {
            "observable english candidate category",
            "observable english object category",
            "one factual english sentence",
        }:
            raise ValueError("schema placeholder is not semantic evidence")
    groups = (
        "integral_parts",
        "external_connections",
        "supporting_objects",
        "contained_objects",
        "uncertainty",
    )
    ownership = set()
    for group in groups:
        if not isinstance(result.get(group), list) or not all(
            isinstance(x, str) for x in result[group]
        ):
            raise ValueError(f"invalid semantic group {group}")
        if group != "uncertainty":
            normalized = {" ".join(x.lower().split()) for x in result[group]}
            if ownership & normalized:
                raise ValueError("component assigned to conflicting ownership groups")
            ownership |= normalized
    ids = result.get("observed_image_ids")
    if (
        not isinstance(ids, list)
        or len(ids) != len(image_ids)
        or set(ids) != set(image_ids)
    ):
        raise ValueError("wrong semantic evidence image IDs")
    function = result.get("functional_identity")
    if not isinstance(function, dict) or not isinstance(
        function.get("direct_visual_evidence"), str
    ):
        raise ValueError("functional evidence required")
    if function.get("label") is not None and (
        not isinstance(function["label"], str)
        or not function["direct_visual_evidence"].strip()
    ):
        raise ValueError("functional identity must cite direct visual evidence")
    return result


def select_review_views(rows, budget=2):
    """Select visible, distinct timestamps; never duplicate one timestamp's views."""
    if type(budget) is not int or not 1 <= budget <= 3:
        raise ValueError("view budget must be an integer from 1 to 3")
    candidates = [r for r in rows if r["foreground_pixels"] > 0]
    if not candidates:
        return []
    if any(not r["clipped"] for r in candidates):
        candidates = [r for r in candidates if not r["clipped"]]
    candidates.sort(key=lambda r: (-r["foreground_pixels"], r["image_id"]))
    chosen = [candidates.pop(0)]
    while candidates and len(chosen) < budget:
        candidates = [
            r
            for r in candidates
            if r["timestamp"] not in {x["timestamp"] for x in chosen}
        ]
        if not candidates:
            break
        # Prefer a different physical sensor at comparable visibility. The
        # timestamp separation prevents counting virtual views as new evidence.
        sensors = {
            x["source_image"]["path"].split("/")[-1].split("_")[0] for x in chosen
        }
        maximum = max(r["foreground_pixels"] for r in candidates)

        def score(r):
            sensor = r["source_image"]["path"].split("/")[-1].split("_")[0]
            return math.sqrt(r["foreground_pixels"] / maximum) + 0.15 * (
                sensor not in sensors
            )

        candidate = max(candidates, key=score)
        chosen.append(candidate)
        candidates.remove(candidate)
    return chosen


def validate_review(text, image_ids):
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(value)
    for key in ("label", "caption", "uncertainty"):
        if not isinstance(result.get(key), str):
            raise ValueError(f"invalid {key}")
    if not result["label"].strip() or not result["caption"].strip():
        raise ValueError("empty object description")
    for key in ("include", "exclude"):
        if not isinstance(result.get(key), list) or not all(
            isinstance(x, str) for x in result[key]
        ):
            raise ValueError(f"invalid {key}")
    views = result.get("views")
    if not isinstance(views, list) or len(views) != len(image_ids):
        raise ValueError("all and only requested views must be reviewed")
    if {r.get("image_id") for r in views} != set(image_ids):
        raise ValueError("wrong or duplicate image IDs")
    for row in views:
        if type(row.get("choice")) is not int or row["choice"] not in (-1, 0, 1, 2, 3):
            raise ValueError("invalid candidate choice")
        if not isinstance(row.get("reason"), str):
            raise ValueError("missing visual reason")
    return result


class LocalObjectReviewer:
    def __init__(self, model_path):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
        ).eval()

    def review(self, images, image_ids, *, semantic_only=False):
        return self.ask(
            SEMANTIC_PROMPT if semantic_only else REVIEW_PROMPT,
            images,
            image_ids,
            validate_semantics if semantic_only else validate_review,
        )

    def ask(self, prompt, images, image_ids, validator, *, max_new_tokens=500):
        if not 1 <= len(images) <= 3 or len(images) != len(image_ids):
            raise ValueError("one to three aligned image/ID pairs required")
        torch = self.torch
        content = [
            {
                "type": "text",
                "text": prompt,
            }
        ]
        for image, image_id in zip(images, image_ids):
            content += [
                {"type": "text", "text": f"Sheet image_id={image_id}"},
                {"type": "image", "image": image},
            ]
        messages = [{"role": "user", "content": content}]
        batch = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to("cuda")
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(
                **batch, max_new_tokens=max_new_tokens, do_sample=False
            )
        torch.cuda.synchronize()
        raw = self.processor.decode(
            output[0, batch.input_ids.shape[1] :], skip_special_tokens=True
        )
        try:
            parsed, error = validator(raw, image_ids), None
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            parsed, error = None, str(exc)
        return dict(
            raw=raw,
            parsed=parsed,
            validation_error=error,
            seconds=time.monotonic() - started,
            input_tokens=batch.input_ids.shape[1],
            output_tokens=output.shape[1] - batch.input_ids.shape[1],
        )
