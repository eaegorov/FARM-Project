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

    def review(self, images, image_ids):
        if not 1 <= len(images) <= 3 or len(images) != len(image_ids):
            raise ValueError("one to three aligned image/ID pairs required")
        torch = self.torch
        content = [{"type": "text", "text": REVIEW_PROMPT}]
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
            output = self.model.generate(**batch, max_new_tokens=500, do_sample=False)
        torch.cuda.synchronize()
        raw = self.processor.decode(
            output[0, batch.input_ids.shape[1] :], skip_special_tokens=True
        )
        try:
            parsed, error = validate_review(raw, image_ids), None
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
