"""Bounded visual-prompt refinement with one SAM image encoder per crop."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def interior_points(mask, count=5):
    mask = np.asarray(mask, bool)
    distance = ndimage.distance_transform_edt(mask)
    if not mask.any():
        return []
    selected = []
    score = distance.copy()
    ys, xs = np.where(mask)
    separation = max(2.0, min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) / 3)
    yy, xx = np.indices(mask.shape)
    for _ in range(count):
        y, x = np.unravel_index(np.argmax(score), score.shape)
        if score[y, x] <= 0:
            break
        selected.append([float(x), float(y)])
        score *= np.minimum(1.0, np.hypot(xx - x, yy - y) / separation)
        score[y, x] = 0
    return selected


def prompt_variants(probability):
    """Offer competing prompts; uncertain background is not a universal veto."""
    probability = np.asarray(probability, np.float32)
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("finite foreground projection required")
    support = probability >= 0.1
    if support.sum() < 8:
        return []
    y, x = np.where(support)
    h, w = support.shape
    margin = max(2.0, 0.025 * max(x.max() - x.min(), y.max() - y.min()))
    box = [
        max(0.0, x.min() - margin),
        max(0.0, y.min() - margin),
        min(w - 1.0, x.max() + margin),
        min(h - 1.0, y.max() + margin),
    ]
    core = probability >= max(0.2, 0.65 * float(probability.max()))
    if not core.any():
        return []
    positives = interior_points(core, 5)
    outside_distance = ndimage.distance_transform_edt(~support)
    ring = (outside_distance >= max(3.0, margin)) & (
        outside_distance <= max(6.0, 3 * margin)
    )
    negatives = interior_points(ring, 5)
    return [
        dict(name="box", box=box, points=[], labels=[]),
        dict(name="foreground", box=box, points=positives, labels=[1] * len(positives)),
        dict(
            name="foreground_background",
            box=box,
            points=positives + negatives,
            labels=[1] * len(positives) + [0] * len(negatives),
        ),
    ]


class CachedSAMRefiner:
    def __init__(self, model_path):
        import torch
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        self.torch = torch
        self.processor = Sam3TrackerProcessor.from_pretrained(
            str(model_path), local_files_only=True
        )
        self.model = (
            Sam3TrackerModel.from_pretrained(
                str(model_path),
                local_files_only=True,
                dtype=torch.bfloat16,
            )
            .eval()
            .to("cuda")
        )

    def predict(self, image, variants):
        torch = self.torch
        inputs = self.processor(images=image, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            embeddings = self.model.get_image_embeddings(
                inputs.pixel_values.to(torch.bfloat16)
            )
            results = []
            for variant in variants:
                kwargs = dict(input_boxes=[[variant["box"]]])
                if variant["points"]:
                    kwargs.update(
                        input_points=[[variant["points"]]],
                        input_labels=[[variant["labels"]]],
                    )
                prompts = self.processor(
                    original_sizes=inputs.original_sizes, return_tensors="pt", **kwargs
                ).to("cuda")
                outputs = self.model(
                    image_embeddings=embeddings, multimask_output=True, **prompts
                )
                logits = self.processor.post_process_masks(
                    outputs.pred_masks.float(),
                    inputs.original_sizes,
                    binarize=False,
                )[0][0]
                scores = outputs.iou_scores[0, 0].float().cpu().numpy()
                for index in np.argsort(-scores):
                    results.append(
                        dict(
                            variant=variant["name"],
                            predicted_iou=float(scores[index]),
                            logits=logits[int(index)].float().cpu().numpy(),
                        )
                    )
        return results


def mask_iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if a.shape != b.shape:
        raise ValueError("mask shapes must agree")
    return float((a & b).sum() / max(1, (a | b).sum()))


def candidate_score(candidate):
    """Keep SAM tracker IoU prediction distinct from concept detection confidence."""
    kind = candidate.get("score_kind", "tracker_predicted_iou")
    if kind == "tracker_predicted_iou":
        score = float(candidate["predicted_iou"])
    elif kind == "sam3_presence_instance":
        score = float(candidate["model_score"])
    else:
        raise ValueError("unknown segmentation score kind")
    if not np.isfinite(score):
        raise ValueError("finite segmentation score required")
    return score


def consensus_score_passes(candidate):
    threshold = 0.5 if candidate.get("score_kind") == "sam3_presence_instance" else 0.85
    return candidate_score(candidate) >= threshold


def select_mask_proposal(row, arrays, review_choice=None, reviewed=False):
    """Choose conservative consensus, with explicit per-view review precedence.

    Abstention preserves the original observation; a rendered prior is never
    written back as independent 2D evidence. Model IoU is only a proposal score.
    """
    shown = row["display_candidates"]
    if reviewed:
        if type(review_choice) is not int or review_choice not in (-1, 0, 1, 2, 3):
            raise ValueError("invalid reviewed mask choice")
        if review_choice <= 0:
            return None, "review_preserve_original"
        selected = shown[review_choice]
        if not selected["geometry_eligible"]:
            return None, "review_failed_core_guard"
        return selected["key"], "bounded_vlm_with_core_guard"
    candidates = shown[1:]
    if len(candidates) != 3:
        return None, "proposal_disagreement_or_low_score"
    # A text query with no instance is unavailable evidence, not a conflicting
    # mask. Detected but geometrically invalid alternatives still veto changes.
    concept = all(c.get("score_kind") == "sam3_presence_instance" for c in candidates)
    if concept:
        candidates = [
            c for c in candidates if c.get("abstention") != "no_concept_instance"
        ]
    if len(candidates) < (2 if concept else 3) or not all(
        c["geometry_eligible"] and consensus_score_passes(c) for c in candidates
    ):
        return None, "proposal_disagreement_or_low_score"
    minimum_iou = min(
        mask_iou(arrays[a["key"]], arrays[b["key"]])
        for i, a in enumerate(candidates)
        for b in candidates[i + 1 :]
    )
    if minimum_iou < 0.94:
        return None, "proposal_disagreement_or_low_score"
    selected = max(candidates, key=candidate_score)
    reason = "available_concept_consensus" if concept else "three_prompt_consensus"
    return selected["key"], reason


def restore_crop_mask(mask, row):
    """Inverse exact rotation, then nearest-neighbor resize to the source grid."""
    from PIL import Image
    from farm_runtime.angular_discovery import rotate_image

    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError("boolean crop mask required")
    box = row["crop_grid_xyxy"]
    h, w = row["grid_shape_hw"]
    if len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError("integer crop box required")
    x0, y0, x1, y1 = box
    if not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h:
        raise ValueError("crop outside source grid")
    upright = rotate_image(mask, -row["turns"])
    local = np.asarray(
        Image.fromarray(upright).resize((x1 - x0, y1 - y0), Image.Resampling.NEAREST)
    )
    return box, local
