"""Class-agnostic duplicate suppression for overlapping instance masks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def suppress_contained_masks(
    masks: torch.Tensor,
    scores: torch.Tensor,
    *,
    containment_threshold: float = 0.82,
    minimum_larger_area_ratio: float = 1.35,
    comparison_size: int = 160,
) -> torch.Tensor:
    """Return indices after suppressing lower-score masks contained in a larger one.

    The comparison is deliberately class agnostic.  It targets duplicate labels
    and part detections inside an already detected complete assembly.  Nearest
    downsampling bounds memory while preserving mask topology for this gate.
    """

    if masks.ndim != 3:
        raise ValueError(f"masks must have shape [N,H,W], got {tuple(masks.shape)}")
    count = int(masks.shape[0])
    if scores.ndim != 1 or int(scores.shape[0]) != count:
        raise ValueError("scores must be row-aligned with masks")
    if count <= 1:
        return torch.arange(count, device=masks.device, dtype=torch.long)
    if not 0.0 < containment_threshold <= 1.0:
        raise ValueError("containment threshold must lie in (0,1]")
    if minimum_larger_area_ratio <= 1.0 or comparison_size < 16:
        raise ValueError("invalid containment comparison limits")

    height, width = int(masks.shape[1]), int(masks.shape[2])
    scale = min(1.0, comparison_size / float(max(height, width)))
    target = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    sampled = masks.to(dtype=torch.float32)
    if target != (height, width):
        sampled = F.interpolate(
            sampled.unsqueeze(1), size=target, mode="nearest"
        ).squeeze(1)
    flat = (sampled > 0.5).to(dtype=torch.float32).flatten(1)
    areas = flat.sum(dim=1)
    intersections = flat @ flat.t()
    keep = torch.ones(count, dtype=torch.bool, device=masks.device)
    order = torch.argsort(scores, descending=True, stable=True)
    for position, larger_index in enumerate(order.tolist()):
        if not bool(keep[larger_index]) or float(areas[larger_index]) <= 0.0:
            continue
        for smaller_index in order[position + 1 :].tolist():
            if not bool(keep[smaller_index]) or float(areas[smaller_index]) <= 0.0:
                continue
            if float(areas[larger_index]) < (
                minimum_larger_area_ratio * float(areas[smaller_index])
            ):
                continue
            contained = float(intersections[larger_index, smaller_index]) / float(
                areas[smaller_index]
            )
            if contained >= containment_threshold:
                keep[smaller_index] = False
    return torch.nonzero(keep, as_tuple=False).squeeze(1)
