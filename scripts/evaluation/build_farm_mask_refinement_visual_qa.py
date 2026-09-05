#!/usr/bin/env python3
"""Build deterministic object-centred QA sheets for FARM mask refinement.

This is a visual inspection aid, not a release-acceptance gate.  It consumes
the four-panel JPEGs written by ``refine_farm_full_colmap_masks.py`` and never
runs SAM3 or mutates the source refinement result.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


SOURCE_SCHEMA = "farm.full-colmap-mask-refinement.v1"
INDEX_SCHEMA = "farm.mask-refinement-object-centered-visual-qa.v1"
PANEL_LABELS = ("SOURCE", "SEED", "SAM3", "FINAL")
DEFAULT_CONTEXT_FRACTION = 0.18
DEFAULT_MINIMUM_CONTEXT_PIXELS = 32
DEFAULT_OVERLAY_THRESHOLD = 40

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class ViewInspection:
    object_id: int
    view: Mapping[str, Any]
    visual_path: Path
    mask_path: Path | None
    image_width: int
    image_height: int
    seed_bbox: BBox
    final_bbox: BBox
    prompt_bbox: BBox | None
    crop_bbox: BBox
    seed_bbox_source: str
    final_bbox_source: str
    visual_sha256: str
    mask_sha256: str | None
    crop_panel_stddev: tuple[float, float, float, float]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bbox_from_values(
    values: Sequence[Any], width: int, height: int, *, name: str
) -> BBox:
    _require(len(values) == 4, f"{name} must contain four coordinates")
    numeric = [float(value) for value in values]
    _require(all(math.isfinite(value) for value in numeric), f"{name} is not finite")
    x0 = max(0, min(width, math.floor(numeric[0])))
    y0 = max(0, min(height, math.floor(numeric[1])))
    x1 = max(0, min(width, math.ceil(numeric[2])))
    y1 = max(0, min(height, math.ceil(numeric[3])))
    _require(x1 > x0 and y1 > y0, f"{name} is empty after clipping")
    return x0, y0, x1, y1


def _bbox_from_mask(mask: np.ndarray, *, name: str) -> BBox:
    ys, xs = np.nonzero(mask)
    _require(xs.size > 0, f"{name} contains no foreground pixels")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def bbox_contains(outer: BBox, inner: BBox) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def compute_context_crop(
    bboxes: Iterable[BBox],
    image_width: int,
    image_height: int,
    *,
    context_fraction: float = DEFAULT_CONTEXT_FRACTION,
    minimum_context_pixels: int = DEFAULT_MINIMUM_CONTEXT_PIXELS,
) -> BBox:
    """Return a clipped contextual union that contains every input bbox."""

    boxes = list(bboxes)
    _require(bool(boxes), "at least one bbox is required")
    _require(image_width > 0 and image_height > 0, "image dimensions must be positive")
    _require(context_fraction >= 0.0, "context_fraction must be non-negative")
    _require(minimum_context_pixels >= 0, "minimum_context_pixels must be non-negative")
    for box in boxes:
        _require(
            0 <= box[0] < box[2] <= image_width
            and 0 <= box[1] < box[3] <= image_height,
            f"bbox {box} lies outside {image_width}x{image_height}",
        )
    union = (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )
    pad_x = max(minimum_context_pixels, math.ceil((union[2] - union[0]) * context_fraction))
    pad_y = max(minimum_context_pixels, math.ceil((union[3] - union[1]) * context_fraction))
    crop = (
        max(0, union[0] - pad_x),
        max(0, union[1] - pad_y),
        min(image_width, union[2] + pad_x),
        min(image_height, union[3] + pad_y),
    )
    _require(all(bbox_contains(crop, box) for box in boxes), "context crop cuts an input bbox")
    return crop


def split_visual(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _require(image.ndim == 3 and image.shape[2] == 3, "visual must be a three-channel image")
    height, width = image.shape[:2]
    _require(height > 0 and width > 0, "visual is empty")
    _require(width % 4 == 0, f"visual width {width} is not divisible into four panels")
    panel_width = width // 4
    _require(
        panel_width == height,
        f"expected four square full-frame panels, got panel {panel_width}x{height}",
    )
    return tuple(
        image[:, index * panel_width : (index + 1) * panel_width]
        for index in range(4)
    )  # type: ignore[return-value]


def overlay_bbox(
    source: np.ndarray,
    overlay: np.ndarray,
    *,
    threshold: int = DEFAULT_OVERLAY_THRESHOLD,
    name: str,
) -> tuple[BBox, int]:
    _require(source.shape == overlay.shape, f"{name} overlay shape differs from SOURCE")
    _require(1 <= threshold <= 255, "overlay threshold must be in [1, 255]")
    difference = np.max(cv2.absdiff(source, overlay), axis=2)
    foreground = difference >= threshold
    return _bbox_from_mask(foreground, name=name), int(foreground.sum())


def _load_final_bbox(
    result_root: Path,
    view: Mapping[str, Any],
    source: np.ndarray,
    final_panel: np.ndarray,
    *,
    overlay_threshold: int,
) -> tuple[BBox, str, Path | None, str | None]:
    relative = str(view.get("mask_relative") or "")
    if relative:
        path = result_root / relative
        _require(path.is_file(), f"stored mask sidecar is missing: {path}")
        actual_sha256 = sha256_file(path)
        declared_sha256 = str(view.get("mask_sha256") or "")
        if declared_sha256:
            _require(
                actual_sha256 == declared_sha256,
                f"stored mask hash differs for {path}",
            )
        with np.load(path, allow_pickle=False) as payload:
            required = {"image_shape", "raw_bits", "raw_shape", "raw_bbox_xyxy"}
            _require(required.issubset(payload.files), f"incomplete stored mask sidecar: {path}")
            image_shape = np.asarray(payload["image_shape"], dtype=np.int64).reshape(-1)
            raw_shape = np.asarray(payload["raw_shape"], dtype=np.int64).reshape(-1)
            raw_bbox_values = np.asarray(payload["raw_bbox_xyxy"], dtype=np.int64).reshape(-1)
            _require(image_shape.tolist() == list(source.shape[:2]), f"image_shape differs in {path}")
            _require(raw_shape.size == 2 and np.all(raw_shape > 0), f"invalid raw_shape in {path}")
            _require(raw_bbox_values.size == 4, f"invalid raw_bbox_xyxy in {path}")
            bbox = _bbox_from_values(
                raw_bbox_values.tolist(), source.shape[1], source.shape[0], name=f"final bbox in {path}"
            )
            _require(
                [bbox[3] - bbox[1], bbox[2] - bbox[0]] == raw_shape.tolist(),
                f"raw_shape and raw_bbox_xyxy disagree in {path}",
            )
            bits = np.asarray(payload["raw_bits"], dtype=np.uint8).reshape(-1)
            required_bits = int(np.prod(raw_shape))
            _require(bits.size * 8 >= required_bits, f"raw_bits is truncated in {path}")
            pixel_count = int(
                np.unpackbits(bits, bitorder="little")[:required_bits].sum()
            )
            _require(pixel_count > 0, f"stored final mask is empty: {path}")
            if view.get("stored_mask_pixels") is not None:
                _require(
                    pixel_count == int(view["stored_mask_pixels"]),
                    f"stored_mask_pixels differs from sidecar for {path}",
                )
        return bbox, "stored_mask_sidecar.raw_bbox_xyxy", path, actual_sha256

    for key in ("final_bbox_xyxy", "raw_bbox_xyxy"):
        values = view.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            return (
                _bbox_from_values(values, source.shape[1], source.shape[0], name=key),
                f"result_view.{key}",
                None,
                None,
            )
    bbox, _ = overlay_bbox(
        source, final_panel, threshold=overlay_threshold, name="FINAL visual overlay"
    )
    return bbox, "visual_difference.SOURCE_FINAL", None, None


def validate_result(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    _require(payload.get("schema") == SOURCE_SCHEMA, f"unsupported result schema: {payload.get('schema')!r}")
    _require(payload.get("status") == "PASS", f"source result status is not PASS: {payload.get('status')!r}")
    accepted_ids = payload.get("accepted_object_ids")
    objects = payload.get("objects")
    _require(isinstance(accepted_ids, list), "accepted_object_ids must be a list")
    _require(isinstance(objects, list), "objects must be a list")
    normalized_ids = [int(value) for value in accepted_ids]
    _require(len(normalized_ids) == len(set(normalized_ids)), "accepted_object_ids contains duplicates")
    _require(int(payload.get("accepted_objects", -1)) == len(normalized_ids), "accepted_objects count differs")

    by_id: dict[int, Mapping[str, Any]] = {}
    for obj in objects:
        _require(isinstance(obj, Mapping), "objects entries must be mappings")
        object_id = int(obj.get("object_id", -1))
        _require(object_id not in by_id, f"duplicate object_id {object_id}")
        by_id[object_id] = obj
    actual_ids = [
        object_id for object_id, obj in by_id.items() if bool(obj.get("accepted"))
    ]
    _require(set(normalized_ids) == set(actual_ids), "accepted_object_ids differs from accepted object rows")

    accepted_objects: list[Mapping[str, Any]] = []
    for object_id in normalized_ids:
        obj = by_id[object_id]
        _require(obj.get("status") == "accepted", f"accepted object {object_id} has non-accepted status")
        views = obj.get("views")
        _require(isinstance(views, list), f"object {object_id} views must be a list")
        accepted_views = [view for view in views if isinstance(view, Mapping) and bool(view.get("accepted"))]
        _require(bool(accepted_views), f"accepted object {object_id} has no accepted views")
        _require(
            int(obj.get("accepted_views", -1)) == len(accepted_views),
            f"accepted view count differs for object {object_id}",
        )
        global_ids = [int(view.get("global_image_id", -1)) for view in accepted_views]
        source_images = [str(view.get("source_image") or "") for view in accepted_views]
        _require(all(value >= 0 for value in global_ids), f"object {object_id} has an invalid global_image_id")
        _require(all(source_images), f"object {object_id} has an empty source_image")
        _require(len(global_ids) == len(set(global_ids)), f"object {object_id} repeats a global_image_id")
        _require(len(source_images) == len(set(source_images)), f"object {object_id} repeats a source_image")
        accepted_objects.append(obj)
    return accepted_objects


def inspect_view(
    result_root: Path,
    object_id: int,
    view: Mapping[str, Any],
    *,
    context_fraction: float,
    minimum_context_pixels: int,
    overlay_threshold: int,
) -> ViewInspection:
    relative = str(view.get("visual_relative") or "")
    _require(bool(relative), f"object {object_id} accepted view lacks visual_relative")
    visual_path = result_root / relative
    _require(visual_path.is_file(), f"visual is missing: {visual_path}")
    image = cv2.imread(str(visual_path), cv2.IMREAD_COLOR)
    _require(image is not None, f"cannot decode visual: {visual_path}")
    panels = split_visual(image)
    source, seed_panel, sam_panel, final_panel = panels
    seed_bbox, seed_difference_pixels = overlay_bbox(
        source, seed_panel, threshold=overlay_threshold, name=f"SEED visual overlay {relative}"
    )
    _require(seed_difference_pixels > 0, f"SEED visual overlay is empty: {relative}")
    _, sam_difference_pixels = overlay_bbox(
        source, sam_panel, threshold=overlay_threshold, name=f"SAM3 visual overlay {relative}"
    )
    _require(sam_difference_pixels > 0, f"SAM3 visual overlay is empty: {relative}")
    visual_final_bbox, final_difference_pixels = overlay_bbox(
        source, final_panel, threshold=overlay_threshold, name=f"FINAL visual overlay {relative}"
    )
    _require(final_difference_pixels > 0, f"FINAL visual overlay is empty: {relative}")
    final_bbox, final_bbox_source, mask_path, mask_sha256 = _load_final_bbox(
        result_root,
        view,
        source,
        final_panel,
        overlay_threshold=overlay_threshold,
    )
    _require(
        not (
            final_bbox[2] <= visual_final_bbox[0]
            or final_bbox[0] >= visual_final_bbox[2]
            or final_bbox[3] <= visual_final_bbox[1]
            or final_bbox[1] >= visual_final_bbox[3]
        ),
        f"stored FINAL bbox does not overlap the FINAL panel overlay: {relative}",
    )
    prompt_values = view.get("prompt_box_xyxy")
    prompt_bbox = None
    if isinstance(prompt_values, Sequence) and not isinstance(prompt_values, (str, bytes)):
        prompt_bbox = _bbox_from_values(
            prompt_values,
            source.shape[1],
            source.shape[0],
            name=f"prompt_box_xyxy for object {object_id}",
        )
    boxes = [seed_bbox, final_bbox]
    if prompt_bbox is not None:
        boxes.append(prompt_bbox)
    crop_bbox = compute_context_crop(
        boxes,
        source.shape[1],
        source.shape[0],
        context_fraction=context_fraction,
        minimum_context_pixels=minimum_context_pixels,
    )
    x0, y0, x1, y1 = crop_bbox
    stddev = tuple(float(np.std(panel[y0:y1, x0:x1])) for panel in panels)
    _require(all(value > 1.0 for value in stddev), f"one or more panel crops are blank: {relative}")
    return ViewInspection(
        object_id=object_id,
        view=view,
        visual_path=visual_path,
        mask_path=mask_path,
        image_width=source.shape[1],
        image_height=source.shape[0],
        seed_bbox=seed_bbox,
        final_bbox=final_bbox,
        prompt_bbox=prompt_bbox,
        crop_bbox=crop_bbox,
        seed_bbox_source="visual_difference.SOURCE_SEED",
        final_bbox_source=final_bbox_source,
        visual_sha256=sha256_file(visual_path),
        mask_sha256=mask_sha256,
        crop_panel_stddev=stddev,
    )


def _metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.{digits}f}"


def _gate_summary(view: Mapping[str, Any], key: str) -> tuple[int, int]:
    gates = view.get(key)
    if not isinstance(gates, Mapping):
        return 0, 0
    values = [bool(value) for value in gates.values()]
    return sum(values), len(values)


def _put_fitted_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    max_width: int,
    *,
    scale: float = 0.49,
    color: tuple[int, int, int] = (225, 229, 235),
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    fitted = text
    while fitted and cv2.getTextSize(fitted, font, scale, thickness)[0][0] > max_width:
        fitted = fitted[:-1]
    if fitted != text and len(fitted) >= 3:
        fitted = fitted[:-3] + "..."
    cv2.putText(canvas, fitted, origin, font, scale, color, thickness, cv2.LINE_AA)


def _fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    output = np.full((height, width, 3), (13, 15, 18), dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, int(round(image.shape[1] * scale)))
    resized_height = max(1, int(round(image.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    output[top : top + resized_height, left : left + resized_width] = resized
    return output


def render_object_sheet(
    object_id: int,
    inspections: Sequence[ViewInspection],
    *,
    panel_width: int,
    panel_height: int,
) -> np.ndarray:
    _require(panel_width >= 160 and panel_height >= 160, "rendered panels must be at least 160 pixels")
    margin, gap = 24, 12
    object_header_height = 66
    metadata_height = 94
    label_height = 34
    row_gap = 16
    row_height = metadata_height + label_height + panel_height + row_gap
    sheet_width = 2 * margin + 4 * panel_width + 3 * gap
    sheet_height = object_header_height + len(inspections) * row_height + margin
    canvas = np.full((sheet_height, sheet_width, 3), (27, 29, 34), dtype=np.uint8)
    cv2.putText(
        canvas,
        f"FARM SAM3 v2 OBJECT-CENTERED VISUAL QA  |  OBJECT {object_id:06d}",
        (margin, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (246, 248, 251),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"QA ONLY - NOT RELEASE ACCEPTANCE  |  accepted views: {len(inspections)}",
        (margin, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (100, 185, 255),
        1,
        cv2.LINE_AA,
    )
    label_colors = ((72, 78, 88), (77, 73, 208), (53, 142, 213), (91, 166, 65))
    max_text_width = sheet_width - 2 * margin
    for view_index, inspection in enumerate(inspections, start=1):
        top = object_header_height + (view_index - 1) * row_height
        background = (31, 34, 40) if view_index % 2 else (35, 38, 44)
        cv2.rectangle(
            canvas,
            (10, top),
            (sheet_width - 11, top + row_height - row_gap),
            background,
            -1,
        )
        view = inspection.view
        hard_passed, hard_total = _gate_summary(view, "quality_gates")
        advisory_passed, advisory_total = _gate_summary(view, "advisory_gates")
        raw_pixels = view.get("raw_pixels", view.get("propagated_metric_seed_pixels"))
        if raw_pixels is None and isinstance(view.get("projection_seed_metrics"), Mapping):
            raw_pixels = view["projection_seed_metrics"].get("seed_pixels")
        seed_recall = view.get("raw_recall", view.get("propagated_metric_seed_recall"))
        crop = inspection.crop_bbox
        lines = (
            f"view {view_index:02d}/{len(inspections):02d} | source={view.get('source_image')} | "
            f"global_id={int(view.get('global_image_id', -1)):06d} | seed_origin={view.get('seed_origin', 'n/a')}",
            f"hard_gates={'PASS' if hard_passed == hard_total and hard_total else 'CHECK'} "
            f"({hard_passed}/{hard_total}) | advisory={advisory_passed}/{advisory_total} | "
            f"confidence={_metric(view.get('confidence'))} | seed_recall={_metric(seed_recall)} | "
            f"multiview_recall={_metric(view.get('multiview_seed_recall'))}",
            f"pixels seed/SAM3/final={raw_pixels}/{view.get('sam_pixels')}/{view.get('final_pixels')} | "
            f"area={_metric(view.get('area_expansion'))} | positive={_metric(view.get('positive_inclusion'))} | "
            f"negative={_metric(view.get('negative_inclusion'))} | other={_metric(view.get('other_mask_fraction'))}",
            f"depth_inlier={_metric(view.get('depth_inlier_fraction'))} | "
            f"OBB_inside={_metric(view.get('obb_inside_fraction'))} | crop={list(crop)} | "
            f"bbox inputs=seed+final{' +prompt' if inspection.prompt_bbox else ''}",
        )
        for line_index, line in enumerate(lines):
            _put_fitted_text(
                canvas,
                line,
                (margin, top + 19 + 21 * line_index),
                max_text_width,
            )
        label_top = top + metadata_height
        image_top = label_top + label_height
        image = cv2.imread(str(inspection.visual_path), cv2.IMREAD_COLOR)
        _require(image is not None, f"cannot re-decode visual: {inspection.visual_path}")
        panels = split_visual(image)
        x0, y0, x1, y1 = crop
        for panel_index, (label, color, panel) in enumerate(zip(PANEL_LABELS, label_colors, panels)):
            left = margin + panel_index * (panel_width + gap)
            cv2.rectangle(
                canvas,
                (left, label_top),
                (left + panel_width - 1, label_top + label_height - 2),
                color,
                -1,
            )
            text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[0][0]
            cv2.putText(
                canvas,
                label,
                (left + (panel_width - text_width) // 2, label_top + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            fitted = _fit_panel(panel[y0:y1, x0:x1], panel_width, panel_height)
            canvas[image_top : image_top + panel_height, left : left + panel_width] = fitted
    return canvas


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=".farm-mask-qa-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    _require(bool(ok), f"failed to encode contact sheet: {path}")
    _atomic_write_bytes(path, encoded.tobytes())
    decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    _require(decoded is not None, f"failed to decode written contact sheet: {path}")
    _require(decoded.shape == image.shape, f"written contact sheet has wrong dimensions: {path}")
    _require(float(np.std(decoded)) > 1.0, f"written contact sheet is blank: {path}")


def _inspection_json(inspection: ViewInspection) -> dict[str, Any]:
    view = inspection.view
    hard_passed, hard_total = _gate_summary(view, "quality_gates")
    advisory_passed, advisory_total = _gate_summary(view, "advisory_gates")
    crop = inspection.crop_bbox
    return {
        "source_image": str(view.get("source_image")),
        "global_image_id": int(view.get("global_image_id", -1)),
        "physical_timestamp_ns": int(view.get("physical_timestamp_ns", 0)),
        "seed_origin": str(view.get("seed_origin") or ""),
        "visual_relative": str(view.get("visual_relative") or ""),
        "visual_sha256": inspection.visual_sha256,
        "mask_relative": str(view.get("mask_relative") or "") or None,
        "mask_sha256": inspection.mask_sha256,
        "frame_shape": [inspection.image_height, inspection.image_width],
        "bbox_inputs_xyxy": {
            "seed": list(inspection.seed_bbox),
            "final": list(inspection.final_bbox),
            "prompt": list(inspection.prompt_bbox) if inspection.prompt_bbox else None,
        },
        "bbox_sources": {
            "seed": inspection.seed_bbox_source,
            "final": inspection.final_bbox_source,
            "prompt": "result_view.prompt_box_xyxy" if inspection.prompt_bbox else None,
        },
        "crop_bbox_xyxy": list(crop),
        "crop_shape": [crop[3] - crop[1], crop[2] - crop[0]],
        "crop_area_fraction": (crop[2] - crop[0])
        * (crop[3] - crop[1])
        / (inspection.image_width * inspection.image_height),
        "metrics": {
            key: view.get(key)
            for key in (
                "confidence",
                "raw_pixels",
                "sam_pixels",
                "final_pixels",
                "raw_recall",
                "multiview_seed_recall",
                "positive_inclusion",
                "negative_inclusion",
                "other_mask_fraction",
                "area_expansion",
                "border_fraction",
                "depth_valid_fraction",
                "depth_inlier_fraction",
                "obb_inside_fraction",
            )
            if key in view
        },
        "quality_gates": dict(view.get("quality_gates") or {}),
        "advisory_gates": dict(view.get("advisory_gates") or {}),
        "gate_summary": {
            "quality_passed": hard_passed,
            "quality_total": hard_total,
            "advisory_passed": advisory_passed,
            "advisory_total": advisory_total,
        },
        "qa_checks": {
            "accepted_view": bool(view.get("accepted")),
            "same_crop_all_four_panels": True,
            "all_bbox_inputs_contained": all(
                bbox_contains(crop, box)
                for box in (inspection.seed_bbox, inspection.final_bbox, inspection.prompt_bbox)
                if box is not None
            ),
            "panel_crops_nonblank": all(value > 1.0 for value in inspection.crop_panel_stddev),
            "panel_crop_stddev": [round(value, 6) for value in inspection.crop_panel_stddev],
        },
    }


def _report_ru(index: Mapping[str, Any]) -> str:
    ids = [int(value) for value in index["accepted_object_ids"]]
    lines = [
        "# Object-centered visual QA масок FARM / SAM3 v2",
        "",
        "**Это вспомогательный артефакт визуального QA, а не release acceptance и не разрешение на публикацию.**",
        "",
        f"Источник: `{index['source_result']}`",
        f"SHA-256 источника: `{index['source_result_sha256']}`",
        f"Проверенная схема/статус: `{index['source_schema']}` / `{index['source_status']}`.",
        "",
        f"Принятые object ID ({len(ids)}): " + ", ".join(f"`{value}`" for value in ids) + ".",
        f"Всего принятых views: **{index['accepted_views']}**; contact sheets: **{len(index['objects'])}**.",
        "",
        "## Что показано",
        "",
        "Для каждого принятого view исходный четырёхпанельный visual разбит в порядке "
        "`SOURCE / SEED / SAM3 / FINAL`. Один и тот же crop применён ко всем четырём панелям. "
        "Crop содержит union bbox seed, final и prompt (если prompt сохранён), а затем расширен контекстом; "
        "все bbox проверены на полное попадание в crop.",
        "",
        "BBox seed восстановлен из цветового overlay `SOURCE↔SEED` с фиксированным порогом. "
        "BBox final взят из сохранённого mask sidecar, когда он доступен; sidecar и его SHA-256 проверены. "
        "В заголовке каждого view приведены источник seed, ключевые метрики и сводка hard/advisory gates.",
        "",
        "## Contact sheets",
        "",
        "| object ID | accepted views | файл |",
        "|---:|---:|:---|",
    ]
    for obj in index["objects"]:
        lines.append(
            f"| {obj['object_id']} | {obj['accepted_views']} | [{obj['sheet_relative']}]({obj['sheet_relative']}) |"
        )
    lines.extend(
        [
            "",
            "## Автоматические QA-проверки",
            "",
            "- схема источника и `status=PASS` валидны; declared/actual accepted counts совпадают;",
            "- внутри объекта нет повторов `global_image_id` или `source_image`;",
            "- все visuals декодируются как четыре квадратные панели одинакового размера;",
            "- принятые mask sidecars непустые, их shape/bbox/pixel count и заявленный hash согласованы;",
            "- каждый crop содержит seed/final/prompt bbox, четыре cropped panels непустые;",
            "- каждый записанный JPEG повторно декодирован, проверены его размер и непустое содержимое.",
            "",
            "Полные координаты, метрики, gates, hashes и результаты проверок находятся в [`index.json`](index.json).",
            "",
        ]
    )
    return "\n".join(lines)


def build_qa(
    result_path: Path,
    output_dir: Path,
    *,
    context_fraction: float = DEFAULT_CONTEXT_FRACTION,
    minimum_context_pixels: int = DEFAULT_MINIMUM_CONTEXT_PIXELS,
    overlay_threshold: int = DEFAULT_OVERLAY_THRESHOLD,
    panel_width: int = 300,
    panel_height: int = 340,
    jpeg_quality: int = 94,
) -> dict[str, Any]:
    _require(result_path.is_file(), f"result JSON is missing: {result_path}")
    _require(1 <= jpeg_quality <= 100, "jpeg_quality must be in [1, 100]")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "result JSON root must be an object")
    accepted_objects = validate_result(payload)
    result_root = result_path.parent

    inspections_by_object: list[tuple[Mapping[str, Any], list[ViewInspection]]] = []
    for obj in accepted_objects:
        object_id = int(obj["object_id"])
        accepted_views = [view for view in obj["views"] if bool(view.get("accepted"))]
        accepted_views.sort(key=lambda view: (str(view.get("source_image") or ""), int(view.get("global_image_id", -1))))
        inspections = [
            inspect_view(
                result_root,
                object_id,
                view,
                context_fraction=context_fraction,
                minimum_context_pixels=minimum_context_pixels,
                overlay_threshold=overlay_threshold,
            )
            for view in accepted_views
        ]
        inspections_by_object.append((obj, inspections))

    output_dir.mkdir(parents=True, exist_ok=True)
    for temporary in output_dir.glob(".farm-mask-qa-*.tmp"):
        temporary.unlink(missing_ok=True)
    object_entries: list[dict[str, Any]] = []
    expected_sheets: set[Path] = set()
    for obj, inspections in inspections_by_object:
        object_id = int(obj["object_id"])
        sheet_relative = Path(f"object_{object_id:06d}.jpg")
        sheet_path = output_dir / sheet_relative
        sheet = render_object_sheet(
            object_id,
            inspections,
            panel_width=panel_width,
            panel_height=panel_height,
        )
        _write_jpeg(sheet_path, sheet, jpeg_quality)
        expected_sheets.add(sheet_path)
        object_entries.append(
            {
                "object_id": object_id,
                "accepted_views": len(inspections),
                "sheet_relative": sheet_relative.as_posix(),
                "sheet_shape": list(sheet.shape[:2]),
                "sheet_sha256": sha256_file(sheet_path),
                "views": [_inspection_json(inspection) for inspection in inspections],
            }
        )

    for stale in output_dir.glob("object_[0-9][0-9][0-9][0-9][0-9][0-9].jpg"):
        if stale not in expected_sheets:
            stale.unlink()

    index: dict[str, Any] = {
        "schema": INDEX_SCHEMA,
        "purpose": "visual_qa_only_not_release_acceptance",
        "release_acceptance": False,
        "source_result": str(result_path.resolve()),
        "source_result_sha256": sha256_file(result_path),
        "source_schema": str(payload["schema"]),
        "source_status": str(payload["status"]),
        "accepted_object_ids": [int(value) for value in payload["accepted_object_ids"]],
        "accepted_objects": len(object_entries),
        "accepted_views": sum(entry["accepted_views"] for entry in object_entries),
        "panel_order": list(PANEL_LABELS),
        "crop_policy": {
            "inputs": ["seed_bbox", "final_bbox", "prompt_bbox_when_available"],
            "union": True,
            "context_fraction_per_axis": context_fraction,
            "minimum_context_pixels_per_axis": minimum_context_pixels,
            "overlay_difference_threshold": overlay_threshold,
            "same_crop_for_all_panels": True,
            "letterbox_without_distortion": True,
        },
        "render": {
            "panel_width": panel_width,
            "panel_height": panel_height,
            "jpeg_quality": jpeg_quality,
        },
        "objects": object_entries,
    }
    index_bytes = (json.dumps(index, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_write_bytes(output_dir / "index.json", index_bytes)
    _atomic_write_bytes(output_dir / "REPORT_RU.md", _report_ru(index).encode("utf-8"))
    _require(not list(output_dir.glob(".farm-mask-qa-*.tmp")), "temporary files remain in output")
    return index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path, help="Mask-refinement result.json")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (default: <result-dir>/object_centered_qa)",
    )
    parser.add_argument("--context-fraction", type=float, default=DEFAULT_CONTEXT_FRACTION)
    parser.add_argument("--minimum-context-pixels", type=int, default=DEFAULT_MINIMUM_CONTEXT_PIXELS)
    parser.add_argument("--overlay-threshold", type=int, default=DEFAULT_OVERLAY_THRESHOLD)
    parser.add_argument("--panel-width", type=int, default=300)
    parser.add_argument("--panel-height", type=int, default=340)
    parser.add_argument("--jpeg-quality", type=int, default=94)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result_path = args.result.resolve()
    output_dir = args.output.resolve() if args.output else result_path.parent / "object_centered_qa"
    index = build_qa(
        result_path,
        output_dir,
        context_fraction=args.context_fraction,
        minimum_context_pixels=args.minimum_context_pixels,
        overlay_threshold=args.overlay_threshold,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
        jpeg_quality=args.jpeg_quality,
    )
    print(
        f"Wrote {index['accepted_objects']} object sheets / {index['accepted_views']} views "
        f"to {output_dir} (visual QA only; not release acceptance)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
