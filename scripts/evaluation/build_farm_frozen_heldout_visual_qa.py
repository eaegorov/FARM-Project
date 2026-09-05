#!/usr/bin/env python3
"""Build source-backed object-centred PNG QA for frozen heldout projections.

The tool is deliberately read-only.  It visualises the exact observations
already evaluated by ``frozen_heldout_qc`` and refuses unverified RGB/mask
artifacts, partial candidate coverage, or changed inputs.  It is an inspection
aid, never a substitute for the numeric frozen-heldout release gate.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from farm_runtime.frozen_heldout_qc import (  # noqa: E402
    _load_mask,
    sha256_file,
)

QC_SCHEMA = "farm.frozen-heldout-qc.v1"
EVIDENCE_SCHEMA = "farm.frozen-heldout-evidence.v1"
FOLD_SCHEMA = "farm.full-colmap-rgbd-folds.v1"
FRAMES_SCHEMA = "farm_frames_json_v1"
REPORT_SCHEMA = "farm.frozen-heldout-object-centered-visual-qa.v1"
RESULT_SCHEMA = "farm.frozen-heldout-object-centered-visual-qa.result.v1"

BBox = tuple[int, int, int, int]
ObservationKey = tuple[int, str]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / filename,
        Path("/usr/share/fonts/dejavu") / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _sha256(value: object, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    _require(
        len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"{field} must be a lowercase SHA-256",
    )
    return digest


def _resolve_path(value: object, *, base: Path, field: str) -> Path:
    text = str(value or "").strip()
    _require(bool(text), f"{field} has no path")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve(strict=True)
    _require(path.is_file(), f"{field} is not a regular file: {path}")
    return path


def _remember_hash(path: Path, hashes: dict[Path, str]) -> str:
    digest = hashes.get(path)
    if digest is None:
        digest = sha256_file(path)
        hashes[path] = digest
    return digest


def _verify_artifact(
    spec: object,
    *,
    base: Path,
    field: str,
    hashes: dict[Path, str],
) -> tuple[Path, Mapping[str, Any]]:
    _require(isinstance(spec, Mapping), f"{field} must be an artifact object")
    path = _resolve_path(spec.get("path"), base=base, field=field)
    expected = _sha256(spec.get("sha256"), field=f"{field}.sha256")
    _require(
        _remember_hash(path, hashes) == expected,
        f"{field} SHA-256 mismatch: {path}",
    )
    if "bytes" in spec:
        size = spec.get("bytes")
        _require(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0,
            f"{field}.bytes must be a non-negative integer",
        )
        _require(path.stat().st_size == size, f"{field} byte-size mismatch")
    return path, spec


def _verified_json_from_provenance(
    provenance: Mapping[str, Any],
    name: str,
    hashes: dict[Path, str],
) -> tuple[Path, dict[str, Any]]:
    spec = provenance.get(name)
    _require(isinstance(spec, Mapping), f"QC provenance lacks {name}")
    path = _resolve_path(spec.get("path"), base=Path.cwd(), field=name)
    expected = _sha256(spec.get("sha256"), field=f"{name}.sha256")
    _require(_remember_hash(path, hashes) == expected, f"{name} SHA-256 mismatch")
    return path, _json_object(path)


def _observation_key(row: Mapping[str, Any], *, field: str) -> ObservationKey:
    object_id = int(row.get("object_id"))
    source = str(row.get("source_image") or "").strip()
    _require(bool(source), f"{field} has an empty source_image")
    return object_id, source


def _bbox_from_masks(*masks: np.ndarray) -> BBox:
    union = np.logical_or.reduce(masks)
    ys, xs = np.nonzero(union)
    _require(bool(xs.size), "target and prediction masks are both empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def compute_context_crop(
    masks: Sequence[np.ndarray],
    *,
    context_fraction: float,
    minimum_context_pixels: int,
) -> BBox:
    """Return a clipped contextual bbox around the target/prediction union."""

    _require(bool(masks), "at least one mask is required")
    shape = masks[0].shape
    _require(len(shape) == 2, "mask must be two-dimensional")
    _require(all(mask.shape == shape for mask in masks), "mask shapes differ")
    _require(
        math.isfinite(context_fraction) and context_fraction >= 0.0,
        "context_fraction must be finite and non-negative",
    )
    _require(minimum_context_pixels >= 0, "minimum_context_pixels is negative")
    x0, y0, x1, y1 = _bbox_from_masks(*masks)
    pad_x = max(minimum_context_pixels, math.ceil((x1 - x0) * context_fraction))
    pad_y = max(minimum_context_pixels, math.ceil((y1 - y0) * context_fraction))
    height, width = shape
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    )


def _load_rgb(path: Path, *, expected_shape: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    _require(
        (rgb.height, rgb.width) == expected_shape,
        f"RGB shape {(rgb.height, rgb.width)} differs from masks {expected_shape}",
    )
    return rgb


def _tinted_overlay(
    rgb: Image.Image, mask: np.ndarray, color: tuple[int, int, int]
) -> Image.Image:
    source = np.asarray(rgb, dtype=np.float32)
    result = source.copy()
    selected = np.asarray(mask, dtype=bool)
    result[selected] = source[selected] * 0.34 + np.asarray(color) * 0.66
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _error_overlay(
    rgb: Image.Image, predicted: np.ndarray, target: np.ndarray
) -> Image.Image:
    source = np.asarray(rgb, dtype=np.float32)
    result = source * 0.24
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    regions = (
        (predicted & target, np.asarray((37, 214, 115), dtype=np.float32)),
        (predicted & ~target, np.asarray((244, 76, 76), dtype=np.float32)),
        (~predicted & target, np.asarray((66, 153, 245), dtype=np.float32)),
    )
    for selected, color in regions:
        result[selected] = source[selected] * 0.20 + color * 0.80
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _fit_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (7, 11, 18))
    _require(image.width > 0 and image.height > 0, "panel image is empty")
    scale = min(width / image.width, height / image.height)
    fitted = image.resize(
        (
            max(1, min(width, round(image.width * scale))),
            max(1, min(height, round(image.height * scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return canvas


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def _median(object_row: Mapping[str, Any], metric: str) -> object:
    summary = (object_row.get("summaries") or {}).get(metric)
    return summary.get("median") if isinstance(summary, Mapping) else None


def _render_object_sheet(
    object_row: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]
) -> Image.Image:
    panel_width, panel_height = 374, 300
    margin, gap = 18, 8
    title_height, column_height, row_label_height = 112, 42, 70
    columns = 4
    width = margin * 2 + columns * panel_width + (columns - 1) * gap
    row_height = row_label_height + panel_height + gap
    height = margin * 2 + title_height + column_height + len(observations) * row_height
    sheet = Image.new("RGB", (width, height), (7, 11, 18))
    draw = ImageDraw.Draw(sheet)
    title_font = _font(26, bold=True)
    label_font = _font(18, bold=True)
    small_font = _font(14)
    tiny_font = _font(12)

    object_id = int(object_row["object_id"])
    status = str(object_row.get("status") or "unknown").upper()
    route = str(object_row.get("route") or "")
    status_color = (54, 211, 153) if status == "VERIFIED" else (251, 191, 36)
    draw.text(
        (margin, margin),
        f"FROZEN HELDOUT QA · OBJECT {object_id:06d}",
        font=title_font,
        fill=(242, 246, 252),
    )
    draw.text((margin, margin + 37), status, font=label_font, fill=status_color)
    draw.text((margin + 112, margin + 40), route, font=small_font, fill=(169, 183, 204))
    summary = (
        f"timestamps {int(object_row.get('good_timestamps', 0))}/"
        f"{int(object_row.get('independent_physical_timestamps', 0))} good  ·  "
        f"median IoU {_fmt(_median(object_row, 'iou'))}  ·  "
        f"P {_fmt(_median(object_row, 'precision'))}  ·  "
        f"R {_fmt(_median(object_row, 'recall'))}"
    )
    draw.text((margin, margin + 68), summary, font=small_font, fill=(205, 214, 227))
    reasons = ", ".join(str(value) for value in object_row.get("reasons") or [])
    if reasons:
        draw.text(
            (margin, margin + 88),
            f"gate reasons: {reasons}",
            font=tiny_font,
            fill=(251, 146, 60),
        )

    headings = (
        ("SOURCE RGB", "SHA-verified heldout frame", (82, 171, 255)),
        ("HELDOUT TARGET", "evaluation-only SAM3 pseudo-label", (52, 211, 153)),
        ("GAUSSIAN PREDICTION", "frozen post-lift projection", (251, 146, 60)),
        ("PIXEL ERROR", "TP green · FP red · FN blue", (196, 181, 253)),
    )
    heading_y = margin + title_height
    for column, (heading, subtitle, color) in enumerate(headings):
        x = margin + column * (panel_width + gap)
        draw.rectangle(
            (x, heading_y, x + panel_width, heading_y + column_height),
            fill=(15, 23, 35),
        )
        draw.text((x + 9, heading_y + 5), heading, font=small_font, fill=color)
        draw.text(
            (x + 9, heading_y + 23), subtitle, font=tiny_font, fill=(137, 151, 173)
        )

    for row_index, observation in enumerate(observations):
        row_y = heading_y + column_height + row_index * row_height
        draw.rectangle(
            (margin, row_y, width - margin, row_y + row_label_height), fill=(11, 18, 29)
        )
        metrics = observation["metrics"]
        source = str(observation["source_image"])
        timestamp = str(observation["physical_timestamp"])
        draw.text(
            (margin + 9, row_y + 8),
            f"{source} · physical timestamp {timestamp}",
            font=label_font,
            fill=(238, 242, 248),
        )
        line = (
            f"IoU {_fmt(metrics.get('iou'))}   precision {_fmt(metrics.get('precision'))}   "
            f"recall {_fmt(metrics.get('recall'))}   area ratio {_fmt(metrics.get('area_ratio'))}   "
            f"competing {_fmt(metrics.get('competing_prediction_fraction'))}   "
            f"depth median {_fmt(metrics.get('depth_error_median'))} m / p90 {_fmt(metrics.get('depth_error_p90'))} m"
        )
        draw.text((margin + 9, row_y + 38), line, font=small_font, fill=(171, 186, 207))
        panel_y = row_y + row_label_height
        crop = observation["crop_bbox_xyxy"]
        images = (
            observation["rgb"],
            observation["target_overlay"],
            observation["prediction_overlay"],
            observation["error_overlay"],
        )
        for column, image in enumerate(images):
            x = margin + column * (panel_width + gap)
            cropped = image.crop(crop)
            sheet.paste(_fit_panel(cropped, panel_width, panel_height), (x, panel_y))
    return sheet


def _render_summary(
    objects: Sequence[Mapping[str, Any]],
    observations: Mapping[int, Sequence[Mapping[str, Any]]],
) -> Image.Image:
    thumb_width, thumb_height = 300, 210
    margin, row_gap = 20, 10
    header_height, row_header = 86, 64
    width = 1280
    height = (
        margin * 2
        + header_height
        + len(objects) * (row_header + thumb_height + row_gap)
    )
    sheet = Image.new("RGB", (width, height), (7, 11, 18))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (margin, margin),
        "FROZEN HELDOUT GAUSSIAN MASK QA",
        font=_font(29, bold=True),
        fill=(244, 247, 252),
    )
    draw.text(
        (margin, margin + 42),
        "Best-IoU heldout view per exact candidate · inspection only",
        font=_font(16),
        fill=(153, 168, 190),
    )
    label_font = _font(19, bold=True)
    small_font = _font(14)
    for index, object_row in enumerate(objects):
        object_id = int(object_row["object_id"])
        choices = list(observations[object_id])
        representative = max(
            choices, key=lambda row: float(row["metrics"].get("iou") or -1.0)
        )
        y = margin + header_height + index * (row_header + thumb_height + row_gap)
        status = str(object_row.get("status") or "unknown")
        color = (54, 211, 153) if status == "verified" else (251, 191, 36)
        draw.rectangle((margin, y, width - margin, y + row_header), fill=(13, 21, 33))
        draw.text(
            (margin + 10, y + 8),
            f"OBJECT {object_id:06d} · {status.upper()}",
            font=label_font,
            fill=color,
        )
        draw.text(
            (margin + 10, y + 35),
            f"median IoU {_fmt(_median(object_row, 'iou'))} · P {_fmt(_median(object_row, 'precision'))} · R {_fmt(_median(object_row, 'recall'))} · {object_row.get('route')}",
            font=small_font,
            fill=(177, 191, 211),
        )
        crop = representative["crop_bbox_xyxy"]
        previews = (
            representative["rgb"],
            representative["target_overlay"],
            representative["prediction_overlay"],
            representative["error_overlay"],
        )
        for column, image in enumerate(previews):
            x = margin + column * (thumb_width + 6)
            sheet.paste(
                _fit_panel(image.crop(crop), thumb_width, thumb_height),
                (x, y + row_header),
            )
    return sheet


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_png(path: Path, image: Image.Image) -> None:
    image.save(path, format="PNG", optimize=True)
    with Image.open(path) as decoded:
        decoded.verify()
    _require(path.stat().st_size > 1024, f"rendered PNG is implausibly small: {path}")


def _load_inputs(
    qc_report_path: Path,
) -> tuple[
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[Path, str],
]:
    hashes: dict[Path, str] = {}
    qc_report_path = qc_report_path.expanduser().resolve(strict=True)
    qc = _json_object(qc_report_path)
    _remember_hash(qc_report_path, hashes)
    _require(qc.get("schema") == QC_SCHEMA, "unsupported frozen-heldout QC schema")
    provenance = qc.get("provenance")
    _require(isinstance(provenance, Mapping), "QC report lacks provenance")
    evidence_path, evidence = _verified_json_from_provenance(
        provenance, "evidence_manifest", hashes
    )
    _require(
        evidence.get("schema") == EVIDENCE_SCHEMA
        and evidence.get("status") == "frozen",
        "evidence is not a frozen v1 manifest",
    )
    fold_path, fold = _verified_json_from_provenance(
        provenance, "fold_manifest", hashes
    )
    _require(
        fold.get("schema") == FOLD_SCHEMA and fold.get("status") == "PASS",
        "fold is not a PASS v1 manifest",
    )
    heldout_path, heldout_document = _verified_json_from_provenance(
        provenance, "heldout_frames", hashes
    )
    _require(
        heldout_document.get("schema_version") == FRAMES_SCHEMA,
        "unsupported heldout frames schema",
    )
    fold_descriptor = (fold.get("folds") or {}).get("heldout")
    _require(isinstance(fold_descriptor, Mapping), "fold lacks heldout descriptor")
    declared_heldout = _resolve_path(
        fold_descriptor.get("frames_json"),
        base=fold_path.parent,
        field="fold heldout frames",
    )
    _require(
        declared_heldout == heldout_path,
        "QC and fold refer to different heldout frames",
    )
    _require(
        _sha256(fold_descriptor.get("sha256"), field="fold heldout frames SHA")
        == hashes[heldout_path],
        "fold heldout frames SHA-256 mismatch",
    )

    frames: dict[str, Mapping[str, Any]] = {}
    for raw in heldout_document.get("frames") or []:
        _require(isinstance(raw, Mapping), "heldout frames contain a non-object row")
        source = str(raw.get("source_image") or "").strip()
        _require(
            source and source not in frames,
            "heldout frames contain empty/duplicate source",
        )
        frames[source] = raw
    rendered = ((fold.get("provenance") or {}).get("rendered_artifacts") or {}).get(
        "heldout"
    )
    _require(isinstance(rendered, list), "fold lacks heldout RGB artifact provenance")
    rgb_specs: dict[str, Mapping[str, Any]] = {}
    for raw in rendered:
        _require(isinstance(raw, Mapping), "heldout artifact row is not an object")
        source = str(raw.get("source_image") or "").strip()
        _require(
            source in frames and source not in rgb_specs,
            "unknown/duplicate heldout artifact source",
        )
        rgb = raw.get("rgb")
        _require(
            isinstance(rgb, Mapping), f"heldout RGB provenance missing for {source}"
        )
        rgb_specs[source] = rgb
    _require(
        set(rgb_specs) == set(frames),
        "heldout RGB provenance does not exactly cover frames",
    )
    return (
        qc,
        evidence_path,
        evidence,
        heldout_path,
        heldout_document,
        frames,
        rgb_specs,
        hashes,
    )


def build_visual_qa(
    qc_report_path: Path,
    output_dir: Path,
    *,
    context_fraction: float = 0.24,
    minimum_context_pixels: int = 36,
    summary: bool = True,
) -> dict[str, Any]:
    """Validate inputs and atomically publish object-centred QA sheets."""

    (
        qc,
        evidence_path,
        evidence,
        heldout_path,
        _heldout_document,
        frames,
        rgb_specs,
        hashes,
    ) = _load_inputs(qc_report_path)
    qc_report_path = qc_report_path.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    _require(not output_dir.exists(), f"refusing to overwrite output: {output_dir}")

    raw_objects = qc.get("objects")
    _require(
        isinstance(raw_objects, list) and raw_objects,
        "QC report has no candidate objects",
    )
    objects: list[Mapping[str, Any]] = []
    qc_views: dict[ObservationKey, Mapping[str, Any]] = {}
    for raw_object in raw_objects:
        _require(isinstance(raw_object, Mapping), "QC contains a non-object row")
        object_id = int(raw_object.get("object_id"))
        _require(
            all(int(existing["object_id"]) != object_id for existing in objects),
            f"duplicate QC object {object_id}",
        )
        views = raw_object.get("views")
        _require(
            isinstance(views, list) and views, f"QC object {object_id} has no views"
        )
        for view in views:
            _require(
                isinstance(view, Mapping),
                f"QC object {object_id} has a non-object view",
            )
            key = _observation_key(view, field="QC view")
            _require(
                key[0] == object_id and key not in qc_views,
                f"duplicate/mismatched QC observation {key}",
            )
            qc_views[key] = view
        objects.append(raw_object)
    objects.sort(key=lambda row: int(row["object_id"]))
    declared_count = int((qc.get("counts") or {}).get("candidate_objects", -1))
    _require(declared_count == len(objects), "QC candidate object count differs")

    evidence_rows: dict[ObservationKey, Mapping[str, Any]] = {}
    for row in evidence.get("observations") or []:
        _require(isinstance(row, Mapping), "evidence contains a non-object observation")
        key = _observation_key(row, field="evidence observation")
        _require(key not in evidence_rows, f"duplicate evidence observation {key}")
        evidence_rows[key] = row
    _require(set(evidence_rows) == set(qc_views), "QC/evidence observation sets differ")

    rendered_by_object: dict[int, list[dict[str, Any]]] = defaultdict(list)
    input_rows: list[dict[str, Any]] = []
    for key in sorted(qc_views):
        object_id, source = key
        view = qc_views[key]
        evidence_row = evidence_rows[key]
        frame = frames.get(source)
        _require(isinstance(frame, Mapping), f"heldout frame missing for {source}")
        timestamp = str(frame.get("physical_timestamp") or "")
        _require(
            timestamp
            and timestamp == str(view.get("physical_timestamp") or "")
            and timestamp == str(evidence_row.get("physical_timestamp") or ""),
            f"physical timestamp mismatch for {key}",
        )
        target_path, target_spec = _verify_artifact(
            evidence_row.get("target_mask"),
            base=evidence_path.parent,
            field=f"{key} target mask",
            hashes=hashes,
        )
        prediction_path, prediction_spec = _verify_artifact(
            evidence_row.get("prediction_mask"),
            base=evidence_path.parent,
            field=f"{key} prediction mask",
            hashes=hashes,
        )
        target = _load_mask(target_path, target_spec, field=f"{key} target mask")
        prediction = _load_mask(
            prediction_path, prediction_spec, field=f"{key} prediction mask"
        )
        expected_shape = tuple(int(value) for value in frame.get("depth_size") or [])
        _require(
            len(expected_shape) == 2
            and target.shape == prediction.shape == expected_shape,
            f"mask/frame shapes differ for {key}",
        )

        rgb_path, rgb_spec = _verify_artifact(
            rgb_specs[source],
            base=heldout_path.parent,
            field=f"{key} RGB",
            hashes=hashes,
        )
        frame_rgb = _resolve_path(
            frame.get("rgb_path"), base=heldout_path.parent, field=f"{key} frame RGB"
        )
        _require(rgb_path == frame_rgb, f"frame/provenance RGB paths differ for {key}")
        rgb = _load_rgb(rgb_path, expected_shape=expected_shape)
        crop = compute_context_crop(
            (target, prediction),
            context_fraction=context_fraction,
            minimum_context_pixels=minimum_context_pixels,
        )
        artifacts = view.get("artifacts") or {}
        _require(isinstance(artifacts, Mapping), f"QC view {key} lacks artifacts")
        for name, expected_path in (
            ("target_mask", target_path),
            ("prediction_mask", prediction_path),
        ):
            declared = artifacts.get(name)
            _require(isinstance(declared, Mapping), f"QC view {key} lacks {name}")
            _require(
                Path(str(declared.get("path"))).resolve() == expected_path,
                f"QC/evidence {name} path differs for {key}",
            )
            _require(
                _sha256(declared.get("sha256"), field=f"QC {name} SHA")
                == hashes[expected_path],
                f"QC/evidence {name} SHA differs for {key}",
            )
        rendered_by_object[object_id].append(
            {
                "source_image": source,
                "physical_timestamp": timestamp,
                "rgb": rgb,
                "target_overlay": _tinted_overlay(rgb, target, (35, 211, 145)),
                "prediction_overlay": _tinted_overlay(rgb, prediction, (249, 145, 48)),
                "error_overlay": _error_overlay(rgb, prediction, target),
                "crop_bbox_xyxy": crop,
                "metrics": view,
                "artifacts": {
                    "rgb": {
                        "path": str(rgb_path),
                        "bytes": rgb_path.stat().st_size,
                        "sha256": hashes[rgb_path],
                    },
                    "target_mask": {
                        "path": str(target_path),
                        "bytes": target_path.stat().st_size,
                        "sha256": hashes[target_path],
                    },
                    "prediction_mask": {
                        "path": str(prediction_path),
                        "bytes": prediction_path.stat().st_size,
                        "sha256": hashes[prediction_path],
                    },
                },
            }
        )
        input_rows.append(
            {
                "object_id": object_id,
                "source_image": source,
                "physical_timestamp": timestamp,
                "crop_bbox_xyxy": list(crop),
                "frame_shape": list(expected_shape),
                "metrics": {
                    name: view.get(name)
                    for name in (
                        "iou",
                        "precision",
                        "recall",
                        "area_ratio",
                        "largest_component_fraction",
                        "competing_prediction_fraction",
                        "depth_coverage",
                        "depth_error_median",
                        "depth_error_p90",
                    )
                },
                "artifacts": rendered_by_object[object_id][-1]["artifacts"],
            }
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.", suffix=".tmp", dir=output_dir.parent
        )
    )
    try:
        output_rows = []
        for object_row in objects:
            object_id = int(object_row["object_id"])
            observations = sorted(
                rendered_by_object[object_id],
                key=lambda row: (
                    str(row["physical_timestamp"]),
                    str(row["source_image"]),
                ),
            )
            _require(
                bool(observations),
                f"candidate object {object_id} has no rendered observations",
            )
            relative = Path("objects") / f"object_{object_id:06d}.png"
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_png(path, _render_object_sheet(object_row, observations))
            output_rows.append(
                {
                    "object_id": object_id,
                    "status": str(object_row.get("status") or ""),
                    "route": str(object_row.get("route") or ""),
                    "views": len(observations),
                    "path": relative.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        summary_spec = None
        if summary:
            relative = Path("summary.png")
            path = temporary / relative
            _write_png(path, _render_summary(objects, rendered_by_object))
            summary_spec = {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

        changed = [
            str(path) for path, digest in hashes.items() if sha256_file(path) != digest
        ]
        _require(not changed, f"frozen input changed during visual QA: {changed[:8]}")
        report = {
            "schema": REPORT_SCHEMA,
            "status": "PASS",
            "purpose": "read_only_visual_inspection_not_release_acceptance",
            "source_qc_status": qc.get("status"),
            "contract": {
                "read_only": True,
                "candidate_objects_exactly_covered": True,
                "qc_and_evidence_observations_exactly_matched": True,
                "rgb_and_masks_sha256_verified_and_rechecked": True,
                "heldout_updates_candidate": False,
                "target_is_evaluation_only_pseudo_label": True,
                "visuals_do_not_override_numeric_release_gate": True,
            },
            "counts": {
                "candidate_objects": len(objects),
                "observations": len(input_rows),
                "verified_input_files": len(hashes),
                "object_sheets": len(output_rows),
            },
            "render_config": {
                "context_fraction": context_fraction,
                "minimum_context_pixels": minimum_context_pixels,
                "error_colors_rgb": {
                    "true_positive": [37, 214, 115],
                    "false_positive": [244, 76, 76],
                    "false_negative": [66, 153, 245],
                },
            },
            "inputs": {
                "qc_report": {
                    "path": str(qc_report_path),
                    "sha256": hashes[qc_report_path],
                },
                "evidence_manifest": {
                    "path": str(evidence_path),
                    "sha256": hashes[evidence_path],
                },
                "heldout_frames": {
                    "path": str(heldout_path),
                    "sha256": hashes[heldout_path],
                },
                "consumed_files": [
                    {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}
                    for path, digest in sorted(
                        hashes.items(), key=lambda item: str(item[0])
                    )
                ],
                "observations": input_rows,
            },
            "outputs": {"objects": output_rows, "summary": summary_spec},
        }
        report_path = temporary / "manifest.json"
        _write_json(report_path, report)
        _write_json(
            temporary / "_RESULT.json",
            {
                "schema": RESULT_SCHEMA,
                "status": "PASS",
                "manifest": "manifest.json",
                "manifest_sha256": sha256_file(report_path),
            },
        )
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qc-report",
        type=Path,
        required=True,
        help="report.json from evaluate_farm_frozen_heldout_qc.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new output directory; existing paths are never overwritten",
    )
    parser.add_argument("--context-fraction", type=float, default=0.24)
    parser.add_argument("--minimum-context-pixels", type=int, default=36)
    parser.add_argument(
        "--summary", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_visual_qa(
        args.qc_report,
        args.output,
        context_fraction=args.context_fraction,
        minimum_context_pixels=args.minimum_context_pixels,
        summary=args.summary,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "counts": report["counts"],
                "output": str(args.output.expanduser().resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
