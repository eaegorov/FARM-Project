"""Deterministic, inspectable heldout exports for Gaussian Grouping identity runs."""

from __future__ import annotations

import colorsys
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EXPORT_SCHEMA = "farm.gaussian-grouping-heldout-export.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def compact_id_palette() -> np.ndarray:
    """Return a deterministic 256-entry RGB palette with a dark background."""

    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[0] = (18, 21, 27)
    for identity in range(1, 256):
        hue = (identity * 0.6180339887498949) % 1.0
        saturation = 0.68 + 0.14 * ((identity * 37) % 3) / 2.0
        value = 0.88 + 0.10 * ((identity * 53) % 2)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        palette[identity] = np.rint(np.asarray(rgb) * 255.0).astype(np.uint8)
    return palette


def validate_compact_mask(
    value: np.ndarray,
    *,
    expected_size_wh: tuple[int, int],
    maximum_id: int,
) -> np.ndarray:
    array = np.asarray(value)
    width, height = (int(item) for item in expected_size_wh)
    if not 0 <= int(maximum_id) <= 254:
        raise ValueError("maximum compact identity ID must be in [0, 254]")
    if array.ndim != 2 or array.shape != (height, width):
        raise ValueError(
            f"compact identity mask has shape {array.shape}, expected {(height, width)}"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("compact identity mask must have an integer dtype")
    minimum = int(array.min(initial=0))
    maximum = int(array.max(initial=0))
    if minimum < 0 or maximum > int(maximum_id) or maximum > 254:
        raise ValueError("compact identity mask contains an undeclared ID")
    return array.astype(np.uint8, copy=False)


def colorize_compact_mask(
    mask: np.ndarray, palette: np.ndarray | None = None
) -> np.ndarray:
    colors = (
        compact_id_palette() if palette is None else np.asarray(palette, dtype=np.uint8)
    )
    if colors.shape != (256, 3):
        raise ValueError("identity palette must have shape (256, 3)")
    values = np.asarray(mask)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("identity mask must be a 2D integer array")
    if int(values.min(initial=0)) < 0 or int(values.max(initial=0)) > 255:
        raise ValueError("identity mask cannot index the uint8 palette")
    return colors[values.astype(np.uint8, copy=False)]


def _boundaries(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask)
    boundary = np.zeros(values.shape, dtype=bool)
    boundary[1:, :] |= values[1:, :] != values[:-1, :]
    boundary[:-1, :] |= values[:-1, :] != values[1:, :]
    boundary[:, 1:] |= values[:, 1:] != values[:, :-1]
    boundary[:, :-1] |= values[:, :-1] != values[:, 1:]
    return boundary


def make_identity_overlay(
    rgb: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    palette: np.ndarray | None = None,
) -> np.ndarray:
    """Blend prediction with RGB and expose target/prediction disagreement."""

    image = np.asarray(rgb, dtype=np.uint8)
    target_ids = np.asarray(target)
    prediction_ids = np.asarray(prediction)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or target_ids.shape != image.shape[:2]
        or prediction_ids.shape != image.shape[:2]
    ):
        raise ValueError("RGB, target and prediction must share one HxW resolution")
    predicted_colors = colorize_compact_mask(prediction_ids, palette)
    foreground = prediction_ids > 0
    overlay = image.astype(np.float32)
    overlay[foreground] = (
        0.58 * overlay[foreground] + 0.42 * predicted_colors[foreground]
    )
    mismatch = target_ids != prediction_ids
    overlay[mismatch] = 0.35 * overlay[mismatch] + 0.65 * np.asarray(
        [255.0, 55.0, 45.0]
    )
    overlay = np.clip(np.rint(overlay), 0, 255).astype(np.uint8)
    target_boundary = _boundaries(target_ids)
    prediction_boundary = _boundaries(prediction_ids)
    overlay[target_boundary] = (0, 255, 255)
    overlay[prediction_boundary & ~target_boundary] = (255, 0, 255)
    return overlay


def _atomic_png(path: Path, pixels: np.ndarray, *, mode: str) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    Image.fromarray(np.asarray(pixels), mode=mode).save(temporary, format="PNG")
    os.replace(temporary, destination)
    with Image.open(destination) as image:
        size = list(image.size)
        actual_mode = image.mode
        actual_format = image.format
    if actual_mode != mode or actual_format != "PNG":
        raise RuntimeError(f"invalid PNG export: {destination}")
    return {
        "relative_path": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "size_wh": size,
        "mode": actual_mode,
    }


def export_heldout_frame(
    output_root: Path,
    *,
    source_name: str,
    source_sha256: str,
    rgb: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    maximum_id: int,
    palette: np.ndarray | None = None,
) -> dict[str, Any]:
    """Write exact-resolution prediction and QA PNGs for one frozen frame."""

    if Path(source_name).name != source_name:
        raise ValueError("heldout source_name must be a basename")
    normalized_source_sha256 = str(source_sha256).lower()
    if len(normalized_source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in normalized_source_sha256
    ):
        raise ValueError("heldout source SHA256 must be 64 hexadecimal characters")
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("heldout RGB must be HxWx3 uint8")
    height, width = image.shape[:2]
    expected = (width, height)
    target_ids = validate_compact_mask(
        target, expected_size_wh=expected, maximum_id=maximum_id
    )
    prediction_ids = validate_compact_mask(
        prediction, expected_size_wh=expected, maximum_id=maximum_id
    )
    colors = compact_id_palette() if palette is None else palette
    stem = Path(source_name).stem
    root = Path(output_root)
    paths = {
        "compact_prediction": root / "predictions" / f"{stem}.png",
        "target_color": root / "qa" / "target" / f"{stem}.png",
        "prediction_color": root / "qa" / "prediction" / f"{stem}.png",
        "overlay": root / "qa" / "overlay" / f"{stem}.png",
    }
    artifacts = {
        "compact_prediction": _atomic_png(
            paths["compact_prediction"], prediction_ids, mode="L"
        ),
        "target_color": _atomic_png(
            paths["target_color"], colorize_compact_mask(target_ids, colors), mode="RGB"
        ),
        "prediction_color": _atomic_png(
            paths["prediction_color"],
            colorize_compact_mask(prediction_ids, colors),
            mode="RGB",
        ),
        "overlay": _atomic_png(
            paths["overlay"],
            make_identity_overlay(image, target_ids, prediction_ids, palette=colors),
            mode="RGB",
        ),
    }
    for key, path in paths.items():
        artifacts[key]["relative_path"] = path.relative_to(root).as_posix()
    return {
        "source_name": source_name,
        "source_sha256": normalized_source_sha256,
        "training_resolution_wh": [width, height],
        "target_ids": sorted(int(value) for value in np.unique(target_ids)),
        "prediction_ids": sorted(int(value) for value in np.unique(prediction_ids)),
        "pixel_accuracy": float(np.mean(target_ids == prediction_ids)),
        "artifacts": artifacts,
    }


def build_contact_sheet(
    output_root: Path,
    frames: Iterable[dict[str, Any]],
    *,
    preview_width: int = 180,
    columns: int = 2,
) -> dict[str, Any]:
    rows = list(frames)
    if not rows:
        raise ValueError("heldout contact sheet requires at least one frame")
    if int(preview_width) < 64 or int(columns) < 1:
        raise ValueError("invalid contact sheet layout")
    root = Path(output_root)
    font = ImageFont.load_default()
    tiles: list[Image.Image] = []
    title_height = 30
    for row in rows:
        panels: list[Image.Image] = []
        for key in ("target_color", "prediction_color", "overlay"):
            path = root / row["artifacts"][key]["relative_path"]
            with Image.open(path) as source:
                panel = source.convert("RGB")
                height = max(1, int(round(panel.height * preview_width / panel.width)))
                panels.append(
                    panel.resize((preview_width, height), Image.Resampling.LANCZOS)
                )
        panel_height = min(panel.height for panel in panels)
        panels = [
            panel.resize((preview_width, panel_height), Image.Resampling.LANCZOS)
            for panel in panels
        ]
        tile = Image.new(
            "RGB", (preview_width * 3, title_height + panel_height), (10, 14, 20)
        )
        draw = ImageDraw.Draw(tile)
        draw.text((8, 3), str(row["source_name"]), fill=(238, 242, 248), font=font)
        draw.text(
            (8, 15),
            f"target | prediction | overlay    acc={row['pixel_accuracy']:.3f}",
            fill=(150, 164, 182),
            font=font,
        )
        for index, panel in enumerate(panels):
            tile.paste(panel, (index * preview_width, title_height))
        tiles.append(tile)
    tile_width = max(tile.width for tile in tiles)
    tile_height = max(tile.height for tile in tiles)
    row_count = (len(tiles) + int(columns) - 1) // int(columns)
    sheet = Image.new(
        "RGB",
        (tile_width * int(columns), tile_height * row_count),
        (6, 9, 14),
    )
    for index, tile in enumerate(tiles):
        sheet.paste(
            tile,
            (
                (index % int(columns)) * tile_width,
                (index // int(columns)) * tile_height,
            ),
        )
    artifact = _atomic_png(
        root / "heldout_contact_sheet.png", np.asarray(sheet), mode="RGB"
    )
    artifact["relative_path"] = "heldout_contact_sheet.png"
    artifact["preview_panel_width"] = int(preview_width)
    artifact["columns"] = int(columns)
    return artifact


def write_export_manifest(
    output_root: Path,
    frames: Iterable[dict[str, Any]],
    contact_sheet: dict[str, Any],
    *,
    maximum_id: int,
    resolution_factor: int,
) -> dict[str, Any]:
    rows = list(frames)
    names = [str(row.get("source_name") or "") for row in rows]
    if not rows or len(names) != len(set(names)) or any(not name for name in names):
        raise ValueError("heldout export frames must have unique source names")
    stems = [Path(name).stem for name in names]
    if len(stems) != len(set(stems)):
        raise ValueError("heldout export source names must have unique output stems")
    manifest = {
        "schema": EXPORT_SCHEMA,
        "status": "complete",
        "frame_count": len(rows),
        "maximum_compact_id": int(maximum_id),
        "resolution_factor": int(resolution_factor),
        "training_resolutions_wh": sorted(
            {tuple(row["training_resolution_wh"]) for row in rows}
        ),
        "contact_sheet": contact_sheet,
        "frames": rows,
        "invariants": {
            "all_heldout_frames_exported": True,
            "compact_predictions_are_uint8_l_png": True,
            "per_frame_artifacts_match_training_resolution": True,
            "heldout_export_updates_training": False,
            "heldout_export_updates_geometry": False,
            "source_gaussian_order_preserved": True,
        },
    }
    destination = Path(output_root) / "manifest.json"
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "schema": EXPORT_SCHEMA,
        "frame_count": len(rows),
        "contact_sheet": contact_sheet,
    }
