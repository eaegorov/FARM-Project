#!/usr/bin/env python3
"""Visualise full 8-corner OBB reprojection against source detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


BG = "#0b111c"
PANEL = "#111a28"
TEXT = "#e8edf4"
MUTED = "#a8b1c0"
CYAN = "#38c8e8"
GREEN = "#4bd982"
GOLD = "#f3b63f"
EDGES = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
SIGNS = np.asarray([
    [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
    [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
], dtype=np.float64)


def _numpy(value: object, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _wxyz_to_matrix(value: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(value, dtype=float) / max(float(np.linalg.norm(value)), 1.0e-12)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _corners(center: np.ndarray, dimensions: np.ndarray, wxyz: np.ndarray) -> np.ndarray:
    return center + (SIGNS * (0.5 * dimensions)) @ _wxyz_to_matrix(wxyz).T


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    lo = np.maximum(first[:2], second[:2])
    hi = np.minimum(first[2:], second[2:])
    intersection = max(0.0, float(hi[0] - lo[0])) * max(0.0, float(hi[1] - lo[1]))
    area_first = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    area_second = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    return intersection / max(area_first + area_second - intersection, 1.0e-12)


def _record_value(record: object, key: str, default: object = None) -> object:
    return record.get(key, default) if isinstance(record, dict) else getattr(record, key, default)


def _resolve_mask(raw_path: str, mapping_dir: Path) -> Path | None:
    raw = Path(raw_path)
    candidates = [raw, mapping_dir / "masks" / raw.parent.name / raw.name]
    return next((path for path in candidates if path.is_file()), None)


def _project(points_world: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray) -> np.ndarray | None:
    homogeneous = np.column_stack((points_world, np.ones((len(points_world),), dtype=np.float64)))
    camera = (np.linalg.inv(pose) @ homogeneous.T).T[:, :3]
    if not np.isfinite(camera).all() or np.any(camera[:, 2] <= 1.0e-5):
        return None
    image = (intrinsics @ camera.T).T
    return image[:, :2] / image[:, 2:3]


def _render_overlay(image_path: Path, detection: np.ndarray, projected: np.ndarray, center_uv: np.ndarray) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    detection_tuple = tuple(float(value) for value in detection)
    draw.rectangle(detection_tuple, outline=GOLD, width=5)
    for first, second in EDGES:
        draw.line((tuple(projected[first]), tuple(projected[second])), fill=CYAN, width=4)
    radius = 7
    draw.ellipse((center_uv[0] - radius, center_uv[1] - radius, center_uv[0] + radius, center_uv[1] + radius), fill=GREEN, outline="#07110b", width=2)

    projected_bbox = np.r_[projected.min(axis=0), projected.max(axis=0)]
    union_lo = np.minimum(detection[:2], projected_bbox[:2])
    union_hi = np.maximum(detection[2:], projected_bbox[2:])
    extent = np.maximum(union_hi - union_lo, 40.0)
    padding = np.maximum(0.30 * extent, 36.0)
    crop_lo = np.maximum(np.floor(union_lo - padding), 0).astype(int)
    crop_hi = np.minimum(np.ceil(union_hi + padding), np.asarray(image.size)).astype(int)
    if np.any(crop_hi - crop_lo < 16):
        return image
    return image.crop((int(crop_lo[0]), int(crop_lo[1]), int(crop_hi[0]), int(crop_hi[1])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--geometry-report", type=Path, required=True)
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument("--mapping-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-id", default="scene")
    parser.add_argument("--objects", type=int, default=4)
    parser.add_argument("--views", type=int, default=3)
    args = parser.parse_args()

    payload = torch.load(args.scene_state, map_location="cpu", weights_only=False)
    state = payload["state"] if isinstance(payload, dict) and isinstance(payload.get("state"), dict) else payload
    report = json.loads(args.geometry_report.read_text(encoding="utf-8"))
    report_by_id = {int(row["object_id"]): row for row in report.get("objects", []) if row.get("status") == "geometry_pass"}
    frames_payload = json.loads(args.frames_json.read_text(encoding="utf-8"))
    frames_root = args.frames_json.parent
    frames_by_stem: dict[str, dict] = {}
    for frame in frames_payload.get("frames", []):
        for key in ("rgb_path", "source_image"):
            if frame.get(key):
                frames_by_stem[Path(str(frame[key])).stem] = frame

    active = _numpy(state["active"]).astype(bool)
    object_ids = _numpy(state["object_id"]).astype(np.int64)
    observations = state.get("object_mask_observations") or []
    images = list(state.get("images") or [])
    categories = state.get("object_category") or []
    candidates: list[tuple[float, int, int]] = []
    for index in np.flatnonzero(active).tolist():
        object_id = int(object_ids[index])
        row = report_by_id.get(object_id)
        if row is None:
            continue
        score = (
            float(row.get("valid_observations") or 0)
            * float(row.get("median_projected_box_iou") or 0.0)
            * max(float(row.get("orientation_confidence") or 0.0), 0.10)
        )
        candidates.append((score, index, object_id))
    candidates.sort(reverse=True)
    selected = candidates[: max(1, int(args.objects))]

    rendered_rows: list[dict] = []
    for _, index, object_id in selected:
        geometry = report_by_id[object_id]
        center = np.asarray(geometry["box_center_m"], dtype=np.float64)
        corners = _corners(center, np.asarray(geometry["box_dimensions_m"], dtype=np.float64), np.asarray(geometry["box_wxyz"], dtype=np.float64))
        projected_views: list[dict] = []
        for observation in observations[index] if index < len(observations) else []:
            if not isinstance(observation, dict):
                continue
            mask_path = _resolve_mask(str(observation.get("path") or ""), args.mapping_dir)
            image_id = int(observation.get("image_id", -1))
            if mask_path is None or image_id < 0 or image_id >= len(images):
                continue
            record = images[image_id]
            source_ref = str(_record_value(record, "source_ref", "") or _record_value(record, "storage_path", ""))
            frame = frames_by_stem.get(Path(source_ref).stem)
            if frame is None:
                continue
            image_path = frames_root / str(frame.get("rgb_path") or "")
            if not image_path.is_file():
                continue
            pose = _numpy(_record_value(record, "pose"), np.float64)
            intrinsics = np.asarray(frame.get("K"), dtype=np.float64)
            projected = _project(corners, pose, intrinsics)
            center_projected = _project(center.reshape(1, 3), pose, intrinsics)
            if projected is None or center_projected is None:
                continue
            with np.load(mask_path, allow_pickle=False) as archive:
                detection = np.asarray(archive["raw_bbox_xyxy"], dtype=np.float64).reshape(4)
            projected_bbox = np.r_[projected.min(axis=0), projected.max(axis=0)]
            projected_views.append({
                "image_id": image_id,
                "iou": _bbox_iou(projected_bbox, detection),
                "image": _render_overlay(image_path, detection, projected, center_projected[0]),
            })
        projected_views.sort(key=lambda view: float(view["iou"]))
        count = min(max(1, int(args.views)), len(projected_views))
        indices = np.linspace(0, len(projected_views) - 1, count).round().astype(int) if projected_views else []
        rendered_rows.append({
            "object_id": object_id,
            "category": str(categories[index] if index < len(categories) else "object"),
            "dimensions": geometry["box_dimensions_m"],
            "wxyz": geometry["box_wxyz"],
            "views": [projected_views[int(view_index)] for view_index in indices],
        })

    row_count = len(rendered_rows)
    column_count = max(1, int(args.views))
    fig = plt.figure(figsize=(19.2, 10.8), dpi=200, facecolor=BG)
    grid = fig.add_gridspec(row_count, column_count, left=0.035, right=0.985, top=0.895, bottom=0.07, hspace=0.34, wspace=0.05)
    for row_index, row in enumerate(rendered_rows):
        for column_index in range(column_count):
            ax = fig.add_subplot(grid[row_index, column_index])
            ax.set_facecolor(PANEL)
            ax.axis("off")
            if column_index < len(row["views"]):
                view = row["views"][column_index]
                ax.imshow(view["image"])
                ax.set_title(f"view {int(view['image_id']):03d} · projected OBB IoU {float(view['iou']):.0%}", color=MUTED, fontsize=9, loc="right", pad=4)
            if column_index == 0:
                dims = " × ".join(f"{float(value):.2f}" for value in row["dimensions"])
                ax.text(0.0, 1.09, f"#{int(row['object_id']):03d}  {row['category']}   ·   OBB {dims} m", transform=ax.transAxes, color=TEXT, fontsize=11, weight="bold", va="bottom")

    scene_label = str(args.scene_id).replace("_", " ").strip().upper() or "SCENE"
    fig.text(0.035, 0.965, f"{scene_label} | METRIC OBB REPROJECTION CHECK", color=TEXT, fontsize=22, weight="bold", ha="left")
    fig.text(0.035, 0.932, "cyan = projected 8-corner oriented box · gold = source 2D detection · green = projected 3D centre", color=MUTED, fontsize=11, ha="left")
    fig.text(0.985, 0.026, "rows are selected automatically by evidence × projected IoU × orientation confidence; views show low / median / high overlap", color=MUTED, fontsize=9, ha="right")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=BG, dpi=200)
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
