#!/usr/bin/env python3
"""Render a management-readable QA dashboard for FARM object geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from scipy.spatial import ConvexHull


BG = "#0b111c"
PANEL = "#111a28"
GRID = "#4d596b"
TEXT = "#e8edf4"
MUTED = "#a8b1c0"
CYAN = "#38c8e8"
GREEN = "#4bd982"
RED = "#ef6a79"
GOLD = "#f3b63f"


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.22, linewidth=0.6)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _wxyz_to_matrix(value: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(value, dtype=float) / max(float(np.linalg.norm(value)), 1.0e-12)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _box_corners(center: np.ndarray, dimensions: np.ndarray, wxyz: np.ndarray) -> np.ndarray:
    signs = np.asarray([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=float)
    return center + (signs * (0.5 * dimensions)) @ _wxyz_to_matrix(wxyz).T


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-id", default="scene")
    parser.add_argument("--max-cloud-points", type=int, default=120_000)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = list(report.get("objects") or [])
    thresholds = report.get("thresholds") or {}
    cloud = np.load(args.cloud)["xyz"].astype(np.float32, copy=False)
    if args.max_cloud_points > 0 and cloud.shape[0] > args.max_cloud_points:
        rng = np.random.default_rng(20260803)
        cloud = cloud[rng.choice(cloud.shape[0], args.max_cloud_points, replace=False)]

    passed = np.asarray([row.get("status") == "geometry_pass" for row in rows], dtype=bool)
    ids = np.asarray([int(row["object_id"]) for row in rows])
    centers = np.asarray([row["box_center_m"] for row in rows], dtype=float)
    dimensions = np.asarray([row["box_dimensions_m"] for row in rows], dtype=float)
    orientations = np.asarray([row.get("box_wxyz") or [1.0, 0.0, 0.0, 0.0] for row in rows], dtype=float)
    inside = np.asarray([float(row.get("center_inside_detection_rate") or 0.0) for row in rows])
    errors = np.asarray([
        float(row["median_reprojection_error_normalized"])
        if row.get("median_reprojection_error_normalized") is not None else np.nan
        for row in rows
    ])
    old_volume = np.asarray([float(row.get("original_box_volume_m3") or np.nan) for row in rows])
    new_volume = np.asarray([float(row.get("refined_box_volume_m3") or np.nan) for row in rows])
    box_ious = np.asarray([float(row.get("median_projected_box_iou") or 0.0) for row in rows])
    support_rates = np.asarray([float(row.get("box_support_rate") or 0.0) for row in rows])
    voxel_inside_rates = np.asarray([float(row.get("voxel_inside_obb_rate") or 0.0) for row in rows])
    orientation_confidences = np.asarray([float(row.get("orientation_confidence") or 0.0) for row in rows])

    fig = plt.figure(figsize=(19.2, 10.8), dpi=200, facecolor=BG)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.45, 1.0, 1.0], height_ratios=[1.0, 1.0], hspace=0.26, wspace=0.22)
    map_ax = fig.add_subplot(grid[:, 0])
    inside_ax = fig.add_subplot(grid[0, 1])
    error_ax = fig.add_subplot(grid[0, 2])
    volume_ax = fig.add_subplot(grid[1, 1])
    notes_ax = fig.add_subplot(grid[1, 2])
    for ax in (map_ax, inside_ax, error_ax, volume_ax):
        _style_axis(ax)
    notes_ax.set_facecolor(PANEL)
    notes_ax.axis("off")

    finite_cloud = np.isfinite(cloud).all(axis=1)
    cloud = cloud[finite_cloud]
    map_ax.scatter(cloud[:, 0], cloud[:, 2], s=0.09, c="#8793a5", alpha=0.16, linewidths=0, rasterized=True)
    projected_boxes: list[np.ndarray] = []
    for idx, row in enumerate(rows):
        corners_xz = _box_corners(centers[idx], dimensions[idx], orientations[idx])[:, [0, 2]]
        hull = corners_xz[ConvexHull(corners_xz).vertices]
        projected_boxes.append(hull)
        x, z = centers[idx, 0], centers[idx, 2]
        color = GREEN if passed[idx] else RED
        map_ax.add_patch(patches.Polygon(hull, closed=True, fill=False, ec=color, lw=1.0, alpha=0.8))
        map_ax.scatter([x], [z], s=19 if passed[idx] else 32, c=color, edgecolors=BG, linewidths=0.5, zorder=4)
        if not passed[idx]:
            map_ax.text(x, z, f"  #{ids[idx]}", color=RED, fontsize=7, va="center")
    map_ax.set_title("Metric oriented boxes after RGB-D refinement (X–Z)", fontsize=13, pad=10)
    map_ax.set_xlabel("world X, m")
    map_ax.set_ylabel("world Z, m")
    map_ax.set_aspect("equal", adjustable="box")
    cloud_lo, cloud_hi = np.percentile(cloud[:, [0, 2]], [1.0, 99.0], axis=0)
    all_projected = np.concatenate(projected_boxes, axis=0)
    box_lo, box_hi = all_projected.min(axis=0), all_projected.max(axis=0)
    plot_lo, plot_hi = np.minimum(cloud_lo, box_lo), np.maximum(cloud_hi, box_hi)
    padding = np.maximum((plot_hi - plot_lo) * 0.06, 0.5)
    map_ax.set_xlim(plot_lo[0] - padding[0], plot_hi[0] + padding[0])
    map_ax.set_ylim(plot_lo[1] - padding[1], plot_hi[1] + padding[1])

    order = np.argsort(inside)
    y = np.arange(len(rows))
    colors = np.where(passed[order], GREEN, RED)
    inside_ax.barh(y, inside[order] * 100.0, color=colors, alpha=0.88)
    inside_ax.axvline(float(thresholds.get("min_center_inside_rate", 0.70)) * 100.0, color=GOLD, ls="--", lw=1.2)
    inside_ax.set_xlim(0, 102)
    inside_ax.set_yticks(y[::3], [f"#{value}" for value in ids[order][::3]])
    inside_ax.set_xlabel("observations containing projected 3D centre, %")
    inside_ax.set_title("Multi-view centre alignment", fontsize=11)

    iou_order = np.argsort(box_ious)
    error_ax.barh(y, box_ious[iou_order] * 100.0, color=np.where(passed[iou_order], CYAN, RED), alpha=0.88)
    error_ax.axvline(float(thresholds.get("min_median_projected_box_iou", 0.20)) * 100.0, color=GOLD, ls="--", lw=1.2)
    error_ax.set_yticks(y[::3], [f"#{value}" for value in ids[iou_order][::3]])
    error_ax.set_xlabel("median projected OBB / detection IoU, %")
    error_ax.set_title("Position + size + rotation reprojection", fontsize=11)

    finite_volume = np.isfinite(old_volume) & np.isfinite(new_volume) & (old_volume > 0) & (new_volume > 0)
    vmax = max(float(np.nanmax(old_volume[finite_volume])), float(np.nanmax(new_volume[finite_volume])), 0.1)
    vmin = min(float(np.nanmin(old_volume[finite_volume])), float(np.nanmin(new_volume[finite_volume])), 0.1)
    volume_ax.loglog([vmin, vmax], [vmin, vmax], color=MUTED, ls="--", lw=1.0, alpha=0.7)
    volume_ax.scatter(old_volume[finite_volume & passed], new_volume[finite_volume & passed], s=34, c=GREEN, alpha=0.8, label="retained")
    volume_ax.scatter(old_volume[finite_volume & ~passed], new_volume[finite_volume & ~passed], s=42, c=RED, marker="x", label="rejected")
    volume_ax.set_xlabel("original covariance-box volume, m³")
    volume_ax.set_ylabel("refined RGB-D box volume, m³")
    volume_ax.set_title("Box fit before → after", fontsize=11)
    volume_ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    rejected = [row for row in rows if row.get("status") != "geometry_pass"]
    median_inside = float(np.median(inside[passed])) if passed.any() else 0.0
    median_error = float(np.nanmedian(errors[passed])) if passed.any() else float("nan")
    median_iou = float(np.median(box_ious[passed])) if passed.any() else 0.0
    median_support = float(np.median(support_rates[passed])) if passed.any() else 0.0
    median_voxel_inside = float(np.median(voxel_inside_rates[passed])) if passed.any() else 0.0
    median_orientation = float(np.median(orientation_confidences[passed])) if passed.any() else 0.0
    lines = [
        "GEOMETRY QA SUMMARY",
        "",
        f"Input reviewed objects     {len(rows)}",
        f"Retained after geometry    {int(passed.sum())}",
        f"Rejected automatically     {int((~passed).sum())}",
        "",
        f"Median centre-in-mask      {median_inside:.0%}",
        f"Median normalized residual {median_error:.3f}",
        f"Median projected OBB IoU   {median_iou:.0%}",
        f"Median supported views     {median_support:.0%}",
        f"Median voxel agreement     {median_voxel_inside:.0%}",
        f"Median orientation conf.   {median_orientation:.0%}",
        "",
        "REJECTED OBJECTS",
    ]
    for row in rejected[:10]:
        lines.append(
            f"#{int(row['object_id']):03d}  {str(row.get('category') or 'object')[:23]:23s}  "
            f"2D={float(row.get('median_projected_box_iou') or 0.0):.0%} "
            f"vox={float(row.get('voxel_inside_obb_rate') or 0.0):.0%}"
        )
    if len(rejected) > 10:
        lines.append(f"... {len(rejected) - 10} more in geometry_audit.json")
    lines += [
        "",
        "Policy is category-agnostic:",
        "multi-view depth consistency + robust refit +",
        "8-corner OBB reprojection support.",
        "No object names or text prompts are used for geometry gating.",
    ]
    notes_ax.text(0.06, 0.94, "\n".join(lines), transform=notes_ax.transAxes, va="top", ha="left", color=TEXT, fontsize=8.5, family="monospace", linespacing=1.24)

    scene_label = str(args.scene_id).replace("_", " ").strip().upper() or "SCENE"
    fig.text(0.035, 0.965, f"{scene_label} | OBJECT GEOMETRY REFINEMENT QA", color=TEXT, fontsize=22, weight="bold", ha="left")
    fig.text(0.035, 0.935, "Saved inlier masks + metric 3DGS depth + COLMAP camera poses", color=MUTED, fontsize=11, ha="left")
    fig.text(0.965, 0.025, "green = retained  ·  red = automatically rejected  ·  gold dashed line = gate", color=MUTED, fontsize=9, ha="right")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=BG, dpi=200)
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
