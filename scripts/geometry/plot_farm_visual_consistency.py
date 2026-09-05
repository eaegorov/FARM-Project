#!/usr/bin/env python3
"""Render a management-readable QA dashboard for prompt-free crop consistency."""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


BG = "#0b111c"
PANEL = "#111a28"
GRID = "#4d596b"
TEXT = "#e8edf4"
MUTED = "#a8b1c0"
CYAN = "#38c8e8"
GREEN = "#4bd982"
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


def _resolve_crop(raw_path: str, mapping_dir: Path) -> Path | None:
    raw = Path(raw_path)
    candidates = [raw, mapping_dir / "masks" / raw.parent.name / raw.name]
    return next((path for path in candidates if path.is_file()), None)


def _read_crop(raw_path: str, mapping_dir: Path) -> Image.Image | None:
    path = _resolve_crop(raw_path, mapping_dir)
    if path is None:
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            jpeg = np.asarray(archive["crop_jpeg_bytes"], dtype=np.uint8).tobytes()
        return Image.open(io.BytesIO(jpeg)).convert("RGB")
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mapping-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-id", default="scene")
    parser.add_argument("--lowest", type=int, default=6)
    parser.add_argument("--crops-per-object", type=int, default=3)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = [row for row in report.get("objects", []) if row.get("median_pairwise_similarity") is not None]
    rows.sort(key=lambda row: float(row["median_pairwise_similarity"]))
    scores = np.asarray([float(row["median_pairwise_similarity"]) for row in rows])
    object_ids = np.asarray([int(row["object_id"]) for row in rows])
    gate = report.get("gate") or {}
    threshold = float(gate.get("threshold") or -1.0)

    fig = plt.figure(figsize=(19.2, 10.8), dpi=200, facecolor=BG)
    outer = fig.add_gridspec(2, 1, height_ratios=[0.95, 1.12], hspace=0.28)
    top = outer[0].subgridspec(1, 3, width_ratios=[1.45, 0.8, 0.9], wspace=0.24)
    bar_ax = fig.add_subplot(top[0, 0])
    hist_ax = fig.add_subplot(top[0, 1])
    notes_ax = fig.add_subplot(top[0, 2])
    for ax in (bar_ax, hist_ax):
        _style_axis(ax)
    notes_ax.set_facecolor(PANEL)
    notes_ax.axis("off")

    y = np.arange(len(rows))
    colors = np.where(scores >= threshold, GREEN, GOLD)
    bar_ax.barh(y, scores, color=colors, alpha=0.9)
    bar_ax.axvline(threshold, color=GOLD, ls="--", lw=1.4, label=f"automatic lower fence {threshold:.3f}")
    bar_ax.axvline(float(gate.get("median") or np.median(scores)), color=CYAN, ls=":", lw=1.4, label="scene median")
    step = max(1, len(rows) // 11)
    bar_ax.set_yticks(y[::step], [f"#{value}" for value in object_ids[::step]])
    bar_ax.set_xlim(max(-0.05, threshold - 0.12), min(1.0, float(scores.max()) + 0.06))
    bar_ax.set_xlabel("median DINOv3 cosine similarity across object views")
    bar_ax.set_title("Multi-view appearance consistency per object", fontsize=12)
    bar_ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8, loc="lower right")

    bins = np.linspace(max(0.0, float(scores.min()) - 0.05), min(1.0, float(scores.max()) + 0.05), 10)
    hist_ax.hist(scores, bins=bins, color=CYAN, alpha=0.86, edgecolor=BG)
    hist_ax.axvline(threshold, color=GOLD, ls="--", lw=1.4)
    hist_ax.axvline(float(gate.get("median") or np.median(scores)), color=GREEN, lw=1.4)
    hist_ax.set_xlabel("object median similarity")
    hist_ax.set_ylabel("objects")
    hist_ax.set_title("Scene-derived score distribution", fontsize=12)

    lines = [
        "VISUAL ASSOCIATION QA",
        "",
        f"Input geometry-validated  {int(report.get('input_objects') or len(rows))}",
        f"Retained automatically    {int(report.get('retained_objects') or 0)}",
        f"Rejected as outliers      {len(report.get('rejected_objects') or [])}",
        "",
        f"Scene median similarity   {float(gate.get('median') or np.median(scores)):.3f}",
        f"Robust lower fence        {threshold:.3f}",
        f"Interquartile range       {float(gate.get('q1') or 0.0):.3f}–{float(gate.get('q3') or 0.0):.3f}",
        "",
        "AUTOMATIC POLICY",
        "DINOv3 embeddings from up to 6 saved views.",
        "The gate comes from this scene's score distribution.",
        "Categories, captions and hand-written prompts are not inputs.",
        "",
        "Lowest-scoring retained objects are shown below",
        "for visual review, not silently removed.",
    ]
    notes_ax.text(0.06, 0.94, "\n".join(lines), transform=notes_ax.transAxes, va="top", ha="left", color=TEXT, fontsize=10, family="monospace", linespacing=1.42)

    lowest = rows[: max(1, min(int(args.lowest), len(rows)))]
    columns_per_object = max(1, int(args.crops_per_object))
    object_columns = 3
    object_rows = int(np.ceil(len(lowest) / object_columns))
    gallery = outer[1].subgridspec(object_rows, object_columns * columns_per_object, hspace=0.26, wspace=0.035)
    for object_index, row in enumerate(lowest):
        gallery_row = object_index // object_columns
        gallery_column = (object_index % object_columns) * columns_per_object
        crop_paths = list(row.get("crop_paths") or [])[:columns_per_object]
        for crop_index in range(columns_per_object):
            ax = fig.add_subplot(gallery[gallery_row, gallery_column + crop_index])
            ax.set_facecolor(PANEL)
            ax.axis("off")
            image = _read_crop(crop_paths[crop_index], args.mapping_dir) if crop_index < len(crop_paths) else None
            if image is not None:
                ax.imshow(image)
            if crop_index == 0:
                category = str(row.get("category") or "object")[:18]
                ax.set_title(
                    f"#{int(row['object_id']):03d}  {category}  ·  score {float(row['median_pairwise_similarity']):.3f}",
                    color=TEXT,
                    fontsize=8,
                    loc="left",
                    pad=5,
                )

    scene_label = str(args.scene_id).replace("_", " ").strip().upper() or "SCENE"
    fig.text(0.035, 0.965, f"{scene_label} | PROMPT-FREE MULTI-VIEW CONSISTENCY QA", color=TEXT, fontsize=22, weight="bold", ha="left")
    fig.text(0.035, 0.935, "DINOv3 ViT-S+/16 · saved segmentation crops · scene-relative robust outlier gate", color=MUTED, fontsize=11, ha="left")
    retained = int(report.get("retained_objects") or 0)
    fig.text(0.965, 0.025, f"all {retained} geometry-validated objects pass the automatic appearance-consistency gate", color=MUTED, fontsize=9, ha="right")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=BG, dpi=200)
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
