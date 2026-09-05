#!/usr/bin/env python3
"""Build a compact, readable QA report for a saved FARM scene state."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from farm_pipeline.final_acceptance import (  # noqa: E402
    classify_pair_records as classify_acceptance_pairs,
    merge_pair_evidence,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
try:
    from scripts.geometry.farm_geometry_axes import horizontal_plane_basis, normalize_up_vector
except ModuleNotFoundError:  # package import
    from scripts.geometry.farm_geometry_axes import horizontal_plane_basis, normalize_up_vector


BG = (7, 9, 14)
PANEL = (14, 18, 27)
CARD = (23, 29, 42)
TEXT = (245, 247, 250)
MUTED = (170, 180, 198)
ACCENT = (235, 190, 65)
IMAGE_ID_RE = re.compile(r"^img_(\d+)")
OBJECT_ID_RE = re.compile(r"^object_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=Path, required=True)
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument("--segmentation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--presentation-catalog", type=Path, default=None,
        help="Authoritative final presentation catalog; falls back to active state rows.",
    )
    parser.add_argument(
        "--dedup-audit", type=Path, default=None,
        help="Optional duplicate audit used for the residual-overlap review visual.",
    )
    parser.add_argument("--acceptance-report", type=Path, default=None)
    parser.add_argument("--acceptance-clusters", type=Path, default=None)
    parser.add_argument("--acceptance-labels", type=Path, default=None)
    parser.add_argument("--segmentation-samples", type=int, default=10)
    parser.add_argument("--catalog-size", type=int, default=10)
    parser.add_argument(
        "--scene-id", default=None,
        help="Optional display name; defaults to the scene_id stored in frames.json.",
    )
    return parser.parse_args()


def as_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def put_text(
    canvas: np.ndarray,
    value: object,
    xy: tuple[int, int],
    scale: float = 0.8,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 2,
) -> None:
    cv2.putText(
        canvas, str(value), xy, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA,
    )


def place(
    canvas: np.ndarray, image: np.ndarray, x: int, y: int, width: int, height: int
) -> None:
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (
        max(1, int(round(image.shape[1] * scale))),
        max(1, int(round(image.shape[0] * scale))),
    )
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    xx = x + (width - resized.shape[1]) // 2
    yy = y + (height - resized.shape[0]) // 2
    canvas[yy : yy + resized.shape[0], xx : xx + resized.shape[1]] = resized


def embedding_dim(rows: list, index: int) -> int:
    if index >= len(rows):
        return 0
    value = rows[index]
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    try:
        return len(value)
    except TypeError:
        return 0


def state_arrays(state: dict) -> dict[str, object]:
    means = as_numpy(state.get("means", np.zeros((0, 3)))).reshape(-1, 3)
    refined = state.get("object_box_centers_m")
    centers = means if refined is None else as_numpy(refined).reshape(-1, 3)
    if centers.shape != means.shape:
        raise ValueError("object_box_centers_m must align with means")
    return {
        "means": means,
        "centers": centers,
        "cov6": as_numpy(state.get("cov6", np.zeros((len(means), 6)))).reshape(-1, 6),
        "active": as_numpy(state.get("active", np.ones(len(means), dtype=bool))).astype(bool),
        "object_ids": as_numpy(state.get("object_id", np.arange(len(means)))).reshape(-1),
        "counts": as_numpy(state.get("count", np.zeros(len(means), dtype=int))).reshape(-1),
        "dimensions": as_numpy(
            state.get("object_box_dimensions_m", np.full((len(means), 3), np.nan))
        ).reshape(-1, 3),
        "wxyz": as_numpy(
            state.get(
                "object_box_wxyz",
                np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]]), (len(means), 1)),
            )
        ).reshape(-1, 4),
        "categories": list(state.get("object_category") or []),
        "supercategories": list(state.get("object_supercategory") or []),
        "captions": list(state.get("object_caption") or []),
        "semantic_tiers": list(state.get("object_semantic_tier") or []),
        "display_statuses": list(state.get("object_display_status") or []),
        "qwen": list(state.get("object_qwen3_vl_embedding") or []),
        "siglip": list(state.get("object_siglip2_embedding") or []),
    }


def covisibility_edges(state: dict, active: np.ndarray) -> list[tuple[int, int, float]]:
    result: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for left, row in (state.get("covisibility_weights") or {}).items():
        if not isinstance(row, dict):
            continue
        try:
            i = int(left)
        except (TypeError, ValueError):
            continue
        for right, value in row.items():
            try:
                j, weight = int(right), float(value)
            except (TypeError, ValueError):
                continue
            edge = tuple(sorted((i, j)))
            if (
                edge in seen or i == j or i >= len(active) or j >= len(active)
                or not active[i] or not active[j] or weight <= 0
            ):
                continue
            seen.add(edge)
            result.append((edge[0], edge[1], weight))
    return sorted(result, key=lambda item: item[2], reverse=True)


def load_presentation_ids(path: Path | None, object_ids: np.ndarray) -> set[int]:
    """Load the exported presentation catalog as the display source of truth."""
    if path is None:
        return set()
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("presentation catalog must be a JSON array")
    values: list[int] = []
    for row in payload:
        if not isinstance(row, dict) or row.get("id") is None:
            raise ValueError("every presentation catalog row must contain an object id")
        values.append(int(row["id"]))
    if len(values) != len(set(values)):
        raise ValueError("presentation catalog contains duplicate object ids")
    known = set(map(int, np.asarray(object_ids).reshape(-1)))
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"presentation catalog references unknown object ids: {unknown[:8]}")
    return set(values)


def strongest_incident_edges(
    edges: list[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    """Keep each node's strongest link, preserving coverage without graph hairballs."""
    best: dict[int, tuple[float, int]] = {}
    weights: dict[tuple[int, int], float] = {}
    for i, j, weight in edges:
        edge = (min(i, j), max(i, j))
        weights[edge] = max(float(weight), weights.get(edge, -math.inf))
        for source, target in ((i, j), (j, i)):
            previous = best.get(source)
            if previous is None or (float(weight), -target) > (previous[0], -previous[1]):
                best[source] = (float(weight), target)
    selected = {
        (min(source, target), max(source, target))
        for source, (_, target) in best.items()
    }
    return sorted(
        [(i, j, weights[(i, j)]) for i, j in selected],
        key=lambda row: (-row[2], row[0], row[1]),
    )


def spatial_label_indices(
    indices: np.ndarray,
    points: np.ndarray,
    counts: np.ndarray,
    object_ids: np.ndarray,
    limit: int = 16,
) -> list[int]:
    """Choose high-evidence labels while spreading them over the ground plane."""
    candidates = sorted(
        map(int, indices), key=lambda index: (-int(counts[index]), int(object_ids[index]))
    )[: max(limit * 4, limit)]
    if len(candidates) <= limit:
        return candidates
    sample = points[candidates]
    low = np.quantile(sample, 0.05, axis=0)
    high = np.quantile(sample, 0.95, axis=0)
    scale = np.maximum(high - low, 1.0e-6)
    normalized = np.clip((sample - low) / scale, 0.0, 1.0)
    evidence = np.log1p(np.asarray([counts[index] for index in candidates], dtype=float))
    evidence /= max(float(evidence.max()), 1.0)
    selected_positions = [0]
    while len(selected_positions) < min(limit, len(candidates)):
        remaining = [i for i in range(len(candidates)) if i not in selected_positions]
        scored = []
        for position in remaining:
            distance = min(
                float(np.linalg.norm(normalized[position] - normalized[chosen]))
                for chosen in selected_positions
            )
            scored.append((distance + 0.18 * evidence[position], -candidates[position], position))
        selected_positions.append(max(scored)[2])
    return [candidates[position] for position in selected_positions]


def _wxyz_to_matrix(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(quaternion).all() or norm <= 1.0e-8:
        raise ValueError("invalid OBB quaternion")
    w, x, y, z = quaternion / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def obb_ground_polygon(
    center: np.ndarray,
    dimensions: np.ndarray,
    wxyz: np.ndarray,
    plane_basis: np.ndarray,
) -> np.ndarray | None:
    dimensions = np.asarray(dimensions, dtype=np.float64).reshape(3)
    if not np.isfinite(center).all() or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        return None
    try:
        rotation = _wxyz_to_matrix(wxyz)
    except ValueError:
        return None
    signs = np.asarray([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float64)
    corners = np.asarray(center) + (signs * (0.5 * dimensions)) @ rotation.T
    projected = (corners @ plane_basis).astype(np.float32)
    hull = cv2.convexHull(projected).reshape(-1, 2)
    return hull if len(hull) >= 3 else None


def camera_positions(state: dict) -> np.ndarray:
    rows = []
    for value in state.get("image_positions") or []:
        point = as_numpy(value).reshape(-1)
        if len(point) >= 3 and np.isfinite(point[:3]).all():
            rows.append(point[:3])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 3)


def make_object_map(
    state: dict,
    arrays: dict[str, object],
    presentation_mask: np.ndarray,
    path: Path,
) -> dict[str, object]:
    centers = arrays["centers"]
    active = arrays["active"]
    counts = arrays["counts"]
    categories = arrays["categories"]
    semantic_tiers = arrays["semantic_tiers"]
    dimensions = arrays["dimensions"]
    orientations = arrays["wxyz"]
    object_ids = arrays["object_ids"]
    if presentation_mask.shape != active.shape:
        raise ValueError("presentation mask must align with active state rows")
    up = normalize_up_vector(state.get("object_geometry_up_vector", [0.0, 1.0, 0.0]))
    plane_basis = horizontal_plane_basis(up)
    centers_2d = centers @ plane_basis
    active_ids = np.flatnonzero(active)
    presentation_ids = np.flatnonzero(presentation_mask)
    hidden_ids = np.flatnonzero(active & ~presentation_mask)
    metric_edges = covisibility_edges(state, active)
    presentation_edges = covisibility_edges(state, presentation_mask)
    rendered_edges = strongest_incident_edges(presentation_edges)
    positions = camera_positions(state)
    positions_2d = positions @ plane_basis if len(positions) else np.zeros((0, 2))

    fig, ax = plt.subplots(figsize=(15, 9), facecolor="#090b10")
    ax.set_facecolor("#0e121b")
    for i, j, weight in rendered_edges:
        ax.plot(
            [centers_2d[i, 0], centers_2d[j, 0]],
            [centers_2d[i, 1], centers_2d[j, 1]],
            color="#526172", linewidth=0.38 + min(weight, 6.0) * 0.10,
            alpha=0.22, zorder=1,
        )
    if len(positions):
        ax.plot(
            positions_2d[:, 0], positions_2d[:, 1], color="#38bdf8", linewidth=1.0,
            alpha=0.72, label="camera trajectory", zorder=0,
        )
    if len(hidden_ids):
        ax.scatter(
            centers_2d[hidden_ids, 0], centers_2d[hidden_ids, 1],
            marker="x", color="#7b8794", s=20, linewidths=0.7, alpha=0.48,
            label=f"retained, hidden ({len(hidden_ids)})", zorder=2,
        )

    tier_styles = {
        "confirmed": ("#55d6be", "o"),
        "probable": ("#f4b942", "^"),
    }
    for tier, (color, marker) in tier_styles.items():
        indices = np.asarray([
            index for index in presentation_ids
            if index < len(semantic_tiers) and str(semantic_tiers[index]).lower() == tier
        ], dtype=int)
        if not len(indices):
            continue
        sizes = 17 + np.minimum(counts[indices].astype(float), 30.0) * 1.5
        ax.scatter(
            centers_2d[indices, 0], centers_2d[indices, 1], s=sizes,
            marker=marker, color=color, edgecolors="#f2f4f7", linewidths=0.35,
            alpha=0.90, label=f"{tier} ({len(indices)})", zorder=3,
        )
    styled = {
        index for index in presentation_ids
        if index < len(semantic_tiers) and str(semantic_tiers[index]).lower() in tier_styles
    }
    other_ids = np.asarray([index for index in presentation_ids if index not in styled], dtype=int)
    if len(other_ids):
        ax.scatter(
            centers_2d[other_ids, 0], centers_2d[other_ids, 1], marker="s",
            color="#a8b2c1", s=28, edgecolors="#f2f4f7", linewidths=0.35,
            alpha=0.85, label=f"other presentation ({len(other_ids)})", zorder=3,
        )

    label_ids = spatial_label_indices(
        presentation_ids, centers_2d, counts, object_ids, limit=16
    )
    for index in label_ids:
        tier = (
            str(semantic_tiers[index]).lower()
            if index < len(semantic_tiers) else "other"
        )
        color = tier_styles.get(tier, ("#a8b2c1", "s"))[0]
        polygon = obb_ground_polygon(
            centers[index], dimensions[index], orientations[index], plane_basis
        )
        if polygon is not None:
            ax.add_patch(Polygon(
                polygon, closed=True, facecolor=color, edgecolor=color,
                alpha=0.10, linewidth=0.8, zorder=2,
            ))
        category = (
            str(categories[index] or "unlabeled")
            if index < len(categories) else "unlabeled"
        )
        ax.annotate(
            f"{int(object_ids[index])}: {category}", centers_2d[index], xytext=(5, 5),
            textcoords="offset points", color="#f5f7fa", fontsize=6.8,
            bbox={"facecolor": "#0e121b", "edgecolor": "none", "alpha": 0.58, "pad": 1.2},
            zorder=4,
        )

    ax.set_title(
        f"Presentation object map | {len(presentation_ids)} shown + {len(hidden_ids)} retained-hidden\n"
        f"strongest co-visibility links: {len(rendered_edges)} shown / {len(presentation_edges)} total",
        color="white", fontsize=14,
    )
    ax.set_xlabel("ground-plane axis 1, m", color="#d0d5dd")
    ax.set_ylabel("ground-plane axis 2, m", color="#d0d5dd")
    ax.grid(color="#344054", alpha=0.3)
    ax.tick_params(colors="#98a2b3")
    ax.axis("equal")
    ax.legend(facecolor="#101828", labelcolor="white", fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor="#090b10", bbox_inches="tight")
    plt.close(fig)
    return {
        "metric_active_object_count": int(len(active_ids)),
        "presentation_object_count": int(len(presentation_ids)),
        "retained_hidden_object_count": int(len(hidden_ids)),
        "covisibility_edge_count": int(len(presentation_edges)),
        "covisibility_edge_count_metric": int(len(metric_edges)),
        "covisibility_edge_count_rendered": int(len(rendered_edges)),
        "map_label_count": int(len(label_ids)),
        "camera_position_count": int(len(positions)),
    }


def classify_residual_candidates(
    audit: dict,
    presentation_ids: set[int],
) -> dict[str, list[dict]]:
    """Separate duplicate-like review rows from explicitly distinct similarities."""
    rows = audit.get("residual_strong_candidates") or []
    visible: list[dict] = []
    hidden: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            first_id, second_id = int(row["first_id"]), int(row["second_id"])
        except (KeyError, TypeError, ValueError):
            continue
        target = visible if first_id in presentation_ids and second_id in presentation_ids else hidden
        target.append(row)
    duplicate_like = [
        row for row in visible
        if str(row.get("raw_relation") or "").lower().startswith("duplicate")
    ]
    similar_distinct = [row for row in visible if row not in duplicate_like]
    return {
        "visible_duplicate_like": duplicate_like,
        "visible_similar_distinct": similar_distinct,
        "hidden_endpoint": hidden,
    }


def classify_acceptance_candidates(
    state: dict,
    audit: dict,
    presentation_ids: set[int],
    acceptance: dict,
) -> dict[str, list[dict]]:
    """Reproduce the persisted release policy without raw-relation heuristics."""

    initial_ids = set(presentation_ids)
    for row in acceptance.get("holdouts") or []:
        if isinstance(row, dict) and row.get("id") is not None:
            initial_ids.add(int(row["id"]))
    merged, merge_errors = merge_pair_evidence(audit)
    classified, classify_errors = classify_acceptance_pairs(merged, state, initial_ids)
    if merge_errors or classify_errors:
        raise ValueError(
            "acceptance evidence cannot be visualized: "
            + ", ".join(sorted({*merge_errors, *classify_errors}))
        )
    return {
        "auto_holdout": [row for row in classified if row.get("classification") == "auto_holdout"],
        "release_blocking_ambiguous": [
            row for row in classified
            if row.get("classification") == "release_blocking_ambiguous"
        ],
        "similar_distinct": [
            row for row in classified if row.get("classification") == "similar_distinct"
        ],
        "hidden_context": [
            row for row in classified if row.get("classification") == "hidden_context"
        ],
    }


def make_overlap_review_dashboard(
    state: dict,
    arrays: dict[str, object],
    presentation_mask: np.ndarray,
    audit: dict,
    path: Path,
    acceptance: dict | None = None,
) -> dict[str, int]:
    object_ids = arrays["object_ids"]
    centers = arrays["centers"]
    categories = arrays["categories"]
    presentation_ids = set(map(int, object_ids[presentation_mask]))
    acceptance = acceptance or {}
    if acceptance.get("schema") == "farm.final-acceptance.v1":
        policy_groups = classify_acceptance_candidates(
            state, audit, presentation_ids, acceptance
        )
        duplicate_like = policy_groups["auto_holdout"]
        blockers = policy_groups["release_blocking_ambiguous"]
        similar_distinct = policy_groups["similar_distinct"]
        hidden = policy_groups["hidden_context"]
    else:
        groups = classify_residual_candidates(audit, presentation_ids)
        duplicate_like = groups["visible_duplicate_like"]
        blockers = []
        similar_distinct = groups["visible_similar_distinct"]
        hidden = groups["hidden_endpoint"]
    index_by_id = {int(object_id): index for index, object_id in enumerate(object_ids)}
    up = normalize_up_vector(state.get("object_geometry_up_vector", [0.0, 1.0, 0.0]))
    points = centers @ horizontal_plane_basis(up)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(24, 13.5), dpi=160, facecolor="#09111b")
    grid = fig.add_gridspec(1, 5, wspace=0.16)
    ax = fig.add_subplot(grid[:, :3])
    detail = fig.add_subplot(grid[:, 3:])
    ax.set_facecolor("#0f1520")
    indices = np.flatnonzero(presentation_mask)
    if len(indices):
        ax.scatter(
            points[indices, 0], points[indices, 1], s=20, color="#677386",
            alpha=0.46, edgecolors="none", label=f"presentation context ({len(indices)})",
        )

    def draw_pairs(rows: list[dict], color: str, linestyle: str, label: str) -> None:
        labelled = False
        for row in rows:
            first = index_by_id.get(int(row["first_id"]))
            second = index_by_id.get(int(row["second_id"]))
            if first is None or second is None:
                continue
            ax.plot(
                [points[first, 0], points[second, 0]],
                [points[first, 1], points[second, 1]],
                color=color, linewidth=1.8 if linestyle == "-" else 1.0,
                linestyle=linestyle, alpha=0.88 if linestyle == "-" else 0.42,
                label=label if not labelled else None, zorder=3,
            )
            labelled = True
            ax.scatter(
                [points[first, 0], points[second, 0]],
                [points[first, 1], points[second, 1]],
                s=45 if linestyle == "-" else 28, color=color,
                edgecolors="#f8fafc", linewidths=0.45, zorder=4,
            )

    draw_pairs(
        similar_distinct, "#64b5d9", "--",
        f"similar, classified distinct ({len(similar_distinct)})",
    )
    draw_pairs(
        duplicate_like, "#f4b942", "-",
        f"automatic presentation holdout ({len(duplicate_like)})",
    )
    draw_pairs(
        blockers, "#ff6b6b", "-",
        f"release-blocking ambiguity ({len(blockers)})",
    )
    labelled_ids: set[int] = set()
    for row in [*blockers, *duplicate_like]:
        for key in ("first_id", "second_id"):
            object_id = int(row[key])
            if object_id in labelled_ids or object_id not in index_by_id:
                continue
            labelled_ids.add(object_id)
            index = index_by_id[object_id]
            category = (
                str(categories[index] or "unlabeled")
                if index < len(categories) else "unlabeled"
            )
            ax.annotate(
                f"{object_id}: {category}", points[index], xytext=(5, 5),
                textcoords="offset points", fontsize=8, color="#fff3cf",
                bbox={"facecolor": "#111827", "edgecolor": "none", "alpha": 0.72, "pad": 1.4},
                zorder=5,
            )
    ax.set_title(
        "Spatial review locator · duplicate-like evidence is separated from distinct similarity",
        fontsize=15,
    )
    ax.set_xlabel("ground-plane axis 1, m")
    ax.set_ylabel("ground-plane axis 2, m")
    ax.grid(color="#344054", alpha=0.27)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=9, facecolor="#101828")

    detail.axis("off")
    detail.text(0.0, 0.98, "OVERLAP / DUPLICATE REVIEW", fontsize=19, fontweight="bold", va="top")
    detail.text(
        0.0, 0.91,
        f"Resolved groups: {len(audit.get('duplicate_groups') or [])}   ·   "
        f"suppressed: {int(audit.get('suppressed_objects') or 0)}\n"
        f"Automatically held duplicate edges: {len(duplicate_like)}\n"
        f"Release-blocking ambiguous edges: {len(blockers)}\n"
        f"Similar but classified distinct: {len(similar_distinct)}\n"
        f"Rows with a hidden endpoint: {len(hidden)}",
        fontsize=13, color="#dbe8f4", va="top", linespacing=1.45,
    )
    detail.text(
        0.0, 0.72,
        "Amber edges were hidden by the deterministic acceptance pass. Red edges block release.\n"
        "Blue dashed pairs remain useful similarity diagnostics, but are not called duplicates.",
        fontsize=11, color="#9fb0c3", va="top", linespacing=1.35,
    )
    y = 0.61
    for row in [*blockers, *duplicate_like][:10]:
        first_id, second_id = int(row["first_id"]), int(row["second_id"])
        relation = str(row.get("matched_rule") or "acceptance overlap").replace("_", " ")
        feature = float(row.get("feature_cosine") or 0.0)
        mask_iou = float(row.get("mask_iou") or 0.0)
        distance = float(row.get("center_distance_m") or 0.0)
        detail.text(
            0.0, y, f"#{first_id} ↔ #{second_id}   {relation}",
            fontsize=11, color="#f6cf73", fontweight="bold", va="top",
        )
        detail.text(
            0.02, y - 0.035,
            f"feature={feature:.3f}  mask IoU={mask_iou:.3f}  center={distance:.2f} m",
            fontsize=9.5, color="#c7d2df", va="top",
        )
        y -= 0.083
        if y < 0.06:
            break
    if not duplicate_like and not blockers:
        detail.text(
            0.0, 0.55, "No presentation-visible duplicate-like pairs remain.",
            fontsize=15, color="#55d6be", va="top",
        )
    fig.suptitle(
        "PRESENTATION OVERLAP AUDIT · DETERMINISTIC HOLDOUT + DISTINCT CONTEXT",
        fontsize=23, fontweight="bold", x=0.025, ha="left", y=0.985,
    )
    fig.subplots_adjust(left=0.055, right=0.975, top=0.91, bottom=0.07)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    after = acceptance.get("classification_after") or {}
    authoritative = acceptance.get("schema") == "farm.final-acceptance.v1"
    return {
        "residual_candidates_total": int(
            len(duplicate_like) + len(blockers) + len(similar_distinct) + len(hidden)
        ),
        "acceptance_auto_holdout_edges": int(len(duplicate_like)),
        "acceptance_blocking_edges": int(len(blockers)),
        "residual_duplicate_like_visible": (
            int(after.get("auto_holdout") or 0)
            + int(after.get("release_blocking_ambiguous") or 0)
            if authoritative else int(len(duplicate_like))
        ),
        "residual_similar_distinct_visible": (
            int(after.get("similar_distinct") or 0)
            if authoritative else int(len(similar_distinct))
        ),
        "residual_hidden_endpoint": (
            int(after.get("hidden_context") or 0)
            if authoritative else int(len(hidden))
        ),
    }


def _frame_rgb_path(frames_root: Path, frame: dict) -> Path | None:
    for key in ("rgb_path", "source_image"):
        reference = str(frame.get(key) or "").strip()
        if not reference:
            continue
        candidate = Path(reference).expanduser()
        if not candidate.is_absolute():
            candidate = frames_root / candidate
        if candidate.is_file():
            return candidate
    return None


def _mask_files_by_image(segmentation_dir: Path) -> dict[int, list[tuple[int, Path]]]:
    grouped: dict[int, list[tuple[int, Path]]] = {}
    for mask_path in sorted(segmentation_dir.glob("object_*/img_*.npz")):
        image_match = IMAGE_ID_RE.match(mask_path.stem)
        object_match = OBJECT_ID_RE.match(mask_path.parent.name)
        if image_match is None or object_match is None:
            continue
        image_id = int(image_match.group(1))
        grouped.setdefault(image_id, []).append((int(object_match.group(1)), mask_path))
    return grouped


def _unpack_mask_canvas(mask_path: Path, image_shape: tuple[int, int]) -> np.ndarray | None:
    with np.load(mask_path, allow_pickle=False) as data:
        kind = "raw" if "raw_bits" in data.files else "inlier"
        required = {f"{kind}_bits", f"{kind}_shape", f"{kind}_bbox_xyxy"}
        if not required.issubset(data.files):
            return None
        shape = np.asarray(data[f"{kind}_shape"], dtype=np.int32).reshape(2)
        bbox = np.asarray(data[f"{kind}_bbox_xyxy"], dtype=np.int32).reshape(4)
        height, width = int(shape[0]), int(shape[1])
        if height <= 0 or width <= 0:
            return None
        flat = np.unpackbits(
            np.asarray(data[f"{kind}_bits"], dtype=np.uint8), bitorder="little"
        )
    if flat.size < height * width:
        return None
    crop = flat[: height * width].reshape(height, width).astype(bool, copy=False)
    canvas_height, canvas_width = image_shape
    x0, y0, x1, y1 = [int(value) for value in bbox]
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1, dst_y1 = min(canvas_width, x1), min(canvas_height, y1)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return None
    src_x0, src_y0 = dst_x0 - x0, dst_y0 - y0
    width = min(dst_x1 - dst_x0, crop.shape[1] - src_x0)
    height = min(dst_y1 - dst_y0, crop.shape[0] - src_y0)
    if width <= 0 or height <= 0:
        return None
    canvas = np.zeros((canvas_height, canvas_width), dtype=bool)
    canvas[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = crop[
        src_y0 : src_y0 + height, src_x0 : src_x0 + width
    ]
    return canvas


def _object_color(object_id: int) -> np.ndarray:
    # Stable, high-contrast BGR palette without depending on catalog ordering.
    hue = int((object_id * 47 + 13) % 180)
    hsv = np.asarray([[[hue, 205, 250]]], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)


def _overlay_saved_masks(image: np.ndarray, masks: list[tuple[int, Path]]) -> tuple[np.ndarray, int]:
    result = image.copy()
    rendered = 0
    for object_id, mask_path in masks:
        mask = _unpack_mask_canvas(mask_path, image.shape[:2])
        if mask is None or not np.any(mask):
            continue
        color = _object_color(object_id)
        result[mask] = np.clip(
            result[mask].astype(np.float32) * 0.55 + color * 0.45, 0, 255
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, tuple(int(v) for v in color), 2, cv2.LINE_AA)
        rendered += 1
    return result, rendered


def _sheet_from_cells(cells: list[np.ndarray], path: Path) -> np.ndarray:
    if not cells:
        raise RuntimeError("Could not load any RGB frames or segmentation overlays")
    columns = 5
    rows = int(math.ceil(len(cells) / columns))
    blank = np.full_like(cells[0], PANEL)
    while len(cells) < rows * columns:
        cells.append(blank.copy())
    sheet = np.concatenate(
        [np.concatenate(cells[row * columns : (row + 1) * columns], axis=1) for row in range(rows)],
        axis=0,
    )
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return sheet


def make_segmentation_sheet(
    segmentation_dir: Path,
    frames_root: Path,
    frames: list[dict],
    sample_count: int,
    path: Path,
) -> tuple[np.ndarray, int, str]:
    grouped_masks = _mask_files_by_image(segmentation_dir)
    available = [
        image_id for image_id in sorted(grouped_masks)
        if image_id < len(frames) and _frame_rgb_path(frames_root, frames[image_id]) is not None
    ]
    cells: list[np.ndarray] = []
    if available:
        count = min(max(1, sample_count), len(available))
        selected = np.unique(np.linspace(0, len(available) - 1, count).round().astype(int))
        for selected_index in selected:
            image_id = available[int(selected_index)]
            frame = frames[image_id]
            rgb_path = _frame_rgb_path(frames_root, frame)
            image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR) if rgb_path else None
            if image is None:
                continue
            image, rendered = _overlay_saved_masks(image, grouped_masks[image_id])
            image = cv2.resize(image, (680, 520), interpolation=cv2.INTER_AREA)
            bar = np.full((50, 680, 3), PANEL, dtype=np.uint8)
            put_text(
                bar,
                f"{frame.get('frame_id', image_id)} | {frame.get('camera', '')} | "
                f"saved object masks: {rendered}",
                (14, 33), 0.52, TEXT, 1,
            )
            cells.append(np.concatenate([bar, image], axis=0))
        return _sheet_from_cells(cells, path), len(available), "saved_object_masks"

    legacy_files = sorted(segmentation_dir.glob("frame_*.jpg"))
    if legacy_files:
        count = min(max(1, sample_count), len(legacy_files))
        selected = np.unique(np.linspace(0, len(legacy_files) - 1, count).round().astype(int))
        for selected_index in selected:
            image_path = legacy_files[int(selected_index)]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.resize(image, (680, 520), interpolation=cv2.INTER_AREA)
            bar = np.full((50, 680, 3), PANEL, dtype=np.uint8)
            put_text(bar, f"{image_path.stem} | pre-rendered segmentation overlay", (14, 33), 0.52, TEXT, 1)
            cells.append(np.concatenate([bar, image], axis=0))
        return _sheet_from_cells(cells, path), len(legacy_files), "pre_rendered_overlays"

    # Honest fallback: show source RGB, clearly stating that no mask overlay exists.
    available_rgb = [
        index for index, frame in enumerate(frames)
        if _frame_rgb_path(frames_root, frame) is not None
    ]
    count = min(max(1, sample_count), len(available_rgb))
    selected = (
        np.unique(np.linspace(0, len(available_rgb) - 1, count).round().astype(int))
        if available_rgb else np.asarray([], dtype=int)
    )
    for selected_index in selected:
        image_id = available_rgb[int(selected_index)]
        frame = frames[image_id]
        rgb_path = _frame_rgb_path(frames_root, frame)
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR) if rgb_path else None
        if image is None:
            continue
        image = cv2.resize(image, (680, 520), interpolation=cv2.INTER_AREA)
        bar = np.full((50, 680, 3), PANEL, dtype=np.uint8)
        put_text(
            bar,
            f"{frame.get('frame_id', image_id)} | {frame.get('camera', '')} | SOURCE RGB - NO SAVED MASK",
            (14, 33), 0.52, TEXT, 1,
        )
        cells.append(np.concatenate([bar, image], axis=0))
    return _sheet_from_cells(cells, path), 0, "source_rgb_no_saved_masks"


def build_report(
    state: dict,
    arrays: dict[str, object],
    presentation_mask: np.ndarray,
    frame_count: int,
    overlay_count: int,
    map_stats: dict[str, object],
    catalog_size: int,
) -> dict[str, object]:
    means = arrays["centers"]
    active = arrays["active"]
    counts = arrays["counts"]
    categories = arrays["categories"]
    supercategories = arrays["supercategories"]
    captions = arrays["captions"]
    qwen = arrays["qwen"]
    siglip = arrays["siglip"]
    object_ids = arrays["object_ids"]
    active_ids = np.flatnonzero(active)
    presentation_ids = np.flatnonzero(presentation_mask)
    ranked = presentation_ids[np.argsort(counts[presentation_ids])[::-1]]
    top_objects = []
    for i in ranked[:catalog_size]:
        top_objects.append(
            {
                "id": int(object_ids[i]),
                "observations": int(counts[i]),
                "category": str(categories[i] or "unlabeled") if i < len(categories) else "unlabeled",
                "supercategory": str(supercategories[i] or "unknown") if i < len(supercategories) else "unknown",
                "caption": str(captions[i] or "") if i < len(captions) else "",
                "position_world_m": means[i].astype(float).tolist(),
            }
        )
    return {
        "status": "ok",
        "input_frames": frame_count,
        "segmentation_overlays": overlay_count,
        "objects_total": int(len(means)),
        "objects_active": int(len(active_ids)),
        "objects_presentation": int(len(presentation_ids)),
        "objects_retained_hidden": int(np.count_nonzero(active & ~presentation_mask)),
        "objects_inactive": int(len(means) - len(active_ids)),
        "captioned_active": int(sum(bool(str(captions[i]).strip()) for i in active_ids)),
        "qwen_vl_embedded_active": int(sum(embedding_dim(qwen, i) > 0 for i in active_ids)),
        "siglip2_embedded_active": int(sum(embedding_dim(siglip, i) > 0 for i in active_ids)),
        "region_count": int(len(state.get("region_labels") or [])),
        "region_labels": dict(Counter(map(str, state.get("region_labels") or []))),
        "categories": dict(
            Counter(
                str(categories[i] or "unlabeled")
                for i in presentation_ids if i < len(categories)
            ).most_common()
        ),
        "top_objects": top_objects,
        **map_stats,
    }


def make_dashboard(
    report: dict[str, object],
    map_image: np.ndarray,
    sheet: np.ndarray,
    path: Path,
) -> None:
    canvas = np.full((2160, 3840, 3), BG, dtype=np.uint8)
    scene_label = str(report.get("scene_id") or "scene").replace("_", " ").strip()
    put_text(
        canvas, f"{scene_label.upper()[:52]} | FULL KEYFRAME MAPPING",
        (76, 92), 1.62, TEXT, 4,
    )
    put_text(
        canvas,
        "YOLOE segmentation + DINOv3 ViT-S+/16 association + Qwen3-VL captions + metric 3DGS depth",
        (80, 142), 0.68, MUTED, 2,
    )
    kpis = [
        ("INPUT VIEWS", report["input_frames"], f"{report.get('camera_count', 0)} cameras"),
        ("METRIC OBJECTS", report["objects_active"], f"{report['objects_total']} allocated"),
        (
            "PRESENTATION", report["objects_presentation"],
            f"{report['objects_retained_hidden']} retained-hidden",
        ),
        ("CAPTIONS", f"{report['captioned_active']}/{report['objects_active']}", "Qwen3-VL-8B"),
        (
            "GRAPH LINKS", report["covisibility_edge_count_rendered"],
            f"of {report['covisibility_edge_count']} presentation edges",
        ),
    ]
    for index, (name, value, note) in enumerate(kpis):
        x = 78 + index * 748
        cv2.rectangle(canvas, (x, 178), (x + 700, 330), PANEL, -1)
        cv2.rectangle(canvas, (x, 178), (x + 7, 330), ACCENT, -1)
        put_text(canvas, name, (x + 28, 220), 0.62, MUTED, 2)
        put_text(canvas, value, (x + 28, 284), 1.20, TEXT, 3)
        put_text(canvas, note, (x + 250, 283), 0.48, MUTED, 1)

    cv2.rectangle(canvas, (70, 370), (2120, 1280), PANEL, -1)
    place(canvas, map_image, 86, 386, 2018, 878)
    cv2.rectangle(canvas, (2160, 370), (3770, 1280), PANEL, -1)
    put_text(canvas, "MOST OBSERVED OBJECTS", (2200, 420), 0.84, TEXT, 2)
    for rank, item in enumerate(report["top_objects"][:10]):
        row = rank % 5
        col = rank // 5
        x = 2190 + col * 785
        y = 448 + row * 160
        cv2.rectangle(canvas, (x, y), (x + 745, y + 145), CARD, -1)
        put_text(
            canvas,
            f"#{item['id']:03d}  {str(item['category']).upper()}  obs={item['observations']}",
            (x + 16, y + 34), 0.53, TEXT, 2,
        )
        for line_index, line in enumerate(textwrap.wrap(str(item["caption"]), width=45)[:3]):
            put_text(canvas, line, (x + 16, y + 66 + 25 * line_index), 0.45, MUTED, 1)

    cv2.rectangle(canvas, (70, 1310), (3770, 2090), PANEL, -1)
    place(canvas, sheet, 88, 1328, 3664, 744)
    segmentation_source = str(report.get("segmentation_source") or "")
    if segmentation_source == "saved_object_masks":
        footer = "Representative saved object-mask overlays reconstructed on source RGB frames."
    elif segmentation_source == "pre_rendered_overlays":
        footer = "Representative pre-rendered segmentation overlays sampled across the sequence."
    else:
        footer = "Representative source RGB frames; no saved segmentation masks were available."
    put_text(canvas, footer, (88, 2135), 0.58, MUTED, 1)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 96])


def _write_4k(path: Path, canvas: np.ndarray) -> None:
    if canvas.shape[:2] != (2160, 3840):
        raise ValueError("canonical dashboard canvas must be exactly 3840x2160")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 96]):
        raise RuntimeError(f"could not write dashboard: {path}")


def make_acceptance_dashboard(
    acceptance: dict,
    clusters_payload: dict,
    labels_payload: dict,
    scene_id: str,
    path: Path,
) -> None:
    """Render the release decision and its evidence as an exact 4K dashboard."""

    canvas = np.full((2160, 3840, 3), BG, dtype=np.uint8)
    status = str(acceptance.get("status") or "MISSING").upper()
    status_color = {"PASS": (190, 214, 85), "WARN": (66, 185, 244), "FAIL": (107, 107, 255)}.get(
        status, MUTED
    )
    counts = acceptance.get("counts") or {}
    before = acceptance.get("classification_before") or {}
    after = acceptance.get("classification_after") or {}
    runtime = acceptance.get("runtime") or {}
    labels = labels_payload.get("objects") or []
    clusters = clusters_payload.get("clusters") or []
    put_text(canvas, f"{scene_id.upper()[:50]} | FINAL ACCEPTANCE", (74, 96), 1.55, TEXT, 4)
    put_text(
        canvas,
        "One deterministic presentation-only pass · metric geometry, activation and masks are immutable",
        (78, 150), 0.67, MUTED, 2,
    )
    kpis = [
        ("RELEASE", status, "PASS/WARN publish; FAIL blocks"),
        ("PRESENTATION", counts.get("presentation_after", "n/a"), f"from {counts.get('presentation_before', 'n/a')}"),
        ("HELD OBJECTS", counts.get("held_objects", "n/a"), "duplicate presentation rows"),
        ("OPEN OVERLAP", int(counts.get("remaining_auto_clusters") or 0) + int(counts.get("remaining_blocking_clusters") or 0), "automatic + ambiguous"),
        ("LABEL HARD ERRORS", counts.get("label_hard_errors", "n/a"), f"{counts.get('high_uncertainty_labels', 'n/a')} high uncertainty"),
    ]
    for index, (name, value, note) in enumerate(kpis):
        x = 76 + index * 748
        cv2.rectangle(canvas, (x, 190), (x + 700, 350), PANEL, -1)
        cv2.rectangle(canvas, (x, 190), (x + 7, 350), status_color if index == 0 else ACCENT, -1)
        put_text(canvas, name, (x + 28, 236), 0.61, MUTED, 2)
        put_text(canvas, value, (x + 28, 302), 1.16, status_color if index == 0 else TEXT, 3)
        put_text(canvas, note, (x + 248, 301), 0.43, MUTED, 1)

    cv2.rectangle(canvas, (70, 395), (1880, 1290), PANEL, -1)
    put_text(canvas, "PAIR CLASSIFICATION · BEFORE / AFTER", (112, 452), 0.86, TEXT, 2)
    names = ("auto_holdout", "release_blocking_ambiguous", "similar_distinct", "hidden_context")
    colors = ((66, 185, 244), (107, 107, 255), (217, 181, 100), (130, 139, 151))
    maximum = max(1, *(int(before.get(name) or 0) for name in names), *(int(after.get(name) or 0) for name in names))
    for row_index, (name, color) in enumerate(zip(names, colors, strict=True)):
        y = 525 + row_index * 175
        put_text(canvas, name.replace("_", " ").upper(), (112, y), 0.57, TEXT, 2)
        for offset, (label, values) in enumerate((("before", before), ("after", after))):
            value = int(values.get(name) or 0)
            yy = y + 35 + offset * 52
            put_text(canvas, label, (112, yy + 24), 0.43, MUTED, 1)
            cv2.rectangle(canvas, (300, yy), (1700, yy + 32), CARD, -1)
            width = round(1400 * value / maximum)
            if width:
                cv2.rectangle(canvas, (300, yy), (300 + width, yy + 32), color, -1)
            put_text(canvas, value, (1720, yy + 26), 0.49, TEXT, 1)

    cv2.rectangle(canvas, (1920, 395), (3770, 1290), PANEL, -1)
    put_text(canvas, "AUTOMATIC HOLDOUTS", (1962, 452), 0.86, TEXT, 2)
    holdouts = acceptance.get("holdouts") or []
    y = 512
    for row in holdouts[:12]:
        if not isinstance(row, dict):
            continue
        put_text(
            canvas,
            f"#{row.get('id')}  → canonical #{row.get('canonical_id')}",
            (1970, y), 0.56, (235, 211, 142), 2,
        )
        reason = str(row.get("reason") or "").replace("acceptance:", "")
        put_text(canvas, textwrap.shorten(reason, width=68, placeholder="…"), (2450, y), 0.43, MUTED, 1)
        y += 58
    if not holdouts:
        put_text(canvas, "No presentation rows required automatic holdout.", (1970, 535), 0.67, (190, 214, 85), 2)
    if len(holdouts) > 12:
        put_text(canvas, f"+ {len(holdouts) - 12} more in result.json", (1970, 1225), 0.48, MUTED, 1)

    cv2.rectangle(canvas, (70, 1335), (3770, 2075), PANEL, -1)
    put_text(canvas, "PROVENANCE, LABEL QUALITY AND BUDGET", (112, 1395), 0.86, TEXT, 2)
    lines = [
        f"Merged pair evidence: {counts.get('merged_candidate_pairs', 'n/a')} of {counts.get('raw_candidate_records', 'n/a')} records; overlap clusters persisted: {len(clusters)}",
        f"Label audit: {len(labels)} presentation rows; uncertain {counts.get('uncertain_labels', 'n/a')}; high uncertainty {counts.get('high_uncertainty_labels', 'n/a')}",
        f"Runtime: {float(runtime.get('elapsed_seconds') or 0.0):.2f} s / {float(runtime.get('budget_seconds') or 0.0):.2f} s budget; GPU calls: {runtime.get('model_calls', 0)}",
        "Release invariant: remaining automatic overlap = 0, blocking ambiguity = 0, label hard errors = 0.",
    ]
    for index, line in enumerate(lines):
        put_text(canvas, line, (120, 1480 + index * 78), 0.60, TEXT if index < 3 else MUTED, 2 if index < 3 else 1)
    warning_y = 1835
    warnings = acceptance.get("warnings") or []
    if warnings:
        put_text(canvas, "WARNINGS", (120, warning_y), 0.61, (66, 185, 244), 2)
        for index, row in enumerate(warnings[:3]):
            put_text(canvas, f"• {row.get('code', 'warning')} · count={row.get('count', 'n/a')}", (390, warning_y + index * 55), 0.50, MUTED, 1)
    else:
        put_text(canvas, "No acceptance warnings.", (120, warning_y), 0.58, (190, 214, 85), 2)
    _write_4k(path, canvas)


def _representative_crop(segmentation_dir: Path, object_id: int) -> np.ndarray | None:
    for mask_path in sorted((segmentation_dir / f"object_{object_id:06d}").glob("img_*.npz")):
        try:
            with np.load(mask_path, allow_pickle=False) as archive:
                if "crop_jpeg_bytes" not in archive.files:
                    continue
                encoded = np.asarray(archive["crop_jpeg_bytes"], dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        except (OSError, ValueError):
            continue
        if image is not None and image.size:
            return image
    return None


def make_label_uncertainty_dashboard(
    labels_payload: dict,
    segmentation_dir: Path,
    scene_id: str,
    path: Path,
) -> None:
    """Render a deterministic uncertainty-first crop sample at exact 4K."""

    canvas = np.full((2160, 3840, 3), BG, dtype=np.uint8)
    rows = {
        int(row["id"]): row
        for row in labels_payload.get("objects") or []
        if isinstance(row, dict) and row.get("id") is not None
    }
    selected_ids = [int(value) for value in labels_payload.get("sample_ids") or []]
    selected = [rows[value] for value in selected_ids if value in rows][:16]
    if not selected:
        selected = sorted(
            rows.values(),
            key=lambda row: (-float(row.get("uncertainty_score") or 0.0), int(row["id"])),
        )[:16]
    put_text(canvas, f"{scene_id.upper()[:50]} | LABEL QUALITY SAMPLE", (74, 96), 1.52, TEXT, 4)
    put_text(
        canvas,
        f"Deterministic uncertainty-first stratified sample · {len(selected)} shown / {len(rows)} audited presentation labels",
        (78, 150), 0.67, MUTED, 2,
    )
    columns, card_width, card_height = 4, 900, 430
    for index in range(16):
        column, row_index = index % columns, index // columns
        x, y = 70 + column * 930, 225 + row_index * 465
        cv2.rectangle(canvas, (x, y), (x + card_width, y + card_height), PANEL, -1)
        if index >= len(selected):
            put_text(canvas, "sample slot unused", (x + 28, y + 60), 0.52, MUTED, 1)
            continue
        row = selected[index]
        object_id = int(row["id"])
        score = float(row.get("uncertainty_score") or 0.0)
        crop = _representative_crop(segmentation_dir, object_id)
        if crop is not None:
            place(canvas, crop, x + 18, y + 58, 380, 335)
        else:
            cv2.rectangle(canvas, (x + 18, y + 58), (x + 398, y + 393), CARD, -1)
            put_text(canvas, "NO DECODABLE CROP", (x + 62, y + 235), 0.48, MUTED, 1)
        category = str(row.get("category") or "unresolved")
        tier = str(row.get("semantic_tier") or "unknown")
        put_text(canvas, f"#{object_id} · {category.upper()[:28]}", (x + 20, y + 40), 0.54, TEXT, 2)
        put_text(canvas, f"tier  {tier}", (x + 430, y + 92), 0.49, MUTED, 1)
        put_text(canvas, f"uncertainty  {score:.2f}", (x + 430, y + 132), 0.54, TEXT, 2)
        cv2.rectangle(canvas, (x + 430, y + 152), (x + 850, y + 176), CARD, -1)
        color = (107, 107, 255) if score >= 0.75 else (66, 185, 244) if score >= 0.5 else (190, 214, 85)
        cv2.rectangle(canvas, (x + 430, y + 152), (x + 430 + round(420 * min(1.0, score)), y + 176), color, -1)
        put_text(canvas, f"views  {row.get('unique_view_count', 'n/a')}  · groups  {row.get('independent_group_count', 'n/a')}", (x + 430, y + 222), 0.46, MUTED, 1)
        put_text(canvas, f"decodable crops  {row.get('decodable_crop_count', 'n/a')}", (x + 430, y + 260), 0.46, MUTED, 1)
        reasons = ", ".join(str(value).replace("_", " ") for value in row.get("uncertainty_reasons") or []) or "no uncertainty flags"
        for line_index, line in enumerate(textwrap.wrap(reasons, width=42)[:4]):
            put_text(canvas, line, (x + 430, y + 307 + line_index * 28), 0.39, MUTED, 1)
    _write_4k(path, canvas)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.pt.expanduser(), map_location="cpu", weights_only=False)
    state = payload.get("state", payload) if isinstance(payload, dict) else {}
    frames_payload = json.loads(args.frames_json.expanduser().read_text(encoding="utf-8"))
    frames = list(frames_payload.get("frames") or [])
    scene_id = str(
        args.scene_id or frames_payload.get("scene_id") or state.get("scene_id")
        or args.pt.expanduser().stem
    )
    arrays = state_arrays(state)
    authoritative_ids = load_presentation_ids(args.presentation_catalog, arrays["object_ids"])
    presentation_mask = (
        np.isin(arrays["object_ids"], list(authoritative_ids))
        if args.presentation_catalog is not None else arrays["active"].copy()
    )
    if not np.any(presentation_mask):
        raise ValueError("presentation catalog contains no objects")
    dedup_audit = (
        json.loads(args.dedup_audit.expanduser().read_text(encoding="utf-8"))
        if args.dedup_audit is not None else {}
    )
    if not isinstance(dedup_audit, dict):
        raise TypeError("dedup audit must be a JSON object")
    acceptance = (
        json.loads(args.acceptance_report.expanduser().read_text(encoding="utf-8"))
        if args.acceptance_report is not None else {}
    )
    clusters_payload = (
        json.loads(args.acceptance_clusters.expanduser().read_text(encoding="utf-8"))
        if args.acceptance_clusters is not None else {}
    )
    labels_payload = (
        json.loads(args.acceptance_labels.expanduser().read_text(encoding="utf-8"))
        if args.acceptance_labels is not None else {}
    )
    if not all(isinstance(value, dict) for value in (acceptance, clusters_payload, labels_payload)):
        raise TypeError("acceptance artifacts must be JSON objects")

    map_path = output_dir / "01_object_map.png"
    sheet_path = output_dir / "02_segmentation_contact_sheet.jpg"
    dashboard_path = output_dir / "03_scene_state_dashboard_4k.jpg"
    overlap_path = output_dir / "04_duplicate_overlap_review_4k.jpg"
    acceptance_path = output_dir / "05_final_acceptance_dashboard_4k.jpg"
    label_path = output_dir / "06_label_uncertainty_sample_4k.jpg"
    map_stats = make_object_map(state, arrays, presentation_mask, map_path)
    overlap_stats = make_overlap_review_dashboard(
        state, arrays, presentation_mask, dedup_audit, overlap_path, acceptance
    )
    make_acceptance_dashboard(
        acceptance, clusters_payload, labels_payload, scene_id, acceptance_path
    )
    make_label_uncertainty_dashboard(
        labels_payload, args.segmentation_dir.expanduser(), scene_id, label_path
    )
    sheet, overlay_count, segmentation_source = make_segmentation_sheet(
        args.segmentation_dir.expanduser(), args.frames_json.expanduser().resolve().parent,
        frames, args.segmentation_samples, sheet_path
    )
    report = build_report(
        state, arrays, presentation_mask, len(frames), overlay_count,
        {**map_stats, **overlap_stats}, args.catalog_size,
    )
    report.update({
        "scene_id": scene_id,
        "camera_count": len({str(frame.get("camera") or "") for frame in frames}),
        "segmentation_source": segmentation_source,
        "acceptance_status": acceptance.get("status"),
        "acceptance_counts": acceptance.get("counts") or {},
    })
    map_image = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
    make_dashboard(report, map_image, sheet, dashboard_path)
    report.update(
        {
            "scene_state": str(args.pt.expanduser().resolve()),
            "presentation_catalog": (
                str(args.presentation_catalog.expanduser().resolve())
                if args.presentation_catalog is not None else None
            ),
            "outputs": [
                map_path.name, sheet_path.name, dashboard_path.name, overlap_path.name,
                acceptance_path.name, label_path.name,
            ],
        }
    )
    (output_dir / "scene_state_visual_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "categories"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
