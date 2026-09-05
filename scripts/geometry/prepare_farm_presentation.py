#!/usr/bin/env python3
"""Prepare a compact, presentation-grade FARM scene bundle.

The command is intentionally post-processing only: it never re-runs mapping.
It exports a coloured context cloud from an aligned 3DGS PLY and creates a
filtered copy of ``scene_state.pt`` whose active objects satisfy objective
multi-view / geometry gates.  Optionally, an automatically reviewed catalog
can replace raw captions and categories.

The original scene state and PLY are never modified.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData


SH_C0 = 0.28209479177387814


def validate_meters_per_scene_unit(value: float) -> float:
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("--meters-per-scene-unit must be finite and strictly positive")
    return scale


def _file_signature(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cloud_export_contract(
    source_ply: Path,
    *,
    max_points: int,
    min_opacity: float,
    seed: int,
    meters_per_scene_unit: float,
) -> dict[str, object]:
    return {
        "schema": "farm.presentation-cloud-contract.v1",
        "source": _file_signature(source_ply),
        "max_points": max(0, int(max_points)),
        "min_opacity": float(np.clip(min_opacity, 0.0, 1.0)),
        "seed": int(seed),
        "meters_per_scene_unit": validate_meters_per_scene_unit(meters_per_scene_unit),
    }


def validate_reusable_cloud(
    manifest_path: Path,
    cloud_path: Path,
    requested_contract: dict[str, object],
) -> dict[str, object]:
    """Fail closed on stale/scaled cloud reuse, while accepting legacy 1:1 bundles."""

    if not cloud_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("--reuse-cloud requires data/cloud.npz and manifest.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = payload.get("cloud") if isinstance(payload, dict) else None
    if not isinstance(summary, dict):
        raise ValueError("Existing manifest has no cloud contract")
    requested_fingerprint = _fingerprint(requested_contract)
    existing_fingerprint = summary.get("fingerprint_sha256")
    if existing_fingerprint is not None:
        if str(existing_fingerprint) != requested_fingerprint:
            raise ValueError("Existing cloud fingerprint does not match the requested export contract")
        return {**summary, "reused": True, "reuse_validation": "fingerprint"}

    # Backward compatibility for presentation bundles written before export
    # fingerprints existed. Never treat a missing legacy scale as non-unit.
    source = str(summary.get("source_ply") or "")
    requested_source = str(requested_contract["source"]["path"])
    legacy_scale = validate_meters_per_scene_unit(summary.get("meters_per_scene_unit", 1.0))
    requested_scale = float(requested_contract["meters_per_scene_unit"])
    if Path(source).expanduser().resolve() != Path(requested_source).resolve():
        raise ValueError("Existing cloud was not exported from the requested scene PLY")
    if not math.isclose(legacy_scale, requested_scale, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Existing cloud metric scale does not match --meters-per-scene-unit")
    return {**summary, "reused": True, "reuse_validation": "legacy-source-and-scale"}


def _load_payload(path: Path) -> tuple[dict, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("state"), dict):
        return payload, payload["state"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported scene-state payload: {type(payload)!r}")
    return {"format_version": 1, "state": payload, "meta": {}}, payload


def _unique_observation_count(value: object) -> int:
    if isinstance(value, (list, tuple)):
        ids: set[int] = set()
        for item in value:
            try:
                ids.add(int(item))
            except (TypeError, ValueError):
                continue
        return len(ids)
    try:
        return 1 if int(value) >= 0 else 0
    except (TypeError, ValueError):
        return 0


def _tier(count: int) -> str:
    if count >= 10:
        return "stable"
    if count >= 5:
        return "robust"
    if count >= 3:
        return "multi-view"
    if count == 2:
        return "tentative"
    return "single-view"


def _load_reviewed_catalog(path: Path | None) -> dict[int, dict]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Reviewed catalog must be a JSON list: {path}")
    out: dict[int, dict] = {}
    for row in raw:
        if isinstance(row, dict) and "id" in row:
            out[int(row["id"])] = row
    return out


def _camera_positions(state: dict) -> np.ndarray:
    values = state.get("image_positions") or []
    rows: list[np.ndarray] = []
    for value in values:
        try:
            arr = np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value, dtype=np.float32)
        except Exception:
            continue
        if arr.shape == (3,) and np.isfinite(arr).all():
            rows.append(arr)
    return np.stack(rows) if rows else np.zeros((0, 3), dtype=np.float32)


def _ensure_list(state: dict, key: str, size: int, default: object) -> list:
    value = state.get(key)
    if not isinstance(value, list):
        value = list(value) if isinstance(value, tuple) else []
    if len(value) < size:
        value.extend(default() if callable(default) else default for _ in range(size - len(value)))
    state[key] = value
    return value


def build_demo_state(
    source: Path,
    destination: Path,
    *,
    reviewed_catalog: Path | None,
    min_observations: int,
    resolved_only: bool,
    max_camera_distance_m: float,
    metadata_source_scene_state: str | None = None,
    metadata_reviewed_catalog: str | None = None,
) -> dict:
    payload, state = _load_payload(source)
    object_ids_t = state.get("object_id")
    means_t = state.get("means")
    active_t = state.get("active")
    if not all(isinstance(v, torch.Tensor) for v in (object_ids_t, means_t, active_t)):
        raise ValueError("scene_state is missing object_id/means/active tensors")

    object_ids = object_ids_t.detach().cpu().numpy().astype(np.int64, copy=False)
    means = means_t.detach().cpu().numpy().astype(np.float32, copy=False)
    source_active = active_t.detach().cpu().numpy().astype(bool, copy=False)
    count = int(object_ids.shape[0])
    image_ids = state.get("object_image_ids") or []
    observations = np.asarray(
        [_unique_observation_count(image_ids[i] if i < len(image_ids) else []) for i in range(count)],
        dtype=np.int32,
    )
    tiers = [_tier(int(value)) for value in observations]

    reviewed = _load_reviewed_catalog(reviewed_catalog)
    captions = _ensure_list(state, "object_caption", count, "")
    categories = _ensure_list(state, "object_category", count, "")
    supercategories = _ensure_list(state, "object_supercategory", count, "")
    decisions = _ensure_list(state, "object_caption_decision", count, "")
    attributes = _ensure_list(state, "object_key_attributes", count, list)
    semantic_tiers = _ensure_list(state, "object_semantic_tier", count, "")
    semantic_status = ["raw_mapping"] * count
    review_confidence = np.zeros((count,), dtype=np.float32)

    resolved_ids: set[int] = set()
    for index, object_id in enumerate(object_ids.tolist()):
        row = reviewed.get(int(object_id))
        if row is None:
            continue
        status = str(row.get("semantic_status") or "").strip()
        decision = str(row.get("review_decision") or "").strip().lower()
        category = str(row.get("category") or "").strip()
        description = str(row.get("description") or "").strip()
        row_attributes = [
            str(value).strip() for value in (row.get("attributes") or [])
            if str(value).strip()
        ]
        semantic_tier = str(row.get("semantic_tier") or "").strip()
        confidence = float(row.get("review_confidence") or 0.0)
        is_resolved = bool(category and description and category.lower() != "unknown" and decision in {"keep", "relabel"})
        # Every evidence-tier row carries useful, source-backed semantics.
        # In particular ``geometry_only`` rows intentionally use neutral
        # values ("unresolved object" + a visible-description fallback).  Do
        # not erase them merely because identity review remained unresolved:
        # downstream embeddings and QA must preserve all metric evidence.
        if category:
            categories[index] = category
        if description:
            captions[index] = description
        attributes[index] = row_attributes
        semantic_tiers[index] = semantic_tier or ("confirmed" if is_resolved else "geometry_only")
        decisions[index] = decision or ("keep" if is_resolved else "unknown")
        supercategories[index] = (
            "open-vocabulary object" if is_resolved
            else "open-vocabulary unresolved"
        )
        if is_resolved:
            resolved_ids.add(int(object_id))
        semantic_status[index] = status or ("vlm_multi_crop_resolved" if is_resolved else "needs_review")
        review_confidence[index] = confidence

    finite_geometry = np.isfinite(means).all(axis=1)
    distance_to_camera = np.full((count,), np.inf, dtype=np.float32)
    cameras = _camera_positions(state)
    if cameras.shape[0]:
        # Object counts are small; the dense distance matrix is only a few MB.
        distance_to_camera = np.linalg.norm(means[:, None, :] - cameras[None, :, :], axis=-1).min(axis=1)

    keep = source_active & finite_geometry & (observations >= int(min_observations))
    if max_camera_distance_m > 0.0:
        keep &= distance_to_camera <= float(max_camera_distance_m)
    if resolved_only:
        keep &= np.asarray([int(object_id) in resolved_ids for object_id in object_ids], dtype=bool)

    state["active"] = torch.as_tensor(keep, dtype=torch.bool)
    # Persist reviewed semantics alongside their evidence metadata.  Without
    # these assignments a derived presentation kept stale source labels even
    # after accepting a newer reviewed catalog.
    state["object_caption"] = captions
    state["object_category"] = categories
    state["object_supercategory"] = supercategories
    state["object_caption_decision"] = decisions
    state["object_key_attributes"] = attributes
    state["object_evidence_count"] = torch.as_tensor(observations, dtype=torch.int32)
    state["object_evidence_tier"] = tiers
    state["object_camera_distance_m"] = torch.as_tensor(distance_to_camera, dtype=torch.float32)
    state["object_semantic_status"] = semantic_status
    state["object_semantic_tier"] = semantic_tiers
    state["object_review_confidence"] = torch.as_tensor(review_confidence, dtype=torch.float32)

    payload["state"] = state
    payload["saved_unix_s"] = time.time()
    meta = payload.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        payload["meta"] = meta
    meta["presentation"] = {
        "source_scene_state": metadata_source_scene_state or str(source.resolve()),
        "reviewed_catalog": (
            metadata_reviewed_catalog
            if metadata_reviewed_catalog is not None
            else (str(reviewed_catalog.resolve()) if reviewed_catalog else None)
        ),
        "source_active_objects": int(source_active.sum()),
        "presentation_active_objects": int(keep.sum()),
        "min_observations": int(min_observations),
        "resolved_only": bool(resolved_only),
        "max_camera_distance_m": float(max_camera_distance_m),
        "automatic_policy": True,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return meta["presentation"]
def build_presentation_catalog(
    state_path: Path,
    reviewed_catalog: Path | None,
    destination: Path,
) -> dict[str, object]:
    """Materialize the viewer catalog from the already-filtered final state."""

    _, state = _load_payload(state_path)
    object_ids = np.asarray(state["object_id"].detach().cpu(), dtype=np.int64).reshape(-1)
    active = np.asarray(state["active"].detach().cpu(), dtype=bool).reshape(-1)
    count = int(object_ids.size)
    if active.shape != (count,) or len(set(object_ids.tolist())) != count:
        raise ValueError("presentation state has invalid active/object_id alignment")

    def metric_rows(name: str, width: int, fallback: str | None = None) -> np.ndarray:
        value = state.get(name)
        if value is None and fallback:
            value = state.get(fallback)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (count, width):
            raise ValueError(f"presentation state field {name!r} must be Nx{width}")
        return array

    centers = metric_rows("object_box_centers_m", 3, fallback="means")
    dimensions = metric_rows("object_box_dimensions_m", 3)
    rotations = metric_rows("object_box_wxyz", 4)
    reviewed = _load_reviewed_catalog(reviewed_catalog)
    categories = list(state.get("object_category") or [])
    captions = list(state.get("object_caption") or [])
    decisions = list(state.get("object_caption_decision") or [])
    tiers = list(state.get("object_semantic_tier") or [])
    statuses = list(state.get("object_semantic_status") or [])
    image_ids = list(state.get("object_image_ids") or [])
    display_statuses = list(state.get("object_display_status") or [])
    geometry_statuses = list(state.get("object_geometry_status") or [])
    compound_boxes = list(state.get("object_compound_boxes") or [])

    rows: list[dict[str, object]] = []
    for index in range(count):
        tier = str(tiers[index] if index < len(tiers) else "").strip().lower()
        display = str(
            display_statuses[index] if index < len(display_statuses) else ""
        ).strip().lower()
        geometry_status = str(
            geometry_statuses[index] if index < len(geometry_statuses) else ""
        ).strip().lower()
        boxes = compound_boxes[index] if index < len(compound_boxes) else []
        probable_compound = (
            "compound_geometry_probable" in geometry_status
            and "rejected" not in geometry_status
            and isinstance(boxes, (list, tuple))
            and len(boxes) == 2
        )
        if display.endswith("_suppressed") or not (
            (bool(active[index]) and tier in {"confirmed", "probable"})
            or probable_compound
        ):
            continue
        object_id = int(object_ids[index])
        center = centers[index]
        dimension = dimensions[index]
        rotation = rotations[index]
        if (
            not np.isfinite(center).all()
            or not np.isfinite(dimension).all()
            or np.any(dimension <= 0.0)
            or not np.isfinite(rotation).all()
            or np.linalg.norm(rotation) <= 1.0e-8
        ):
            raise ValueError(f"presentation object {object_id} has invalid metric OBB")
        row = copy.deepcopy(reviewed.get(object_id) or {"id": object_id})
        row.update({
            "id": object_id,
            "category": str(categories[index] if index < len(categories) else "object"),
            "description": str(captions[index] if index < len(captions) else ""),
            "review_decision": str(decisions[index] if index < len(decisions) else ""),
            "semantic_tier": str(tiers[index] if index < len(tiers) else ""),
            "semantic_status": str(statuses[index] if index < len(statuses) else ""),
            "center_m": center.astype(float).tolist(),
            "dimensions_m": dimension.astype(float).tolist(),
            "wxyz": (rotation / np.linalg.norm(rotation)).astype(float).tolist(),
            "observation_count": _unique_observation_count(
                image_ids[index] if index < len(image_ids) else []
            ),
            "presentation_visible": True,
        })
        rows.append(row)

    if not rows:
        raise ValueError("presentation state contains no active objects")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "schema": "farm.presentation-catalog-summary.v1",
        "visible_objects": len(rows),
        "object_ids": [int(row["id"]) for row in rows],
        "output": str(destination.resolve()),
        "fingerprint_sha256": _fingerprint(rows),
    }




def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable enough for 3DGS opacity logits while avoiding scipy dependency.
    out = np.empty_like(x, dtype=np.float32)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def export_cloud(
    source_ply: Path,
    destination: Path,
    *,
    max_points: int,
    min_opacity: float,
    seed: int,
    meters_per_scene_unit: float = 1.0,
) -> dict:
    scale = validate_meters_per_scene_unit(meters_per_scene_unit)
    contract = cloud_export_contract(
        source_ply,
        max_points=max_points,
        min_opacity=min_opacity,
        seed=seed,
        meters_per_scene_unit=scale,
    )
    vertex = PlyData.read(str(source_ply), mmap=True)["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {"x", "y", "z"}
    if not required.issubset(names):
        raise ValueError(f"PLY has no XYZ fields: {source_ply}")
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32, copy=False)
    valid = np.isfinite(xyz).all(axis=1)

    alpha = np.ones((xyz.shape[0],), dtype=np.float32)
    if "opacity" in names:
        alpha = _sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))
        valid &= np.isfinite(alpha) & (alpha >= float(min_opacity))

    indices = np.nonzero(valid)[0]
    if indices.size == 0:
        raise ValueError("No finite points survived the opacity gate")

    rng = np.random.default_rng(int(seed))
    if max_points > 0 and indices.size > max_points:
        # Weighted reservoir approximation: opacity improves visible-surface
        # density without turning this into an expensive renderer.
        weights = np.clip(alpha[indices], 1.0e-4, 1.0)
        keys = np.log(np.clip(rng.random(indices.size), 1.0e-12, 1.0)) / weights
        chosen = np.argpartition(keys, -int(max_points))[-int(max_points):]
        indices = indices[chosen]
    indices.sort()
    points = np.asarray(xyz[indices], dtype=np.float32) * scale

    if {"red", "green", "blue"}.issubset(names):
        colors = np.column_stack([vertex["red"][indices], vertex["green"][indices], vertex["blue"][indices]])
        colors = np.asarray(colors, dtype=np.uint8)
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        dc = np.column_stack(
            [vertex["f_dc_0"][indices], vertex["f_dc_1"][indices], vertex["f_dc_2"][indices]]
        ).astype(np.float32, copy=False)
        colors = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
        colors = np.rint(colors * 255.0).astype(np.uint8)
    else:
        colors = np.full((points.shape[0], 3), 150, dtype=np.uint8)

    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, xyz=points, rgb=colors)
    lo, hi = np.percentile(points, [1.0, 99.0], axis=0)
    return {
        "source_ply": str(source_ply.resolve()),
        "source_vertices": int(xyz.shape[0]),
        "exported_points": int(points.shape[0]),
        "min_opacity": float(min_opacity),
        "meters_per_scene_unit": scale,
        "coordinate_unit": "metre",
        "contract": contract,
        "fingerprint_sha256": _fingerprint(contract),
        "bounds_p01_m": lo.round(5).tolist(),
        "bounds_p99_m": hi.round(5).tolist(),
    }


def main() -> int:
    overall_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--scene-ply", type=Path, required=True)
    parser.add_argument("--reviewed-catalog", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-filename", default="scene_state_presentation.pt")
    parser.add_argument("--catalog-filename", default="presentation_catalog.json")
    parser.add_argument(
        "--metadata-source-scene-state",
        help="Portable provenance label stored instead of the absolute input state path.",
    )
    parser.add_argument(
        "--metadata-reviewed-catalog",
        help="Portable provenance label stored instead of the absolute reviewed-catalog path.",
    )
    parser.add_argument(
        "--reuse-cloud",
        action="store_true",
        help="Reuse data/cloud.npz and its existing manifest metadata instead of exporting the PLY again.",
    )
    parser.add_argument("--min-observations", type=int, default=5)
    parser.add_argument("--resolved-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-camera-distance-m", type=float, default=9.0)
    parser.add_argument("--max-cloud-points", type=int, default=1_500_000)
    parser.add_argument("--min-opacity", type=float, default=0.08)
    parser.add_argument("--meters-per-scene-unit", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    # Validate the entire cloud contract before creating or overwriting any
    # output. A bad metric scale must be side-effect free.
    scale = validate_meters_per_scene_unit(args.meters_per_scene_unit)
    source_state = args.scene_state.expanduser().resolve()
    source_ply = args.scene_ply.expanduser().resolve()
    reviewed_catalog = args.reviewed_catalog.expanduser().resolve() if args.reviewed_catalog else None
    max_cloud_points = max(0, int(args.max_cloud_points))
    min_opacity = float(np.clip(args.min_opacity, 0.0, 1.0))
    cloud_contract = cloud_export_contract(
        source_ply,
        max_points=max_cloud_points,
        min_opacity=min_opacity,
        seed=int(args.seed),
        meters_per_scene_unit=scale,
    )
    out = args.output_dir.expanduser().resolve()
    data_dir = out / "data"
    visuals_dir = out / "visuals"
    reused_cloud_summary = None
    if args.reuse_cloud:
        reused_cloud_summary = validate_reusable_cloud(
            out / "manifest.json", data_dir / "cloud.npz", cloud_contract
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    state_started = time.perf_counter()
    state_summary = build_demo_state(
        source_state,
        data_dir / args.state_filename,
        reviewed_catalog=reviewed_catalog,
        min_observations=max(1, int(args.min_observations)),
        resolved_only=bool(args.resolved_only),
        max_camera_distance_m=max(0.0, float(args.max_camera_distance_m)),
        metadata_source_scene_state=args.metadata_source_scene_state,
        metadata_reviewed_catalog=args.metadata_reviewed_catalog,
    )
    state_summary["coordinate_unit"] = "metre"
    state_summary["coordinate_transform"] = "identity; mapping state is already metric"
    catalog_summary = build_presentation_catalog(
        data_dir / args.state_filename,
        reviewed_catalog,
        data_dir / args.catalog_filename,
    )
    state_seconds = time.perf_counter() - state_started
    cloud_started = time.perf_counter()
    if args.reuse_cloud:
        assert reused_cloud_summary is not None
        cloud_summary = reused_cloud_summary
    else:
        cloud_summary = export_cloud(
            source_ply,
            data_dir / "cloud.npz",
            max_points=max_cloud_points,
            min_opacity=min_opacity,
            seed=int(args.seed),
            meters_per_scene_unit=scale,
        )
    cloud_seconds = time.perf_counter() - cloud_started
    manifest = {
        "schema": "farm.presentation-bundle.v1",
        "created_unix_s": time.time(),
        "metric_frame": {
            "scene_state_unit": "metre",
            "cloud_unit": "metre",
            "cloud_source_unit": "3DGS scene unit",
            "meters_per_scene_unit": scale,
            "scene_state_rescaled": False,
        },
        "fingerprint_sha256": _fingerprint({
            "source_state": _file_signature(source_state),
            "reviewed_catalog": _file_signature(reviewed_catalog) if reviewed_catalog else None,
            "cloud": cloud_summary.get("fingerprint_sha256", _fingerprint(cloud_contract)),
            "policy": {
                "min_observations": max(1, int(args.min_observations)),
                "resolved_only": bool(args.resolved_only),
                "max_camera_distance_m": max(0.0, float(args.max_camera_distance_m)),
            },
        }),
        "state": state_summary,
        "catalog": catalog_summary,
        "cloud": cloud_summary,
        "viewer": {
            "scene_state": f"data/{args.state_filename}",
            "presentation_catalog": f"data/{args.catalog_filename}",
            "cloud": "data/cloud.npz",
            "presentation_mode": True,
        },
        "timing": {
            "state_filter_seconds": state_seconds,
            "cloud_export_seconds": cloud_seconds,
            "duration_seconds": time.perf_counter() - overall_started,
            "stage": "presentation_bundle_export",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# Scene presentation bundle\n\n"
        "Generated automatically from the saved experiment. The full source mapping remains unchanged and auditable.\n\n"
        f"- source cloud: `{cloud_summary['source_ply']}`\n"
        f"- context cloud: {cloud_summary['exported_points']:,} points\n"
        f"- full geometry/appearance-valid layer: {state_summary['source_active_objects']} objects\n"
        f"- visible resolved layer: {state_summary['presentation_active_objects']} objects\n"
        f"- semantic unknowns retained outside the demo layer: "
        f"{state_summary['source_active_objects'] - state_summary['presentation_active_objects']} objects\n"
        f"- policy: >= {state_summary['min_observations']} observations, "
        f"resolved_only={state_summary['resolved_only']}\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
