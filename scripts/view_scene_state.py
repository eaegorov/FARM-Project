#!/usr/bin/env python3
"""View a saved ``scene_state.pt`` in the browser (viser) — no re-mapping needed.

Loads a scene graph produced by ``python -m scene_graph.offline.run`` (or one
of the prebuilt graphs shipped with FARM-Scenes) and serves it interactively:
per-object voxel clouds and 3D boxes with captions, click-to-inspect, and the
**Query** panel, which runs the full relational retrieval pipeline
(``parse_query`` -> ``execute_spatial_query``) against the loaded scene.

Scene-state files use PyTorch pickle serialization. Only open files produced
by FARM or downloaded from a source you trust.

Examples (inside the container)::

    # Objects only
    python scripts/view_scene_state.py --pt /data/out/warehouse.pt

    # With the dataset's accumulated point cloud as background context
    python scripts/view_scene_state.py \
        --pt /data/scene_graphs/grandtour/2024-11-25_warehouse.pt \
        --cloud /data/scenes/grandtour/2024-11-25_warehouse/cloud.npz

Then open http://localhost:8080. For language queries, either click
"Start vLLM retrieval backend" in the Query panel (launches the servers on
this machine) or run ``./run.sh vllm`` first / point ``VLLM_BASE_URL`` +
``VLLM_EMBED_BASE_URL`` at running servers.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_state(pt_path: Path) -> dict:
    """Load a scene_state payload, normalising through the library loader."""
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    feature_dim = None
    if isinstance(payload, dict):
        feature_dim = payload.get("feature_dim")
        if feature_dim is None and isinstance(payload.get("state"), dict):
            feats = payload["state"].get("features")
            if isinstance(feats, torch.Tensor) and feats.ndim == 2:
                feature_dim = int(feats.shape[1])
    if feature_dim is not None:
        from scene_graph.scene_state_io import load_scene_state

        state = load_scene_state(pt_path, feature_dim=int(feature_dim), device="cpu")
        # The library loader intentionally knows only the upstream schema.
        # Preserve post-processing extensions (validated OBBs, evidence QA,
        # review confidence) that would otherwise be silently discarded.
        raw_state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
        for key, value in raw_state.items():
            if key not in state:
                state[key] = value
        return state
    # Fallback: raw dict without the save_scene_state wrapper.
    return payload["state"] if isinstance(payload, dict) and "state" in payload else payload


def _record_field(rec: object, key: str) -> object:
    """Field access for image records, which are dicts in raw payloads and
    ``ImageRecord`` dataclasses after ``load_scene_state`` normalisation."""
    if isinstance(rec, dict):
        return rec.get(key)
    return getattr(rec, key, None)


def remap_image_refs(state: dict, frames_dir: Path) -> int:
    """Re-point saved image references at *frames_dir* (a scene directory with
    ``rgb/<camera>/`` frames) so click-to-inspect can show each object's anchor
    view. Saved graphs reference the reconstruction machine's paths; the same
    frames ship with the dataset."""
    remapped = 0
    for rec in state.get("images") or []:
        ref = str(_record_field(rec, "source_ref") or _record_field(rec, "storage_path") or "")
        base = Path(ref).name
        if not base:
            continue
        camera = str(_record_field(rec, "camera_id") or "")
        candidates = [frames_dir / "rgb" / camera / base] if camera else []
        candidates += [frames_dir / "rgb" / base, frames_dir / base]
        for cand in candidates:
            if cand.is_file():
                if isinstance(rec, dict):
                    rec["source_ref"] = str(cand)
                else:
                    rec.source_ref = str(cand)
                remapped += 1
                break
    return remapped


def load_cloud(cloud_path: Path, max_points: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Read points (+ optional colors) from a ``cloud.npz``-style archive."""
    with np.load(cloud_path) as data:
        pts = None
        for key in ("points", "xyz", "cloud"):
            if key in data.files:
                pts = np.asarray(data[key], dtype=np.float32)
                break
        if pts is None:
            pts = np.asarray(data[data.files[0]], dtype=np.float32)
        pts = pts.reshape(-1, pts.shape[-1])[:, :3]
        cols = None
        for key in ("colors", "rgb", "color"):
            if key in data.files:
                cols = np.asarray(data[key], dtype=np.float32).reshape(-1, 3)
                break
    if max_points > 0 and pts.shape[0] > max_points:
        keep = np.random.default_rng(0).choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[keep]
        if cols is not None and cols.shape[0] >= keep.max() + 1:
            cols = cols[keep]
    return pts, cols


def resolve_world_up(
    explicit: object | None,
    resolved_context: Path | None,
) -> np.ndarray | None:
    """Resolve exact up direction with CLI > run-context > legacy priority."""

    value = explicit
    if value is None and resolved_context is not None:
        try:
            payload = json.loads(resolved_context.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid --resolved-context: {resolved_context}") from exc
        if not isinstance(payload, dict):
            raise ValueError("--resolved-context must contain a JSON object")
        value = payload.get("resolved_up", payload.get("world_up"))
        if value is None:
            raise ValueError("--resolved-context has no resolved_up/world_up vector")
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.float64).reshape(3)
    except (TypeError, ValueError) as exc:
        raise ValueError("world-up must contain exactly three numbers") from exc
    norm = float(np.linalg.norm(result))
    if not np.isfinite(result).all() or not np.isfinite(norm) or norm <= 1.0e-8:
        raise ValueError("world-up must be finite and non-zero")
    return (result / norm).astype(np.float32)


def attach_object_crops(
    state: dict,
    mapping_dir: Path,
    max_crops: int,
    *,
    additional_roots: tuple[Path, ...] = (),
) -> int:
    """Decode the strongest saved detection crops for click-to-inspect.

    Mask archives contain compact JPEG bytes, so this adds useful visual
    evidence without keeping full frames or re-running perception.
    """
    if max_crops <= 0:
        return 0
    observations = state.get("object_mask_observations") or []
    means = state.get("means")
    count = int(means.shape[0]) if isinstance(means, torch.Tensor) else len(observations)
    galleries: list[list[np.ndarray]] = [[] for _ in range(count)]
    active = state.get("active")
    active_mask = (
        active.detach().cpu().numpy().astype(bool)
        if isinstance(active, torch.Tensor)
        else np.ones((count,), dtype=bool)
    )
    geometry_statuses = state.get("object_geometry_status") or []
    loaded = 0
    for obj_idx in range(min(count, len(observations))):
        status = str(
            geometry_statuses[obj_idx] if obj_idx < len(geometry_statuses) else ""
        ).strip().lower()
        probable_compound = (
            "compound_geometry_probable" in status and "rejected" not in status
        )
        if (
            obj_idx >= active_mask.shape[0]
            or (not bool(active_mask[obj_idx]) and not probable_compound)
        ):
            continue
        rows = observations[obj_idx]
        if not isinstance(rows, (list, tuple)):
            continue
        ranked = sorted(
            (row for row in rows if isinstance(row, dict)),
            key=lambda row: (float(row.get("score") or 0.0), int(row.get("crop_jpeg_bytes_len") or 0)),
            reverse=True,
        )
        seen_images: set[int] = set()
        for row in ranked:
            try:
                image_id = int(row.get("image_id", -1))
            except (TypeError, ValueError):
                image_id = -1
            if image_id in seen_images:
                continue
            raw_path = Path(str(row.get("path") or ""))
            roots = (mapping_dir, *additional_roots)
            candidates = [raw_path]
            if raw_path.name:
                for root in roots:
                    candidates.extend(
                        (
                            root / raw_path,
                            root / "masks" / raw_path,
                            root / "masks" / "assemblies" / raw_path,
                            root / "masks" / raw_path.parent.name / raw_path.name,
                        )
                    )
            path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if path is None:
                continue
            try:
                with np.load(path, allow_pickle=False) as archive:
                    jpeg = np.asarray(archive["crop_jpeg_bytes"], dtype=np.uint8).tobytes()
                image = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"), dtype=np.uint8)
            except Exception:
                continue
            galleries[obj_idx].append(image)
            seen_images.add(image_id)
            loaded += 1
            if len(galleries[obj_idx]) >= max_crops:
                break
    state["rgb_observations"] = galleries
    return loaded


def standard_run_crop_roots(scene_state_path: Path) -> tuple[Path, ...]:
    """Additional evidence roots for a canonical standard-run final state."""

    state_path = scene_state_path.expanduser()
    # Canonical bundles have used both ``final/scene_state.pt`` and
    # ``final/data/scene_state.pt``. Resolve the run root from the named
    # ``final`` directory instead of relying on a fixed parent depth.
    final_dir = next(
        (parent for parent in state_path.parents if parent.name == "final"),
        state_path.parent,
    )
    run_root = final_dir.parent
    return (run_root, run_root / "qa" / "assemblies")


def has_geometry_layers(state: dict) -> bool:
    """Whether the state contains audited V8 geometry display extensions."""

    statuses = state.get("object_geometry_status") or []
    if any("compound_geometry_" in str(status).lower() for status in statuses):
        return True
    compound_rows = state.get("object_compound_boxes") or []
    if any(isinstance(row, (list, tuple)) and bool(row) for row in compound_rows):
        return True
    consensus_rows = state.get("object_geometry_consensus_points") or []
    for row in consensus_rows:
        if isinstance(row, torch.Tensor) and row.numel() > 0:
            return True
        with np.errstate(all="ignore"):
            try:
                if np.asarray(row).size > 0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def choose_initial_object(state: dict) -> int | None:
    """Select a representative object from saved evidence, without class prompts."""
    active = state.get("active")
    object_ids = state.get("object_id")
    if not isinstance(active, torch.Tensor) or not isinstance(object_ids, torch.Tensor):
        return None
    active_np = active.detach().cpu().numpy().astype(bool, copy=False)
    indices = np.flatnonzero(active_np)
    if indices.size == 0:
        return None

    def values(key: str, default: float) -> np.ndarray:
        raw = state.get(key)
        if isinstance(raw, torch.Tensor) and raw.ndim > 0 and int(raw.shape[0]) == active_np.shape[0]:
            return raw.detach().cpu().numpy().astype(np.float64, copy=False)
        return np.full(active_np.shape[0], default, dtype=np.float64)

    observations = values("object_evidence_count", 1.0)
    inside = values("object_geometry_inside_rate", 1.0)
    box_iou = values("object_geometry_projected_box_iou", 1.0)
    orientation = values("object_geometry_orientation_confidence", 0.5)
    geometry_statuses = state.get("object_geometry_status") or []
    probable_compounds = np.asarray(
        [
            "compound_geometry_probable" in str(
                geometry_statuses[index] if index < len(geometry_statuses) else ""
            ).lower()
            and "rejected" not in str(
                geometry_statuses[index] if index < len(geometry_statuses) else ""
            ).lower()
            for index in range(active_np.shape[0])
        ],
        dtype=bool,
    )
    compound_indices = np.flatnonzero(probable_compounds)
    if compound_indices.size:
        # A validated compound is the most informative prompt-free first card:
        # it demonstrates part aggregation and multi-box geometry. Parent-box
        # reprojection IoU is intentionally excluded because articulated
        # assemblies are represented by their child OBBs.
        compound_scores = (
            np.log1p(np.maximum(observations, 0.0))
            * (0.5 + 0.5 * np.clip(inside, 0.0, 1.0))
            * (0.5 + 0.5 * np.clip(orientation, 0.0, 1.0))
        )
        best_compound = int(
            compound_indices[int(np.nanargmax(compound_scores[compound_indices]))]
        )
        return int(object_ids[best_compound].item())
    dimensions_raw = state.get("object_box_dimensions_m")
    if isinstance(dimensions_raw, torch.Tensor) and dimensions_raw.shape == (active_np.shape[0], 3):
        volumes = np.prod(dimensions_raw.detach().cpu().numpy(), axis=1)
    else:
        volumes = np.ones(active_np.shape[0], dtype=np.float64)
    scores = (
        np.log1p(np.maximum(observations, 0.0))
        * np.clip(inside, 0.0, 1.0)
        * np.clip(box_iou, 0.0, 1.0)
        * (0.5 + 0.5 * np.clip(orientation, 0.0, 1.0))
        / (1.0 + 0.15 * np.maximum(volumes - 1.0, 0.0))
    )
    best_index = int(indices[int(np.nanargmax(scores[indices]))])
    return int(object_ids[best_index].item())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve a saved scene_state.pt in viser (objects, captions, retrieval).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pt", type=Path, required=True, help="Path to scene_state.pt")
    parser.add_argument("--cloud", type=Path, default=None, help="Optional cloud.npz shown as static background")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Scene directory with rgb/<camera>/ frames for click-to-inspect anchor views (default: the --cloud directory)")
    parser.add_argument("--host", default="127.0.0.1", help="Interface for the viser server")
    parser.add_argument("--port", type=int, default=8080, help="Port for the viser server")
    parser.add_argument("--point-size", type=float, default=0.004, help="Background cloud point size (m)")
    parser.add_argument("--max-cloud-points", type=int, default=2_000_000, help="Random-subsample the background cloud to this many points (0 = keep all)")
    parser.add_argument("--voxel-points-per-object", type=int, default=0, help="Per-object voxel evidence points to render (0 = clean default: boxes + cloud + trajectory only)")
    parser.add_argument("--object-point-size", type=float, default=0.018, help="Rendered object-evidence point size (m)")
    parser.add_argument("--query-examples", type=Path, default=None, help="Text file with one query per line for the Query panel's Examples dropdown (default: derive examples from the scene's own captioned objects)")
    parser.add_argument("--mapping-dir", type=Path, default=None, help="Mapping directory containing masks/ for object crop galleries (default: parent of --pt)")
    parser.add_argument("--max-object-crops", type=int, default=3, help="Best saved crops shown after clicking an object (0 disables)")
    parser.add_argument("--presentation", action=argparse.BooleanOptionalAction, default=False, help="Use a clean dark presentation theme and compact object cards")
    parser.add_argument("--title", default="Scene Memory", help="Browser/server title")
    parser.add_argument("--query", action=argparse.BooleanOptionalAction, default=True, help="Show language-query controls")
    parser.add_argument("--show-trajectory", action=argparse.BooleanOptionalAction, default=False, help="Draw the camera path through the scene")
    parser.add_argument("--show-trajectory-axes", action=argparse.BooleanOptionalAction, default=False, help="Draw every camera coordinate frame in addition to the trajectory line")
    parser.add_argument("--initial-object-id", default="auto", help="Object selected on load; 'auto' uses geometric evidence, 'none' disables")
    parser.add_argument(
        "--world-up", type=float, nargs=3, metavar=("X", "Y", "Z"),
        help="Exact world-up vector; overrides --resolved-context.",
    )
    parser.add_argument(
        "--resolved-context", type=Path,
        help="Standard-run resolved_context.json providing resolved_up.",
    )
    args = parser.parse_args()

    from scene_graph.visualization.viser_visualizer import PipelineViserVisualizer

    world_up = resolve_world_up(
        args.world_up,
        args.resolved_context.expanduser() if args.resolved_context is not None else None,
    )
    state = load_state(args.pt.expanduser())
    frames_dir = args.frames_dir
    if frames_dir is None and args.cloud is not None:
        frames_dir = args.cloud.expanduser().parent
    if frames_dir is not None:
        remapped = remap_image_refs(state, Path(frames_dir).expanduser())
        if remapped:
            print(f"Resolved {remapped} image references into {frames_dir}")
    mapping_dir = args.mapping_dir.expanduser() if args.mapping_dir is not None else args.pt.expanduser().parent
    loaded_crops = attach_object_crops(
        state,
        mapping_dir,
        max(0, int(args.max_object_crops)),
        additional_roots=standard_run_crop_roots(args.pt),
    )
    if loaded_crops:
        print(f"Object crop gallery: {loaded_crops} saved crops from {mapping_dir}")
    means = state.get("means")
    n_objects = int(means.shape[0]) if isinstance(means, torch.Tensor) else 0
    captions = [c for c in (state.get("object_caption") or []) if isinstance(c, str) and c.strip()]
    active = state.get("active")
    active_objects = int(active.sum().item()) if isinstance(active, torch.Tensor) else n_objects
    print(f"Loaded {n_objects} objects / {active_objects} active ({len(captions)} captioned) from {args.pt}")

    background = None
    if args.cloud is not None:
        try:
            background = load_cloud(args.cloud.expanduser(), args.max_cloud_points)
        except Exception as exc:  # noqa: BLE001 - cloud is optional context
            print(f"Skipping background cloud ({exc})")
    image_records = state.get("images") or []
    camera_ids = {
        str(_record_field(record, "camera_id") or "") for record in image_records
        if str(_record_field(record, "camera_id") or "")
    }
    semantic_tiers = state.get("object_semantic_tier") or []
    geometry_statuses = state.get("object_geometry_status") or []
    display_statuses = state.get("object_display_status") or []
    tier_counts: dict[str, int] = {}
    if isinstance(active, torch.Tensor) and isinstance(semantic_tiers, list):
        active_np = active.detach().cpu().numpy().astype(bool, copy=False)
        for index in np.flatnonzero(active_np).tolist():
            tier = str(
                semantic_tiers[index]
                if index < len(semantic_tiers)
                else "confirmed"
            )
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
    presentation_objects = 0
    if isinstance(active, torch.Tensor):
        active_np = active.detach().cpu().numpy().astype(bool, copy=False)
        for index in range(active_np.shape[0]):
            geometry_status = str(
                geometry_statuses[index] if index < len(geometry_statuses) else ""
            ).strip().lower()
            semantic_tier = str(
                semantic_tiers[index] if index < len(semantic_tiers) else "confirmed"
            ).strip().lower()
            display_status = str(
                display_statuses[index] if index < len(display_statuses) else ""
            ).strip().lower()
            geometry_pass = geometry_status == "geometry_pass" or geometry_status.endswith("_geometry_pass")
            direct = active_np[index] and geometry_pass and semantic_tier in {"confirmed", "probable"}
            compound = "compound_geometry_probable" in geometry_status and "rejected" not in geometry_status
            presentation_objects += int(not display_status.endswith("_suppressed") and (direct or compound))
    summary = {
        "active_objects": active_objects,
        "metric_objects": active_objects,
        "confirmed_objects": tier_counts.get("confirmed") if tier_counts else None,
        "probable_objects": tier_counts.get("probable", 0) if tier_counts else None,
        "geometry_only_objects": tier_counts.get("geometry_only", 0) if tier_counts else None,
        "presentation_objects": presentation_objects,
        "views": len(image_records),
        "timestamps": len(image_records) // max(1, len(camera_ids)),
        "cloud_points": f"{background[0].shape[0] / 1_000_000:.1f}M" if background is not None else "0",
        "geometry_layers_available": has_geometry_layers(state),
    }

    query_examples = None
    if args.query_examples is not None:
        lines = args.query_examples.expanduser().read_text().splitlines()
        query_examples = [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
        print(f"Loaded {len(query_examples)} query examples from {args.query_examples}")

    visualizer = PipelineViserVisualizer(
        enabled=True,
        host=args.host,
        port=args.port,
        server_label=args.title,
        presentation_mode=bool(args.presentation),
        scene_summary=summary,
        query_enabled=bool(args.query),
        query_examples=query_examples,
        live_rgb_enabled=False,
        image_pose_axes_enabled=False,
        object_image_connections_enabled=False,
        covisibility_connections_enabled=False,
        regions_enabled=False,
        object_voxel_cloud_enabled=args.voxel_points_per_object > 0,
        object_voxel_max_points_per_object=max(0, args.voxel_points_per_object),
        object_voxel_point_size=max(0.004, float(args.object_point_size)),
        object_box_from_voxels=not bool(args.presentation),
        object_box_max_volume_m3=30.0 if args.presentation else 0.0,
        object_box_max_side_m=0.0,
        object_box_large_side_threshold_m=4.0 if args.presentation else 0.0,
        object_box_max_large_sides=1 if args.presentation else 0,
        world_up=world_up,
    )
    if not visualizer.enabled:
        print("viser is not available in this environment — aborting.")
        return 1

    frame_points = None
    if background is not None:
        points, colors = background
        visualizer.add_background_point_cloud(points, colors, point_size=args.point_size)
        print(f"Background cloud: {points.shape[0]:,} points from {args.cloud}")
        frame_points = points

    visualizer.update(colors=[], depths=[], intrinsics=[], poses=[], scene_state=state)
    initial_token = str(args.initial_object_id or "none").strip().lower()
    initial_object_id = choose_initial_object(state) if initial_token == "auto" else None
    if initial_token not in {"", "auto", "none", "off", "disabled"}:
        try:
            initial_object_id = int(initial_token)
        except ValueError:
            initial_object_id = None
    if initial_object_id is not None:
        visualizer._handle_object_click(initial_object_id)
        print(f"Initial representative object: {initial_object_id}")
    poses = []
    for rec in state.get("images") or []:
        pose = _record_field(rec, "pose")
        if pose is None:
            continue
        try:
            arr = np.asarray(pose.cpu().numpy() if hasattr(pose, "cpu") else pose, dtype=np.float32)
        except Exception:
            continue
        if arr.shape == (4, 4) and np.isfinite(arr).all():
            poses.append(arr)
    if poses and bool(args.show_trajectory):
        try:
            visualizer.add_trajectory(np.stack(poses), show_axes=bool(args.show_trajectory_axes))
        except Exception as exc:  # noqa: BLE001 - trajectory is optional context
            print(f"Skipping trajectory ({exc})")
    if frame_points is None and isinstance(means, torch.Tensor) and means.numel():
        frame_points = means.detach().cpu().numpy().reshape(-1, 3)
    visualizer.set_home_view(frame_points)
    print(f"Serving on http://localhost:{args.port} — Ctrl+C to stop.")
    try:
        while True:
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("Bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
