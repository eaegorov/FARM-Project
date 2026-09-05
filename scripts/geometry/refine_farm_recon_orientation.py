#!/usr/bin/env python3
"""Refine MV-SAM3D orientation with gravity and real PINHOLE silhouettes.

The original reconstruction assembly aligned PCA axes to a FARM OBB.  That is
not sufficient to establish semantic up for incomplete or approximately
symmetric generated geometry.  This post-process keeps the reconstructed
shape rigid, tests a minimal gravity correction in the actual COLMAP cameras,
and accepts only corrections supported by the saved full object masks.

The script belongs to FARM.  It imports the versioned reconstruction helpers
from a sibling ``splatica_demo_app`` checkout but never edits that checkout's
source code.  Only the requested reconstruction output directory is mutated.
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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial.transform import Rotation


SCHEMA = "splatica.farm-recon-orientation.v1"


@dataclass(frozen=True)
class SilhouetteEvidence:
    median_iou: float
    per_view_iou: tuple[float, ...]


def _unit(vector: Any, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or length < 1.0e-10:
        raise ValueError(f"{name} must be one finite non-zero 3-vector")
    return value / length


def vector_angle_degrees(first: Any, second: Any) -> float:
    a = _unit(first, name="first direction")
    b = _unit(second, name="second direction")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def minimal_row_rotation(source: Any, target: Any) -> np.ndarray:
    """Return a right-handed row-vector rotation mapping source to target."""

    start = _unit(source, name="source direction")
    finish = _unit(target, name="target direction")
    cross = np.cross(start, finish)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(start, finish), -1.0, 1.0))
    if sine < 1.0e-10:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.asarray([1.0, 0.0, 0.0])
        if abs(float(start[0])) > 0.8:
            axis = np.asarray([0.0, 1.0, 0.0])
        axis = _unit(np.cross(start, axis), name="antiparallel rotation axis")
        column_rotation = Rotation.from_rotvec(axis * math.pi).as_matrix()
    else:
        axis = cross / sine
        column_rotation = Rotation.from_rotvec(
            axis * math.atan2(sine, cosine)
        ).as_matrix()
    row_rotation = column_rotation.T
    if not np.allclose(start @ row_rotation, finish, atol=1.0e-7):
        raise ValueError("minimal rotation does not map reconstruction up to gravity")
    if not np.allclose(row_rotation.T @ row_rotation, np.eye(3), atol=1.0e-8):
        raise ValueError("minimal rotation is not rigid")
    if not math.isclose(float(np.linalg.det(row_rotation)), 1.0, abs_tol=1.0e-8):
        raise ValueError("minimal rotation changes handedness")
    return row_rotation


def fractional_row_rotation(full_rotation: np.ndarray, fraction: float) -> np.ndarray:
    value = float(fraction)
    if not 0.0 <= value <= 1.0:
        raise ValueError("rotation fraction must be in [0, 1]")
    column = np.asarray(full_rotation, dtype=np.float64).reshape(3, 3).T
    return Rotation.from_rotvec(Rotation.from_matrix(column).as_rotvec() * value).as_matrix().T


def choose_correction_fraction(
    fractions: Sequence[float],
    scores: Sequence[float],
    *,
    angle_degrees: float,
    tolerance: float,
    minimum_angle_degrees: float,
) -> float:
    """Choose the strongest mask-supported correction, or keep small tilts."""

    if len(fractions) != len(scores) or len(fractions) == 0:
        raise ValueError("fractions and scores must have the same non-zero length")
    pairs = [(float(fraction), float(score)) for fraction, score in zip(fractions, scores, strict=True)]
    if any(not 0.0 <= fraction <= 1.0 or not np.isfinite(score) for fraction, score in pairs):
        raise ValueError("invalid orientation candidate")
    if float(angle_degrees) < float(minimum_angle_degrees):
        return 0.0
    best = max(score for _, score in pairs)
    eligible = [fraction for fraction, score in pairs if score + float(tolerance) >= best]
    return max(eligible)


def _quaternion_wxyz_matrix(value: Any) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1.0e-12)
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_applied_canonical_up(params_path: Path) -> np.ndarray:
    with np.load(params_path, allow_pickle=False) as payload:
        rotation = _quaternion_wxyz_matrix(payload["rotation"])
        scale = np.asarray(payload["scale"], dtype=np.float64).reshape(3)
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError(f"invalid reconstruction scale: {params_path}")
    if not np.allclose(scale, scale[0], rtol=1.0e-4, atol=1.0e-6):
        raise ValueError(f"non-uniform reconstruction scale: {params_path}")
    return _unit(np.asarray([0.0, 0.0, scale[2]]) @ rotation, name="pose-applied canonical up")


def _deterministic_sample(points: np.ndarray, maximum: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(values) <= int(maximum):
        return values
    return values[np.linspace(0, len(values) - 1, int(maximum), dtype=np.int64)]


def _target_mask_small(mask: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    resized = image.resize((int(size), int(size)), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


def _projected_mask_small(
    points_world: np.ndarray,
    view: Mapping[str, Any],
    *,
    size: int,
    project_world: Any,
) -> np.ndarray:
    pixels, _depth = project_world(points_world, dict(view))
    source_size = tuple(int(value) for value in view["source_size"])
    scaled = pixels * np.asarray([size / source_size[0], size / source_size[1]], dtype=np.float64)
    xy = np.floor(scaled).astype(np.int64, copy=False)
    valid = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < size)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < size)
    )
    raster = np.zeros((size, size), dtype=bool)
    if np.any(valid):
        raster[xy[valid, 1], xy[valid, 0]] = True
    raster = ndimage.binary_dilation(raster, iterations=2)
    raster = ndimage.binary_closing(raster, iterations=2)
    raster = ndimage.binary_fill_holes(raster)
    return np.asarray(raster, dtype=bool)


def binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("silhouette masks must have identical shapes")
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / max(union, 1))


def silhouette_evidence(
    points_world: np.ndarray,
    views: Sequence[Mapping[str, Any]],
    object_dir: Path,
    *,
    size: int,
    project_world: Any,
    snapshot_mask_native: Any,
) -> SilhouetteEvidence:
    values: list[float] = []
    for view in views:
        target = _target_mask_small(snapshot_mask_native(object_dir, dict(view)), size)
        projected = _projected_mask_small(
            points_world, view, size=size, project_world=project_world
        )
        values.append(binary_iou(projected, target))
    return SilhouetteEvidence(float(np.median(values)), tuple(values))


def _orientation_sheet(
    object_dir: Path,
    views: Sequence[Mapping[str, Any]],
    points_world: np.ndarray,
    per_view_iou: Sequence[float],
    *,
    project_world: Any,
    contact_sheet: Any,
) -> None:
    images: list[Image.Image] = []
    labels: list[str] = []
    for view, iou in zip(views, per_view_iou, strict=True):
        rank = int(view["rank"])
        with Image.open(object_dir / "snapshots" / str(rank) / "image-source.png") as source:
            canvas = source.convert("RGB")
        with Image.open(object_dir / "snapshots" / str(rank) / "mask.png") as mask_image:
            mask = mask_image.convert("L").resize(canvas.size, Image.Resampling.NEAREST)
        tint = Image.new("RGB", canvas.size, (255, 40, 120))
        canvas = Image.composite(Image.blend(canvas, tint, 0.30), canvas, mask)
        draw = ImageDraw.Draw(canvas)
        projected, _ = project_world(points_world, dict(view))
        crop = np.asarray(view["crop_xyxy_2048"], dtype=np.float64)
        side = np.maximum(crop[2:] - crop[:2], 1.0)
        local = (projected - crop[:2]) * np.asarray([canvas.width, canvas.height]) / side
        inside = (
            (local[:, 0] >= 0)
            & (local[:, 0] < canvas.width)
            & (local[:, 1] >= 0)
            & (local[:, 1] < canvas.height)
        )
        for x, y in local[inside][:: max(1, int(np.count_nonzero(inside) // 5000))]:
            draw.point((float(x), float(y)), fill=(20, 230, 255))
        draw.rectangle((0, 0, min(canvas.width, 380), 34), fill=(5, 10, 18))
        draw.text((8, 8), f"full-mask silhouette IoU {iou:.3f}", fill=(240, 245, 250))
        images.append(canvas)
        labels.append(f"view {rank} | {view['source_image']}")
    contact_sheet(images, labels, object_dir / "metric/orientation-silhouette-qa.jpg")


def _load_helpers(splatica_root: Path) -> dict[str, Any]:
    scripts = splatica_root.expanduser().resolve(strict=True) / "scripts"
    if not scripts.is_dir():
        raise ValueError(f"splatica_demo_app scripts directory is missing: {scripts}")
    sys.path.insert(0, str(scripts))
    from farm_recon_batch import _contact_sheet, _json, _mesh_from, _ply_table, _sha256, _write_json
    from farm_recon_geometry import (
        bbox_metrics,
        binary_bbox,
        projected_bbox,
        project_world,
        refine_scale_translation,
        snapshot_mask_native,
        world_to_y_up_rotation,
    )
    from farm_recon_refine import _atomic_export, _view_overlay
    from farm_recon_world_ply import combine

    return locals()


def _source_ply_sample(
    path: Path, *, source_center: np.ndarray, maximum: int, ply_table: Any
) -> np.ndarray:
    count, offset, dtype = ply_table(path)
    source = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))
    indices = (
        np.arange(count, dtype=np.int64)
        if count <= maximum
        else np.linspace(0, count - 1, maximum, dtype=np.int64)
    )
    rows = source[indices]
    points = np.column_stack((rows["x"], rows["y"], rows["z"])).astype(np.float64)
    del source
    if not np.isfinite(points).all():
        raise ValueError(f"non-finite reconstruction PLY centres: {path}")
    return points


def refine(
    root: Path,
    splatica_root: Path,
    *,
    silhouette_size: int,
    maximum_splats: int,
    candidate_steps: int,
    silhouette_tolerance: float,
    minimum_angle_degrees: float,
    dry_run: bool,
) -> dict[str, Any]:
    import trimesh

    helpers = _load_helpers(splatica_root)
    load_json = helpers["_json"]
    sha256 = helpers["_sha256"]
    write_json = helpers["_write_json"]
    mesh_from = helpers["_mesh_from"]
    ply_table = helpers["_ply_table"]
    project_world = helpers["project_world"]
    snapshot_mask_native = helpers["snapshot_mask_native"]
    refine_scale_translation = helpers["refine_scale_translation"]
    world_to_y_up_rotation = helpers["world_to_y_up_rotation"]
    projected_bbox = helpers["projected_bbox"]
    binary_bbox = helpers["binary_bbox"]
    bbox_metrics = helpers["bbox_metrics"]
    atomic_export = helpers["_atomic_export"]
    view_overlay = helpers["_view_overlay"]
    contact_sheet = helpers["_contact_sheet"]
    combine_world_ply = helpers["combine"]

    root = root.expanduser().resolve(strict=True)
    marker_path = root / "_SUCCESS.json"
    manifest_path = root / "manifest.json"
    marker = load_json(marker_path)
    manifest = load_json(manifest_path)
    if marker.get("status") != "PASS" or marker.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("reconstruction completion marker does not bind manifest.json")
    rows = list(manifest.get("objects") or [])
    assembly_rows = list((manifest.get("assembly") or {}).get("objects") or [])
    if len(rows) != 21 or len(assembly_rows) != 21:
        raise ValueError("orientation refinement expects the complete 21-object deliverable")
    current_records = {int(row["id"]): row for row in assembly_rows}
    run = Path(str(manifest["farm_run"])).expanduser().resolve(strict=True)
    context_path = run / "input/resolved_context.json"
    context = load_json(context_path)
    resolved_up = _unit(context["resolved_up"], name="FARM resolved_up")
    export_rotation = world_to_y_up_rotation(resolved_up)
    export_matrix = np.eye(4, dtype=np.float64)
    export_matrix[:3, :3] = export_rotation

    stage = root / ".orientation-stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    fractions = np.linspace(0.0, 1.0, int(candidate_steps) + 1)
    planned: list[dict[str, Any]] = []
    started = time.perf_counter()

    for position, row in enumerate(rows, start=1):
        object_id = int(row["id"])
        object_dir = root / str(row["slug"])
        reconstruction = load_json(object_dir / "reconstruction/result.json")
        artifacts = reconstruction.get("artifacts") or {}
        mesh_path = object_dir / str(artifacts["glb_path"])
        ply_path = object_dir / str(artifacts["ply_path"])
        params_path = object_dir / str(artifacts["params_path"])
        mesh = mesh_from(mesh_path)
        source_vertices = np.asarray(mesh.vertices, dtype=np.float64)
        previous = current_records[object_id]
        old_transform = previous["transform"]
        source_center = np.asarray(old_transform["source_center"], dtype=np.float64)
        current_linear = np.asarray(old_transform["row_linear_transform"], dtype=np.float64)
        current_center = np.asarray(old_transform["world_center_m"], dtype=np.float64)
        source_sample = _source_ply_sample(
            ply_path, source_center=source_center, maximum=maximum_splats, ply_table=ply_table
        )
        pose_up = pose_applied_canonical_up(params_path)
        current_up = _unit(pose_up @ current_linear, name="current reconstruction up")
        angle_before = vector_angle_degrees(current_up, resolved_up)
        full_correction = minimal_row_rotation(current_up, resolved_up)
        views = list(row["views"])

        candidate_scores: list[float] = []
        for fraction in fractions:
            correction = fractional_row_rotation(full_correction, float(fraction))
            candidate_linear = current_linear @ correction
            candidate_world = (source_sample - source_center) @ candidate_linear + current_center
            candidate_scores.append(
                silhouette_evidence(
                    candidate_world,
                    views,
                    object_dir,
                    size=silhouette_size,
                    project_world=project_world,
                    snapshot_mask_native=snapshot_mask_native,
                ).median_iou
            )
        fraction = choose_correction_fraction(
            fractions,
            candidate_scores,
            angle_degrees=angle_before,
            tolerance=silhouette_tolerance,
            minimum_angle_degrees=minimum_angle_degrees,
        )
        correction = fractional_row_rotation(full_correction, fraction)
        oriented_linear = current_linear @ correction
        oriented_world = (source_vertices - source_center) @ oriented_linear + current_center
        oriented_sample = (source_sample - source_center) @ oriented_linear + current_center
        oriented_evidence = silhouette_evidence(
            oriented_sample,
            views,
            object_dir,
            size=silhouette_size,
            project_world=project_world,
            snapshot_mask_native=snapshot_mask_native,
        )

        scale_result = refine_scale_translation(
            oriented_world, current_center, views, object_dir
        )
        scale_result.pop("points_world", None)
        uniform_scale = float(scale_result["uniform_scale"])
        translation = np.asarray(scale_result["translation_delta_m"], dtype=np.float64)
        refined_linear = oriented_linear * uniform_scale
        refined_center = current_center + translation
        refined_sample = (source_sample - source_center) @ refined_linear + refined_center
        refined_evidence = silhouette_evidence(
            refined_sample,
            views,
            object_dir,
            size=silhouette_size,
            project_world=project_world,
            snapshot_mask_native=snapshot_mask_native,
        )
        scale_mask_accepted = (
            bool(scale_result["accepted"])
            and refined_evidence.median_iou + silhouette_tolerance >= oriented_evidence.median_iou
        )
        if not scale_mask_accepted:
            uniform_scale = 1.0
            translation = np.zeros(3, dtype=np.float64)
            refined_linear = oriented_linear
            refined_center = current_center
            refined_evidence = oriented_evidence
        world_vertices = (source_vertices - source_center) @ refined_linear + refined_center
        final_up = _unit(pose_up @ refined_linear, name="refined reconstruction up")
        angle_after = vector_angle_degrees(final_up, resolved_up)

        per_view: list[dict[str, Any]] = []
        overlays: list[Image.Image] = []
        for view, silhouette_iou in zip(views, refined_evidence.per_view_iou, strict=True):
            metrics = bbox_metrics(
                projected_bbox(world_vertices, view),
                binary_bbox(snapshot_mask_native(object_dir, view)),
            )
            record = {
                "rank": int(view["rank"]),
                "source_image": str(view["source_image"]),
                "silhouette_iou": float(silhouette_iou),
                **metrics,
            }
            per_view.append(record)
            overlays.append(view_overlay(object_dir, view, world_vertices, metrics))
        medians = {
            key: float(np.median([item[key] for item in per_view]))
            for key in ("silhouette_iou", "bbox_iou", "center_error_px", "linear_area_ratio")
        }
        quality_reasons: list[str] = []
        if str(row.get("surface_status")) != "surface_pass":
            quality_reasons.append("farm_surface_support_not_pass")
        if medians["silhouette_iou"] < 0.40:
            quality_reasons.append("median_full_mask_silhouette_iou_below_0_40")
        if medians["bbox_iou"] < 0.58:
            quality_reasons.append("median_reverse_projection_bbox_iou_below_0_58")
        quality_status = "PASS" if not quality_reasons else "REVIEW"

        transform = {
            **old_transform,
            "method": "farm_gravity_plus_full_mask_silhouette_v4",
            "row_linear_transform": refined_linear.tolist(),
            "world_center_m": refined_center.tolist(),
            "uniform_refinement_scale": uniform_scale,
            "translation_refinement_m": translation.tolist(),
            "farm_world_to_gltf_y_up": export_matrix.tolist(),
            "resolved_up": resolved_up.tolist(),
            "orientation_refinement": {
                "schema": SCHEMA,
                "canonical_up_before": current_up.tolist(),
                "angle_before_degrees": angle_before,
                "accepted_fraction": fraction,
                "angle_after_degrees": angle_after,
                "candidate_fractions": fractions.tolist(),
                "candidate_median_silhouette_iou": candidate_scores,
                "median_silhouette_iou_before": candidate_scores[0],
                "median_silhouette_iou_after": refined_evidence.median_iou,
                "silhouette_size": silhouette_size,
                "maximum_sampled_splats": maximum_splats,
                "selection_tolerance": silhouette_tolerance,
                "minimum_corrected_angle_degrees": minimum_angle_degrees,
                "scale_translation_mask_accepted": scale_mask_accepted,
            },
        }
        result = {
            "id": object_id,
            "category": row["category"],
            "slug": row["slug"],
            "status": quality_status,
            "quality_reasons": quality_reasons,
            "metric_glb": str(object_dir.relative_to(root) / "metric/model_metric.glb"),
            "vertices": int(len(world_vertices)),
            "faces": int(len(mesh.faces)),
            "transform": transform,
            "refinement": {
                **scale_result,
                "accepted": bool(scale_mask_accepted),
                "uniform_scale": uniform_scale,
                "translation_delta_m": translation.tolist(),
            },
            "reverse_projection": {"summary": medians, "views": per_view},
            "qa_contact_sheet": str(object_dir.relative_to(root) / "metric/reverse-projection-qa.jpg"),
            "orientation_qa_contact_sheet": str(object_dir.relative_to(root) / "metric/orientation-silhouette-qa.jpg"),
        }
        planned.append(
            {
                "row": row,
                "object_dir": object_dir,
                "mesh": mesh,
                "world_vertices": world_vertices,
                "result": result,
                "overlays": overlays,
                "per_view": per_view,
                "sample_world": refined_sample,
                "evidence": refined_evidence,
            }
        )
        print(
            f"ORIENT {position}/{len(rows)} object {object_id}: "
            f"angle {angle_before:.1f}->{angle_after:.1f} deg; "
            f"fraction={fraction:.1f}; silhouette {candidate_scores[0]:.3f}->{refined_evidence.median_iou:.3f}; "
            f"{quality_status}",
            flush=True,
        )

    summary = {
        "schema": SCHEMA,
        "status": "PASS",
        "dry_run": bool(dry_run),
        "objects": [
            {
                "id": int(item["row"]["id"]),
                "category": item["row"]["category"],
                **item["result"]["transform"]["orientation_refinement"],
            }
            for item in planned
        ],
    }
    if dry_run:
        shutil.rmtree(stage)
        return summary

    marker_path.unlink(missing_ok=True)
    scene = trimesh.Scene()
    farm_vertices: list[np.ndarray] = []
    export_vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_ids: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    face_offset = 0
    for item in planned:
        row = item["row"]
        object_id = int(row["id"])
        object_dir = item["object_dir"]
        mesh = item["mesh"]
        world_vertices = item["world_vertices"]
        export_mesh = mesh.copy()
        export_mesh.vertices = world_vertices @ export_rotation.T
        staged_glb = stage / f"object_{object_id:06d}.glb"
        atomic_export(export_mesh, staged_glb)
        final_glb = object_dir / "metric/model_metric.glb"
        os.replace(staged_glb, final_glb)
        result = item["result"]
        result["metric_glb_sha256"] = sha256(final_glb)
        write_json(object_dir / "metric/result.json", result)
        contact_sheet(
            item["overlays"],
            [f"view {record['rank']} | {record['source_image']}" for record in item["per_view"]],
            object_dir / "metric/reverse-projection-qa.jpg",
        )
        _orientation_sheet(
            object_dir,
            list(row["views"]),
            item["sample_world"],
            item["evidence"].per_view_iou,
            project_world=project_world,
            contact_sheet=contact_sheet,
        )
        records.append(result)
        scene.add_geometry(
            export_mesh,
            geom_name=f"object_{object_id:06d}_{row['category']}",
            node_name=f"object_{object_id:06d}",
        )
        farm = np.asarray(world_vertices, dtype=np.float32)
        exported = np.asarray(export_mesh.vertices, dtype=np.float32)
        local_faces = np.asarray(mesh.faces, dtype=np.int64)
        farm_vertices.append(farm)
        export_vertices.append(exported)
        faces.append(local_faces + face_offset)
        vertex_ids.append(np.full(len(farm), object_id, dtype=np.int32))
        face_offset += len(farm)

    combined_stage = stage / "combined_metric_objects.glb"
    scene.export(combined_stage, file_type="glb")
    os.replace(combined_stage, root / "combined_metric_objects.glb")
    all_faces = np.concatenate(faces)
    all_ids = np.concatenate(vertex_ids)
    np.savez_compressed(
        stage / "combined_metric_objects.npz",
        vertices=np.concatenate(export_vertices),
        faces=all_faces,
        vertex_object_id=all_ids,
        farm_world_to_gltf_y_up=export_matrix,
    )
    np.savez_compressed(
        stage / "combined_world_metric_objects.npz",
        vertices=np.concatenate(farm_vertices),
        faces=all_faces,
        vertex_object_id=all_ids,
    )
    os.replace(stage / "combined_metric_objects.npz", root / "combined_metric_objects.npz")
    os.replace(stage / "combined_world_metric_objects.npz", root / "combined_world_metric_objects.npz")

    pass_count = sum(record["status"] == "PASS" for record in records)
    review_count = len(records) - pass_count
    before_values = [
        record["transform"]["orientation_refinement"]["median_silhouette_iou_before"]
        for record in records
    ]
    after_values = [record["reverse_projection"]["summary"]["silhouette_iou"] for record in records]
    assembly = {
        "status": "PASS" if review_count == 0 else "WARN",
        "quality_status": "PASS" if review_count == 0 else "WARN",
        "object_count": len(records),
        "pass_objects": pass_count,
        "review_objects": review_count,
        "median_silhouette_iou_before": float(np.median(before_values)),
        "median_silhouette_iou_after": float(np.median(after_values)),
        "objects": records,
        "combined_glb": "combined_metric_objects.glb",
        "combined_glb_sha256": sha256(root / "combined_metric_objects.glb"),
        "combined_npz": "combined_metric_objects.npz",
        "combined_npz_sha256": sha256(root / "combined_metric_objects.npz"),
        "combined_farm_world_npz": "combined_world_metric_objects.npz",
        "combined_farm_world_npz_sha256": sha256(root / "combined_world_metric_objects.npz"),
        "coordinate_contract": {
            "FARM_world": "metric source-scene frame used by T_world_cam",
            "glTF": "right-handed +Y-up rigid rotation of FARM_world",
            "farm_world_to_gltf_y_up": export_matrix.tolist(),
            "resolved_context": str(context_path),
            "resolved_context_sha256": sha256(context_path),
        },
        "orientation_refinement": {
            "schema": SCHEMA,
            "policy": "minimal gravity correction selected by real PINHOLE full-mask silhouette IoU",
            "silhouette_size": silhouette_size,
            "maximum_sampled_splats": maximum_splats,
            "candidate_steps": candidate_steps,
            "silhouette_tolerance": silhouette_tolerance,
            "minimum_angle_degrees": minimum_angle_degrees,
        },
        "duration_seconds": time.perf_counter() - started,
        "completed_unix_s": time.time(),
    }
    manifest["assembly"] = assembly
    manifest["status"] = assembly["status"]
    write_json(manifest_path, manifest)
    write_json(
        marker_path,
        {
            "schema": "splatica.farm-mv-sam3d.v1",
            "status": "PASS",
            "quality_status": assembly["quality_status"],
            "manifest_sha256": sha256(manifest_path),
        },
    )

    combine_world_ply(
        root,
        root / "combined_world_metric_gaussians.ply",
        50_000,
    )
    marker_path.unlink(missing_ok=True)
    combine_world_ply(
        root,
        root / "combined_viewer_y_up_gaussians.ply",
        50_000,
        export_rotation=export_rotation,
    )
    manifest = load_json(manifest_path)
    manifest["assembly"]["orientation_refinement_report"] = {
        "path": "orientation_refinement.json",
        "sha256": "pending",
    }
    summary["duration_seconds"] = time.perf_counter() - started
    write_json(root / "orientation_refinement.json", summary)
    manifest["assembly"]["orientation_refinement_report"]["sha256"] = sha256(
        root / "orientation_refinement.json"
    )
    write_json(manifest_path, manifest)
    write_json(
        marker_path,
        {
            "schema": "splatica.farm-mv-sam3d.v1",
            "status": "PASS",
            "quality_status": manifest["assembly"]["quality_status"],
            "manifest_sha256": sha256(manifest_path),
        },
    )
    shutil.rmtree(stage)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recon", type=Path, required=True)
    parser.add_argument(
        "--splatica-root",
        type=Path,
        default=Path("/home/splatica/workspace/splatica_demo_app"),
    )
    parser.add_argument("--silhouette-size", type=int, default=256)
    parser.add_argument("--maximum-splats", type=int, default=150_000)
    parser.add_argument("--candidate-steps", type=int, default=10)
    parser.add_argument("--silhouette-tolerance", type=float, default=0.005)
    parser.add_argument("--minimum-angle-degrees", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.silhouette_size < 64 or args.maximum_splats < 1_000 or args.candidate_steps < 2:
        parser.error("silhouette/sample/candidate limits are too small")
    result = refine(
        args.recon,
        args.splatica_root,
        silhouette_size=args.silhouette_size,
        maximum_splats=args.maximum_splats,
        candidate_steps=args.candidate_steps,
        silhouette_tolerance=args.silhouette_tolerance,
        minimum_angle_degrees=args.minimum_angle_degrees,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
