#!/usr/bin/env python3
"""Build category-agnostic parent assemblies from repeated part/whole tracks.

Accepted relations come from :mod:`analyze_farm_part_whole`; object names never
participate.  Each parent receives a gravity-aligned OBB fitted to the union of
its FARM voxel evidence and a multi-view crop bank for independent VLM review.
The source state is copied, never overwritten.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.geometry.farm_geometry_axes import (
        add_up_arguments,
        horizontal_plane_basis,
        normalize_up_vector,
        policy_from_args,
        write_state_up_policy,
    )
except ModuleNotFoundError:  # package import: scripts.geometry.build_farm_object_assemblies
    from scripts.geometry.farm_geometry_axes import (
        add_up_arguments,
        horizontal_plane_basis,
        normalize_up_vector,
        policy_from_args,
        write_state_up_policy,
    )
from scene_graph.utils.geometry import (
    decode_voxel_keys_numpy,
    merge_voxel_buffers,
)


def _numpy(value: object, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray([0.25 * scale, (m[2, 1] - m[1, 2]) / scale, (m[0, 2] - m[2, 0]) / scale, (m[1, 0] - m[0, 1]) / scale])
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            scale = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1.0e-12)) * 2.0
            quat = np.asarray([(m[2, 1] - m[1, 2]) / scale, 0.25 * scale, (m[0, 1] + m[1, 0]) / scale, (m[0, 2] + m[2, 0]) / scale])
        elif axis == 1:
            scale = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1.0e-12)) * 2.0
            quat = np.asarray([(m[0, 2] - m[2, 0]) / scale, (m[0, 1] + m[1, 0]) / scale, 0.25 * scale, (m[1, 2] + m[2, 1]) / scale])
        else:
            scale = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1.0e-12)) * 2.0
            quat = np.asarray([(m[1, 0] - m[0, 1]) / scale, (m[0, 2] + m[2, 0]) / scale, (m[1, 2] + m[2, 1]) / scale, 0.25 * scale])
    quat /= max(float(np.linalg.norm(quat)), 1.0e-12)
    if quat[0] < 0.0:
        quat *= -1.0
    return quat.astype(np.float32)


def _fit_gravity_obb(
    points: np.ndarray,
    quantile: float,
    *,
    up_axis_index: int = 1,
    up_vector: object | None = None,
) -> dict:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 24:
        raise ValueError("Assembly has fewer than 24 finite voxel points")
    origin = np.median(points, axis=0)
    centered = points - origin
    if up_vector is None:
        up = np.zeros(3, dtype=np.float64)
        up[int(up_axis_index)] = 1.0
    else:
        up = normalize_up_vector(up_vector)
    plane_basis = horizontal_plane_basis(up)
    floor_points = centered @ plane_basis
    covariance = floor_points.T @ floor_points / max(floor_points.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    major_2d = eigenvectors[:, order[0]]
    horizontal_major = plane_basis @ major_2d
    major = int(np.argmax(np.abs(horizontal_major)))
    if horizontal_major[major] < 0.0:
        horizontal_major *= -1.0
    horizontal_major /= max(float(np.linalg.norm(horizontal_major)), 1.0e-12)
    horizontal_minor = np.cross(up, horizontal_major)
    horizontal_minor /= max(float(np.linalg.norm(horizontal_minor)), 1.0e-12)
    basis = np.column_stack((horizontal_major, horizontal_minor, up))
    if np.linalg.det(basis) < 0.0:
        basis[:, 1] *= -1.0
    local = centered @ basis
    q = float(np.clip(quantile, 0.0, 0.20))
    lo, hi = np.quantile(local, [q, 1.0 - q], axis=0)
    dimensions = np.maximum(hi - lo, 0.08)
    padding = np.maximum(0.025, 0.025 * dimensions)
    dimensions += 2.0 * padding
    local_center = 0.5 * (lo + hi)
    center = origin + basis @ local_center
    inside = np.all(np.abs((points - center) @ basis) <= 0.5 * dimensions + 1.0e-6, axis=1)
    confidence = float(np.clip((eigenvalues[0] - eigenvalues[1]) / max(float(eigenvalues[0]), 1.0e-12), 0.0, 1.0))
    sigma = dimensions / 5.0
    covariance_world = basis @ np.diag(sigma * sigma) @ basis.T
    cov6 = np.asarray(
        [covariance_world[0, 0], covariance_world[0, 1], covariance_world[0, 2], covariance_world[1, 1], covariance_world[1, 2], covariance_world[2, 2]],
        dtype=np.float32,
    )
    return {
        "center": center.astype(np.float32),
        "dimensions": dimensions.astype(np.float32),
        "wxyz": _matrix_to_wxyz(basis),
        "rotation_matrix": basis.astype(np.float32),
        "up_vector": up.astype(np.float32),
        "cov6": cov6,
        "voxel_inside_rate": float(inside.mean()),
        "orientation_confidence": confidence,
    }


def _components(relations: list[dict]) -> list[tuple[list[int], list[dict]]]:
    accepted = [
        row
        for row in relations
        if isinstance(row, dict)
        and (bool(row.get("accepted")) or str(row.get("relation_type")) == "duplicate_overlap")
    ]
    adjacency: dict[int, set[int]] = {}
    for row in accepted:
        a, b = int(row["object_a"]), int(row["object_b"])
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    seen: set[int] = set()
    out: list[tuple[list[int], list[dict]]] = []
    for root in sorted(adjacency):
        if root in seen:
            continue
        stack, members = [root], set()
        while stack:
            node = stack.pop()
            if node in members:
                continue
            members.add(node)
            stack.extend(adjacency.get(node, ()))
        seen.update(members)
        rows = [row for row in accepted if int(row["object_a"]) in members and int(row["object_b"]) in members]
        out.append((sorted(members), rows))
    return out


def _tier(count: int) -> str:
    return "stable" if count >= 10 else "robust" if count >= 5 else "multi-view" if count >= 3 else "tentative"


def canonical_member_tuple(value: object) -> tuple[int, ...]:
    """Stable provenance key for an assembly, independent of transient IDs."""
    if value is None:
        return ()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (str, bytes)):
        return ()
    try:
        return tuple(sorted({int(member) for member in value}))
    except (TypeError, ValueError):
        return ()


def _review_map(path: Path | None) -> dict[tuple[int, ...], dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("objects") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"No review object list in {path}")
    mapped: dict[tuple[int, ...], dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = canonical_member_tuple(row.get("member_object_ids"))
        if key:
            mapped[key] = row
    return mapped



def reviewed_assembly_eligible(review: object) -> tuple[bool, str]:
    """Require an explicit positive multi-view review before reviewed materialisation."""

    if not isinstance(review, dict):
        return False, "missing_review"
    decision = str(review.get("review_decision") or "").strip().lower()
    category = str(review.get("review_category") or "").strip()
    description = str(review.get("review_description") or "").strip()
    if decision not in {"keep", "relabel"}:
        return False, f"review_decision:{decision or 'missing'}"
    if not category or category.lower() == "unknown":
        return False, "review_category_missing"
    if not description:
        return False, "review_description_missing"
    contract = review.get("review_label_contract")
    if not isinstance(contract, dict) or contract.get("contract_valid") is not True:
        return False, "review_label_contract_invalid"
    if contract.get("complete_bounded") is not True:
        return False, "review_not_complete_bounded"
    if str(contract.get("topology") or "") not in {
        "standalone_whole", "carrier_payload"
    }:
        return False, "review_not_whole_topology"
    return True, "reviewed_multi_view_whole"


def _unpack_mask_canvas(data: np.lib.npyio.NpzFile, kind: str, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    canvas = np.zeros((height, width), dtype=bool)
    if f"{kind}_bits" not in data.files:
        return canvas
    shape = np.asarray(data[f"{kind}_shape"], dtype=np.int32).reshape(2)
    bbox = np.asarray(data[f"{kind}_bbox_xyxy"], dtype=np.int32).reshape(4)
    mask_height, mask_width = int(shape[0]), int(shape[1])
    flat = np.unpackbits(np.asarray(data[f"{kind}_bits"], dtype=np.uint8), bitorder="little")
    mask = flat[: mask_height * mask_width].reshape(mask_height, mask_width).astype(bool, copy=False)
    x0, y0, x1, y1 = [int(value) for value in bbox]
    paste_height = min(mask_height, max(0, y1 - y0), max(0, height - y0))
    paste_width = min(mask_width, max(0, x1 - x0), max(0, width - x0))
    if paste_height > 0 and paste_width > 0:
        canvas[y0 : y0 + paste_height, x0 : x0 + paste_width] |= mask[:paste_height, :paste_width]
    return canvas


def _packed_crop(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        bbox = np.asarray([0, 0, 0, 0], dtype=np.int32)
        return np.zeros((0,), dtype=np.uint8), np.asarray([0, 0], dtype=np.int32), bbox
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = mask[y0:y1, x0:x1]
    bits = np.packbits(crop.reshape(-1).astype(np.uint8), bitorder="little")
    return bits, np.asarray(crop.shape, dtype=np.int32), np.asarray([x0, y0, x1, y1], dtype=np.int32)


def _merge_mask_files(source_paths: list[Path], destination: Path) -> dict[str, object]:
    """Union member masks in image space and retain one crop for semantics."""
    if not source_paths:
        raise ValueError("No member masks to merge")
    image_shape: tuple[int, int] | None = None
    raw_canvas: np.ndarray | None = None
    inlier_canvas: np.ndarray | None = None
    crop_payload: dict[str, np.ndarray] | None = None
    for source_path in source_paths:
        with np.load(source_path, allow_pickle=False) as data:
            shape = tuple(int(value) for value in np.asarray(data["image_shape"], dtype=np.int32).reshape(2))
            if image_shape is None:
                image_shape = shape
                raw_canvas = np.zeros(shape, dtype=bool)
                inlier_canvas = np.zeros(shape, dtype=bool)
            if shape != image_shape:
                raise ValueError(f"Inconsistent image shapes in assembly masks: {image_shape} vs {shape}")
            raw_canvas |= _unpack_mask_canvas(data, "raw", image_shape)
            inlier_canvas |= _unpack_mask_canvas(data, "inlier", image_shape)
            crop_bytes = np.asarray(data.get("crop_jpeg_bytes", np.zeros((0,), dtype=np.uint8)), dtype=np.uint8)
            if crop_payload is None or crop_bytes.size > crop_payload["crop_jpeg_bytes"].size:
                crop_payload = {
                    "crop_jpeg_bytes": crop_bytes.copy(),
                    "crop_bbox_xyxy": np.asarray(data.get("crop_bbox_xyxy", [0, 0, 0, 0]), dtype=np.int32).copy(),
                    "crop_shape": np.asarray(data.get("crop_shape", [0, 0]), dtype=np.int32).copy(),
                }
    assert image_shape is not None and raw_canvas is not None and inlier_canvas is not None
    raw_bits, raw_shape, raw_bbox = _packed_crop(raw_canvas)
    inlier_bits, inlier_shape, inlier_bbox = _packed_crop(inlier_canvas)
    crop_payload = crop_payload or {
        "crop_jpeg_bytes": np.zeros((0,), dtype=np.uint8),
        "crop_bbox_xyxy": np.zeros((4,), dtype=np.int32),
        "crop_shape": np.zeros((2,), dtype=np.int32),
    }
    np.savez_compressed(
        destination,
        image_shape=np.asarray(image_shape, dtype=np.int32),
        raw_bits=raw_bits,
        raw_shape=raw_shape,
        raw_bbox_xyxy=raw_bbox,
        inlier_bits=inlier_bits,
        inlier_shape=inlier_shape,
        inlier_bbox_xyxy=inlier_bbox,
        **crop_payload,
    )
    return {
        "image_shape": [int(image_shape[0]), int(image_shape[1])],
        "raw_pixels": int(raw_canvas.sum(dtype=np.int64)),
        "inlier_pixels": int(inlier_canvas.sum(dtype=np.int64)),
        "crop_jpeg_bytes_len": int(crop_payload["crop_jpeg_bytes"].size),
        "crop_bbox_xyxy": np.asarray(
            crop_payload["crop_bbox_xyxy"], dtype=np.int64
        ).reshape(4).tolist(),
        "crop_shape": np.asarray(
            crop_payload["crop_shape"], dtype=np.int64
        ).reshape(2).tolist(),
    }


def _copy_member_review_masks(
    candidates: list[tuple[dict, int, Path]], destination_dir: Path, image_id: int
) -> int:
    """Copy one crop per source member without altering union geometry masks."""
    by_member: dict[int, list[Path]] = {}
    for _, member_id, source_path in candidates:
        by_member.setdefault(int(member_id), []).append(source_path)
    copied = 0
    for member_id, paths in sorted(by_member.items()):
        source = max(paths, key=lambda path: path.stat().st_size)
        destination = destination_dir / (
            f"img_{int(image_id):06d}_member_{int(member_id):06d}.npz"
        )
        shutil.copyfile(source, destination)
        copied += 1
    return copied


def _append_tensor(state: dict, key: str, values: np.ndarray | list, *, dtype: torch.dtype | None = None) -> None:
    current = state.get(key)
    if not isinstance(current, torch.Tensor):
        return
    addition = torch.as_tensor(values, dtype=dtype or current.dtype)
    state[key] = torch.cat((current, addition), dim=0)


def _ensure_list(state: dict, key: str, count: int, factory: object) -> list:
    value = state.get(key)
    if not isinstance(value, list):
        value = list(value) if isinstance(value, tuple) else []
    while len(value) < count:
        value.append(factory() if callable(factory) else copy.deepcopy(factory))
    state[key] = value
    return value


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--part-whole-report", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-catalog", type=Path, required=True)
    parser.add_argument("--output-mask-root", type=Path, required=True)
    parser.add_argument(
        "--output-review-mask-root", type=Path,
        help="Optional member-balanced crop bank for VLM review; union masks remain geometry-only.",
    )
    parser.add_argument("--review-report", type=Path)
    parser.add_argument("--require-reviewed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quantile", type=float, default=0.025)
    parser.add_argument("--voxel-cap", type=int, default=2000)
    parser.add_argument("--max-assembly-members", type=int, default=6)
    add_up_arguments(parser, default_axis="y")
    args = parser.parse_args()
    if args.max_assembly_members < 2:
        parser.error("--max-assembly-members must be at least 2")
    up_policy = policy_from_args(args)

    source = args.scene_state.expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
        raise TypeError(f"Unsupported scene state: {source}")
    payload = copy.deepcopy(payload)
    state = payload["state"]
    ids = _numpy(state["object_id"], np.int64).reshape(-1)
    count = int(ids.size)
    id_to_index = {int(object_id): index for index, object_id in enumerate(ids.tolist())}
    part_whole = json.loads(args.part_whole_report.read_text(encoding="utf-8"))
    groups = _components(part_whole.get("relations") or [])
    reviews = _review_map(args.review_report.expanduser().resolve() if args.review_report else None)

    flat = state["object_voxel_keys_flat"].detach().cpu()
    offsets = _numpy(state["object_voxel_keys_offsets"], np.int64)
    levels = _numpy(state["object_voxel_levels"], np.int64)
    features = state["features"].detach().cpu()
    observations_all = state.get("object_mask_observations") or []
    image_ids_all = state.get("object_image_ids") or []
    camera_distances = _numpy(state.get("object_camera_distance_m", np.full(count, np.inf)), np.float32)

    output_mask_root = args.output_mask_root.expanduser().resolve()
    output_mask_root.mkdir(parents=True, exist_ok=True)
    review_mask_root = (
        args.output_review_mask_root.expanduser().resolve()
        if args.output_review_mask_root else None
    )
    if review_mask_root is not None:
        review_mask_root.mkdir(parents=True, exist_ok=True)
    if "object_assembly_containment" not in state:
        state["object_assembly_containment"] = torch.zeros((count,), dtype=torch.float32)
    if "object_assembly_shared_frames" not in state:
        state["object_assembly_shared_frames"] = torch.zeros((count,), dtype=torch.int32)
    new_ids: list[int] = []
    rows: list[dict] = []
    catalogs: list[dict] = []
    merged_keys_list: list[torch.Tensor] = []
    merged_levels: list[int] = []
    next_id = int(ids.max(initial=-1)) + 1

    tensor_values: dict[str, list] = {
        "count": [], "means": [], "cov6": [], "features": [], "active": [], "object_id": [], "class_ids": [],
        "object_evidence_count": [], "object_camera_distance_m": [], "object_review_confidence": [],
        "object_box_centers_m": [], "object_box_dimensions_m": [], "object_box_wxyz": [],
        "object_geometry_inside_rate": [], "object_geometry_reprojection_error": [],
        "object_geometry_valid_observations": [], "object_geometry_consistent_observations": [],
        "object_geometry_projected_box_iou": [], "object_geometry_box_support_rate": [],
        "object_geometry_voxel_inside_rate": [], "object_geometry_voxel_volume_expansion": [],
        "object_geometry_orientation_confidence": [], "object_geometry_gravity_tilt_degrees": [],
        "object_assembly_containment": [], "object_assembly_shared_frames": [],
    }

    list_fields: dict[str, tuple[object, list]] = {
        "object_caption": ("", []), "object_caption_decision": ("", []), "object_category": ("", []),
        "object_supercategory": ("", []), "object_category_candidates": (list, []), "object_key_attributes": (list, []),
        "object_detection_category_conf": (dict, []), "loser_object_ids": (list, []), "is_locked": (False, []),
        "high_quality_captioning": (False, []), "object_image_ids": (list, []), "viewpoint_image_ids": (list, []),
        "object_mask_observations": (list, []), "object_evidence_tier": ("", []),
        "object_semantic_status": ("", []), "object_semantic_tier": ("", []),
        "object_geometry_status": ("", []),
        "object_geometry_voxel_extent_candidate": ("", []), "object_geometry_orientation_mode": ("gravity_yaw", []),
        "object_assembly_member_ids": (list, []),
        "object_assembly_relation_type": ("", []),
    }
    for key, (default, _) in list_fields.items():
        _ensure_list(state, key, count, default)

    skipped_groups: list[dict] = []
    for members, relations in groups:
        if any(member not in id_to_index for member in members):
            continue
        member_key = canonical_member_tuple(members)
        review = reviews.get(member_key, {})
        if len(members) > int(args.max_assembly_members):
            skipped_groups.append({
                "members": members,
                "reason": "member_count_exceeds_limit",
            })
            continue
        review_eligible, review_reason = reviewed_assembly_eligible(review)
        if args.require_reviewed and not review_eligible:
            skipped_groups.append({
                "members": members,
                "reason": review_reason,
            })
            continue
        indices = [id_to_index[member] for member in members]
        relation_types = {str(row.get("relation_type") or "part_whole") for row in relations}
        assembly_relation_type = (
            "duplicate_track_consolidation"
            if relation_types == {"duplicate_overlap"}
            else "multi_view_whole_candidate"
            if "complementary_parts" in relation_types
            else "part_whole_candidate"
        )
        merged_keys = torch.empty((0,), dtype=torch.int64)
        merged_level = 0
        for member_index in indices:
            start, end = int(offsets[member_index]), int(offsets[member_index + 1])
            keys = flat[start:end].to(torch.int64)
            level = int(levels[member_index])
            merged_keys, merged_level = merge_voxel_buffers(
                merged_keys, merged_level, keys, level, cap=max(128, int(args.voxel_cap))
            )
        points = decode_voxel_keys_numpy(merged_keys.numpy(), merged_level)
        geometry = _fit_gravity_obb(
            points,
            float(args.quantile),
            up_axis_index=up_policy.axis_index,
            up_vector=up_policy.vector,
        )
        assembly_id = next_id
        next_id += 1
        new_ids.append(assembly_id)
        merged_keys_list.append(merged_keys)
        merged_levels.append(int(merged_level))

        candidate_observations: dict[int, list[tuple[dict, int, Path]]] = {}
        for member, member_index in zip(members, indices):
            for observation in observations_all[member_index] if member_index < len(observations_all) else []:
                if not isinstance(observation, dict):
                    continue
                image_id = int(observation.get("image_id", -1))
                recorded = Path(str(observation.get("path") or ""))
                source_mask = args.mask_root.expanduser().resolve() / recorded.parent.name / recorded.name
                if image_id >= 0 and source_mask.is_file():
                    candidate_observations.setdefault(image_id, []).append(
                        (copy.deepcopy(observation), int(member), source_mask)
                    )
        merged_observations: list[dict] = []
        merged_image_ids = sorted(candidate_observations)
        assembly_mask_dir = output_mask_root / f"object_{assembly_id:06d}"
        assembly_mask_dir.mkdir(parents=True, exist_ok=True)
        for stale in assembly_mask_dir.glob("*.npz"):
            stale.unlink()
        review_mask_dir = None
        if review_mask_root is not None:
            review_mask_dir = review_mask_root / f"object_{assembly_id:06d}"
            review_mask_dir.mkdir(parents=True, exist_ok=True)
            for stale in review_mask_dir.glob("*.npz"):
                stale.unlink()
        copied = 0
        review_copied = 0
        for image_id in merged_image_ids:
            candidates = candidate_observations[image_id]
            destination = assembly_mask_dir / f"img_{image_id:06d}_assembly.npz"
            mask_stats = _merge_mask_files(
                [source_path for _, _, source_path in candidates], destination
            )
            representative = max(
                candidates,
                key=lambda item: int(item[0].get("crop_jpeg_bytes_len") or 0),
            )[0]
            representative["path"] = str(destination.relative_to(output_mask_root))
            representative.update(mask_stats)
            representative["assembly_member_ids_in_frame"] = sorted(
                {int(member) for _, member, _ in candidates}
            )
            merged_observations.append(representative)
            copied += 1
            if review_mask_dir is not None:
                review_copied += _copy_member_review_masks(
                    candidates, review_mask_dir, image_id
                )

        feature_weights = torch.as_tensor(
            [max(1, len(image_ids_all[index]) if index < len(image_ids_all) else 1) for index in indices],
            dtype=torch.float32,
        )
        merged_feature = (features[indices] * feature_weights[:, None]).sum(dim=0) / feature_weights.sum()
        merged_feature = merged_feature / torch.clamp(torch.linalg.vector_norm(merged_feature), min=1.0e-12)
        category = str(review.get("review_category") or "").strip()
        description = str(review.get("review_description") or "").strip()
        decision = str(review.get("review_decision") or "unknown").strip().lower()
        confidence = float(review.get("review_confidence") or 0.0)
        resolved = bool(category and category.lower() != "unknown" and description and decision in {"keep", "relabel"})
        # An assembly is provisional until a separate RGB-D geometry pass.
        # Semantic review must never activate unvalidated metric geometry.
        active = False
        attributes = [str(value) for value in (review.get("review_attributes") or []) if str(value).strip()]
        observation_count = len(merged_image_ids)
        containment = float(np.median([float(row.get("median_containment") or 0.0) for row in relations]))
        shared_frames = int(sum(int(row.get("contained_frames") or 0) for row in relations))

        tensor_values["count"].append(observation_count)
        tensor_values["means"].append(geometry["center"])
        tensor_values["cov6"].append(geometry["cov6"])
        tensor_values["features"].append(merged_feature.numpy())
        tensor_values["active"].append(active)
        tensor_values["object_id"].append(assembly_id)
        tensor_values["class_ids"].append(-1)
        tensor_values["object_evidence_count"].append(observation_count)
        tensor_values["object_camera_distance_m"].append(float(np.min(camera_distances[indices])))
        tensor_values["object_review_confidence"].append(confidence)
        tensor_values["object_box_centers_m"].append(geometry["center"])
        tensor_values["object_box_dimensions_m"].append(geometry["dimensions"])
        tensor_values["object_box_wxyz"].append(geometry["wxyz"])
        tensor_values["object_geometry_inside_rate"].append(0.0)
        tensor_values["object_geometry_reprojection_error"].append(float("inf"))
        tensor_values["object_geometry_valid_observations"].append(0)
        tensor_values["object_geometry_consistent_observations"].append(0)
        tensor_values["object_geometry_projected_box_iou"].append(0.0)
        tensor_values["object_geometry_box_support_rate"].append(0.0)
        tensor_values["object_geometry_voxel_inside_rate"].append(0.0)
        tensor_values["object_geometry_voxel_volume_expansion"].append(1.0)
        tensor_values["object_geometry_orientation_confidence"].append(0.0)
        tensor_values["object_geometry_gravity_tilt_degrees"].append(0.0)
        tensor_values["object_assembly_containment"].append(containment)
        tensor_values["object_assembly_shared_frames"].append(shared_frames)

        list_fields["object_caption"][1].append(
            description if resolved
            else "Metric multi-view assembly with unresolved semantic identity."
        )
        list_fields["object_caption_decision"][1].append("keep" if resolved else "unknown")
        list_fields["object_category"][1].append(category if resolved else "unresolved assembly")
        list_fields["object_supercategory"][1].append(
            "open-vocabulary assembly" if resolved else "open-vocabulary unresolved"
        )
        list_fields["object_category_candidates"][1].append([])
        list_fields["object_key_attributes"][1].append(attributes)
        list_fields["object_detection_category_conf"][1].append({})
        list_fields["loser_object_ids"][1].append([])
        list_fields["is_locked"][1].append(False)
        list_fields["high_quality_captioning"][1].append(False)
        list_fields["object_image_ids"][1].append(merged_image_ids)
        list_fields["viewpoint_image_ids"][1].append(merged_image_ids)
        list_fields["object_mask_observations"][1].append(merged_observations)
        list_fields["object_evidence_tier"][1].append(_tier(observation_count))
        list_fields["object_semantic_status"][1].append("vlm_multi_crop_resolved" if resolved else "assembly_needs_review")
        list_fields["object_semantic_tier"][1].append("probable" if resolved else "geometry_only")
        list_fields["object_geometry_status"][1].append("assembly_geometry_pending")
        list_fields["object_geometry_voxel_extent_candidate"][1].append("assembly_union_voxels_candidate")
        list_fields["object_geometry_orientation_mode"][1].append("gravity_yaw")
        list_fields["object_assembly_member_ids"][1].append(members)
        list_fields["object_assembly_relation_type"][1].append(assembly_relation_type)

        row = {
            "id": assembly_id,
            "members": members,
            "relation_type": assembly_relation_type,
            "observations": observation_count,
            "copied_crop_observations": copied,
            "member_review_crop_observations": review_copied,
            "position_world_m": geometry["center"].round(5).tolist(),
            "dimensions_lwh_m": geometry["dimensions"].round(5).tolist(),
            "wxyz": geometry["wxyz"].round(7).tolist(),
            "voxel_count": int(merged_keys.numel()),
            "voxel_inside_obb_rate": geometry["voxel_inside_rate"],
            "geometry_status": "assembly_geometry_pending",
            "independent_geometry_validated": False,
            "orientation_confidence": geometry["orientation_confidence"],
            "up_vector": up_policy.vector.astype(float).tolist(),
            "median_mask_containment": containment,
            "contained_shared_frames": shared_frames,
            "category": category,
            "description": description,
            "review_confidence": confidence,
            "review_decision": decision if review else "not_reviewed",
            "active": False,
        }
        rows.append(row)
        catalogs.append(
            {
                "id": assembly_id,
                "observations": observation_count,
                "position_world_m": geometry["center"].round(5).tolist(),
                "cov6": geometry["cov6"].tolist(),
                "category": "",
                "caption": "",
                "member_object_ids": members,
            }
        )

    new_count = len(rows)
    if new_count:
        for key, values in tensor_values.items():
            if key in state:
                _append_tensor(state, key, np.asarray(values))
        for key, (_, values) in list_fields.items():
            state[key].extend(values)
        old_flat = state["object_voxel_keys_flat"].detach().cpu().to(torch.int64)
        new_flat = torch.cat([old_flat, *merged_keys_list])
        new_offsets = state["object_voxel_keys_offsets"].detach().cpu().to(torch.int64).tolist()
        running = int(new_offsets[-1])
        for keys in merged_keys_list:
            running += int(keys.numel())
            new_offsets.append(running)
        state["object_voxel_keys_flat"] = new_flat
        state["object_voxel_keys_offsets"] = torch.as_tensor(new_offsets, dtype=torch.int64)
        state["object_voxel_levels"] = torch.cat(
            (
                state["object_voxel_levels"].detach().cpu(),
                torch.as_tensor(merged_levels, dtype=state["object_voxel_levels"].dtype),
            )
        )
    write_state_up_policy(state, up_policy)
    payload["saved_unix_s"] = time.time()
    payload.setdefault("meta", {})["object_assemblies"] = {
        "source": str(source),
        "part_whole_report": str(args.part_whole_report.expanduser().resolve()),
        "category_agnostic": True,
        "assemblies_added": new_count,
        "assembly_object_ids": new_ids,
        "assemblies_skipped": len(skipped_groups),
        "skipped_groups": skipped_groups,
        "max_assembly_members": int(args.max_assembly_members),
        "require_reviewed": bool(args.require_reviewed),
        "up": up_policy.to_dict(),
    }
    report = {
        "schema": "farm.object-assembly.v1",
        "created_unix_s": time.time(),
        "source_scene_state": str(source),
        "category_agnostic": True,
        "up": up_policy.to_dict(),
        "assemblies_added": new_count,
        "assemblies_skipped": len(skipped_groups),
        "skipped_groups": skipped_groups,
        "max_assembly_members": int(args.max_assembly_members),
        "active_assemblies": sum(bool(row["active"]) for row in rows),
        "objects": rows,
        "timing": {"duration_seconds": time.perf_counter() - started, "stage": "part_whole_assembly"},
    }
    args.output_state.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_catalog.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_state)
    args.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_catalog.write_text(json.dumps(catalogs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("assemblies_added", "active_assemblies", "timing")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
