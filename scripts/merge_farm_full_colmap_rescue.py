#!/usr/bin/env python3
"""Merge accepted full-COLMAP SAM3 masks into a canonical direct-object state.

The script never appends rescue tracks as new objects.  It attaches only
refined, geometry-associated observations to the existing canonical object ID,
keeps original object rows intact, and builds a single hard-linked mask tree and
frames manifest for all downstream consumers.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_state(path: Path) -> tuple[Any, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state") if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"unsupported scene state: {path}")
    return payload, state


def _array(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _link_file(source: Path, destination: Path) -> None:
    resolved = source.resolve(strict=True)
    if source.is_symlink():
        raise ValueError(f"mask source must not be a symlink: {source}")
    if not resolved.is_file():
        raise ValueError(f"mask source must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    try:
        os.link(resolved, destination)
    except OSError:
        import shutil

        shutil.copy2(resolved, destination)


def _link_mask_tree(source_root: Path, destination_root: Path) -> int:
    root = source_root.resolve(strict=True)
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"mask tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file() or path.suffix.lower() != ".npz":
            raise ValueError(f"mask tree contains an unexpected file: {path}")
        relative = path.relative_to(root)
        _link_file(path, destination_root / relative)
        count += 1
    return count


def _relative_artifact(path_text: str, manifest_path: Path, output: Path) -> str:
    source = (manifest_path.parent / path_text).resolve(strict=True)
    return os.path.relpath(source, output).replace(os.sep, "/")


def _review_row(state: dict, index: int, object_id: int) -> dict[str, Any]:
    def row_value(key: str, default: Any) -> Any:
        values = state.get(key)
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().tolist()
        if isinstance(values, (list, tuple)) and index < len(values):
            return values[index]
        return default

    centers = _array(state.get("object_box_centers_m", state["means"]))
    dimensions = _array(state.get("object_box_dimensions_m"))
    cov6 = _array(state["cov6"])
    observations = row_value("object_image_ids", []) or []
    return {
        "id": object_id,
        "observations": len(set(map(int, observations))),
        "position_world_m": np.asarray(centers[index], dtype=float).tolist(),
        "cov6": np.asarray(cov6[index], dtype=float).tolist(),
        "metric_dimensions_m": np.asarray(dimensions[index], dtype=float).tolist(),
        "category": str(row_value("object_category", "unresolved object") or "unresolved object"),
        "description": str(row_value("object_caption", "") or ""),
        "attributes": list(row_value("object_key_attributes", []) or []),
        "semantic_tier": str(row_value("object_semantic_tier", "geometry_only") or "geometry_only"),
        "semantic_status": str(
            row_value("object_semantic_status", "full_colmap_rescue_pending_vlm")
            or "full_colmap_rescue_pending_vlm"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--source-frames", type=Path, required=True)
    parser.add_argument("--source-mask-root", type=Path, required=True)
    parser.add_argument("--rescue-state", type=Path, required=True)
    parser.add_argument("--rescue-frames", type=Path, required=True)
    parser.add_argument("--refinement", type=Path, required=True)
    parser.add_argument("--prior-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite combined rescue output: {output}")
    output.mkdir(parents=True)
    mask_output = output / "masks"
    mask_output.mkdir()
    started = time.perf_counter()

    source_path = args.source_state.expanduser().resolve(strict=True)
    rescue_path = args.rescue_state.expanduser().resolve(strict=True)
    source_payload, source_state_raw = _load_state(source_path)
    _, rescue_state = _load_state(rescue_path)
    payload = copy.deepcopy(source_payload)
    state = payload["state"] if isinstance(payload, dict) and isinstance(payload.get("state"), dict) else payload
    if not isinstance(state, dict):
        raise TypeError("copied scene state is invalid")

    source_ids = _array(state["object_id"]).astype(np.int64).reshape(-1)
    index_by_id = {int(value): index for index, value in enumerate(source_ids.tolist())}
    if len(index_by_id) != source_ids.size:
        raise ValueError("source state object IDs are not unique")
    source_images = list(state.get("images") or [])
    source_positions = list(state.get("image_positions") or [])
    rescue_images = list(rescue_state.get("images") or [])
    rescue_positions = list(rescue_state.get("image_positions") or [])
    if len(source_images) != len(source_positions) or len(rescue_images) != len(rescue_positions):
        raise ValueError("state images and image_positions are not row-aligned")

    source_frames_path = args.source_frames.expanduser().resolve(strict=True)
    rescue_frames_path = args.rescue_frames.expanduser().resolve(strict=True)
    source_frames_payload = json.loads(source_frames_path.read_text(encoding="utf-8"))
    rescue_frames_payload = json.loads(rescue_frames_path.read_text(encoding="utf-8"))
    source_frames = list(source_frames_payload.get("frames") or [])
    rescue_frames = list(rescue_frames_payload.get("frames") or [])
    if len(source_frames) != len(source_images) or len(rescue_frames) != len(rescue_images):
        raise ValueError("frames manifests and state image tables have different lengths")

    combined_frames = []
    for row in source_frames:
        item = dict(row)
        item["rgb_path"] = _relative_artifact(str(item["rgb_path"]), source_frames_path, output)
        item["depth_path"] = _relative_artifact(str(item["depth_path"]), source_frames_path, output)
        combined_frames.append(item)
    source_image_count = len(source_images)
    for rescue_image_id, row in enumerate(rescue_frames):
        item = dict(row)
        item["frame_id"] = source_image_count + rescue_image_id
        item["rgb_path"] = _relative_artifact(str(item["rgb_path"]), rescue_frames_path, output)
        item["depth_path"] = _relative_artifact(str(item["depth_path"]), rescue_frames_path, output)
        combined_frames.append(item)
    frames_payload = dict(source_frames_payload)
    frames_payload["frames"] = combined_frames
    frames_payload["full_colmap_rescue"] = {
        "source_frames": len(source_frames),
        "rescue_frames": len(rescue_frames),
        "combined_frames": len(combined_frames),
    }
    _atomic_json(output / "frames.json", frames_payload)

    for rescue_image_id, image in enumerate(rescue_images):
        item = copy.deepcopy(image)
        global_image_id = source_image_count + rescue_image_id
        if isinstance(item, dict):
            item["image_id"] = global_image_id
            item["source_ref"] = str(combined_frames[global_image_id]["rgb_path"])
        source_images.append(item)
        source_positions.append(copy.deepcopy(rescue_positions[rescue_image_id]))
    state["images"] = source_images
    state["image_positions"] = source_positions

    original_mask_count = _link_mask_tree(
        args.source_mask_root.expanduser().resolve(strict=True), mask_output
    )
    refinement_path = args.refinement.expanduser().resolve(strict=True)
    refinement = json.loads(refinement_path.read_text(encoding="utf-8"))
    if refinement.get("schema") != "farm.full-colmap-mask-refinement.v1" or refinement.get("status") != "PASS":
        raise ValueError("invalid full-COLMAP refinement report")
    accepted_rows = [
        row for row in refinement.get("objects") or []
        if isinstance(row, dict) and bool(row.get("accepted"))
    ]
    accepted_ids = sorted(int(row["object_id"]) for row in accepted_rows)
    if len(accepted_ids) != len(set(accepted_ids)):
        raise ValueError("refinement report repeats accepted object IDs")
    unknown = sorted(set(accepted_ids) - set(index_by_id))
    if unknown:
        raise ValueError(f"refinement report contains unknown object IDs: {unknown}")

    mask_observations = list(state.get("object_mask_observations") or [])
    object_image_ids = list(state.get("object_image_ids") or [])
    if len(mask_observations) != source_ids.size or len(object_image_ids) != source_ids.size:
        raise ValueError("source state mask/image observation tables are not object-aligned")
    refined_mask_count = 0
    for object_row in accepted_rows:
        object_id = int(object_row["object_id"])
        object_index = index_by_id[object_id]
        observations = list(mask_observations[object_index] or [])
        image_ids = list(map(int, object_image_ids[object_index] or []))
        seen_images = set(image_ids)
        for view in object_row.get("views") or []:
            if not isinstance(view, dict) or not bool(view.get("accepted")):
                continue
            global_image_id = int(view["global_image_id"])
            if not source_image_count <= global_image_id < len(combined_frames):
                raise ValueError(f"refined image ID is outside combined frames: {global_image_id}")
            relative = Path(str(view["mask_relative"]))
            if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".npz":
                raise ValueError(f"unsafe refined mask path: {relative}")
            source_mask = (refinement_path.parent / relative).resolve(strict=True)
            destination = mask_output / f"object_{object_id:06d}" / relative.name
            _link_file(source_mask, destination)
            observation = {
                "image_id": global_image_id,
                "path": (
                    f"/farm-run/qa/full_colmap_rescue/combined/masks/"
                    f"object_{object_id:06d}/{relative.name}"
                ),
                "image_shape": list(map(int, np.load(destination, allow_pickle=False)["image_shape"])),
                "detection_idx": 0,
                "object_idx": object_index,
                "object_id": object_id,
                "mask_kinds": ["raw", "inlier", "crop"],
                "raw_pixels": int(view["final_pixels"]),
                "inlier_pixels": int(view["final_pixels"]),
                "crop_jpeg_bytes_len": int(view.get("crop_jpeg_bytes_len") or 0),
                "crop_bbox_xyxy": list(map(int, view.get("crop_bbox_xyxy") or [])),
                "crop_shape": list(map(int, view.get("crop_shape") or [])),
                "score": float(view.get("confidence") or 0.0),
                "class_id": -1,
                "batch_id": -1,
                "source": "full_colmap_sam3_refinement",
            }
            observations.append(observation)
            if global_image_id not in seen_images:
                image_ids.append(global_image_id)
                seen_images.add(global_image_id)
            refined_mask_count += 1
        mask_observations[object_index] = observations
        object_image_ids[object_index] = sorted(image_ids)
    state["object_mask_observations"] = mask_observations
    state["object_image_ids"] = object_image_ids

    prior_catalog_path = args.prior_catalog.expanduser().resolve(strict=True)
    prior_catalog = json.loads(prior_catalog_path.read_text(encoding="utf-8"))
    if not isinstance(prior_catalog, list):
        raise TypeError("prior semantic catalog must be a JSON array")
    prior_by_id = {int(row["id"]): dict(row) for row in prior_catalog if isinstance(row, dict)}
    review_queue = []
    for object_id in accepted_ids:
        index = index_by_id[object_id]
        row = prior_by_id.get(object_id, _review_row(state, index, object_id))
        row["observations"] = len(set(map(int, object_image_ids[index] or [])))
        row["metric_dimensions_m"] = _array(state["object_box_dimensions_m"])[index].astype(float).tolist()
        row["full_colmap_rescue_pending"] = True
        review_queue.append(row)
    _atomic_json(output / "review_queue.json", review_queue)
    (output / "rescue_object_ids.txt").write_text(
        "".join(f"{object_id}\n" for object_id in accepted_ids), encoding="utf-8"
    )

    if isinstance(payload, dict) and isinstance(payload.get("state"), dict):
        payload["state"] = state
        payload["saved_unix_s"] = time.time()
        payload.setdefault("meta", {})["full_colmap_rescue_merge"] = {
            "accepted_object_ids": accepted_ids,
            "refinement_sha256": _sha256(refinement_path),
            "combined_frames": len(combined_frames),
        }
    else:
        payload = state
    state_path = output / "scene_state_pre_geometry.pt"
    temporary_state = state_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary_state)
    temporary_state.replace(state_path)

    report = {
        "schema": "farm.full-colmap-rescue-merge.v1",
        "status": "PASS",
        "accepted_object_ids": accepted_ids,
        "accepted_objects": len(accepted_ids),
        "source_frames": len(source_frames),
        "rescue_frames": len(rescue_frames),
        "combined_frames": len(combined_frames),
        "original_masks_linked": original_mask_count,
        "refined_masks_linked": refined_mask_count,
        "review_queue_objects": len(review_queue),
        "hardlink_preferred": True,
        "timing_seconds": time.perf_counter() - started,
        "hashes": {
            "source_state_sha256": _sha256(source_path),
            "rescue_state_sha256": _sha256(rescue_path),
            "refinement_sha256": _sha256(refinement_path),
            "combined_state_sha256": _sha256(state_path),
            "combined_frames_sha256": _sha256(output / "frames.json"),
        },
    }
    _atomic_json(output / "merge_report.json", report)
    _atomic_json(output / "_SUCCESS.json", {
        "schema": "farm.full-colmap-rescue-merge.success.v1",
        "status": "success",
        "report": "merge_report.json",
        "report_sha256": _sha256(output / "merge_report.json"),
    })
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
