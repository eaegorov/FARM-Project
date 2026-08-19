"""Persistence helpers for per-object 2D mask observations.

These records are evaluation evidence, not reconstruction geometry.  They keep
the detector's image-space masks attached to the object that consumed the
detection, so offline metrics can compare predicted objects in the image plane
without re-projecting the object's 3D voxel support.

Optionally, the same sidecar also embeds a JPEG-encoded padded RGB crop
(reusing the per-detection crop already computed for the captioner). This
gives later VL-reranking passes the visual context for each observation
without a second pass over the original frame source.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch

try:  # Pillow is already a runtime dep of the imaging pipeline.
    from PIL import Image
except Exception:  # pragma: no cover - defensive import
    Image = None  # type: ignore[assignment]


DEFAULT_MAX_MASK_OBSERVATIONS_PER_OBJECT = 256
DEFAULT_MAX_PAIRWISE_MASK_IMAGES_PER_PAIR = 16
DEFAULT_CROP_JPEG_QUALITY = 85
PAIRWISE_MASK_OVERLAP_FIELD = "object_pair_mask_overlap_evidence"
PAIRWISE_MASK_OVERLAP_SCHEMA = "farm.object-pair-mask-overlap.v1"
OBJECT_DIRECTORY_RE = re.compile(r"^object_(\d+)$")
IMAGE_ID_RE = re.compile(r"img_(\d+)")


@dataclass
class ResolvedMaskObservationIndex:
    paths_by_index: List[List[Path]]
    paths_by_object_id: Dict[int, List[Path]]
    diagnostics: Dict[str, Any]

    def for_index(self, index: int) -> List[Path]:
        index = int(index)
        return list(self.paths_by_index[index]) if 0 <= index < len(self.paths_by_index) else []

    def for_object_id(self, object_id: int) -> List[Path]:
        return list(self.paths_by_object_id.get(int(object_id), []))


def _state_object_ids(state: Mapping[str, Any], row_count: int) -> List[int]:
    values = state.get("object_id")
    if isinstance(values, torch.Tensor):
        with contextlib.suppress(Exception):
            return [int(value) for value in values.detach().cpu().reshape(-1).tolist()]
    if values is not None:
        with contextlib.suppress(Exception):
            return [int(value) for value in list(values)]
    return list(range(max(0, int(row_count))))


def _bounded_existing_sidecar(path: Path, root: Path) -> Optional[Path]:
    root = root.expanduser().resolve()
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if candidate.suffix.lower() != ".npz" or not candidate.is_file():
        return None
    return candidate


def _object_tail(path: Path) -> Optional[Path]:
    for index, part in enumerate(path.parts):
        if OBJECT_DIRECTORY_RE.fullmatch(part):
            return Path(*path.parts[index:])
    return None


def resolve_object_mask_observations(
    state: Mapping[str, Any],
    mask_roots: Path | Iterable[Path],
    *,
    expand_image_matches: bool = False,
    include_object_ids: Optional[Iterable[int]] = None,
) -> ResolvedMaskObservationIndex:
    """Resolve canonical state observations strictly inside supplied roots.

    Archived absolute paths are remapped by their ``object_NNNNNN`` relative
    tail. Filesystem globbing is used only when the state has no canonical
    observation-list key.
    """
    roots_raw = (
        [Path(mask_roots)]
        if isinstance(mask_roots, (str, Path))
        else [Path(value) for value in mask_roots]
    )
    roots: List[Path] = []
    for value in roots_raw:
        root = value.expanduser().resolve()
        if root not in roots:
            roots.append(root)
    if not roots:
        raise ValueError("at least one mask root is required")
    included_ids = (
        None
        if include_object_ids is None
        else {int(value) for value in include_object_ids}
    )

    inventory: set[Path] = set()
    image_inventory: Dict[tuple[int, int], List[Path]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for candidate in sorted(root.glob("object_*/*.npz")):
            path = _bounded_existing_sidecar(candidate, root)
            if path is None:
                continue
            object_match = OBJECT_DIRECTORY_RE.fullmatch(path.parent.name)
            if included_ids is not None and (
                object_match is None or int(object_match.group(1)) not in included_ids
            ):
                continue
            inventory.add(path)
            image_match = IMAGE_ID_RE.search(path.name)
            if object_match and image_match:
                image_inventory.setdefault(
                    (int(object_match.group(1)), int(image_match.group(1))), []
                ).append(path)

    canonical_present = "object_mask_observations" in state
    raw_rows = state.get("object_mask_observations") if canonical_present else None
    rows = raw_rows if isinstance(raw_rows, list) else []
    object_ids = _state_object_ids(state, len(rows))
    count = max(len(object_ids), len(rows))
    if len(object_ids) < count:
        object_ids.extend(range(len(object_ids), count))
    by_index: List[List[Path]] = [[] for _ in range(count)]
    by_id: Dict[int, List[Path]] = {int(value): [] for value in object_ids}
    referenced: set[Path] = set()
    stats = {
        key: 0
        for key in (
            "canonical_record_count",
            "resolved_record_count",
            "remapped_archived_absolute_record_count",
            "rooted_relative_record_count",
            "expanded_derivative_sidecar_count",
            "missing_record_count",
            "invalid_record_count",
            "duplicate_sidecar_reference_count",
        )
    }
    missing_examples: List[str] = []

    def add(index: int, object_id: int, path: Path) -> None:
        if path in by_index[index]:
            stats["duplicate_sidecar_reference_count"] += 1
            return
        by_index[index].append(path)
        by_id.setdefault(object_id, []).append(path)
        referenced.add(path)

    if canonical_present:
        if not isinstance(raw_rows, list):
            stats["invalid_record_count"] += 1
        for index, row in enumerate(rows):
            if index >= count:
                break
            object_id = int(object_ids[index])
            if included_ids is not None and object_id not in included_ids:
                continue
            if row is None:
                continue
            if not isinstance(row, (list, tuple)):
                stats["invalid_record_count"] += 1
                continue
            for observation in row:
                stats["canonical_record_count"] += 1
                if not isinstance(observation, Mapping):
                    stats["invalid_record_count"] += 1
                    continue
                text = str(observation.get("path") or observation.get("mask_path") or "").strip()
                if not text:
                    stats["invalid_record_count"] += 1
                    continue
                recorded = Path(text)
                tail = _object_tail(recorded)
                candidates: List[Path] = []
                for root in roots:
                    path = _bounded_existing_sidecar(tail if tail is not None else recorded, root)
                    if path is not None and path not in candidates:
                        candidates.append(path)
                if not candidates and expand_image_matches:
                    with contextlib.suppress(Exception):
                        candidates.extend(
                            image_inventory.get((object_id, int(observation.get("image_id"))), [])
                        )
                if not candidates:
                    stats["missing_record_count"] += 1
                    if len(missing_examples) < 20:
                        missing_examples.append(text)
                    continue
                stats["resolved_record_count"] += 1
                if recorded.is_absolute():
                    original = recorded.expanduser().resolve(strict=False)
                    stats["remapped_archived_absolute_record_count"] += int(
                        any(value != original for value in candidates)
                    )
                else:
                    stats["rooted_relative_record_count"] += 1
                stats["expanded_derivative_sidecar_count"] += max(0, len(candidates) - 1)
                for path in candidates:
                    add(index, object_id, path)
    else:
        id_to_indices: Dict[int, List[int]] = {}
        for index, object_id in enumerate(object_ids):
            id_to_indices.setdefault(int(object_id), []).append(index)
        for path in sorted(inventory):
            match = OBJECT_DIRECTORY_RE.fullmatch(path.parent.name)
            if match:
                for index in id_to_indices.get(int(match.group(1)), []):
                    add(index, int(match.group(1)), path)

    orphans = sorted(inventory - referenced)
    diagnostics = {
        "mode": "canonical_state" if canonical_present else "legacy_filesystem_fallback",
        "canonical_field_present": canonical_present,
        "legacy_fallback_used": not canonical_present,
        "mask_roots": [str(value) for value in roots],
        "object_id_scope": "all" if included_ids is None else sorted(included_ids),
        **stats,
        "resolved_sidecar_count": len(referenced),
        "filesystem_sidecar_count": len(inventory),
        "orphan_sidecar_count": len(orphans),
        "missing_record_examples": missing_examples,
        "orphan_sidecar_examples": [str(value) for value in orphans[:20]],
    }
    return ResolvedMaskObservationIndex(by_index, by_id, diagnostics)


def _delete_evicted_sidecar(record: object, object_dir: Path) -> bool:
    if not isinstance(record, Mapping):
        return False
    text = str(record.get("path") or record.get("mask_path") or "").strip()
    if not text:
        return False
    allowed = object_dir.expanduser().resolve()
    recorded = Path(text).expanduser()
    if not recorded.is_absolute():
        tail = _object_tail(recorded)
        if tail is None or tail.parent.name != allowed.name:
            return False
        recorded = allowed.parent / tail
    candidate = recorded.resolve(strict=False)
    if candidate.suffix.lower() != ".npz" or candidate.parent != allowed:
        return False
    try:
        candidate.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def _encode_padded_crop_jpeg(
    obs: Any,
    quality: int = DEFAULT_CROP_JPEG_QUALITY,
) -> Optional[Dict[str, Any]]:
    """Pull the padded RGB crop out of a caption-observation entry and JPEG-encode it.

    Returns ``{"crop_jpeg_bytes": <uint8 ndarray>, "crop_bbox_xyxy": <int32[4]>,
    "crop_shape": <int32[2]>}`` or ``None`` if no usable crop is present.
    """
    if obs is None or Image is None:
        return None
    image = obs.get("image") if isinstance(obs, Mapping) else None
    bbox = obs.get("bbox") if isinstance(obs, Mapping) else None
    if image is None or bbox is None:
        return None
    if isinstance(image, torch.Tensor):
        arr = image.detach()
        if arr.is_cuda:
            arr = arr.cpu()
        arr = arr.numpy()
    else:
        arr = np.asarray(image)
    # Accept HWC uint8; if CHW float, transpose+cast.
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        return None
    buf = io.BytesIO()
    try:
        Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=int(quality), optimize=True)
    except Exception:
        return None
    return {
        "crop_jpeg_bytes": np.frombuffer(buf.getvalue(), dtype=np.uint8),
        "crop_bbox_xyxy": np.asarray(
            [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])], dtype=np.int32
        ),
        "crop_shape": np.asarray([int(arr.shape[0]), int(arr.shape[1])], dtype=np.int32),
    }


def _as_numpy_bool(mask: Any) -> Optional[np.ndarray]:
    if mask is None:
        return None
    value = mask
    if isinstance(value, torch.Tensor):
        value = value.detach().to("cpu")
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        value = value.numpy()
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        return None
    return arr.astype(bool, copy=False)


def _mask_bbox_xyxy(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if ys.size == 0 or xs.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return [x0, y0, x1, y1]


def _pack_crop(mask: np.ndarray, bbox_xyxy: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    crop = np.asarray(mask[y0:y1, x0:x1], dtype=np.uint8)
    packed = np.packbits(crop.reshape(-1), bitorder="little")
    shape = np.asarray([int(crop.shape[0]), int(crop.shape[1])], dtype=np.int32)
    return packed, shape


def _score_at(scores: Any, det_i: int) -> Optional[float]:
    if scores is None:
        return None
    with contextlib.suppress(Exception):
        if isinstance(scores, torch.Tensor):
            return float(scores.detach().to("cpu").view(-1)[int(det_i)].item())
        return float(np.asarray(scores).reshape(-1)[int(det_i)])
    return None


def _int_at(values: Any, det_i: int) -> Optional[int]:
    if values is None:
        return None
    with contextlib.suppress(Exception):
        if isinstance(values, torch.Tensor):
            return int(values.detach().to("cpu").view(-1)[int(det_i)].item())
        return int(np.asarray(values).reshape(-1)[int(det_i)])
    return None


def _canonical_pair_key(first_object_id: int, second_object_id: int) -> str:
    first, second = sorted((int(first_object_id), int(second_object_id)))
    return f"{first}:{second}"


def _resolve_redirect_object_id(state: Mapping[str, Any], object_id: int) -> int:
    redirects = state.get("id_redirect")
    if not isinstance(redirects, Mapping):
        return int(object_id)
    current = int(object_id)
    seen: set[int] = set()
    while current not in seen:
        seen.add(current)
        target = redirects.get(current)
        if target is None:
            target = redirects.get(str(current))
        if target is None:
            break
        try:
            target_id = int(target)
        except (TypeError, ValueError):
            break
        if target_id == current:
            break
        current = target_id
    return current


def _object_id_at(state: Mapping[str, Any], object_index: int) -> Optional[int]:
    values = state.get("object_id")
    try:
        if isinstance(values, torch.Tensor):
            if 0 <= int(object_index) < int(values.numel()):
                return _resolve_redirect_object_id(
                    state, int(values.detach().cpu().reshape(-1)[int(object_index)].item())
                )
        elif values is not None and 0 <= int(object_index) < len(values):
            return _resolve_redirect_object_id(state, int(values[int(object_index)]))
    except (IndexError, TypeError, ValueError):
        return None
    return None


def _raw_mask_intersection_pixels(
    first_mask: np.ndarray,
    first_bbox: Sequence[int],
    second_mask: np.ndarray,
    second_bbox: Sequence[int],
) -> int:
    first_box = [int(value) for value in first_bbox]
    second_box = [int(value) for value in second_bbox]
    x0 = max(first_box[0], second_box[0])
    y0 = max(first_box[1], second_box[1])
    x1 = min(first_box[2], second_box[2])
    y1 = min(first_box[3], second_box[3])
    if x1 <= x0 or y1 <= y0:
        return 0
    first = first_mask[y0:y1, x0:x1]
    second = second_mask[y0:y1, x0:x1]
    height = min(first.shape[0], second.shape[0])
    width = min(first.shape[1], second.shape[1])
    if height <= 0 or width <= 0:
        return 0
    return int(np.count_nonzero(first[:height, :width] & second[:height, :width]))


def _pair_image_record_rank(record: Mapping[str, Any]) -> tuple[float, ...]:
    """Deterministic, policy-neutral ordering for coherent mask comparisons."""

    def finite_float(key: str) -> float:
        try:
            value = float(record.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0
        return value if np.isfinite(value) else 0.0

    return (
        finite_float("raw_iou"),
        finite_float("raw_containment"),
        finite_float("raw_area_ratio"),
        finite_float("raw_intersection_pixels"),
        finite_float("comparison_count"),
        -finite_float("image_id"),
    )


def _mask_pair_fingerprint(
    *,
    first_object_id: int,
    second_object_id: int,
    image_id: int,
    first_detection_idx: int,
    second_detection_idx: int,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        np.asarray(
            [
                int(first_object_id),
                int(second_object_id),
                int(image_id),
                int(first_detection_idx),
                int(second_detection_idx),
                int(first_mask.shape[0]),
                int(first_mask.shape[1]),
            ],
            dtype=np.int64,
        ).tobytes()
    )
    digest.update(np.packbits(first_mask.reshape(-1), bitorder="little").tobytes())
    digest.update(np.packbits(second_mask.reshape(-1), bitorder="little").tobytes())
    return digest.hexdigest()


def _empty_pairwise_mask_overlap_store(max_images_per_pair: int) -> Dict[str, Any]:
    return {
        "schema": PAIRWISE_MASK_OVERLAP_SCHEMA,
        "max_images_per_pair": max(1, int(max_images_per_pair)),
        "pairs": {},
    }


def _ensure_pairwise_mask_overlap_store(
    state: Dict[str, Any],
    *,
    max_images_per_pair: int,
) -> Dict[str, Any]:
    store = state.get(PAIRWISE_MASK_OVERLAP_FIELD)
    if not isinstance(store, dict) or store.get("schema") != PAIRWISE_MASK_OVERLAP_SCHEMA:
        store = _empty_pairwise_mask_overlap_store(max_images_per_pair)
        state[PAIRWISE_MASK_OVERLAP_FIELD] = store
    if not isinstance(store.get("pairs"), dict):
        store["pairs"] = {}
    store["max_images_per_pair"] = max(1, int(max_images_per_pair))
    return store


def _merge_pair_image_record(
    existing: Optional[Mapping[str, Any]],
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    candidate_record = dict(candidate)
    if not isinstance(existing, Mapping):
        return candidate_record
    comparison_count = max(0, int(existing.get("comparison_count", 0))) + max(
        0, int(candidate_record.get("comparison_count", 0))
    )
    update_count = max(0, int(existing.get("update_count", 0))) + max(
        1, int(candidate_record.get("update_count", 1))
    )
    existing_rank = _pair_image_record_rank(existing)
    candidate_rank = _pair_image_record_rank(candidate_record)
    if existing_rank > candidate_rank:
        merged = dict(existing)
    elif existing_rank == candidate_rank:
        existing_fingerprint = str(existing.get("evidence_fingerprint") or "")
        candidate_fingerprint = str(candidate_record.get("evidence_fingerprint") or "")
        merged = dict(existing if existing_fingerprint <= candidate_fingerprint else candidate_record)
    else:
        merged = candidate_record
    merged["comparison_count"] = int(comparison_count)
    merged["update_count"] = int(update_count)
    return merged


def _cap_pair_image_records(
    records: Iterable[Mapping[str, Any]],
    *,
    cap: int,
) -> List[Dict[str, Any]]:
    by_image: Dict[int, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        try:
            image_id = int(record.get("image_id", -1))
        except (TypeError, ValueError):
            continue
        if image_id < 0:
            continue
        by_image[image_id] = _merge_pair_image_record(by_image.get(image_id), record)
    strongest = sorted(
        by_image.values(),
        key=lambda row: (_pair_image_record_rank(row), str(row.get("evidence_fingerprint") or "")),
        reverse=True,
    )[: max(1, int(cap))]
    return sorted(strongest, key=lambda row: int(row["image_id"]))


def _normalize_pairwise_mask_overlap_redirects(
    state: Mapping[str, Any],
    store: Dict[str, Any],
) -> None:
    redirects = state.get("id_redirect")
    normalized_redirects: List[tuple[int, int]] = []
    if isinstance(redirects, Mapping):
        for source, target in redirects.items():
            try:
                normalized_redirects.append((int(source), int(target)))
            except (TypeError, ValueError):
                continue
    redirect_fingerprint = hashlib.sha256(
        repr(sorted(normalized_redirects)).encode("utf-8")
    ).hexdigest()
    if store.get("redirect_fingerprint") == redirect_fingerprint:
        return
    cap = max(
        1,
        int(store.get("max_images_per_pair", DEFAULT_MAX_PAIRWISE_MASK_IMAGES_PER_PAIR)),
    )
    normalized_pairs: Dict[str, Dict[str, Any]] = {}
    pairs = store.get("pairs")
    for entry in pairs.values() if isinstance(pairs, Mapping) else []:
        if not isinstance(entry, Mapping):
            continue
        try:
            old_first = int(entry.get("first_object_id"))
            old_second = int(entry.get("second_object_id"))
        except (TypeError, ValueError):
            continue
        first = _resolve_redirect_object_id(state, old_first)
        second = _resolve_redirect_object_id(state, old_second)
        if first == second:
            continue
        canonical_first, canonical_second = sorted((first, second))
        swap = (first, second) != (canonical_first, canonical_second)
        key = _canonical_pair_key(canonical_first, canonical_second)
        target = normalized_pairs.setdefault(
            key,
            {
                "first_object_id": canonical_first,
                "second_object_id": canonical_second,
                "images": [],
            },
        )
        incoming: List[Dict[str, Any]] = []
        raw_images = entry.get("images")
        for raw_record in raw_images if isinstance(raw_images, list) else []:
            if not isinstance(raw_record, Mapping):
                continue
            record = dict(raw_record)
            if swap:
                for first_key, second_key in (
                    ("first_area_pixels", "second_area_pixels"),
                    ("first_detection_idx", "second_detection_idx"),
                ):
                    record[first_key], record[second_key] = (
                        record.get(second_key),
                        record.get(first_key),
                    )
            incoming.append(record)
        target["images"] = _cap_pair_image_records(
            [*target.get("images", []), *incoming], cap=cap
        )
    store["pairs"] = normalized_pairs
    store["redirect_fingerprint"] = redirect_fingerprint


def record_pairwise_mask_overlap_evidence(
    state: Dict[str, Any],
    raw_masks: Sequence[Any],
    detection_image_ids: Sequence[Optional[int]],
    det_to_obj: Sequence[Optional[int]],
    *,
    max_images_per_pair: int = DEFAULT_MAX_PAIRWISE_MASK_IMAGES_PER_PAIR,
) -> Dict[str, int]:
    """Persist bounded same-image raw-mask evidence before sidecar eviction."""

    cap = max(1, int(max_images_per_pair))
    store = _ensure_pairwise_mask_overlap_store(state, max_images_per_pair=cap)
    _normalize_pairwise_mask_overlap_redirects(state, store)
    grouped: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
    for detection_idx, object_index_raw in enumerate(det_to_obj):
        if object_index_raw is None or detection_idx >= len(raw_masks):
            continue
        if detection_idx >= len(detection_image_ids):
            continue
        try:
            object_index = int(object_index_raw)
            image_id = int(detection_image_ids[detection_idx])
        except (TypeError, ValueError):
            continue
        if object_index < 0 or image_id < 0:
            continue
        object_id = _object_id_at(state, object_index)
        raw = _as_numpy_bool(raw_masks[detection_idx])
        if object_id is None or raw is None:
            continue
        bbox = _mask_bbox_xyxy(raw)
        area = int(np.count_nonzero(raw))
        if bbox is None or area <= 0:
            continue
        grouped.setdefault(image_id, {}).setdefault(int(object_id), []).append(
            {
                "detection_idx": int(detection_idx),
                "mask": raw,
                "bbox": bbox,
                "area": area,
            }
        )

    pair_images_updated = 0
    comparisons = 0
    pairs = store["pairs"]
    for image_id, by_object in sorted(grouped.items()):
        object_ids = sorted(by_object)
        for first_pos, first_object_id in enumerate(object_ids):
            for second_object_id in object_ids[first_pos + 1 :]:
                best: Optional[Dict[str, Any]] = None
                image_comparisons = 0
                for first in by_object[first_object_id]:
                    for second in by_object[second_object_id]:
                        first_mask = first["mask"]
                        second_mask = second["mask"]
                        if first_mask.shape != second_mask.shape:
                            continue
                        image_comparisons += 1
                        intersection = _raw_mask_intersection_pixels(
                            first_mask,
                            first["bbox"],
                            second_mask,
                            second["bbox"],
                        )
                        first_area = int(first["area"])
                        second_area = int(second["area"])
                        union = first_area + second_area - intersection
                        iou = intersection / float(union) if union > 0 else 0.0
                        containment = intersection / float(min(first_area, second_area))
                        area_ratio = min(first_area, second_area) / float(
                            max(first_area, second_area)
                        )
                        candidate = {
                            "image_id": int(image_id),
                            "raw_iou": float(iou),
                            "raw_containment": float(containment),
                            "raw_area_ratio": float(area_ratio),
                            "raw_intersection_pixels": int(intersection),
                            "first_area_pixels": int(first_area),
                            "second_area_pixels": int(second_area),
                            "comparison_count": 1,
                            "update_count": 1,
                            "first_detection_idx": int(first["detection_idx"]),
                            "second_detection_idx": int(second["detection_idx"]),
                            "image_shape": [int(first_mask.shape[0]), int(first_mask.shape[1])],
                            "source": "online_raw_detection_masks",
                            "evidence_fingerprint": _mask_pair_fingerprint(
                                first_object_id=first_object_id,
                                second_object_id=second_object_id,
                                image_id=image_id,
                                first_detection_idx=int(first["detection_idx"]),
                                second_detection_idx=int(second["detection_idx"]),
                                first_mask=first_mask,
                                second_mask=second_mask,
                            ),
                        }
                        candidate_key = (
                            _pair_image_record_rank(candidate),
                            str(candidate["evidence_fingerprint"]),
                        )
                        best_key = (
                            _pair_image_record_rank(best),
                            str(best["evidence_fingerprint"]),
                        ) if best is not None else None
                        if best_key is None or candidate_key > best_key:
                            best = candidate
                if best is None:
                    continue
                best["comparison_count"] = int(image_comparisons)
                comparisons += int(image_comparisons)
                key = _canonical_pair_key(first_object_id, second_object_id)
                pair_entry = pairs.setdefault(
                    key,
                    {
                        "first_object_id": int(first_object_id),
                        "second_object_id": int(second_object_id),
                        "images": [],
                    },
                )
                pair_entry["images"] = _cap_pair_image_records(
                    [*pair_entry.get("images", []), best], cap=cap
                )
                pair_images_updated += 1
    return {
        "pair_images_updated": int(pair_images_updated),
        "mask_pairs_compared": int(comparisons),
    }


def get_pairwise_mask_overlap_records(
    state: Mapping[str, Any],
    first_object_id: int,
    second_object_id: int,
) -> List[Dict[str, Any]]:
    """Return validated, filesystem-independent evidence for one object pair."""

    first = _resolve_redirect_object_id(state, int(first_object_id))
    second = _resolve_redirect_object_id(state, int(second_object_id))
    if first == second:
        return []
    store = state.get(PAIRWISE_MASK_OVERLAP_FIELD)
    if not isinstance(store, Mapping) or store.get("schema") != PAIRWISE_MASK_OVERLAP_SCHEMA:
        return []
    pairs = store.get("pairs")
    if not isinstance(pairs, Mapping):
        return []
    entry = pairs.get(_canonical_pair_key(first, second))
    if not isinstance(entry, Mapping):
        return []
    records: List[Dict[str, Any]] = []
    raw_images = entry.get("images")
    for raw_record in raw_images if isinstance(raw_images, list) else []:
        if not isinstance(raw_record, Mapping):
            continue
        try:
            image_id = int(raw_record.get("image_id", -1))
            values = [
                float(raw_record.get("raw_iou", -1.0)),
                float(raw_record.get("raw_containment", -1.0)),
                float(raw_record.get("raw_area_ratio", -1.0)),
            ]
        except (TypeError, ValueError):
            continue
        if image_id < 0 or not all(
            np.isfinite(value) and 0.0 <= value <= 1.0 for value in values
        ):
            continue
        records.append(dict(raw_record))
    return sorted(records, key=lambda row: int(row["image_id"]))


def ensure_object_mask_observation_rows(state: Dict[str, Any], min_len: int) -> List[List[Dict[str, Any]]]:
    rows = state.get("object_mask_observations")
    if not isinstance(rows, list):
        rows = []
    while len(rows) < int(min_len):
        rows.append([])
    state["object_mask_observations"] = rows
    return rows


def register_detection_mask_observations(
    state: Dict[str, Any],
    seg_outputs: Mapping[str, Any],
    detection_image_ids: Sequence[Optional[int]],
    det_to_obj: Sequence[Optional[int]],
    mask_storage_dir: Path,
    *,
    max_per_object: int = DEFAULT_MAX_MASK_OBSERVATIONS_PER_OBJECT,
    max_pair_images_per_pair: int = DEFAULT_MAX_PAIRWISE_MASK_IMAGES_PER_PAIR,
    detection_rgb_observations: Optional[Sequence[Any]] = None,
    save_crops: bool = False,
    crop_jpeg_quality: int = DEFAULT_CROP_JPEG_QUALITY,
) -> int:
    """Save detector masks and attach observation records to scene-state rows.

    Args:
        state: Mutable scene graph state.
        seg_outputs: Filtered segmentation outputs for the current batch.
        detection_image_ids: Image id per detection.
        det_to_obj: Canonical object index per detection after state update.
        mask_storage_dir: Directory where compact ``.npz`` sidecars are stored.
        max_per_object: Keep at most this many records referenced per object.
        max_pair_images_per_pair: Keep at most this many filesystem-independent
            same-frame overlap records for each distinct object pair.
        detection_rgb_observations: Optional list aligned with detections,
            holding the padded RGB crops already prepared for the captioner
            (each entry is the dict from
            :func:`scene_graph.captioning.crop_util.compute_caption_observations`
            with ``image`` HWC uint8 + ``bbox`` xyxy in original frame).
        save_crops: If True and ``detection_rgb_observations`` is provided,
            JPEG-encode the padded crop into the same ``.npz`` sidecar as the
            mask, so downstream rerank passes get visual context without
            reloading the source frame.
        crop_jpeg_quality: JPEG quality for the saved crop (default 85).

    Returns:
        Number of observation records appended to the state.
    """

    raw_masks = list(seg_outputs.get("masks") or [])
    if not raw_masks:
        return 0
    inlier_masks = list(seg_outputs.get("masks_inlier") or [])
    scores = seg_outputs.get("scores")
    class_ids = seg_outputs.get("class_ids")
    batch_ids = seg_outputs.get("batch_ids")
    object_ids = state.get("object_id")
    n_objects = int(object_ids.shape[0]) if isinstance(object_ids, torch.Tensor) else len(det_to_obj)
    rows = ensure_object_mask_observation_rows(state, n_objects)
    mask_storage_dir = Path(mask_storage_dir)
    mask_storage_dir.mkdir(parents=True, exist_ok=True)

    # Preserve same-frame evidence while every raw detection mask is still in
    # memory. This must precede the per-object sidecar cap below: a later cap
    # eviction may otherwise erase the only comparable frame for dedup.
    record_pairwise_mask_overlap_evidence(
        state,
        raw_masks,
        detection_image_ids,
        det_to_obj,
        max_images_per_pair=max_pair_images_per_pair,
    )

    added = 0
    cap = max(1, int(max_per_object))
    for det_i, obj_idx_raw in enumerate(det_to_obj):
        if obj_idx_raw is None:
            continue
        try:
            obj_idx = int(obj_idx_raw)
        except Exception:
            continue
        if obj_idx < 0:
            continue
        if det_i >= len(raw_masks) or det_i >= len(detection_image_ids):
            continue
        image_id_raw = detection_image_ids[det_i]
        if image_id_raw is None:
            continue
        try:
            image_id = int(image_id_raw)
        except Exception:
            continue
        raw = _as_numpy_bool(raw_masks[det_i])
        if raw is None:
            continue
        raw_bbox = _mask_bbox_xyxy(raw)
        if raw_bbox is None:
            continue

        inlier = None
        inlier_bbox = None
        if det_i < len(inlier_masks):
            inlier = _as_numpy_bool(inlier_masks[det_i])
            if inlier is not None and inlier.shape == raw.shape:
                inlier_bbox = _mask_bbox_xyxy(inlier)
            else:
                inlier = None

        ensure_object_mask_observation_rows(state, obj_idx + 1)
        rows = state["object_mask_observations"]
        object_id = obj_idx
        if isinstance(object_ids, torch.Tensor) and obj_idx < int(object_ids.shape[0]):
            with contextlib.suppress(Exception):
                object_id = int(object_ids[obj_idx].detach().to("cpu").item())

        object_dir = mask_storage_dir / f"object_{int(object_id):06d}"
        object_dir.mkdir(parents=True, exist_ok=True)
        path = object_dir / f"img_{image_id:06d}_det_{int(det_i):04d}.npz"

        raw_bits, raw_shape = _pack_crop(raw, raw_bbox)
        payload: Dict[str, Any] = {
            "image_shape": np.asarray([int(raw.shape[0]), int(raw.shape[1])], dtype=np.int32),
            "raw_bits": raw_bits,
            "raw_shape": raw_shape,
            "raw_bbox_xyxy": np.asarray(raw_bbox, dtype=np.int32),
        }
        mask_kinds = ["raw"]
        if inlier is not None and inlier_bbox is not None:
            inlier_bits, inlier_shape = _pack_crop(inlier, inlier_bbox)
            payload.update(
                {
                    "inlier_bits": inlier_bits,
                    "inlier_shape": inlier_shape,
                    "inlier_bbox_xyxy": np.asarray(inlier_bbox, dtype=np.int32),
                }
            )
            mask_kinds.append("inlier")

        crop_record_extra: Dict[str, Any] = {}
        if save_crops and detection_rgb_observations is not None:
            obs_entry = (
                detection_rgb_observations[det_i]
                if det_i < len(detection_rgb_observations)
                else None
            )
            crop_payload = _encode_padded_crop_jpeg(obs_entry, quality=crop_jpeg_quality)
            if crop_payload is not None:
                payload.update(crop_payload)
                mask_kinds.append("crop")
                crop_record_extra = {
                    "crop_jpeg_bytes_len": int(crop_payload["crop_jpeg_bytes"].size),
                    "crop_bbox_xyxy": [int(v) for v in crop_payload["crop_bbox_xyxy"].tolist()],
                    "crop_shape": [
                        int(crop_payload["crop_shape"][0]),
                        int(crop_payload["crop_shape"][1]),
                    ],
                }

        np.savez_compressed(path, **payload)

        record: Dict[str, Any] = {
            "image_id": int(image_id),
            "path": str(path),
            "image_shape": [int(raw.shape[0]), int(raw.shape[1])],
            "detection_idx": int(det_i),
            "object_idx": int(obj_idx),
            "object_id": int(object_id),
            "mask_kinds": mask_kinds,
            "raw_pixels": int(raw.sum()),
        }
        if inlier is not None:
            record["inlier_pixels"] = int(inlier.sum())
        record.update(crop_record_extra)
        score = _score_at(scores, det_i)
        if score is not None:
            record["score"] = float(score)
        class_id = _int_at(class_ids, det_i)
        if class_id is not None:
            record["class_id"] = int(class_id)
        batch_id = _int_at(batch_ids, det_i)
        if batch_id is not None:
            record["batch_id"] = int(batch_id)

        row = rows[obj_idx]
        if not isinstance(row, list):
            row = []
            rows[obj_idx] = row
        row.append(record)
        if len(row) > cap:
            evicted = row[: len(row) - cap]
            del row[: len(row) - cap]
            retained_paths = {
                str(item.get("path") or item.get("mask_path") or "")
                for item in row
                if isinstance(item, Mapping)
            }
            for old_record in evicted:
                old_path = (
                    str(old_record.get("path") or old_record.get("mask_path") or "")
                    if isinstance(old_record, Mapping)
                    else ""
                )
                if old_path and old_path not in retained_paths:
                    _delete_evicted_sidecar(old_record, object_dir)
        added += 1
    return int(added)


def load_mask_observation(
    observation: Mapping[str, Any],
    *,
    kind: str = "raw",
    root: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Load a persisted mask observation as a full image-shaped bool array."""

    path_raw = str(observation.get("path") or observation.get("mask_path") or "")
    if not path_raw:
        return None
    path = Path(path_raw)
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    elif path.is_absolute() and root is not None:
        # Some archived scene states store absolute sidecar paths from the
        # original artifact tree.  If the caller loaded the state from a local
        # mirror, prefer the matching mask sidecar under that mirror.
        parts = path.parts
        for idx, part in enumerate(parts):
            if part.endswith("_masks"):
                local_path = Path(root).joinpath(*parts[idx:])
                if local_path.exists():
                    path = local_path
                break
    if not path.exists():
        return None
    desired = str(kind or "raw").strip().lower()
    if desired not in {"raw", "inlier"}:
        desired = "raw"

    try:
        with np.load(str(path), allow_pickle=False) as data:
            selected = desired
            if f"{selected}_bits" not in data.files:
                selected = "raw"
            if f"{selected}_bits" not in data.files:
                return None
            bits = np.asarray(data[f"{selected}_bits"], dtype=np.uint8)
            crop_shape = np.asarray(data[f"{selected}_shape"], dtype=np.int32).reshape(2)
            bbox = np.asarray(data[f"{selected}_bbox_xyxy"], dtype=np.int32).reshape(4)
            image_shape = np.asarray(data["image_shape"], dtype=np.int32).reshape(2)
            crop_h, crop_w = int(crop_shape[0]), int(crop_shape[1])
            if crop_h <= 0 or crop_w <= 0:
                return None
            flat = np.unpackbits(bits, bitorder="little")[: crop_h * crop_w]
            crop = flat.reshape(crop_h, crop_w).astype(bool, copy=False)
            h, w = int(image_shape[0]), int(image_shape[1])
            if h <= 0 or w <= 0:
                return None
            x0, y0, x1, y1 = [int(v) for v in bbox.tolist()]
            x0 = max(0, min(w, x0))
            x1 = max(0, min(w, x1))
            y0 = max(0, min(h, y0))
            y1 = max(0, min(h, y1))
            if x1 <= x0 or y1 <= y0:
                return None
            out = np.zeros((h, w), dtype=bool)
            out[y0:y1, x0:x1] = crop[: y1 - y0, : x1 - x0]
            return out
    except Exception:
        return None


def load_mask_observation_crop(
    observation: Mapping[str, Any],
    *,
    root: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Load the JPEG-encoded padded RGB crop from a mask observation sidecar.

    Returns the crop as an ``HxWx3`` uint8 array, or ``None`` if the sidecar
    has no crop payload (e.g., older runs that only saved masks).
    """
    if Image is None:
        return None
    path_raw = str(observation.get("path") or observation.get("mask_path") or "")
    if not path_raw:
        return None
    path = Path(path_raw)
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    if not path.exists():
        return None
    try:
        with np.load(str(path), allow_pickle=False) as data:
            if "crop_jpeg_bytes" not in data.files:
                return None
            blob = np.asarray(data["crop_jpeg_bytes"], dtype=np.uint8)
            img = Image.open(io.BytesIO(blob.tobytes())).convert("RGB")
            return np.array(img, dtype=np.uint8)
    except Exception:
        return None
