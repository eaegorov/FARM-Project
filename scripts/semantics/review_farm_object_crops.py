#!/usr/bin/env python3
"""Visually and optionally VLM-review robust FARM objects from saved crop sidecars."""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import base64
import concurrent.futures
import hashlib
import heapq
import http.client
import itertools
import json
import math
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

from scene_graph.captioning.evidence import (
    DEFAULT_CROP_DETAIL_FLOOR_RATIO,
    DEFAULT_MIN_NORMALIZED_CAMERA_BASELINE,
    DEFAULT_MIN_VIEWPOINT_SEPARATION_DEGREES,
    crop_evidence_manifest,
    crop_image_id,
    evidence_pose_diversity,
    evidence_view_overlap,
    gravity_upright_evidence_manifest,
    gravity_upright_orientation,
    load_frame_pose_index,
    partition_crop_candidates,
    select_pose_diverse_crop_candidates,
)
from scene_graph.captioning.label_contract import (
    OPEN_VOCABULARY_JSON_INSTRUCTIONS,
    OPEN_VOCABULARY_RESPONSE_FORMAT,
    assess_open_vocabulary_label,
    assess_open_vocabulary_whole_identity,
    normalize_open_noun,
    parse_open_vocabulary_label,
)
from scene_graph.map_update.mask_observations import resolve_object_mask_observations

try:
    from scene_graph.map_update.filtering import UNINFORMATIVE_YOLOE_LABELS
except ImportError:
    UNINFORMATIVE_YOLOE_LABELS = set()


PROMPT = """The images are automatically selected observations associated with one
physical 3D object in a reconstructed industrial/workshop interior. The target keeps
its natural colours, surrounding context is dimmed but remains readable, and a thin
light outline marks the saved segmentation mask.

Use the target pixels across views as the primary identity evidence. Use stable
context (mounting, support, scale, interaction with nearby equipment, and placement)
as supporting evidence for the object's role, but never name an object from scene
type or a neighbour alone. Prefer the most specific common physical noun supported
by diagnostic target-owned shape, parts, markings, or repeated spatial relationship
in at least two views. Avoid vague nouns such as machine, device, fixture, equipment,
box, plate, or bar when the supplied evidence supports a more informative noun; keep
the supported vague noun rather than inventing a function when it does not.

First decide whether the mask is one whole, a carrier with payload, a standalone
component, an attached component, or an incomplete/mixed target. Then name the whole
bounded physical target. Broad structural surfaces and background are not object
instances.
""" + OPEN_VOCABULARY_JSON_INSTRUCTIONS

VERIFICATION_PROMPT = """Audit the candidate label below against a separate
deterministic view fold. This request is conditioned on the candidate and is a
verification aid, not an independent semantic vote. Use the
multi-view images. Use only pixels that consistently belong to the same object and
do not assume a scene type. A functional identity is supported only when visible
diagnostic parts or readable text distinguish that function from other possible
uses; material, rectangular shape, covering, or general resemblance alone are not
enough. If the candidate overstates the evidence, replace it with an independently
supported physical head noun, or "unknown" when identity is unstable across views.
Do not preserve the candidate merely for consistency. This event is never eligible
to confirm a category.

Candidate category: {category}
Candidate description: {description}
""" + OPEN_VOCABULARY_JSON_INSTRUCTIONS

# Qwen3-VL uses 16 px patches followed by a 2x2 spatial merge.  Capping every
# source crop at 512x512 keeps six-view reviews within the pinned 3,084-token
# context without dropping views or changing the stored crop evidence.
VLM_MAX_IMAGE_PIXELS = 512 * 512
DEFAULT_WORLD_UP_VECTOR = (0.0, -1.0, 0.0)
PUBLICATION_FORBIDDEN_GENERIC_NOUNS = {
    "",
    "unknown",
    "unresolved object",
    "object",
    "machine",
    "unit",
    "device",
    "equipment",
    "container",
    "item",
    "thing",
}
BLIND_INITIAL_EVENT_SOURCES = {
    "initial_blind_review",
    "contextual_dual_panel_review",
}

_FRAME_INDEX_CACHE: dict[Path, list[Path | None]] = {}


def _source_frames(frames_json: Path) -> list[Path | None]:
    """Resolve mapping image indices to the explicitly supplied RGB frames."""

    frames_json = frames_json.expanduser().resolve()
    if frames_json in _FRAME_INDEX_CACHE:
        return _FRAME_INDEX_CACHE[frames_json]
    result: list[Path | None] = []
    try:
        payload = json.loads(frames_json.read_text())
        for row in payload.get("frames", []):
            rel = str(row.get("rgb_path") or "").strip()
            path = (frames_json.parent / rel).resolve() if rel else None
            result.append(path if path is not None and path.is_file() else None)
    except (OSError, ValueError, TypeError):
        result = []
    _FRAME_INDEX_CACHE[frames_json] = result
    return result


def _source_frame_for_crop(frames_json: Path, crop_path: Path) -> np.ndarray | None:
    match = re.search(r"img_(\d+)", crop_path.name)
    if not match:
        return None
    frames = _source_frames(frames_json)
    index = int(match.group(1))
    if not 0 <= index < len(frames) or frames[index] is None:
        return None
    return cv2.imread(str(frames[index]), cv2.IMREAD_COLOR)


def resolve_world_up_contract(
    cli_value: str | None,
    geometry_payload: dict | None,
) -> dict:
    """Resolve one explicit semantic world-up contract with auditable priority."""

    source = "default:[0,-1,0]"
    raw: object = DEFAULT_WORLD_UP_VECTOR
    if geometry_payload is not None:
        up = geometry_payload.get("up")
        if isinstance(up, dict) and up.get("vector") is not None:
            raw = up["vector"]
            source = "geometry_report:$.up.vector"
        elif geometry_payload.get("up_vector") is not None:
            raw = geometry_payload["up_vector"]
            source = "geometry_report:$.up_vector"
    if cli_value is not None:
        raw = [part.strip() for part in str(cli_value).split(",")]
        source = "cli:--world-up-vector"
    try:
        vector = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        vector = np.empty(0, dtype=np.float64)
    if vector.size != 3 or not np.isfinite(vector).all():
        return {
            "available": False,
            "vector": [],
            "source": source,
            "reason": "invalid_world_up_vector",
        }
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return {
            "available": False,
            "vector": [],
            "source": source,
            "reason": "zero_world_up_vector",
        }
    return {
        "available": True,
        "vector": (vector / norm).astype(float).tolist(),
        "source": source,
        "reason": "explicit_world_up_resolved",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--object-id-file", type=Path)
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument("--scene-state", type=Path)
    parser.add_argument(
        "--active-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Review only rows active in --scene-state (fail closed without a state).",
    )
    parser.add_argument(
        "--expand-image-matches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resolve derivative crop-bank files for canonical observation image IDs.",
    )
    parser.add_argument(
        "--preferred-observation-source",
        default="",
        help=(
            "Prefer a complete provenance-tagged observation subset, for example "
            "full_colmap_sam3_refinement. Falls back to all observations unless "
            "--minimum-preferred-observations are available."
        ),
    )
    parser.add_argument("--minimum-preferred-observations", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vllm-url", default=None)
    parser.add_argument("--model", default="qwen3-vl-8b")
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--crops-per-object", type=int, default=3)
    parser.add_argument(
        "--natural-context-panel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compose each VLM image from mask-grounded and natural-context panels."
        ),
    )
    parser.add_argument(
        "--candidate-label-catalog",
        type=Path,
        action="append",
        default=[],
        help="Prior catalog/review supplying auditable label hypotheses; repeatable.",
    )
    parser.add_argument(
        "--candidate-conflicts-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Review only objects with at least two distinct informative hypotheses.",
    )
    parser.add_argument(
        "--selection-report",
        type=Path,
        help="Optional geometry/visual audit; review only object rows whose status ends in '_pass'.",
    )
    parser.add_argument(
        "--geometry-report",
        type=Path,
        help="Optional geometry audit supplying validated metric OBB dimensions for semantic scale checks.",
    )
    parser.add_argument(
        "--world-up-vector",
        default=None,
        metavar="X,Y,Z",
        help=(
            "Exact world-space gravity-up vector for deterministic crop orientation. "
            "Priority: CLI, geometry report up.vector, default 0,-1,0."
        ),
    )
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--verify-kept",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a second generic evidence audit over first-pass kept labels.",
    )
    parser.add_argument(
        "--verification-mode",
        choices=("candidate_conditioned", "independent_blind"),
        default="candidate_conditioned",
        help=(
            "candidate_conditioned preserves the legacy audit; independent_blind "
            "uses the same unconditioned prompt on a disjoint crop fold and "
            "requires canonical noun agreement."
        ),
    )
    parser.add_argument(
        "--require-category-consensus",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Deprecated compatibility flag. Independent verification may refine the "
            "open-vocabulary noun; literal string equality is not a validity test."
        ),
    )
    return parser.parse_args()


def load_object_id_file(path: Path | None) -> set[int] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve(strict=True)
    values: set[int] = set()
    for line_number, raw in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        try:
            object_id = int(value)
        except ValueError as exc:
            raise ValueError(
                f"invalid object ID at {resolved}:{line_number}: {value!r}"
            ) from exc
        if object_id < 0:
            raise ValueError(f"object IDs must be non-negative: {object_id}")
        values.add(object_id)
    if not values:
        raise ValueError(f"object ID allowlist is empty: {resolved}")
    return values


def candidate_label_sets(paths: list[Path]) -> dict[int, set[str]]:
    """Collect hypotheses without treating any prior label as ground truth."""

    result: dict[int, set[str]] = {}
    for path in paths:
        resolved = path.expanduser().resolve(strict=True)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        rows = payload.get("objects") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise TypeError(f"candidate label catalog has no object rows: {resolved}")
        for row in rows:
            if not isinstance(row, dict) or "id" not in row:
                continue
            values = [
                row.get("category"),
                row.get("review_category"),
                (row.get("label_contract") or {}).get("category")
                if isinstance(row.get("label_contract"), dict) else None,
                (row.get("review_label_contract") or {}).get("category")
                if isinstance(row.get("review_label_contract"), dict) else None,
            ]
            labels = result.setdefault(int(row["id"]), set())
            for value in values:
                label = re.sub(r"\s+", " ", str(value or "").lower()).strip()
                if label and label not in {
                    "unknown", "unresolved", "unresolved object", "object"
                }:
                    labels.add(label)
    return result


def candidate_adjudication_prompt(prompt: str, labels: set[str]) -> str:
    hypotheses = ", ".join(sorted(labels))
    return (
        prompt.rstrip()
        + "\n\nPrior independent runs proposed these hypotheses: "
        + hypotheses
        + ". They are untrusted alternatives, not hints or a closed vocabulary. "
          "Inspect the target and natural context yourself; select a hypothesis "
          "only if its diagnostic physical parts and placement are visible, or "
          "return a better supported noun/unknown. Explain the decisive target-"
          "owned and contextual evidence in description."
    )


def filter_active_rows(rows: list[dict], state: dict) -> list[dict]:
    """Return catalog rows whose object IDs are active in the supplied state."""

    if "object_id" not in state or "active" not in state:
        raise KeyError("active-only review requires object_id and active state fields")
    object_ids = np.asarray(state["object_id"]).reshape(-1)
    active = np.asarray(state["active"]).reshape(-1)
    if object_ids.size != active.size:
        raise ValueError("scene-state object_id/active arrays have different lengths")
    active_ids = {
        int(object_id)
        for object_id, is_active in zip(object_ids.tolist(), active.tolist())
        if bool(is_active)
    }
    return [row for row in rows if int(row["id"]) in active_ids]


def hydrate_rows_from_state(rows: list[dict], state: dict) -> dict:
    """Attach canonical metric position/dimensions used by pose-aware crop selection."""

    values = state.get("object_id")
    if isinstance(values, torch.Tensor):
        object_ids = np.asarray(values.detach().cpu()).reshape(-1)
    else:
        object_ids = np.asarray(values).reshape(-1)
    centers_value = state.get("object_box_centers_m", state.get("means"))
    dimensions_value = state.get("object_box_dimensions_m")
    if centers_value is None:
        return {"matched_rows": 0, "position_rows": 0, "dimension_rows": 0}
    if isinstance(centers_value, torch.Tensor):
        centers_value = centers_value.detach().cpu().numpy()
    if isinstance(dimensions_value, torch.Tensor):
        dimensions_value = dimensions_value.detach().cpu().numpy()
    centers = np.asarray(centers_value, dtype=np.float64)
    dimensions = (
        np.asarray(dimensions_value, dtype=np.float64)
        if dimensions_value is not None else np.empty((0, 3), dtype=np.float64)
    )
    if centers.ndim != 2 or centers.shape != (object_ids.size, 3):
        raise ValueError("scene-state object centers are not object-aligned Nx3")
    if dimensions.size and (dimensions.ndim != 2 or dimensions.shape != centers.shape):
        raise ValueError("scene-state object dimensions are not object-aligned Nx3")
    index_by_id = {int(value): index for index, value in enumerate(object_ids.tolist())}
    matched = positions = dimension_rows = 0
    for row in rows:
        index = index_by_id.get(int(row["id"]))
        if index is None:
            continue
        matched += 1
        center = centers[index]
        if np.isfinite(center).all():
            row["position_world_m"] = center.astype(float).tolist()
            positions += 1
        if dimensions.size:
            dimension = dimensions[index]
            if np.isfinite(dimension).all() and np.all(dimension > 0.0):
                row["metric_dimensions_m"] = dimension.astype(float).tolist()
                dimension_rows += 1
    return {
        "matched_rows": matched,
        "position_rows": positions,
        "dimension_rows": dimension_rows,
    }


def select_preferred_observation_paths(
    state: dict,
    object_id: int,
    resolved_paths: list[Path],
    *,
    preferred_source: str,
    minimum_preferred: int,
) -> tuple[list[Path], dict]:
    """Prefer complete refined evidence while retaining an explicit fallback."""

    source = str(preferred_source or "").strip()
    all_paths = list(resolved_paths)
    if not source:
        return all_paths, {
            "mode": "all_resolved",
            "preferred_source": None,
            "preferred_records": 0,
            "selected_paths": len(all_paths),
            "fallback_used": False,
        }
    values = state.get("object_id")
    if isinstance(values, torch.Tensor):
        object_ids = [int(value) for value in values.detach().cpu().reshape(-1).tolist()]
    else:
        object_ids = [int(value) for value in np.asarray(values).reshape(-1).tolist()]
    try:
        index = object_ids.index(int(object_id))
    except ValueError:
        index = -1
    rows = state.get("object_mask_observations")
    records = rows[index] if isinstance(rows, list) and 0 <= index < len(rows) else []
    preferred_tails: set[tuple[str, str]] = set()
    preferred_records = 0
    for record in records if isinstance(records, (list, tuple)) else []:
        if not isinstance(record, dict) or str(record.get("source") or "") != source:
            continue
        text = str(record.get("path") or record.get("mask_path") or "").strip()
        path = Path(text)
        if not text or len(path.parts) < 2:
            continue
        preferred_tails.add((path.parts[-2], path.parts[-1]))
        preferred_records += 1
    preferred = [
        path
        for path in all_paths
        if len(path.parts) >= 2
        and (path.parts[-2], path.parts[-1]) in preferred_tails
    ]
    threshold = max(1, int(minimum_preferred))
    use_preferred = len(preferred) >= threshold
    return (preferred if use_preferred else all_paths), {
        "mode": "preferred_source" if use_preferred else "all_resolved_fallback",
        "preferred_source": source,
        "preferred_records": preferred_records,
        "preferred_resolved_paths": len(preferred),
        "minimum_preferred_observations": threshold,
        "selected_paths": len(preferred) if use_preferred else len(all_paths),
        "fallback_used": not use_preferred,
    }


def _mask_grounded_crop(
    payload: np.lib.npyio.NpzFile,
    encoded: bytes,
    source_image: np.ndarray | None = None,
    natural_context_panel: bool = False,
    quarter_turns_ccw: int = 0,
) -> bytes:
    """Return a crop grounded by its mask without erasing scale context.

    FARM sidecars store a padded RGB crop plus the detector's packed raw mask.
    Earlier review code sent the padded crop unchanged while the prompt claimed
    that the views were masked. Small contextual objects could therefore win
    over the actual segmented foreground. Reconstruct the crop-local mask and
    strongly dim non-mask context while retaining target pixels in their
    natural colours. A thin outer outline makes the mask boundary explicit
    without recolouring the target. Legacy incomplete sidecars retain their crop.
    """

    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return encoded

    turns = int(quarter_turns_ccw) % 4

    def rotate(value: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.rot90(value, k=turns)) if turns else value

    def encode(value: np.ndarray) -> bytes:
        ok, buffer = cv2.imencode(
            ".jpg", value, [int(cv2.IMWRITE_JPEG_QUALITY), 94]
        )
        return bytes(buffer) if ok else encoded

    required = {"raw_bits", "raw_shape", "crop_bbox_xyxy"}
    if not required.issubset(payload.files):
        return encode(rotate(image))
    try:
        raw_height, raw_width = (
            np.asarray(payload["raw_shape"], dtype=np.int32).reshape(2).tolist()
        )
        x0, y0, x1, y1 = (
            np.asarray(payload["crop_bbox_xyxy"], dtype=np.int32).reshape(4).tolist()
        )
    except (TypeError, ValueError):
        return encode(rotate(image))
    if raw_height <= 0 or raw_width <= 0:
        return encode(rotate(image))
    flat = np.unpackbits(
        np.asarray(payload["raw_bits"], dtype=np.uint8), bitorder="little"
    )
    if flat.size < raw_height * raw_width:
        return encode(rotate(image))
    raw = flat[: raw_height * raw_width].reshape(raw_height, raw_width).astype(np.uint8)

    # Prefer a wider crop reconstructed from the original RGB frame.  The
    # detector crop alone can remove the placement evidence that distinguishes
    # a poster from a rigid sign, a pallet from covered furniture, or a box
    # from a loose tool lying on it.  This is category-agnostic and uses only
    # the saved mask/image-index contract.
    used_source_context = False
    if source_image is not None and "raw_bbox_xyxy" in payload.files:
        try:
            sx0, sy0, sx1, sy1 = (
                np.asarray(payload["raw_bbox_xyxy"], dtype=np.int32).reshape(4).tolist()
            )
            source_h, source_w = source_image.shape[:2]
            sx0, sy0 = max(0, int(sx0)), max(0, int(sy0))
            sx1, sy1 = min(source_w, int(sx1)), min(source_h, int(sy1))
            if sx1 > sx0 and sy1 > sy0:
                full_mask = np.zeros((source_h, source_w), dtype=np.uint8)
                full_mask[sy0:sy1, sx0:sx1] = cv2.resize(
                    raw, (sx1 - sx0, sy1 - sy0), interpolation=cv2.INTER_NEAREST
                )
                span_x, span_y = sx1 - sx0, sy1 - sy0
                pad_x = max(48, int(round(0.85 * span_x)))
                pad_y = max(48, int(round(0.85 * span_y)))
                wx0, wy0 = max(0, sx0 - pad_x), max(0, sy0 - pad_y)
                wx1, wy1 = min(source_w, sx1 + pad_x), min(source_h, sy1 + pad_y)
                image = source_image[wy0:wy1, wx0:wx1].copy()
                raw = full_mask[wy0:wy1, wx0:wx1]
                x0, y0, x1, y1 = 0, 0, image.shape[1], image.shape[0]
                used_source_context = True
        except (TypeError, ValueError):
            pass

    height, width = image.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(width, int(x1)), min(height, int(y1))
    if x1 <= x0 or y1 <= y0:
        return encode(rotate(image))
    local = cv2.resize(raw, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = local.astype(bool)
    # Apply the exact same discrete transform to source pixels and their mask
    # before constructing either panel. This preserves pixel/mask alignment and
    # keeps MASK/CONTEXT left-to-right after rotation.
    image = rotate(image)
    mask = rotate(mask)
    height, width = image.shape[:2]
    coverage = float(mask.mean())
    if not np.isfinite(coverage) or coverage <= 0.001:
        return encode(image)

    context_gain, context_offset = (
        (0.42, 8.0) if used_source_context else (0.18, 12.0)
    )
    grounded = np.clip(
        image.astype(np.float32) * context_gain + context_offset, 0, 255
    ).astype(np.uint8)
    grounded[mask] = image[mask]
    outer = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1)
    grounded[outer.astype(bool) & ~mask] = np.asarray([235, 235, 235], dtype=np.uint8)
    if natural_context_panel and used_source_context:
        context = image.copy()
        ys, xs = np.nonzero(mask)
        if xs.size and ys.size:
            bx0, by0 = int(xs.min()), int(ys.min())
            bx1, by1 = int(xs.max()), int(ys.max())
            thickness = max(2, int(round(min(height, width) / 220.0)))
            cv2.rectangle(
                context, (bx0, by0), (bx1, by1), (0, 255, 255), thickness
            )
            context[outer.astype(bool) & ~mask] = np.asarray(
                [0, 255, 255], dtype=np.uint8
            )
        cv2.putText(
            grounded, "MASK", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (235, 235, 235), 2, cv2.LINE_AA,
        )
        cv2.putText(
            context, "CONTEXT", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 255), 2, cv2.LINE_AA,
        )
        grounded = np.concatenate((grounded, context), axis=1)
    return encode(grounded)


def crop_candidates(
    paths: list[Path], *, frames_json: Path | None = None,
    natural_context_panel: bool = False,
    frame_pose_index: dict | None = None,
    world_up_vector: object = DEFAULT_WORLD_UP_VECTOR,
    world_up_source: str = "default:[0,-1,0]",
) -> list[tuple[Path, bytes]]:
    result = []
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as payload:
                encoded = bytes(
                    np.asarray(payload["crop_jpeg_bytes"], dtype=np.uint8)
                )
                # Keep semantic identity tied to the saved FARM object crop.
                # Reconstructing a much wider source-frame window made nearby
                # carts, cabinets and tools dominate small target instances.
                # The crop-local mask still preserves dim placement context,
                # while the outlined natural-colour pixels decide identity.
                source_image = (
                    _source_frame_for_crop(frames_json, path)
                    if frames_json is not None else None
                )
                orientation = gravity_upright_orientation(
                    crop_image_id(path),
                    frame_pose_index=frame_pose_index,
                    world_up_vector=world_up_vector,
                    world_up_source=world_up_source,
                )
                encoded = _mask_grounded_crop(
                    payload,
                    encoded,
                    source_image,
                    natural_context_panel,
                    quarter_turns_ccw=int(
                        orientation.get("applied_quarter_turns_ccw") or 0
                    ),
                )
            if encoded:
                result.append((path, encoded))
        except Exception:
            continue
    result.sort(key=lambda item: len(item[1]), reverse=True)
    return result


def choose_crops(
    candidates: list[tuple[Path, bytes]], requested: int
) -> list[tuple[Path, bytes]]:
    if len(candidates) <= requested:
        return candidates

    def image_id(item: tuple[Path, bytes]) -> int:
        match = re.search(r"img_(\d+)", item[0].name)
        return int(match.group(1)) if match else 0

    def temporal_subset(
        rows: list[tuple[Path, bytes]], count: int
    ) -> list[tuple[Path, bytes]]:
        if count <= 0:
            return []
        if len(rows) <= count:
            return list(rows)
        # Retain a high-detail pool, then choose across temporal/image-id
        # strata. This avoids near-duplicate adjacent views.
        pool = rows[: min(len(rows), max(count * 4, count))]
        chronological = sorted(pool, key=image_id)
        strata = np.array_split(np.arange(len(chronological)), count)
        return [
            max(
                (chronological[int(index)] for index in stratum),
                key=lambda item: len(item[1]),
            )
            for stratum in strata
            if len(stratum)
        ]

    # Assembly crop banks encode their automatically discovered source member
    # in the filename. Balance those members before temporal sampling so a
    # large, visually simple component cannot crowd out every other part.
    # Ordinary FARM object paths contain no member token and retain the legacy
    # category-agnostic temporal selection.
    by_member: dict[int, list[tuple[Path, bytes]]] = {}
    for item in candidates:
        match = re.search(r"member_(\d+)", item[0].name)
        if match:
            by_member.setdefault(int(match.group(1)), []).append(item)
    if 1 < len(by_member) <= requested:
        member_ids = sorted(by_member)
        base, remainder = divmod(requested, len(member_ids))
        selected: list[tuple[Path, bytes]] = []
        for index, member_id in enumerate(member_ids):
            allocation = base + (1 if index < remainder else 0)
            selected.extend(temporal_subset(by_member[member_id], allocation))
        return sorted(selected, key=image_id)[:requested]

    return temporal_subset(candidates, requested)


def choose_evidence_crops(
    candidates: list[tuple[Path, bytes]],
    source: str,
    requested: int,
    *,
    frame_pose_index: dict | None = None,
    object_position_world_m: object = None,
    pose_aware: bool = False,
) -> tuple[list[tuple[Path, bytes]], dict]:
    """Select one deterministic source-view fold and retain its provenance."""

    partitioned, partition = partition_crop_candidates(candidates, source)
    partitioned = sorted(partitioned, key=lambda item: len(item[1]), reverse=True)
    fallback = choose_crops(partitioned, requested)
    if not pose_aware:
        return fallback, partition

    # Assembly banks deliberately balance source members in choose_crops.
    # Preserve that independent semantic safeguard instead of allowing camera
    # diversity to crowd every view onto one visually dominant component.
    member_ids = {
        int(match.group(1))
        for candidate in partitioned
        if (match := re.search(r"member_(\d+)", candidate[0].name))
    }
    selection_candidates = fallback if len(member_ids) > 1 else partitioned
    selected, diagnostics = select_pose_diverse_crop_candidates(
        selection_candidates,
        fallback,
        requested,
        frame_pose_index=frame_pose_index,
        object_position_world_m=object_position_world_m,
    )
    if len(member_ids) > 1:
        diagnostics.update(
            {
                "method": "temporal_fallback",
                "reason": "assembly_member_balancing_preserved",
                "partition_candidate_count": len(partitioned),
            }
        )
    partition = dict(partition)
    partition["selection_diagnostics"] = diagnostics
    return selected, partition


def optimize_cross_fold_pose_diversity(
    candidates: list[tuple[Path, bytes]],
    initial_crops: list[tuple[Path, bytes]],
    verification_crops: list[tuple[Path, bytes]],
    requested: int,
    *,
    frame_pose_index: dict | None,
    object_position_world_m: object,
    detail_floor_ratio: float = DEFAULT_CROP_DETAIL_FLOOR_RATIO,
    min_viewpoint_separation_degrees: float = (
        DEFAULT_MIN_VIEWPOINT_SEPARATION_DEGREES
    ),
    min_normalized_camera_baseline: float = (
        DEFAULT_MIN_NORMALIZED_CAMERA_BASELINE
    ),
    maximum_fold_combinations: int = 16384,
    maximum_joint_pairs: int = 65536,
) -> tuple[
    list[tuple[Path, bytes]],
    list[tuple[Path, bytes]],
    dict,
]:
    """Jointly select two strict, disjoint and pose-independent blind folds.

    Optimizing only the verification fold can get trapped by an otherwise good
    initial selection whose nearest views are too similar. This routine keeps
    the deterministic source partitions and the existing per-fold quality and
    pose gates, but searches both folds together before either VLM request is
    made. It never relaxes an evidence threshold and returns both original
    selections when no fully valid joint choice is found.
    """

    diagnostic: dict = {
        "schema": "farm.cross-fold-pose-selection.v1",
        "status": "fallback",
        "reason": "pose_provenance_or_crop_count_incomplete",
        "evaluated_joint_pair_count": 0,
        "maximum_fold_combination_count": max(
            1, int(maximum_fold_combinations)
        ),
        "maximum_joint_pair_count": max(1, int(maximum_joint_pairs)),
        "thresholds": {
            "detail_floor_ratio": float(detail_floor_ratio),
            "min_viewpoint_separation_degrees": float(
                min_viewpoint_separation_degrees
            ),
            "min_normalized_camera_baseline": float(
                min_normalized_camera_baseline
            ),
        },
    }
    count = max(1, int(requested))
    if (
        len(initial_crops) != count
        or len(verification_crops) != count
        or not isinstance(frame_pose_index, dict)
    ):
        return initial_crops, verification_crops, diagnostic
    try:
        target = np.asarray(object_position_world_m, dtype=np.float64)
    except (TypeError, ValueError):
        return initial_crops, verification_crops, diagnostic
    if target.shape != (3,) or not np.isfinite(target).all():
        return initial_crops, verification_crops, diagnostic

    angle_threshold = max(0.0, float(min_viewpoint_separation_degrees))
    baseline_threshold = max(0.0, float(min_normalized_camera_baseline))
    detail_floor = min(1.0, max(0.0, float(detail_floor_ratio)))
    fold_cap = max(1, int(maximum_fold_combinations))
    joint_cap = max(1, int(maximum_joint_pairs))

    def posed_pool(source: str | None) -> list[dict]:
        partitioned = (
            partition_crop_candidates(candidates, source)[0]
            if source is not None
            else list(candidates)
        )
        by_image: dict[int, tuple[Path, bytes]] = {}
        for candidate in sorted(
            partitioned,
            key=lambda item: (
                crop_image_id(item) is None,
                (
                    crop_image_id(item)
                    if crop_image_id(item) is not None
                    else 2**63
                ),
                -len(item[1]),
                str(item[0]),
            ),
        ):
            image_id = crop_image_id(candidate)
            if image_id is not None:
                by_image.setdefault(int(image_id), candidate)
        pool: list[dict] = []
        for image_id in sorted(by_image):
            raw_pose = frame_pose_index.get(image_id)
            if not isinstance(raw_pose, dict):
                continue
            try:
                center = np.asarray(
                    raw_pose.get("camera_center_world_m"), dtype=np.float64
                )
                forward = np.asarray(
                    raw_pose.get("camera_forward_world"), dtype=np.float64
                )
            except (TypeError, ValueError):
                continue
            if (
                center.shape != (3,)
                or forward.shape != (3,)
                or not np.isfinite(center).all()
                or not np.isfinite(forward).all()
                or float(np.linalg.norm(forward)) <= 1.0e-12
            ):
                continue
            offset = target - center
            range_m = float(np.linalg.norm(offset))
            if not np.isfinite(range_m) or range_m <= 1.0e-12:
                continue
            candidate = by_image[image_id]
            pool.append({
                "candidate": candidate,
                "image_id": int(image_id),
                "camera_center_world_m": center,
                "view_ray": offset / range_m,
                "range_m": range_m,
                "detail": int(len(candidate[1])),
            })
        return pool

    initial_pool = posed_pool("initial_blind_review")
    source_verification_pool = posed_pool("independent_verification")
    all_posed = posed_pool(None)
    initial_pool_ids = {int(row["image_id"]) for row in initial_pool}
    source_verification_ids = {
        int(row["image_id"]) for row in source_verification_pool
    }
    # With three or more temporal partitions, reserving only channel 1 can
    # needlessly discard a much stronger unseen channel. The second blind
    # request may use any posed view outside the initial partition. When the
    # bank is too small to form multiple partitions both sources map to the
    # same pool, and the pairwise disjointness check below performs the split.
    verification_pool = (
        [
            row for row in all_posed
            if int(row["image_id"]) not in initial_pool_ids
        ]
        if initial_pool_ids != source_verification_ids
        else all_posed
    )
    diagnostic["verification_pool_policy"] = (
        "all_posed_views_outside_initial_partition"
        if initial_pool_ids != source_verification_ids
        else "joint_disjoint_split_of_single_partition"
    )
    diagnostic["initial_posed_unique_count"] = len(initial_pool)
    diagnostic["verification_posed_unique_count"] = len(verification_pool)
    if len(initial_pool) < count or len(verification_pool) < count:
        diagnostic["reason"] = "insufficient_unique_posed_views"
        return initial_crops, verification_crops, diagnostic

    def within_metrics(rows: list[dict]) -> tuple[float, float, float]:
        angles: list[float] = []
        normalized_baselines: list[float] = []
        for left, right in itertools.combinations(rows, 2):
            cosine = float(np.dot(left["view_ray"], right["view_ray"]))
            angles.append(
                float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
            )
            baseline = float(np.linalg.norm(
                left["camera_center_world_m"]
                - right["camera_center_world_m"]
            ))
            normalized_baselines.append(
                baseline
                / max(
                    1.0e-9,
                    (float(left["range_m"]) + float(right["range_m"]))
                    / 2.0,
                )
            )
        if not angles:
            return 0.0, 0.0, 0.0
        return (
            float(min(angles)),
            float(np.median(np.asarray(angles, dtype=np.float64))),
            float(min(normalized_baselines)),
        )

    def fold_combinations(
        pool: list[dict], fallback: list[tuple[Path, bytes]]
    ) -> tuple[list[dict], dict]:
        combination_count = math.comb(len(pool), count)

        def partial_rank(indices: tuple[int, ...]) -> tuple:
            rows = [pool[index] for index in indices]
            metrics = (
                within_metrics(rows)
                if len(rows) > 1
                else (0.0, 0.0, 0.0)
            )
            details = [int(row["detail"]) for row in rows]
            return (
                metrics[0],
                metrics[2],
                min(details) if details else 0,
                sum(details),
                tuple(-int(row["image_id"]) for row in rows),
            )

        if combination_count <= fold_cap:
            index_combinations = list(
                itertools.combinations(range(len(pool)), count)
            )
            search_mode = "exact"
        else:
            beam: list[tuple[int, ...]] = [tuple()]
            for depth in range(count):
                remaining = count - depth - 1
                expanded = (
                    prefix + (index,)
                    for prefix in beam
                    for index in range(
                        (prefix[-1] + 1) if prefix else 0,
                        len(pool) - remaining,
                    )
                )
                beam = heapq.nlargest(fold_cap, expanded, key=partial_rank)
                if not beam:
                    break
            index_combinations = beam
            search_mode = "beam"

        fallback_detail = sum(len(payload) for _, payload in fallback)
        minimum_detail = detail_floor * float(fallback_detail)
        valid: list[dict] = []
        for indices in index_combinations:
            if len(indices) != count:
                continue
            rows = [pool[index] for index in indices]
            detail = sum(int(row["detail"]) for row in rows)
            if detail + 1.0e-9 < minimum_detail:
                continue
            metrics = within_metrics(rows)
            if metrics[0] < angle_threshold or metrics[2] < baseline_threshold:
                continue
            image_ids = tuple(int(row["image_id"]) for row in rows)
            valid.append({
                "rows": rows,
                "image_ids": image_ids,
                "image_id_set": frozenset(image_ids),
                "metrics": metrics,
                "detail": detail,
                "rank": (
                    metrics[0],
                    metrics[2],
                    min(int(row["detail"]) for row in rows),
                    detail,
                    metrics[1],
                    tuple(-image_id for image_id in image_ids),
                ),
            })
        valid.sort(key=lambda item: item["rank"], reverse=True)
        return valid, {
            "candidate_combination_count": int(combination_count),
            "evaluated_combination_count": int(len(index_combinations)),
            "valid_combination_count": int(len(valid)),
            "search_mode": search_mode,
            "search_complete": combination_count <= fold_cap,
        }

    initial_combinations, initial_search = fold_combinations(
        initial_pool, initial_crops
    )
    verification_combinations, verification_search = fold_combinations(
        verification_pool, verification_crops
    )
    diagnostic["initial_fold_search"] = initial_search
    diagnostic["verification_fold_search"] = verification_search
    if not initial_combinations or not verification_combinations:
        diagnostic["reason"] = "no_strict_within_fold_pose_combination"
        return initial_crops, verification_crops, diagnostic

    total_joint_pairs = len(initial_combinations) * len(
        verification_combinations
    )
    if total_joint_pairs <= joint_cap:
        searched_initial = initial_combinations
        searched_verification = verification_combinations
        joint_search_complete = True
    else:
        initial_limit = min(
            len(initial_combinations),
            max(
                1,
                int(math.sqrt(
                    joint_cap
                    * len(initial_combinations)
                    / max(1, len(verification_combinations))
                )),
            ),
        )
        verification_limit = min(
            len(verification_combinations),
            max(1, joint_cap // initial_limit),
        )
        searched_initial = initial_combinations[:initial_limit]
        searched_verification = verification_combinations[:verification_limit]
        joint_search_complete = False
    diagnostic.update({
        "candidate_joint_pair_count": int(total_joint_pairs),
        "searched_initial_combination_count": len(searched_initial),
        "searched_verification_combination_count": len(searched_verification),
        "joint_search_complete": joint_search_complete,
    })

    def event(source: str, crops: list[tuple[Path, bytes]]) -> dict:
        return {
            "source": source,
            "event_id": source,
            "confirmation_eligible": True,
            **crop_evidence_manifest(
                0,
                crops,
                frame_pose_index=frame_pose_index,
                object_position_world_m=target.tolist(),
            ),
        }

    best: tuple[tuple, dict, dict, dict] | None = None
    for initial_choice in searched_initial:
        for verification_choice in searched_verification:
            diagnostic["evaluated_joint_pair_count"] += 1
            if initial_choice["image_id_set"].intersection(
                verification_choice["image_id_set"]
            ):
                continue
            initial_selected = [
                row["candidate"] for row in initial_choice["rows"]
            ]
            verification_selected = [
                row["candidate"] for row in verification_choice["rows"]
            ]
            cross = evidence_pose_diversity(
                event("initial_blind_review", initial_selected),
                event("independent_verification", verification_selected),
                min_viewpoint_separation_degrees=angle_threshold,
                min_normalized_camera_baseline=baseline_threshold,
            )
            if cross.get("pose_independent") is not True:
                continue
            rank = (
                float(
                    cross.get(
                        "symmetric_median_nearest_viewpoint_angle_degrees"
                    ) or 0.0
                ),
                float(
                    cross.get(
                        "symmetric_median_nearest_normalized_baseline"
                    ) or 0.0
                ),
                min(
                    initial_choice["metrics"][0],
                    verification_choice["metrics"][0],
                ),
                min(
                    initial_choice["metrics"][2],
                    verification_choice["metrics"][2],
                ),
                min(initial_choice["detail"], verification_choice["detail"]),
                initial_choice["detail"] + verification_choice["detail"],
                tuple(-image_id for image_id in initial_choice["image_ids"]),
                tuple(
                    -image_id for image_id in verification_choice["image_ids"]
                ),
            )
            if best is None or rank > best[0]:
                best = (rank, initial_choice, verification_choice, cross)
    if best is None:
        diagnostic["reason"] = "no_strict_cross_fold_pose_combination"
        return initial_crops, verification_crops, diagnostic
    _, initial_choice, verification_choice, cross = best
    selected_initial = [row["candidate"] for row in initial_choice["rows"]]
    selected_verification = [
        row["candidate"] for row in verification_choice["rows"]
    ]
    diagnostic.update({
        "status": "selected",
        "reason": "strict_within_and_cross_fold_pose_gates_met",
        "selected_initial_image_ids": list(initial_choice["image_ids"]),
        "selected_verification_image_ids": list(
            verification_choice["image_ids"]
        ),
        "initial_within_fold": {
            "min_pairwise_viewpoint_angle_degrees": initial_choice["metrics"][0],
            "median_pairwise_viewpoint_angle_degrees": initial_choice["metrics"][1],
            "min_pairwise_normalized_camera_baseline": initial_choice["metrics"][2],
            "selected_detail_bytes": initial_choice["detail"],
        },
        "verification_within_fold": {
            "min_pairwise_viewpoint_angle_degrees": verification_choice["metrics"][0],
            "median_pairwise_viewpoint_angle_degrees": verification_choice["metrics"][1],
            "min_pairwise_normalized_camera_baseline": verification_choice["metrics"][2],
            "selected_detail_bytes": verification_choice["detail"],
        },
        "cross_fold": cross,
    })
    return selected_initial, selected_verification, diagnostic


def enforce_disjoint_verification_fold(
    initial_crops: list[tuple[Path, bytes]],
    verification_crops: list[tuple[Path, bytes]],
    partition: dict,
) -> tuple[list[tuple[Path, bytes]], dict]:
    """Reject any verification fold that reuses first-pass source views."""

    initial_ids = {crop_image_id(item) for item in initial_crops}
    verification_ids = {crop_image_id(item) for item in verification_crops}
    parsed_overlap = sorted(
        int(value)
        for value in initial_ids.intersection(verification_ids)
        if value is not None
    )
    initial_names = {item[0].name for item in initial_crops}
    verification_names = {item[0].name for item in verification_crops}
    name_overlap = sorted(initial_names.intersection(verification_names))
    unparseable = None in initial_ids or None in verification_ids
    unavailable = bool(parsed_overlap or name_overlap or unparseable)
    audit = dict(partition)
    audit["blind_verification_separation"] = {
        "status": "unavailable" if unavailable else "disjoint",
        "initial_image_ids": sorted(
            int(value) for value in initial_ids if value is not None
        ),
        "verification_image_ids": sorted(
            int(value) for value in verification_ids if value is not None
        ),
        "overlap_image_ids": parsed_overlap,
        "overlap_observation_names": name_overlap,
        "all_image_ids_parseable": not unparseable,
        "reason": (
            "verification_reuses_blind_evidence"
            if parsed_overlap or name_overlap
            else (
                "unparseable_source_image_id"
                if unparseable
                else "disjoint_source_image_ids"
            )
        ),
    }
    return ([] if unavailable else verification_crops), audit


def decode_crop(encoded: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode crop JPEG")
    return image


def data_url(encoded: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def parse_json_text(text: str) -> dict:
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)
    try:
        value = json.loads(clean)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                pass
    return {"category": "unknown", "description": clean[:300], "confidence": 0.0, "decision": "unknown"}


def prompt_with_metric_evidence(prompt: str, row: dict) -> str:
    """Append validated geometry only, never a class or scene hint."""

    try:
        dimensions = np.asarray(row.get("metric_dimensions_m"), dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return prompt
    if not np.isfinite(dimensions).all() or np.any(dimensions <= 0.0):
        return prompt
    values = " × ".join(f"{float(value):.3f}" for value in dimensions)
    return (
        prompt.rstrip()
        + "\n\nValidated multi-view metric OBB dimensions (L × W × H): "
        + values
        + " metres. The third value is the gravity-aligned height. Use this only "
          "as a physical-scale sanity check: reject a noun "
          "whose normal real-world size is clearly incompatible with this object."
    )


def semantic_evidence_event(
    row: dict,
    source: str,
    event_id: str,
    *,
    crops: list[tuple[Path, bytes]] | None = None,
    partition: dict | None = None,
    model_id: str = "",
    prompt_id: str = "",
    prompt: str = "",
    confirmation_eligible: bool = True,
    ineligibility_reason: str = "",
    frame_pose_index: dict | None = None,
    world_up_vector: object = DEFAULT_WORLD_UP_VECTOR,
    world_up_source: str = "default:[0,-1,0]",
) -> dict:
    """Serialize one model request as auditable semantic evidence.

    A request remains in the audit trail even when it returned ``unknown``. Later
    consensus stages must inspect ``decision`` before treating it as positive
    evidence and can deduplicate correlated/derived rows by ``event_id``.
    """

    event = {
        "source": str(source),
        "event_id": str(event_id),
        "category": str(row.get("review_category") or "unknown").strip().lower(),
        "description": str(row.get("review_description") or "").strip(),
        "attributes": [
            str(value).strip()
            for value in (row.get("review_attributes") or [])
            if str(value).strip()
        ][:8],
        "confidence": float(row.get("review_confidence") or 0.0),
        "decision": str(row.get("review_decision") or "unknown").strip().lower(),
        "model_id": str(model_id),
        "prompt_id": str(prompt_id),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "confirmation_eligible": bool(confirmation_eligible),
    }
    label_contract = row.get("review_label_contract")
    if isinstance(label_contract, dict):
        event["label_contract"] = dict(label_contract)
    if ineligibility_reason:
        event["confirmation_ineligibility_reason"] = str(ineligibility_reason)
        if "candidate_conditioned" in str(ineligibility_reason):
            event["request_conditioning"] = "candidate_label_or_description"
            event["semantic_vote_independence"] = "not_independent"
    observation_selection = row.get("semantic_observation_selection")
    if isinstance(observation_selection, dict):
        event["observation_selection"] = dict(observation_selection)
    if crops is not None:
        event.update(
            crop_evidence_manifest(
                int(row.get("id", -1)),
                crops,
                partition=partition,
                frame_pose_index=frame_pose_index,
                object_position_world_m=row.get("position_world_m"),
            )
        )
        event["gravity_upright_normalization"] = (
            gravity_upright_evidence_manifest(
                crops,
                frame_pose_index=frame_pose_index,
                world_up_vector=world_up_vector,
                world_up_source=world_up_source,
            )
        )
    return event


def verification_gate_reason(initial: dict, verified: dict) -> str:
    """Describe candidate-conditioned verification without claiming independence."""

    def normalize(value: object) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()

    return (
        "candidate_conditioned_verification_agrees"
        if normalize(initial.get("category")) == normalize(verified.get("review_category"))
        else "candidate_conditioned_verification_revised_category"
    )


def independent_blind_consensus(
    initial: dict,
    verification: dict,
    *,
    minimum_confidence: float,
) -> dict:
    """Require two valid blind contracts to agree on one concrete noun."""

    reasons: list[str] = []

    def assess(value: dict, prefix: str) -> tuple[str, dict]:
        contract = value.get("label_contract")
        if not isinstance(contract, dict):
            contract = value.get("review_label_contract")
        assessment = assess_open_vocabulary_whole_identity(
            contract if isinstance(contract, dict) else None,
            minimum_confidence=float(minimum_confidence),
        )
        assessed_contract = assessment.get("contract") or {}
        if assessed_contract.get("contract_valid") is not True:
            reasons.append(f"{prefix}_label_contract_invalid")
        noun = normalize_open_noun(assessment.get("category"))
        if noun in PUBLICATION_FORBIDDEN_GENERIC_NOUNS:
            reasons.append(f"{prefix}_category_generic_or_unresolved")
        if (
            not assessment.get("usable")
            and assessed_contract.get("contract_valid") is True
            and normalize_open_noun(assessed_contract.get("category"))
            not in PUBLICATION_FORBIDDEN_GENERIC_NOUNS
        ):
            reasons.append(f"{prefix}_whole_identity_not_supported")
        return noun, assessment

    event_pair = any(
        key in initial or key in verification
        for key in ("source", "event_id", "crop_image_ids")
    )
    initial_event_id = str(initial.get("event_id") or "")
    verification_event_id = str(verification.get("event_id") or "")
    initial_ids = {
        int(value)
        for value in initial.get("crop_image_ids") or []
        if not isinstance(value, bool) and str(value).isdigit()
    }
    verification_ids = {
        int(value)
        for value in verification.get("crop_image_ids") or []
        if not isinstance(value, bool) and str(value).isdigit()
    }
    if event_pair:
        if str(initial.get("source") or "") not in BLIND_INITIAL_EVENT_SOURCES:
            reasons.append("initial_event_source_invalid")
        if str(verification.get("source") or "") != "independent_verification":
            reasons.append("verification_event_source_invalid")
        if (
            not initial_event_id
            or not verification_event_id
            or initial_event_id == verification_event_id
        ):
            reasons.append("semantic_event_ids_missing_or_not_unique")
        if (
            initial.get("confirmation_eligible") is not True
            or verification.get("confirmation_eligible") is not True
        ):
            reasons.append("event_pair_not_confirmation_eligible")
        if any(
            str(value.get("semantic_vote_independence") or "") != "independent"
            for value in (initial, verification)
        ):
            reasons.append("event_pair_not_explicitly_independent")
        if any(
            str(value.get("request_conditioning") or "") != "none_blind"
            for value in (initial, verification)
        ):
            reasons.append("event_pair_not_unconditioned")
        if not initial_ids or not verification_ids or initial_ids & verification_ids:
            reasons.append("event_pair_view_folds_not_disjoint")
        initial_fingerprint = str(
            initial.get("evidence_fingerprint_sha256") or ""
        )
        verification_fingerprint = str(
            verification.get("evidence_fingerprint_sha256") or ""
        )
        if (
            not initial_fingerprint
            or not verification_fingerprint
            or initial_fingerprint == verification_fingerprint
        ):
            reasons.append("event_pair_evidence_not_unique")
        overlap = evidence_view_overlap(initial, verification)
        if overlap.get("independent") is not True:
            reasons.append(
                "event_pair_"
                + str(overlap.get("reason") or "provenance_not_independent")
            )

    initial_noun, initial_assessment = assess(initial, "initial")
    verification_noun, verification_assessment = assess(
        verification, "verification"
    )
    if initial_noun and verification_noun and initial_noun != verification_noun:
        reasons.append("independent_blind_category_disagreement")
    accepted = not reasons and initial_noun == verification_noun
    mask_scope_reasons = sorted(set(
        list(initial_assessment.get("mask_scope_reason_codes") or [])
        + list(verification_assessment.get("mask_scope_reason_codes") or [])
    ) - {"mask_scope_complete"})
    result = {
        "schema": "farm.independent-blind-label-consensus.v1",
        "status": "accepted" if accepted else "unresolved",
        "accepted": accepted,
        "initial_category": initial_noun,
        "verification_category": verification_noun,
        "canonical_category": initial_noun if accepted else "",
        "initial_maximum_semantic_tier": initial_assessment.get(
            "maximum_semantic_tier"
        ),
        "verification_maximum_semantic_tier": verification_assessment.get(
            "maximum_semantic_tier"
        ),
        "reason_codes": sorted(set(reasons)) or [
            "exact_safe_canonical_noun_agreement"
        ],
        "identity_contract_source": "raw_label_contract",
        "mask_scope_incomplete": bool(mask_scope_reasons),
        "mask_scope_reason_codes": mask_scope_reasons or [
            "mask_scope_complete"
        ],
        "geometry_refinement_required": bool(mask_scope_reasons),
        "inpainting_eligible": bool(accepted and not mask_scope_reasons),
    }
    if event_pair:
        result.update(
            {
                "evidence_pair_source": "exact_current_semantic_evidence_events",
                "initial_event_id": initial_event_id,
                "verification_event_id": verification_event_id,
                "initial_crop_image_ids": sorted(initial_ids),
                "verification_crop_image_ids": sorted(verification_ids),
                "initial_evidence_fingerprint_sha256": str(
                    initial.get("evidence_fingerprint_sha256") or ""
                ),
                "verification_evidence_fingerprint_sha256": str(
                    verification.get("evidence_fingerprint_sha256") or ""
                ),
            }
        )
    return result


def current_independent_blind_event_pair(
    row: dict, verification_event: dict
) -> tuple[dict, dict]:
    """Bind consensus to exactly the two events emitted by this review run."""

    object_id = int(row.get("id", -1))
    evidence = row.get("semantic_evidence")
    if not isinstance(evidence, list) or len(evidence) != 1:
        raise ValueError("independent blind review requires exactly one current initial event")
    initial_event = evidence[0]
    if not isinstance(initial_event, dict):
        raise ValueError("current initial semantic event is invalid")
    initial_source = str(initial_event.get("source") or "")
    if initial_source not in BLIND_INITIAL_EVENT_SOURCES:
        raise ValueError("current initial semantic event source is invalid")
    expected_initial_id = f"object:{object_id}:{initial_source}"
    expected_verification_id = f"object:{object_id}:independent_verification"
    if str(initial_event.get("event_id") or "") != expected_initial_id:
        raise ValueError("current initial semantic event ID is invalid or stale")
    if str(verification_event.get("source") or "") != "independent_verification":
        raise ValueError("current independent verification event source is invalid")
    if str(verification_event.get("event_id") or "") != expected_verification_id:
        raise ValueError("current independent verification event ID is invalid or stale")
    if expected_initial_id == expected_verification_id:
        raise ValueError("semantic event IDs are not unique")
    return initial_event, verification_event


def verification_request_contract(
    mode: str,
    *,
    blind_prompt: str,
    initial: dict,
) -> dict:
    """Return an auditable verification request without hidden conditioning."""

    if mode == "independent_blind":
        return {
            "prompt": blind_prompt,
            "prompt_id": "independent_blind_verification.v1",
            "confirmation_eligible": True,
            "ineligibility_reason": "",
            "semantic_vote_independence": "independent",
            "request_conditioning": "none_blind",
        }
    if mode != "candidate_conditioned":
        raise ValueError(f"unsupported verification mode: {mode}")
    return {
        "prompt": VERIFICATION_PROMPT.format(
            category=str(initial.get("category") or "unknown"),
            description=str(initial.get("description") or ""),
        ),
        "prompt_id": "candidate_conditioned_verification.v1",
        "confirmation_eligible": False,
        "ineligibility_reason": "candidate_conditioned_on_initial_review",
        "semantic_vote_independence": "not_independent",
        "request_conditioning": "candidate_label_or_description",
    }


SEMANTIC_QUARANTINE_ACTION = "quarantine_unresolved_semantics"


def quarantine_unresolved_semantics(
    row: dict,
    *,
    gate_reason: str,
    reason_codes: list[str] | tuple[str, ...] = (),
) -> None:
    """Make an unresolved blind review impossible to publish accidentally.

    The input category remains available as an explicitly untrusted archival
    prior so geometry-only consumers do not lose provenance. Publication
    fields are replaced with an unresolved value; complete blind votes stay in
    semantic_evidence and review_initial for later adjudication.
    """

    if "semantic_prior_label" not in row:
        row["semantic_prior_label"] = {
            "category": str(row.get("category") or "unresolved object"),
            "description": str(row.get("description") or ""),
            "semantic_tier": str(row.get("semantic_tier") or ""),
            "semantic_status": str(row.get("semantic_status") or ""),
        }
    hypotheses: list[str] = []
    diagnostic_votes = list(row.get("semantic_evidence") or [])
    diagnostic_votes.extend(
        value
        for value in (row.get("review_initial"), row.get("review_verification"))
        if isinstance(value, dict)
    )
    for event in diagnostic_votes:
        if not isinstance(event, dict):
            continue
        contract = event.get("label_contract")
        if not isinstance(contract, dict):
            contract = event.get("review_label_contract")
        noun = normalize_open_noun(
            contract.get("category") if isinstance(contract, dict)
            else event.get("category", event.get("review_category"))
        )
        if noun not in PUBLICATION_FORBIDDEN_GENERIC_NOUNS and noun not in hypotheses:
            hypotheses.append(noun)
    row.update({
        "review_category": "unknown",
        "review_description": (
            "Independent blind semantic evidence is unresolved; retained "
            "hypotheses are diagnostic only and require review."
        ),
        "review_attributes": [],
        "review_label_contract": None,
        "review_confidence": 0.0,
        "review_decision": "unknown",
        "review_gate_reason": str(gate_reason),
        "semantic_publication_action": SEMANTIC_QUARANTINE_ACTION,
        "semantic_status": "independent_blind_unresolved",
        "semantic_quarantined": True,
        "semantic_review_required": True,
        "label_publication_eligible": False,
        "semantic_quarantine_reason_codes": sorted(set(map(str, reason_codes))),
        "unresolved_semantic_hypotheses": hypotheses,
    })


def apply_independent_blind_result(
    row: dict,
    initial: dict,
    verification: dict,
    *,
    minimum_confidence: float,
) -> dict:
    """Publish only exact safe noun agreement and retain both blind votes."""

    consensus = independent_blind_consensus(
        initial,
        verification,
        minimum_confidence=minimum_confidence,
    )
    row["review_verification"] = {
        key: value
        for key, value in verification.items()
        if key != "review_raw_response"
    }
    row.setdefault(
        "review_initial",
        {
            key: value
            for key, value in initial.items()
            if key not in {"raw_response", "review_raw_response"}
        },
    )
    row["independent_blind_consensus"] = consensus

    # Restore the first blind vote before applying consensus. The second vote is
    # evidence, never an implicit replacement for a disagreeing first noun.
    row["review_category"] = str(initial.get("category") or "unknown")
    row["review_description"] = str(initial.get("description") or "")
    row["review_attributes"] = list(initial.get("attributes") or [])
    row["review_label_contract"] = initial.get("label_contract")
    row["review_confidence"] = float(initial.get("confidence") or 0.0)
    if consensus["accepted"]:
        mask_scope_incomplete = bool(consensus.get("mask_scope_incomplete"))
        row["review_category"] = consensus["canonical_category"]
        row["review_confidence"] = min(
            float(initial.get("confidence") or 0.0),
            float(
                verification.get("confidence", verification.get("review_confidence"))
                or 0.0
            ),
        )
        row["review_decision"] = "keep"
        row["review_gate_reason"] = (
            "independent_blind_exact_safe_noun_agreement"
        )
        row["semantic_publication_action"] = (
            "retain_verified_identity_for_mask_refinement"
            if mask_scope_incomplete
            else "publish_independent_blind_consensus"
        )
        row["semantic_status"] = (
            "independent_blind_identity_verified_mask_scope_incomplete"
            if mask_scope_incomplete
            else "independent_blind_verified"
        )
        row["semantic_quarantined"] = False
        row["semantic_review_required"] = False
        row["semantic_identity_verified"] = True
        row["mask_scope_incomplete"] = mask_scope_incomplete
        row["mask_scope_reason_codes"] = list(
            consensus.get("mask_scope_reason_codes") or []
        )
        row["geometry_refinement_required"] = mask_scope_incomplete
        row["geometry_refinement_route"] = (
            "mask_geometry_refinement"
            if mask_scope_incomplete
            else "none"
        )
        row["inpainting_eligible"] = bool(
            consensus.get("inpainting_eligible")
        )
        row["label_publication_eligible"] = not mask_scope_incomplete
        row["semantic_quarantine_reason_codes"] = []
    else:
        quarantine_unresolved_semantics(
            row,
            gate_reason="independent_blind_consensus_failed",
            reason_codes=list(consensus.get("reason_codes") or ()),
        )
    return consensus


def request_review(
    url: str,
    model: str,
    row: dict,
    crops: list[tuple[Path, bytes]],
    prompt: str = PROMPT,
    *,
    strict_open_vocabulary: bool = False,
) -> dict:
    content = [
        {
            "type": "text",
            "text": prompt_with_metric_evidence(prompt, row),
        }
    ]
    content.extend(
        {"type": "image_url", "image_url": {"url": data_url(encoded)}}
        for _, encoded in crops
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 560,
        "response_format": OPEN_VOCABULARY_RESPONSE_FORMAT,
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {"max_pixels": VLM_MAX_IMAGE_PIXELS},
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt == 2:
                raise
            time.sleep(0.5 * (2 ** attempt))
        except (
            urllib.error.URLError, http.client.RemoteDisconnected,
            TimeoutError, ConnectionError, OSError,
        ):
            if attempt == 2:
                raise
            time.sleep(0.5 * (2 ** attempt))
    text = result["choices"][0]["message"]["content"]
    review = parse_json_text(text)
    label_contract: dict = {}
    if strict_open_vocabulary:
        label_contract = parse_open_vocabulary_label(review)
        assessed = assess_open_vocabulary_label(
            label_contract, minimum_confidence=0.0
        )
        category = str(assessed.get("category") or "unknown")
        description = str(
            assessed.get("description")
            or label_contract.get("description")
            or ""
        ).strip()
        decision = "keep" if assessed.get("usable") else "unknown"
        review = label_contract
    else:
        category = str(review.get("category") or "unknown").strip().lower()
        description = str(review.get("description") or "").strip()
        decision = str(review.get("decision") or "unknown").strip().lower()
        # Compatibility is restricted to utility/test callers. Production
        # semantic requests use the strict contract and never coerce invalid JSON.
        if decision not in {"keep", "unknown"}:
            decision = "unknown" if category == "unknown" else "keep"
    try:
        confidence = float(review.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    attributes = (
        assessed.get("attributes")
        if strict_open_vocabulary
        else review.get("attributes")
    ) or []
    if isinstance(attributes, str):
        attributes = [attributes]
    output = {
        "review_category": category,
        "review_description": description,
        "review_attributes": [str(value).strip() for value in attributes if str(value).strip()][:8],
        "review_confidence": float(np.clip(confidence, 0.0, 1.0)),
        "review_decision": decision,
        "review_raw_response": text,
    }
    if label_contract:
        output["review_label_contract"] = label_contract
    return output


def require_successful_transport(report: dict) -> None:
    """Fail the stage when infrastructure errors were recorded as row evidence."""

    initial = [int(value) for value in report.get("initial_request_error_ids") or []]
    verification = [
        int(value) for value in report.get("verification_request_error_ids") or []
    ]
    if initial or verification:
        raise RuntimeError(
            "VLM transport failed for "
            f"{len(initial)} initial and {len(verification)} verification requests; "
            "semantic unknown is reserved for successful model decisions"
        )


def make_pages(rows: list[dict], crop_map: dict[int, list[tuple[Path, bytes]]], output_dir: Path) -> list[str]:
    page_size = 20
    outputs = []
    for page_index in range((len(rows) + page_size - 1) // page_size):
        page_rows = rows[page_index * page_size : (page_index + 1) * page_size]
        canvas = np.full((2160, 3840, 3), (7, 9, 14), dtype=np.uint8)
        cv2.putText(
            canvas, f"OBJECT CROP AUDIT | PAGE {page_index + 1}",
            (62, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (245, 247, 250), 3, cv2.LINE_AA,
        )
        for slot, row in enumerate(page_rows):
            col, grid_row = slot % 5, slot // 5
            x, y = 38 + col * 760, 118 + grid_row * 505
            cv2.rectangle(canvas, (x, y), (x + 724, y + 470), (18, 24, 35), -1)
            crops = crop_map.get(int(row["id"])) or []
            if crops:
                images = []
                for _, encoded in crops[:3]:
                    image = decode_crop(encoded)
                    image = cv2.resize(image, (228, 260), interpolation=cv2.INTER_AREA)
                    images.append(image)
                while len(images) < 3:
                    images.append(np.full((260, 228, 3), (12, 16, 24), dtype=np.uint8))
                strip = np.concatenate(images[:3], axis=1)
                canvas[y + 8 : y + 268, x + 20 : x + 704] = strip
            observations = int(row.get("observations", row.get("observation_count", 0)) or 0)
            raw = f"#{row['id']:03d} obs={observations} RAW: {str(row['category']).upper()}"
            cv2.putText(canvas, raw, (x + 18, y + 302), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (240, 243, 248), 1, cv2.LINE_AA)
            review_category = row.get("review_category")
            if review_category:
                decision = str(row.get("review_decision", ""))
                confidence = float(row.get("review_confidence", 0.0))
                line = f"VLM: {review_category.upper()} | {decision} | {confidence:.2f}"
                color = (128, 230, 170) if decision == "keep" else (75, 205, 245)
                cv2.putText(canvas, line, (x + 18, y + 334), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
                description = str(row.get("review_description") or "")
            else:
                description = str(row.get("caption") or "")
            words = description.split()
            lines, current = [], []
            for word in words:
                if len(" ".join(current + [word])) > 72:
                    lines.append(" ".join(current))
                    current = [word]
                else:
                    current.append(word)
            if current:
                lines.append(" ".join(current))
            for line_index, line in enumerate(lines[:3]):
                cv2.putText(
                    canvas, line, (x + 18, y + 372 + line_index * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (177, 187, 203), 1, cv2.LINE_AA,
                )
        output_path = output_dir / f"object_crop_audit_page_{page_index + 1:02d}.jpg"
        cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        outputs.append(output_path.name)
    return outputs


def main() -> None:
    overall_started = time.perf_counter()
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.catalog.read_text(encoding="utf-8"))
    requested_ids = load_object_id_file(args.object_id_file)
    if requested_ids is not None:
        known_ids = {int(row["id"]) for row in rows}
        missing_ids = sorted(requested_ids - known_ids)
        if missing_ids:
            raise ValueError(
                f"requested object IDs are absent from catalog: {missing_ids}"
            )
        rows = [row for row in rows if int(row["id"]) in requested_ids]

    candidate_labels_by_id = candidate_label_sets(args.candidate_label_catalog)
    for row in rows:
        label = re.sub(
            r"\s+", " ", str(row.get("category") or "").lower()
        ).strip()
        if label and label not in {
            "unknown", "unresolved", "unresolved object", "object"
        }:
            candidate_labels_by_id.setdefault(int(row["id"]), set()).add(label)
    if args.candidate_conflicts_only:
        if not args.candidate_label_catalog:
            raise ValueError(
                "--candidate-conflicts-only requires --candidate-label-catalog"
            )
        rows = [
            row for row in rows
            if len(candidate_labels_by_id.get(int(row["id"]), set())) >= 2
        ]
    geometry: dict | None = None
    if args.geometry_report:
        geometry = json.loads(args.geometry_report.read_text(encoding="utf-8"))
        geometry_by_id = {
            int(item.get("object_id", item.get("id"))): item
            for item in (geometry.get("objects") or [])
            if isinstance(item, dict) and item.get("object_id", item.get("id")) is not None
        }
        for row in rows:
            item = geometry_by_id.get(int(row["id"]))
            if item is not None and item.get("box_dimensions_m") is not None:
                row["metric_dimensions_m"] = item["box_dimensions_m"]
    world_up_contract = resolve_world_up_contract(args.world_up_vector, geometry)
    selected_ids = None
    if args.selection_report:
        selection = json.loads(args.selection_report.read_text(encoding="utf-8"))
        selected_ids = {
            int(row.get("object_id", row.get("id")))
            for row in (selection.get("objects") or [])
            if isinstance(row, dict) and str(row.get("status") or "").endswith("_pass")
        }
        rows = [row for row in rows if int(row["id"]) in selected_ids]
    if args.scene_state:
        wrapper = torch.load(
            args.scene_state.expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        state = (
            wrapper.get("state")
            if isinstance(wrapper, dict) and isinstance(wrapper.get("state"), dict)
            else wrapper
        )
        if not isinstance(state, dict):
            raise TypeError(f"Unsupported scene state: {args.scene_state}")
    else:
        if args.active_only:
            raise ValueError("--active-only requires --scene-state")
        state = {"object_id": [int(row["id"]) for row in rows]}
    metric_context_hydration = hydrate_rows_from_state(rows, state)
    if args.active_only:
        rows = filter_active_rows(rows, state)
    if args.max_objects > 0:
        rows = rows[: args.max_objects]
    mask_index = resolve_object_mask_observations(
        state,
        args.mask_dir.expanduser().resolve(),
        expand_image_matches=bool(args.expand_image_matches),
        include_object_ids=[int(row["id"]) for row in rows],
    )
    frame_pose_index = load_frame_pose_index(args.frames_json.expanduser().resolve())
    crop_map: dict[int, list[tuple[Path, bytes]]] = {}
    crop_partitions: dict[int, dict] = {}
    verification_crop_map: dict[int, list[tuple[Path, bytes]]] = {}
    verification_crop_partitions: dict[int, dict] = {}
    observation_selection: dict[int, dict] = {}
    missing = []
    for row in rows:
        object_id = int(row["id"])
        selected_paths, selection_audit = select_preferred_observation_paths(
            state,
            object_id,
            mask_index.for_object_id(object_id),
            preferred_source=args.preferred_observation_source,
            minimum_preferred=args.minimum_preferred_observations,
        )
        observation_selection[object_id] = selection_audit
        row["semantic_observation_selection"] = dict(selection_audit)
        candidates = crop_candidates(
            selected_paths,
            frames_json=args.frames_json,
            natural_context_panel=bool(args.natural_context_panel),
            frame_pose_index=frame_pose_index,
            world_up_vector=world_up_contract["vector"],
            world_up_source=world_up_contract["source"],
        )
        crops, partition = choose_evidence_crops(
            candidates,
            "initial_blind_review",
            max(1, args.crops_per_object),
            frame_pose_index=frame_pose_index,
            object_position_world_m=row.get("position_world_m"),
            pose_aware=True,
        )
        verification_crops, verification_partition = choose_evidence_crops(
            candidates,
            "independent_verification",
            max(1, args.crops_per_object),
            frame_pose_index=frame_pose_index,
            object_position_world_m=row.get("position_world_m"),
            pose_aware=True,
        )
        crops, verification_crops, cross_fold_selection = (
            optimize_cross_fold_pose_diversity(
                candidates,
                crops,
                verification_crops,
                max(1, args.crops_per_object),
                frame_pose_index=frame_pose_index,
                object_position_world_m=row.get("position_world_m"),
            )
        )
        verification_partition = dict(verification_partition)
        verification_partition["cross_fold_pose_selection"] = (
            cross_fold_selection
        )
        partition = dict(partition)
        partition["cross_fold_pose_selection"] = cross_fold_selection
        if cross_fold_selection.get("status") == "selected":
            fold_specs = (
                (
                    partition,
                    "initial_within_fold",
                    "selected_initial_image_ids",
                    0,
                    int(cross_fold_selection["initial_posed_unique_count"]),
                    cross_fold_selection["initial_fold_search"],
                ),
                (
                    verification_partition,
                    "verification_within_fold",
                    "selected_verification_image_ids",
                    1,
                    int(cross_fold_selection["verification_posed_unique_count"]),
                    cross_fold_selection["verification_fold_search"],
                ),
            )
            for (
                fold_partition,
                metric_key,
                image_key,
                fold_index,
                candidate_count,
                search,
            ) in fold_specs:
                original_partition = {
                    key: fold_partition.get(key)
                    for key in (
                        "partition_id",
                        "partition_index",
                        "partition_count",
                        "partition_image_ids",
                    )
                }
                metrics = cross_fold_selection[metric_key]
                diagnostics = dict(
                    fold_partition.get("selection_diagnostics") or {}
                )
                fallback_detail = int(
                    diagnostics.get("fallback_detail_bytes") or 0
                )
                selected_detail = int(metrics["selected_detail_bytes"])
                diagnostics.update({
                    "method": "joint_cross_fold_pose_selection",
                    "reason": "strict_within_and_cross_fold_pose_gates_met",
                    "partition_candidate_count": candidate_count,
                    "posed_unique_image_count": candidate_count,
                    "selected_detail_bytes": selected_detail,
                    "detail_ratio_to_temporal_fallback": (
                        selected_detail / fallback_detail
                        if fallback_detail > 0
                        else None
                    ),
                    "min_pairwise_viewpoint_angle_degrees": metrics[
                        "min_pairwise_viewpoint_angle_degrees"
                    ],
                    "median_pairwise_viewpoint_angle_degrees": metrics[
                        "median_pairwise_viewpoint_angle_degrees"
                    ],
                    "min_pairwise_normalized_camera_baseline": metrics[
                        "min_pairwise_normalized_camera_baseline"
                    ],
                    "diversity_gate_met": True,
                    "candidate_combination_count": search[
                        "candidate_combination_count"
                    ],
                    "evaluated_combination_count": search[
                        "evaluated_combination_count"
                    ],
                })
                fold_partition.update({
                    "partition_id": f"joint-pose-fold-{fold_index}-of-2",
                    "partition_index": fold_index,
                    "partition_count": 2,
                    "partition_image_ids": list(
                        cross_fold_selection[image_key]
                    ),
                    "partition_derivation": (
                        "joint_strict_pose_optimization_before_vlm_requests"
                    ),
                    "source_partition_before_joint_optimization": (
                        original_partition
                    ),
                    "selection_diagnostics": diagnostics,
                })
        verification_crops, verification_partition = (
            enforce_disjoint_verification_fold(
                crops,
                verification_crops,
                verification_partition,
            )
        )
        # Persist the exact post-optimization/post-disjointness request inputs.
        # ``partition_image_ids`` is the candidate pool in fallback runs and
        # must never be mistaken for the actual VLM request fold.
        partition["selected_image_ids"] = sorted(
            int(value)
            for value in (crop_image_id(candidate) for candidate in crops)
            if value is not None
        )
        partition["selected_crop_count"] = len(crops)
        partition["selected_image_ids_complete"] = bool(
            len(partition["selected_image_ids"]) == len(crops)
        )
        verification_partition["selected_image_ids"] = sorted(
            int(value)
            for value in (
                crop_image_id(candidate) for candidate in verification_crops
            )
            if value is not None
        )
        verification_partition["selected_crop_count"] = len(verification_crops)
        verification_partition["selected_image_ids_complete"] = bool(
            len(verification_partition["selected_image_ids"])
            == len(verification_crops)
        )
        crop_map[object_id] = crops
        crop_partitions[object_id] = partition
        row["semantic_initial_evidence_selection"] = dict(partition)
        verification_crop_map[object_id] = verification_crops
        verification_crop_partitions[object_id] = verification_partition
        row["semantic_verification_evidence_selection"] = dict(
            verification_partition
        )
        if not crops:
            missing.append(object_id)

    review_prompt = PROMPT
    review_source = "initial_blind_review"
    review_prompt_id = "initial_blind_review.v1"
    if args.natural_context_panel:
        review_prompt = (
            PROMPT
            + "\n\nEach supplied image has two panels of the same observation: "
              "MASK on the left preserves target-owned pixels and is primary; "
              "CONTEXT on the right shows the natural-colour surroundings with "
              "the exact target outlined in yellow. Use the context panel to "
              "resolve mounting, support, scale, placement, and real object role, "
              "but never transfer a neighbouring object's identity to the target."
        )
        review_source = "contextual_dual_panel_review"
        review_prompt_id = "contextual_dual_panel_review.v1"

    request_seconds: list[float] = []
    verification_seconds: list[float] = []
    initial_request_error_ids: list[int] = []
    verification_request_error_ids: list[int] = []
    verification_unavailable_ids: list[int] = []
    if args.vllm_url:
        def apply_gate(row: dict) -> None:
            if (
                row.get("review_category") == "unknown"
                or float(row.get("review_confidence", 0.0)) < float(args.min_confidence)
            ):
                row["review_decision"] = "unknown"
                row["review_gate_reason"] = "unknown_category_or_low_confidence"
            elif str(row.get("review_category") or "").strip().lower() in UNINFORMATIVE_YOLOE_LABELS:
                row["review_decision"] = "unknown"
                row["review_gate_reason"] = "farm_uninformative_category_policy"
            else:
                row["review_gate_reason"] = "multi_view_identity_supported"

        def review_one(index: int, row: dict) -> tuple[int, str, float]:
            crops = crop_map[int(row["id"])]
            if not crops:
                row.update(
                    {
                        "review_category": "unknown",
                        "review_description": "No saved crop observations.",
                        "review_confidence": 0.0,
                        "review_decision": "unknown",
                    }
                )
                return index, "missing", 0.0
            object_id = int(row["id"])
            labels = candidate_labels_by_id.get(object_id, set())
            local_prompt = review_prompt
            local_source = review_source
            local_prompt_id = review_prompt_id
            candidate_conditioned = bool(
                args.verification_mode == "candidate_conditioned"
                and args.candidate_label_catalog
                and labels
            )
            if candidate_conditioned:
                local_prompt = candidate_adjudication_prompt(
                    review_prompt, labels
                )
                local_source = "candidate_context_adjudication"
                local_prompt_id = "candidate_context_adjudication.v1"
            started = time.perf_counter()
            try:
                row.update(request_review(
                    args.vllm_url, args.model, row, crops,
                    prompt=local_prompt,
                    strict_open_vocabulary=True,
                ))
                apply_gate(row)
                event = semantic_evidence_event(
                    row,
                    local_source,
                    f"object:{object_id}:{local_source}",
                    crops=crops,
                    partition=crop_partitions[int(row["id"])],
                    model_id=args.model,
                    prompt_id=local_prompt_id,
                    prompt=local_prompt,
                    confirmation_eligible=not candidate_conditioned,
                    ineligibility_reason=(
                        "candidate_conditioned_context_adjudication"
                        if candidate_conditioned else ""
                    ),
                    frame_pose_index=frame_pose_index,
                    world_up_vector=world_up_contract["vector"],
                    world_up_source=world_up_contract["source"],
                )
                if not candidate_conditioned:
                    event["semantic_vote_independence"] = "independent"
                    event["request_conditioning"] = "none_blind"
                row["semantic_evidence"] = [event]
                status = row["review_decision"]
            except Exception as exc:
                row.update(
                    {
                        "review_category": "unknown",
                        "review_description": f"Review request failed: {exc}",
                        "review_confidence": 0.0,
                        "review_decision": "unknown",
                    }
                )
                status = "error"
            elapsed = time.perf_counter() - started
            row["review_seconds"] = elapsed
            return index, status, elapsed

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = [executor.submit(review_one, index, row) for index, row in enumerate(rows)]
            for future in concurrent.futures.as_completed(futures):
                index, status, elapsed = future.result()
                row = rows[index]
                if elapsed > 0.0:
                    request_seconds.append(elapsed)
                if status == "error":
                    initial_request_error_ids.append(int(row["id"]))
                print(
                    f"[crop-review] {index + 1}/{len(rows)} object={row['id']} "
                    f"raw={row['category']} result={row.get('review_category')} "
                    f"status={status} seconds={elapsed:.2f}",
                    flush=True,
                )

        if args.verify_kept:
            verify_indices = [
                index for index, row in enumerate(rows)
                if row.get("review_decision") == "keep"
            ]

            def verify_one(index: int) -> tuple[int, str, float]:
                row = rows[index]
                crops = verification_crop_map[int(row["id"])]
                initial = {
                    "category": str(row.get("review_category") or "unknown"),
                    "description": str(row.get("review_description") or ""),
                    "attributes": list(row.get("review_attributes") or []),
                    "label_contract": (
                        dict(row["review_label_contract"])
                        if isinstance(row.get("review_label_contract"), dict)
                        else None
                    ),
                    "confidence": float(row.get("review_confidence") or 0.0),
                    "decision": str(row.get("review_decision") or "unknown"),
                    "raw_response": str(row.get("review_raw_response") or ""),
                }
                row["review_initial"] = initial
                independent_mode = args.verification_mode == "independent_blind"
                if not crops:
                    separation = verification_crop_partitions[int(row["id"])].get(
                        "blind_verification_separation", {}
                    )
                    row["verification_status"] = "unavailable"
                    row["verification_gate_reason"] = str(
                        separation.get("reason")
                        or "no_disjoint_verification_evidence"
                    )
                    if independent_mode:
                        row["independent_blind_consensus"] = {
                            "schema": "farm.independent-blind-label-consensus.v1",
                            "status": "unresolved",
                            "accepted": False,
                            "initial_category": normalize_open_noun(
                                initial["category"]
                            ),
                            "verification_category": "",
                            "canonical_category": "",
                            "reason_codes": [
                                "verification_fold_not_provably_disjoint"
                            ],
                        }
                        quarantine_unresolved_semantics(
                            row,
                            gate_reason=(
                                "independent_blind_verification_unavailable"
                            ),
                            reason_codes=[
                                "verification_fold_not_provably_disjoint"
                            ],
                        )
                    # A text-only or reused-view request adds no independent
                    # visual information and is never sent.
                    return index, "unavailable", 0.0
                request_contract = verification_request_contract(
                    args.verification_mode,
                    blind_prompt=review_prompt,
                    initial=initial,
                )
                prompt = request_contract["prompt"]
                started = time.perf_counter()
                try:
                    verified = request_review(
                        args.vllm_url, args.model, row, crops, prompt=prompt,
                        strict_open_vocabulary=True,
                    )
                    row["verification_raw_response"] = verified.pop(
                        "review_raw_response", ""
                    )
                    verification_row = dict(row)
                    verification_row.update(verified)
                    apply_gate(verification_row)
                    event = semantic_evidence_event(
                        verification_row,
                        "independent_verification",
                        f"object:{int(row['id'])}:independent_verification",
                        crops=crops,
                        partition=verification_crop_partitions[int(row["id"])],
                        model_id=args.model,
                        prompt_id=request_contract["prompt_id"],
                        prompt=prompt,
                        confirmation_eligible=request_contract[
                            "confirmation_eligible"
                        ],
                        ineligibility_reason=request_contract[
                            "ineligibility_reason"
                        ],
                        frame_pose_index=frame_pose_index,
                        world_up_vector=world_up_contract["vector"],
                        world_up_source=world_up_contract["source"],
                    )
                    event["semantic_vote_independence"] = request_contract[
                        "semantic_vote_independence"
                    ]
                    event["request_conditioning"] = request_contract[
                        "request_conditioning"
                    ]
                    if independent_mode:
                        initial_event, verification_event = (
                            current_independent_blind_event_pair(row, event)
                        )
                        # Replace rather than append so inherited/stale events
                        # can never enter this run's two-vote consensus.
                        row["semantic_evidence"] = [
                            initial_event,
                            verification_event,
                        ]
                        apply_independent_blind_result(
                            row,
                            initial_event,
                            verification_event,
                            minimum_confidence=float(args.min_confidence),
                        )
                    else:
                        row.setdefault("semantic_evidence", []).append(event)
                        row.update(verified)
                        apply_gate(row)
                        if row["review_decision"] == "keep":
                            row["review_gate_reason"] = verification_gate_reason(
                                initial, row
                            )
                    status = row["review_decision"]
                except Exception as exc:
                    if independent_mode:
                        row["verification_error"] = str(exc)
                        quarantine_unresolved_semantics(
                            row,
                            gate_reason=(
                                "independent_blind_verification_request_failed"
                            ),
                            reason_codes=[
                                "verification_transport_or_contract_failure"
                            ],
                        )
                    else:
                        row.update(
                            {
                                "review_category": "unknown",
                                "review_description": (
                                    f"Verification request failed: {exc}"
                                ),
                                "review_confidence": 0.0,
                                "review_decision": "unknown",
                                "review_gate_reason": (
                                    "verification_request_failed"
                                ),
                            }
                        )
                    status = "error"
                elapsed = time.perf_counter() - started
                row["verification_seconds"] = elapsed
                row["review_seconds"] = float(row.get("review_seconds") or 0.0) + elapsed
                return index, status, elapsed

            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
                futures = [executor.submit(verify_one, index) for index in verify_indices]
                for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                    index, status, elapsed = future.result()
                    row = rows[index]
                    if elapsed > 0.0:
                        verification_seconds.append(elapsed)
                    if status == "error":
                        verification_request_error_ids.append(int(row["id"]))
                    elif status == "unavailable":
                        verification_unavailable_ids.append(int(row["id"]))
                    print(
                        f"[crop-verify] {completed}/{len(verify_indices)} object={row['id']} "
                        f"initial={row['review_initial']['category']} final={row.get('review_category')} "
                        f"status={status} seconds={elapsed:.2f}",
                        flush=True,
                    )

        if args.verification_mode == "independent_blind":
            # Initial unknown/low-confidence decisions are not sent to the
            # second pass, but they still need the same explicit fail-closed
            # publication state as a two-pass disagreement.
            for row in rows:
                consensus = row.get("independent_blind_consensus")
                if isinstance(consensus, dict) and consensus.get("accepted") is True:
                    continue
                if row.get("semantic_quarantined") is True:
                    continue
                quarantine_unresolved_semantics(
                    row,
                    gate_reason=str(
                        row.get("review_gate_reason")
                        or "independent_blind_initial_vote_unresolved"
                    ),
                    reason_codes=list(
                        consensus.get("reason_codes") or ()
                        if isinstance(consensus, dict)
                        else ("initial_vote_not_publication_eligible",)
                    ),
                )

    pages = make_pages(rows, crop_map, output_dir)
    reviewed = sum("review_decision" in row for row in rows)
    decisions = {}
    if reviewed:
        decisions = dict(
            sorted(
                __import__("collections").Counter(
                    str(row.get("review_decision", "unknown")) for row in rows
                ).items()
            )
        )
    duration_seconds = time.perf_counter() - overall_started
    initial_request_array = np.asarray(request_seconds, dtype=np.float64)
    verification_array = np.asarray(verification_seconds, dtype=np.float64)
    request_array = np.concatenate([initial_request_array, verification_array])
    consensus_counts = dict(
        sorted(
            __import__("collections").Counter(
                str((row.get("independent_blind_consensus") or {}).get("status"))
                for row in rows
                if isinstance(row.get("independent_blind_consensus"), dict)
            ).items()
        )
    )
    if args.verification_mode == "independent_blind":
        verification_policy = (
            "second request repeats the unconditioned first-pass prompt on a "
            "provably disjoint fold from the same candidate universe; publication "
            "requires exact safe canonical noun agreement; disagreement, unknown "
            "or unavailable verification explicitly quarantines the label"
        )
    else:
        verification_policy = (
            "compatibility verification is candidate-conditioned and explicitly "
            "confirmation-ineligible"
        )
    report = {
        "schema": "farm.open-vocabulary-crop-review.v2",
        "object_count": len(rows),
        "reviewed_count": reviewed,
        "missing_crop_object_ids": missing,
        "decisions": decisions,
        "selection_report": str(args.selection_report.resolve()) if args.selection_report else None,
        "geometry_report": str(args.geometry_report.resolve()) if args.geometry_report else None,
        "gravity_upright_normalization": world_up_contract,
        "scene_state": (
            str(args.scene_state.expanduser().resolve()) if args.scene_state else None
        ),
        "mask_observation_contract": mask_index.diagnostics,
        "metric_context_hydration": metric_context_hydration,
        "observation_selection": {str(key): value for key, value in observation_selection.items()},
        "min_confidence": float(args.min_confidence),
        "workers": max(1, int(args.workers)),
        "crops_per_object": max(1, int(args.crops_per_object)),
        "verification_enabled": bool(args.verify_kept),
        "verification_mode": str(args.verification_mode),
        "verification_reviewed_count": len(verification_seconds),
        "independent_blind_consensus_counts": consensus_counts,
        "category_consensus_required": True,
        "category_consensus_enforcement_stage": "apply_farm_full_colmap_rescue",
        "deprecated_category_consensus_flag_supplied": bool(args.require_category_consensus),
        "initial_request_error_ids": sorted(initial_request_error_ids),
        "verification_request_error_ids": sorted(verification_request_error_ids),
        "verification_unavailable_ids": sorted(verification_unavailable_ids),
        "request_error_count": len(initial_request_error_ids) + len(verification_request_error_ids),
        "policy": (
            "strict mask-grounded open-vocabulary composition and head-noun review; "
            "no existing FARM class, caption, scene type, or object-specific hint "
            "is shown to any blind pass; " + verification_policy
        ),
        "timing": {
            "duration_seconds": duration_seconds,
            "request_total_seconds": float(request_array.sum()) if request_array.size else 0.0,
            "request_mean_seconds": float(request_array.mean()) if request_array.size else None,
            "request_p50_seconds": float(np.percentile(request_array, 50.0)) if request_array.size else None,
            "request_p95_seconds": float(np.percentile(request_array, 95.0)) if request_array.size else None,
            "initial_request_total_seconds": float(initial_request_array.sum()) if initial_request_array.size else 0.0,
            "verification_request_total_seconds": float(verification_array.sum()) if verification_array.size else 0.0,
            "stage": "strict_open_vocabulary_semantic_review",
        },
        "pages": pages,
        "objects": rows,
    }
    (output_dir / "reviewed_robust_objects.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "objects"}, indent=2))
    require_successful_transport(report)


if __name__ == "__main__":
    main()
