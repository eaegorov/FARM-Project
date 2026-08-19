#!/usr/bin/env python3
"""Visually and optionally VLM-review robust FARM objects from saved crop sidecars."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import http.client
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

from scene_graph.captioning.evidence import (
    crop_evidence_manifest,
    load_frame_pose_index,
    partition_crop_candidates,
    select_pose_diverse_crop_candidates,
)
from scene_graph.captioning.label_contract import (
    OPEN_VOCABULARY_JSON_INSTRUCTIONS,
    assess_open_vocabulary_label,
    parse_open_vocabulary_label,
)
from scene_graph.map_update.mask_observations import resolve_object_mask_observations

try:
    from scene_graph.map_update.filtering import UNINFORMATIVE_YOLOE_LABELS
except ImportError:
    UNINFORMATIVE_YOLOE_LABELS = set()


PROMPT = """The images are automatically selected observations associated with one
physical 3D object. The target keeps its natural colours, surrounding context is
dimmed but remains readable for scale and placement, and a thin light outline
marks the saved segmentation mask.
Identify only what is directly supported by target pixels that remain consistent
across the views; use dim context only to understand scale and placement, and do not
assume a scene type. First decide whether the mask is one whole, a carrier with
payload, a standalone component, an attached component, or an incomplete/mixed
target. Then name only the physical head noun supported by target-owned evidence
across views. Broad structural surfaces and background are not object instances.
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

_FRAME_INDEX_CACHE: dict[Path, list[Path | None]] = {}


def _source_frames(mask_dir: Path) -> list[Path | None]:
    """Resolve mapping image indices to RGB-D source frames, when available."""

    root = mask_dir.parent.parent
    frames_json = root / "rgbd" / "frames.json"
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


def _source_frame_for_crop(mask_dir: Path, crop_path: Path) -> np.ndarray | None:
    match = re.search(r"img_(\d+)", crop_path.name)
    if not match:
        return None
    frames = _source_frames(mask_dir)
    index = int(match.group(1))
    if not 0 <= index < len(frames) or frames[index] is None:
        return None
    return cv2.imread(str(frames[index]), cv2.IMREAD_COLOR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument("--scene-state", type=Path)
    parser.add_argument(
        "--expand-image-matches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resolve derivative crop-bank files for canonical observation image IDs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vllm-url", default=None)
    parser.add_argument("--model", default="qwen3-vl-8b")
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--crops-per-object", type=int, default=3)
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
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--verify-kept",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a second generic evidence audit over first-pass kept labels.",
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


def _mask_grounded_crop(
    payload: np.lib.npyio.NpzFile,
    encoded: bytes,
    source_image: np.ndarray | None = None,
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

    required = {"raw_bits", "raw_shape", "crop_bbox_xyxy"}
    if not required.issubset(payload.files):
        return encoded
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return encoded
    try:
        raw_height, raw_width = (
            np.asarray(payload["raw_shape"], dtype=np.int32).reshape(2).tolist()
        )
        x0, y0, x1, y1 = (
            np.asarray(payload["crop_bbox_xyxy"], dtype=np.int32).reshape(4).tolist()
        )
    except (TypeError, ValueError):
        return encoded
    if raw_height <= 0 or raw_width <= 0:
        return encoded
    flat = np.unpackbits(
        np.asarray(payload["raw_bits"], dtype=np.uint8), bitorder="little"
    )
    if flat.size < raw_height * raw_width:
        return encoded
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
        return encoded
    local = cv2.resize(raw, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = local.astype(bool)
    coverage = float(mask.mean())
    if not np.isfinite(coverage) or coverage <= 0.001:
        return encoded

    context_gain, context_offset = (
        (0.42, 8.0) if used_source_context else (0.18, 12.0)
    )
    grounded = np.clip(
        image.astype(np.float32) * context_gain + context_offset, 0, 255
    ).astype(np.uint8)
    grounded[mask] = image[mask]
    outer = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1)
    grounded[outer.astype(bool) & ~mask] = np.asarray([235, 235, 235], dtype=np.uint8)
    ok, buffer = cv2.imencode(
        ".jpg", grounded, [int(cv2.IMWRITE_JPEG_QUALITY), 94]
    )
    return bytes(buffer) if ok else encoded


def crop_candidates(paths: list[Path]) -> list[tuple[Path, bytes]]:
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
                encoded = _mask_grounded_crop(payload, encoded)
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
        "max_tokens": 360,
        "response_format": {"type": "json_object"},
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
            raw = f"#{row['id']:03d} obs={row['observations']} RAW: {str(row['category']).upper()}"
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
    selected_ids = None
    if args.selection_report:
        selection = json.loads(args.selection_report.read_text(encoding="utf-8"))
        selected_ids = {
            int(row.get("object_id", row.get("id")))
            for row in (selection.get("objects") or [])
            if isinstance(row, dict) and str(row.get("status") or "").endswith("_pass")
        }
        rows = [row for row in rows if int(row["id"]) in selected_ids]
    if args.max_objects > 0:
        rows = rows[: args.max_objects]
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
        state = {"object_id": [int(row["id"]) for row in rows]}
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
    missing = []
    for row in rows:
        object_id = int(row["id"])
        candidates = crop_candidates(mask_index.for_object_id(object_id))
        crops, partition = choose_evidence_crops(
            candidates,
            "initial_blind_review",
            max(1, args.crops_per_object),
            frame_pose_index=frame_pose_index,
            object_position_world_m=row.get("position_world_m"),
            pose_aware=True,
        )
        crop_map[object_id] = crops
        crop_partitions[object_id] = partition
        verification_crops, verification_partition = choose_evidence_crops(
            candidates,
            "independent_verification",
            max(1, args.crops_per_object),
        )
        verification_crop_map[object_id] = verification_crops
        verification_crop_partitions[object_id] = verification_partition
        if not crops:
            missing.append(object_id)

    request_seconds: list[float] = []
    verification_seconds: list[float] = []
    initial_request_error_ids: list[int] = []
    verification_request_error_ids: list[int] = []
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
            started = time.perf_counter()
            try:
                row.update(request_review(
                    args.vllm_url, args.model, row, crops,
                    strict_open_vocabulary=True,
                ))
                apply_gate(row)
                row["semantic_evidence"] = [
                    semantic_evidence_event(
                        row,
                        "initial_blind_review",
                        f"object:{int(row['id'])}:initial_blind_review",
                        crops=crops,
                        partition=crop_partitions[int(row["id"])],
                        model_id=args.model,
                        prompt_id="initial_blind_review.v1",
                        prompt=PROMPT,
                        frame_pose_index=frame_pose_index,
                    )
                ]
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
                    "confidence": float(row.get("review_confidence") or 0.0),
                    "decision": str(row.get("review_decision") or "unknown"),
                    "raw_response": str(row.get("review_raw_response") or ""),
                }
                row["review_initial"] = initial
                prompt = VERIFICATION_PROMPT.format(
                    category=initial["category"],
                    description=initial["description"],
                )
                started = time.perf_counter()
                try:
                    verified = request_review(
                        args.vllm_url, args.model, row, crops, prompt=prompt,
                        strict_open_vocabulary=True,
                    )
                    row["verification_raw_response"] = verified.pop("review_raw_response", "")
                    row.update(verified)
                    apply_gate(row)
                    row.setdefault("semantic_evidence", []).append(
                        semantic_evidence_event(
                            row,
                            "independent_verification",
                            f"object:{int(row['id'])}:independent_verification",
                            crops=crops,
                            partition=verification_crop_partitions[int(row["id"])],
                            model_id=args.model,
                            prompt_id="candidate_conditioned_verification.v1",
                            prompt=prompt,
                            confirmation_eligible=False,
                            ineligibility_reason="candidate_conditioned_on_initial_review",
                            frame_pose_index=frame_pose_index,
                        )
                    )
                    if row["review_decision"] == "keep":
                        row["review_gate_reason"] = verification_gate_reason(initial, row)
                    status = row["review_decision"]
                except Exception as exc:
                    row.update(
                        {
                            "review_category": "unknown",
                            "review_description": f"Verification request failed: {exc}",
                            "review_confidence": 0.0,
                            "review_decision": "unknown",
                            "review_gate_reason": "verification_request_failed",
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
                    verification_seconds.append(elapsed)
                    if status == "error":
                        verification_request_error_ids.append(int(row["id"]))
                    print(
                        f"[crop-verify] {completed}/{len(verify_indices)} object={row['id']} "
                        f"initial={row['review_initial']['category']} final={row.get('review_category')} "
                        f"status={status} seconds={elapsed:.2f}",
                        flush=True,
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
    report = {
        "schema": "farm.open-vocabulary-crop-review.v2",
        "object_count": len(rows),
        "reviewed_count": reviewed,
        "missing_crop_object_ids": missing,
        "decisions": decisions,
        "selection_report": str(args.selection_report.resolve()) if args.selection_report else None,
        "geometry_report": str(args.geometry_report.resolve()) if args.geometry_report else None,
        "scene_state": (
            str(args.scene_state.expanduser().resolve()) if args.scene_state else None
        ),
        "mask_observation_contract": mask_index.diagnostics,
        "min_confidence": float(args.min_confidence),
        "workers": max(1, int(args.workers)),
        "crops_per_object": max(1, int(args.crops_per_object)),
        "verification_enabled": bool(args.verify_kept),
        "verification_reviewed_count": len(verification_seconds),
        "category_consensus_required": False,
        "deprecated_category_consensus_flag_supplied": bool(args.require_category_consensus),
        "initial_request_error_ids": sorted(initial_request_error_ids),
        "verification_request_error_ids": sorted(verification_request_error_ids),
        "request_error_count": len(initial_request_error_ids) + len(verification_request_error_ids),
        "policy": (
            "strict mask-grounded open-vocabulary composition and head-noun review; "
            "no existing FARM class, caption, scene type, or object-specific hint "
            "is shown to the first pass; optional compatibility verification is "
            "candidate-conditioned, confirmation-ineligible, and disabled by the "
            "standard adaptive planner"
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
