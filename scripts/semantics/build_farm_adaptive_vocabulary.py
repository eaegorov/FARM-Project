#!/usr/bin/env python3
"""Build a bounded scene-adaptive open vocabulary from real COLMAP frames.

The two-fold inventory follows the useful REST3D pattern without importing its
scene-specific assumptions: one pose-diverse fold proposes complete physical
objects and a disjoint fold verifies visible diagnostic evidence.  The result
only augments the immutable base vocabulary; it never removes base terms.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import base64
import hashlib
import http.client
import json
import math
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


DISCOVERY_PROMPT = """Inspect these real, undistorted COLMAP scene images as a
single multi-view set. Build an open-vocabulary inventory of complete bounded
physical objects that an instance segmenter should detect. Include large
multi-part equipment and functional assemblies when the complete assembly is
visible. Do not list ceilings, floors, walls, beams that are building structure,
generic surfaces, shadows, reflections, object parts, handles, panels, wheels,
or payload as separate objects. Do not assume a scene type.

Return JSON only with key `objects`, a list of at most 12 highest-confidence
complete objects visible in this image batch.
Each row must contain:
`canonical_name` (short singular lower-case noun phrase), `aliases` (at most 2
short visual synonyms), `topology` (one of standalone_whole, carrier_payload,
multi_part_assembly, attached_component, incomplete_or_mixed),
`diagnostic_evidence` (at most 16 words naming specific visible parts or text), and
`confidence` in [0,1]. Prefer recall, but say incomplete_or_mixed for mere parts.
"""

VERIFICATION_PROMPT = """Independently verify the candidate object categories
against this disjoint set of real COLMAP images. A candidate passes only when a
complete bounded object or coherent equipment assembly is visible in at least
two supplied images and diagnostic pixels support the category. Reject building
structure, loose parts, panels, handles, wheels, shadows, reflections, and
shape-only guesses. Do not preserve a candidate merely because it was proposed.

Candidates:
{candidates}

Return JSON only with key `objects`. Return exactly one row per candidate with:
`canonical_name`, `decision` (keep or reject), `visible_views` (integer count),
`topology` (standalone_whole, carrier_payload, multi_part_assembly,
attached_component, or incomplete_or_mixed), `diagnostic_evidence`, and
`confidence` in [0,1].
"""

MAX_IMAGE_PIXELS = 512 * 512
ALLOWED_TOPOLOGIES = {
    "standalone_whole",
    "carrier_payload",
    "multi_part_assembly",
}

GENERIC_DYNAMIC_TERMS = {
    "equipment",
    "industrial equipment",
    "industrial machine",
    "installation",
    "item",
    "machine",
    "machinery",
    "object",
    "production unit",
    "structure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument("--base-vocabulary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--discovery-views", type=int, default=12)
    parser.add_argument("--verification-views", type=int, default=12)
    parser.add_argument("--batch-views", type=int, default=6)
    parser.add_argument("--min-visible-views", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.72)
    parser.add_argument("--max-additions", type=int, default=48)
    return parser.parse_args()


def canonical_term(value: object) -> str:
    text = re.sub(r"[^a-z0-9 -]+", " ", str(value or "").lower())
    text = re.sub(r"\s+", " ", text).strip(" -")
    if not text or len(text) > 64 or len(text.split()) > 5:
        return ""
    return text


def dynamic_term(value: object) -> str:
    term = canonical_term(value)
    return "" if term in GENERIC_DYNAMIC_TERMS else term


def _camera_position(frame: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(frame.get("T_world_cam"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("frame has invalid T_world_cam")
    return matrix[:3, 3]


def physical_representatives(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose one deterministic camera view per physical timestamp."""

    groups: dict[int, list[dict[str, Any]]] = {}
    for row in frames:
        timestamp = int(row["timestamp_ns"])
        groups.setdefault(timestamp, []).append(row)
    result = []
    for timestamp, rows in sorted(groups.items()):
        rows = sorted(
            rows,
            key=lambda row: (
                0 if str(row.get("camera", "")).endswith("_center") else 1,
                str(row.get("camera", "")),
                str(row.get("rgb_path", "")),
            ),
        )
        chosen = dict(rows[0])
        chosen["timestamp_ns"] = timestamp
        result.append(chosen)
    return result


def farthest_pose_order(frames: list[dict[str, Any]]) -> list[int]:
    """Deterministic farthest-point order over camera translation and time."""

    if not frames:
        return []
    positions = np.stack([_camera_position(row) for row in frames])
    scale = np.ptp(positions, axis=0)
    scale[scale < 1e-9] = 1.0
    normalized = (positions - positions.min(axis=0)) / scale
    times = np.linspace(0.0, 1.0, len(frames), dtype=np.float64)[:, None]
    features = np.concatenate([normalized, 0.35 * times], axis=1)
    selected = [0]
    distances = np.linalg.norm(features - features[0], axis=1)
    while len(selected) < len(frames):
        index = int(np.argmax(distances))
        selected.append(index)
        distances = np.minimum(
            distances, np.linalg.norm(features - features[index], axis=1)
        )
        distances[selected] = -1.0
    return selected


def select_disjoint_folds(
    frames: list[dict[str, Any]], discovery_count: int, verification_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    representatives = physical_representatives(frames)
    required = discovery_count + verification_count
    if len(representatives) < required:
        raise ValueError(
            f"need {required} physical timestamps, found {len(representatives)}"
        )
    ordered = [representatives[index] for index in farthest_pose_order(representatives)]
    discovery: list[dict[str, Any]] = []
    verification: list[dict[str, Any]] = []
    for index, row in enumerate(ordered[:required]):
        target = discovery if index % 2 == 0 else verification
        limit = discovery_count if target is discovery else verification_count
        if len(target) < limit:
            target.append(row)
    for row in ordered[required:]:
        if len(discovery) < discovery_count:
            discovery.append(row)
        elif len(verification) < verification_count:
            verification.append(row)
        else:
            break
    if len(discovery) != discovery_count or len(verification) != verification_count:
        raise AssertionError("could not construct requested disjoint folds")
    if {int(r["timestamp_ns"]) for r in discovery} & {
        int(r["timestamp_ns"]) for r in verification
    }:
        raise AssertionError("inventory folds are not physically disjoint")
    return discovery, verification


def resolve_images(frames_json: Path, rows: Iterable[dict[str, Any]]) -> list[tuple[dict[str, Any], bytes]]:
    result = []
    for row in rows:
        path = (frames_json.parent / str(row["rgb_path"])).resolve(strict=True)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode inventory frame: {path}")
        height, width = image.shape[:2]
        ratio = min(1.0, math.sqrt(MAX_IMAGE_PIXELS / float(height * width)))
        if ratio < 1.0:
            image = cv2.resize(
                image,
                (max(1, int(round(width * ratio))), max(1, int(round(height * ratio)))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 94]
        )
        if not ok:
            raise ValueError(f"cannot encode inventory frame: {path}")
        item = dict(row)
        item["resolved_rgb_path"] = str(path)
        item["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result.append((item, bytes(encoded)))
    return result


def parse_json_object(text: str) -> dict[str, Any]:
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, flags=re.S)
        if not match:
            raise ValueError("VLM response is not JSON")
        value = json.loads(match.group(0))
    if not isinstance(value, dict) or not isinstance(value.get("objects"), list):
        raise ValueError("VLM response lacks objects array")
    return value


def request_vlm(
    url: str, model: str, prompt: str, images: list[tuple[dict[str, Any], bytes]]
) -> tuple[dict[str, Any], str, float]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for _, encoded in images:
        data = base64.b64encode(encoded).decode("ascii")
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}
        )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": 800,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {"max_pixels": MAX_IMAGE_PIXELS},
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == 2:
                raise
            time.sleep(0.5 * (2**attempt))
        except (
            urllib.error.URLError,
            http.client.RemoteDisconnected,
            TimeoutError,
            ConnectionError,
            OSError,
        ):
            if attempt == 2:
                raise
            time.sleep(0.5 * (2**attempt))
    raw = str(result["choices"][0]["message"]["content"])
    return parse_json_object(raw), raw, time.monotonic() - started


def discovery_candidates(responses: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for response_index, response in enumerate(responses):
        for raw in response.get("objects", []):
            if not isinstance(raw, dict):
                continue
            name = dynamic_term(raw.get("canonical_name"))
            topology = str(raw.get("topology") or "").strip().lower()
            try:
                confidence = float(raw.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if not name or topology not in ALLOWED_TOPOLOGIES:
                continue
            row = candidates.setdefault(
                name,
                {
                    "canonical_name": name,
                    "aliases": [],
                    "topology": topology,
                    "diagnostic_evidence": [],
                    "discovery_batches": [],
                    "discovery_confidence": 0.0,
                },
            )
            row["discovery_batches"].append(response_index)
            row["discovery_confidence"] = max(
                float(row["discovery_confidence"]), float(np.clip(confidence, 0.0, 1.0))
            )
            evidence = str(raw.get("diagnostic_evidence") or "").strip()
            if evidence and evidence not in row["diagnostic_evidence"]:
                row["diagnostic_evidence"].append(evidence[:300])
            for alias in raw.get("aliases") or []:
                value = dynamic_term(alias)
                if value and value != name and value not in row["aliases"]:
                    row["aliases"].append(value)
    return candidates


def verify_candidates(
    candidates: dict[str, dict[str, Any]],
    responses: Iterable[dict[str, Any]],
    *,
    min_visible_views: int,
    min_confidence: float,
) -> list[dict[str, Any]]:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for response in responses:
        for row in response.get("objects", []):
            if not isinstance(row, dict):
                continue
            name = canonical_term(row.get("canonical_name"))
            if name:
                by_name.setdefault(name, []).append(row)
    audited = []
    for name, candidate in sorted(candidates.items()):
        rows = by_name.get(name) or []
        confidence = 0.0
        visible = 0
        topologies: list[str] = []
        decisions: list[str] = []
        evidence: list[str] = []
        for raw in rows:
            try:
                confidence = max(confidence, float(raw.get("confidence", 0.0)))
                visible += max(0, int(raw.get("visible_views", 0)))
            except (TypeError, ValueError):
                pass
            topologies.append(str(raw.get("topology") or "").strip().lower())
            decisions.append(str(raw.get("decision") or "reject").strip().lower())
            value = str(raw.get("diagnostic_evidence") or "").strip()
            if value and value not in evidence:
                evidence.append(value[:300])
        topology = next(
            (value for value in topologies if value in ALLOWED_TOPOLOGIES), ""
        )
        decision = "keep" if "keep" in decisions else "reject"
        reasons = []
        if decision != "keep":
            reasons.append("verification_rejected")
        if topology not in ALLOWED_TOPOLOGIES:
            reasons.append("not_complete_bounded_object")
        if visible < min_visible_views:
            reasons.append("insufficient_independent_views")
        if not math.isfinite(confidence) or confidence < min_confidence:
            reasons.append("low_verification_confidence")
        audited.append(
            {
                **candidate,
                "verification_decision": decision,
                "verification_visible_views": visible,
                "verification_topology": topology,
                "verification_confidence": float(np.clip(confidence, 0.0, 1.0)),
                "verification_diagnostic_evidence": evidence,
                "verification_batch_count": len(rows),
                "accepted": not reasons,
                "rejection_reasons": reasons,
            }
        )
    return audited

def merged_vocabulary(
    base_terms: Iterable[str], audited: Iterable[dict[str, Any]], max_additions: int
) -> tuple[list[str], list[str]]:
    output = []
    seen = set()
    for raw in base_terms:
        term = canonical_term(raw)
        if term and term not in seen:
            output.append(term)
            seen.add(term)
    additions = []
    ordered = sorted(
        (row for row in audited if row.get("accepted")),
        key=lambda row: (
            -float(row["verification_confidence"]),
            -int(row["verification_visible_views"]),
            str(row["canonical_name"]),
        ),
    )
    for row in ordered:
        for raw in [row["canonical_name"], *(row.get("aliases") or [])]:
            term = dynamic_term(raw)
            if term and term not in seen:
                output.append(term)
                additions.append(term)
                seen.add(term)
                if len(additions) >= max_additions:
                    return output, additions
    return output, additions


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def main() -> None:
    args = parse_args()
    if args.discovery_views < 2 or args.verification_views < 2:
        raise ValueError("both folds require at least two views")
    if args.batch_views < 1 or args.max_additions < 1:
        raise ValueError("batch views and max additions must be positive")
    frames_payload = json.loads(args.frames_json.read_text(encoding="utf-8"))
    frames = frames_payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("frames JSON lacks frames array")
    discovery_rows, verification_rows = select_disjoint_folds(
        frames, args.discovery_views, args.verification_views
    )
    discovery_images = resolve_images(args.frames_json, discovery_rows)
    verification_images = resolve_images(args.frames_json, verification_rows)

    discovery_outputs = []
    raw_requests = []
    for batch_index, images in enumerate(_chunks(discovery_images, args.batch_views)):
        parsed, raw, seconds = request_vlm(
            args.vllm_url, args.model, DISCOVERY_PROMPT, images
        )
        discovery_outputs.append(parsed)
        raw_requests.append(
            {
                "phase": "discovery",
                "batch": batch_index,
                "frame_timestamps_ns": [int(row["timestamp_ns"]) for row, _ in images],
                "duration_seconds": seconds,
                "raw_response": raw,
            }
        )
    candidates = discovery_candidates(discovery_outputs)
    candidate_payload = [
        {
            "canonical_name": name,
            "aliases": row["aliases"],
            "topology": row["topology"],
            "diagnostic_evidence": row["diagnostic_evidence"],
        }
        for name, row in sorted(
            candidates.items(),
            key=lambda item: (-float(item[1]["discovery_confidence"]), item[0]),
        )[: args.max_additions]
    ]
    candidates = {
        str(row["canonical_name"]): candidates[str(row["canonical_name"])]
        for row in candidate_payload
    }
    verified_responses: list[dict[str, Any]] = []
    verification_prompts: list[str] = []
    request_index = 0
    for candidate_batch in _chunks(candidate_payload, 8):
        verification_prompt = VERIFICATION_PROMPT.format(
            candidates=json.dumps(candidate_batch, ensure_ascii=False)
        )
        verification_prompts.append(verification_prompt)
        for images in _chunks(verification_images, args.batch_views):
            parsed, raw, seconds = request_vlm(
                args.vllm_url, args.model, verification_prompt, images
            )
            verified_responses.append(parsed)
            raw_requests.append(
                {
                    "phase": "verification",
                    "batch": request_index,
                    "candidate_names": [
                        row["canonical_name"] for row in candidate_batch
                    ],
                    "frame_timestamps_ns": [
                        int(row["timestamp_ns"]) for row, _ in images
                    ],
                    "duration_seconds": seconds,
                    "raw_response": raw,
                }
            )
            request_index += 1
    audited = verify_candidates(
        candidates,
        verified_responses,
        min_visible_views=args.min_visible_views,
        min_confidence=args.min_confidence,
    )
    base_terms = args.base_vocabulary.read_text(encoding="utf-8").splitlines()
    vocabulary, additions = merged_vocabulary(
        base_terms, audited, args.max_additions
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "yoloe_vocabulary.txt").write_text(
        "\n".join(vocabulary) + "\n", encoding="utf-8"
    )
    selected = {
        "discovery": [row for row, _ in discovery_images],
        "verification": [row for row, _ in verification_images],
    }
    report = {
        "schema": "farm.adaptive-vocabulary.v1",
        "status": "PASS",
        "model": args.model,
        "frames_json": str(args.frames_json.resolve()),
        "base_vocabulary": str(args.base_vocabulary.resolve()),
        "base_vocabulary_sha256": hashlib.sha256(
            args.base_vocabulary.read_bytes()
        ).hexdigest(),
        "discovery_prompt_sha256": hashlib.sha256(
            DISCOVERY_PROMPT.encode("utf-8")
        ).hexdigest(),
        "verification_prompt_sha256": [
            hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for prompt in verification_prompts
        ],
        "folds": selected,
        "requests": raw_requests,
        "candidates": audited,
        "counts": {
            "base_terms": len(
                {canonical_term(value) for value in base_terms if canonical_term(value)}
            ),
            "discovered_candidates": len(candidates),
            "accepted_candidates": sum(bool(row["accepted"]) for row in audited),
            "added_terms": len(additions),
            "final_terms": len(vocabulary),
        },
        "additions": additions,
        "output_vocabulary_sha256": hashlib.sha256(
            ("\n".join(vocabulary) + "\n").encode("utf-8")
        ).hexdigest(),
        "limits": {
            "discovery_views": args.discovery_views,
            "verification_views": args.verification_views,
            "batch_views": args.batch_views,
            "min_visible_views": args.min_visible_views,
            "min_confidence": args.min_confidence,
            "max_additions": args.max_additions,
        },
    }
    (args.output_dir / "inventory.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
