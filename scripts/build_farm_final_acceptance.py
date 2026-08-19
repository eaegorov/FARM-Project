#!/usr/bin/env python3
"""Apply or verify the model-free FARM final presentation acceptance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from farm_pipeline.final_acceptance import (  # noqa: E402
    DEFAULT_POLICY,
    presentation_ids,
    refresh_runtime_status,
    run_acceptance_policy,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_state(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state", payload) if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or not isinstance(state, dict):
        raise TypeError(f"Unsupported FARM scene state: {path}")
    return payload, state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _candidate_paths(
    object_id: int, observation: Mapping[str, Any], roots: list[Path]
) -> list[Path]:
    raw = str(observation.get("path") or observation.get("mask_path") or "")
    name = Path(raw).name
    if not name:
        return []
    paths = [root / f"object_{object_id:06d}" / name for root in roots]
    recorded = Path(raw).expanduser()
    if recorded.is_absolute():
        paths.append(recorded)
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def _decodable_crop_counts(
    state: Mapping[str, Any], visible: set[int], roots: list[Path]
) -> dict[int, int]:
    object_ids = state.get("object_id")
    ids = (
        [int(value) for value in object_ids.detach().cpu().tolist()]
        if isinstance(object_ids, torch.Tensor)
        else [int(value) for value in object_ids or []]
    )
    rows = state.get("object_mask_observations") or []
    result: dict[int, int] = {}
    for index, object_id in enumerate(ids):
        if object_id not in visible:
            continue
        observations = rows[index] if index < len(rows) and isinstance(rows[index], list) else []
        count = 0
        attempts = 0
        seen_names: set[str] = set()
        for observation in observations:
            if not isinstance(observation, Mapping):
                continue
            name = Path(str(observation.get("path") or observation.get("mask_path") or "")).name
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            if attempts >= DEFAULT_POLICY.max_crop_decode_attempts_per_object:
                break
            attempts += 1
            decoded = False
            for path in _candidate_paths(object_id, observation, roots):
                if not path.is_file():
                    continue
                try:
                    with np.load(path, allow_pickle=False) as archive:
                        values = archive["crop_jpeg_bytes"]
                    image = cv2.imdecode(np.asarray(values, dtype=np.uint8), cv2.IMREAD_COLOR)
                    decoded = image is not None and image.size > 0
                except (KeyError, OSError, ValueError):
                    decoded = False
                if decoded:
                    break
            if decoded:
                count += 1
        result[object_id] = count
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--dedup-audit", type=Path, required=True)
    parser.add_argument("--semantic-catalog", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, action="append", required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-clusters", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--pre-qa-wall-seconds", type=float, default=0.0)
    parser.add_argument("--mode", choices=("apply", "verify"), default="apply")
    args = parser.parse_args()

    started = time.monotonic()
    state_path = args.scene_state.expanduser().resolve(strict=True)
    audit_path = args.dedup_audit.expanduser().resolve(strict=True)
    semantic_path = args.semantic_catalog.expanduser().resolve(strict=True)
    output_state_path = args.output_state.expanduser().resolve()
    if output_state_path == state_path:
        raise ValueError("output-state must differ from the immutable input scene-state")
    roots = [path.expanduser().resolve(strict=True) for path in args.mask_root]
    payload, state = _load_state(state_path)
    audit = _load_json(audit_path)
    semantic = _load_json(semantic_path)
    if not isinstance(audit, dict):
        raise TypeError("dedup audit must be a JSON object")
    if not isinstance(semantic, list):
        raise TypeError("semantic catalog must be a JSON array")
    visible, _ = presentation_ids(state)
    crop_counts = _decodable_crop_counts(state, visible, roots)
    output_state, report, clusters, labels = run_acceptance_policy(
        state,
        audit,
        semantic,
        scene_id=args.scene_id,
        decodable_crop_counts=crop_counts,
        pre_qa_wall_seconds=args.pre_qa_wall_seconds,
        apply_holdouts=args.mode == "apply",
    )
    report["mode"] = args.mode
    report["sources"] = {
        "scene_state": {"path": str(state_path), "sha256": _sha256(state_path)},
        "dedup_audit": {"path": str(audit_path), "sha256": _sha256(audit_path)},
        "semantic_catalog": {"path": str(semantic_path), "sha256": _sha256(semantic_path)},
        "mask_roots": [str(path) for path in roots],
    }
    output_payload = dict(payload)
    output_payload["state"] = output_state
    _atomic_torch(output_state_path, output_payload)
    _atomic_json(args.output_clusters.expanduser().resolve(), {
        "schema": "farm.final-acceptance-clusters.v1",
        "scene_id": args.scene_id,
        "clusters": clusters,
    })
    _atomic_json(args.output_labels.expanduser().resolve(), {
        "schema": "farm.final-acceptance-labels.v1",
        "scene_id": args.scene_id,
        "objects": labels,
        "sample_ids": report["label_sample_ids"],
    })
    elapsed = time.monotonic() - started
    report = refresh_runtime_status(report, elapsed)
    _atomic_json(args.output_report.expanduser().resolve(), report)
    print(json.dumps({
        "status": report["status"],
        "presentation_before": report["counts"]["presentation_before"],
        "presentation_after": report["counts"]["presentation_after"],
        "held_objects": report["counts"]["held_objects"],
        "blocking_clusters": report["counts"]["remaining_blocking_clusters"],
        "label_hard_errors": report["counts"]["label_hard_errors"],
        "elapsed_seconds": report["runtime"]["elapsed_seconds"],
    }, indent=2))
    return 2 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
