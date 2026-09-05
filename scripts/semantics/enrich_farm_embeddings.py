#!/usr/bin/env python3
"""Populate pinned text or crop embeddings in a saved FARM state."""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import base64
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scene_graph.map_update.mask_observations import resolve_object_mask_observations


TEXT_TASK = (
    "Embed for object identity and affordance retrieval. "
    "Use the object's category, supercategory, and visible attributes; "
    "ignore viewpoint and wording."
)
VLM_MAX_IMAGE_PIXELS = 512 * 512


def _vl_payload(model: str, image_uri: str) -> dict[str, Any]:
    """Build a bounded single-crop pooling request for the pinned VL model."""

    return {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": "Represent the given image."},
            ],
        }],
        "encoding_format": "float",
        "mm_processor_kwargs": {"max_pixels": VLM_MAX_IMAGE_PIXELS},
    }


def _request(url: str, payload: dict[str, Any], timeout: float = 180.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("embedding endpoint returned a non-object payload")
    return value


def _model_ready(base_url: str, model: str) -> None:
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    served = {
        str(row.get("id"))
        for row in (payload.get("data") or [])
        if isinstance(row, dict)
    }
    if model not in served:
        raise RuntimeError(f"embedding service does not expose expected model {model!r}")


def _rows(payload: dict[str, Any], expected: int) -> list[list[float]]:
    indexed: dict[int, Any] = {}
    for fallback, row in enumerate(payload.get("data") or []):
        if isinstance(row, dict):
            indexed[int(row.get("index", fallback))] = row.get("embedding")
    output: list[list[float]] = []
    dimension: int | None = None
    for index in range(expected):
        vector = np.asarray(indexed.get(index) or [], dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            output.append([])
            continue
        vector /= float(np.linalg.norm(vector) + 1e-12)
        if dimension is None:
            dimension = int(vector.size)
        elif vector.size != dimension:
            raise ValueError("embedding endpoint returned inconsistent dimensions")
        output.append(vector.tolist())
    return output


def _load(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    wrapper = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(wrapper, dict):
        raise ValueError("expected a FARM state dictionary")
    state = wrapper.get("state") if isinstance(wrapper.get("state"), dict) else wrapper
    if not isinstance(state, dict):
        raise ValueError("FARM state payload is missing")
    return wrapper, state


def _ids_and_active(state: dict[str, Any]) -> tuple[list[int], list[bool]]:
    means = state.get("means")
    count = int(means.shape[0]) if isinstance(means, torch.Tensor) else len(state.get("object_id") or [])
    ids_value = state.get("object_id")
    if isinstance(ids_value, torch.Tensor):
        ids = [int(value) for value in ids_value.detach().cpu().tolist()]
    else:
        ids = [int(value) for value in (ids_value or range(count))]
    active_value = state.get("active")
    if isinstance(active_value, torch.Tensor):
        active = [bool(value) for value in active_value.detach().cpu().tolist()]
    else:
        active = [bool(value) for value in (active_value or [False] * count)]
    if len(ids) != count or len(active) != count:
        raise ValueError("object_id/active lengths differ from means")
    return ids, active


def _ensure_list(state: dict[str, Any], key: str, count: int) -> list[Any]:
    value = list(state.get(key) or [])
    if len(value) < count:
        value.extend([] for _ in range(count - len(value)))
    if len(value) != count:
        raise ValueError(f"{key} length differs from object count")
    return value


def _crop_bytes(paths: list[Path], limit: int) -> list[bytes]:
    candidates: list[tuple[int, bytes]] = []
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as archive:
                encoded = bytes(np.asarray(archive["crop_jpeg_bytes"], dtype=np.uint8))
        except Exception:
            continue
        if encoded:
            candidates.append((len(encoded), encoded))
    candidates.sort(reverse=True, key=lambda item: item[0])
    return [encoded for _, encoded in candidates[: max(1, limit)]]


def _mean(vectors: list[list[float]]) -> list[float]:
    rows = [np.asarray(value, dtype=np.float32) for value in vectors if value]
    if not rows:
        return []
    if len({row.size for row in rows}) != 1:
        raise ValueError("crop embedding dimensions differ")
    value = np.mean(np.stack(rows), axis=0)
    value /= float(np.linalg.norm(value) + 1e-12)
    return value.tolist()


def _save_atomic(wrapper: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(wrapper, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--mode", choices=("text", "vl"), required=True)
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mask-root", type=Path, action="append", default=[])
    parser.add_argument("--crops-per-object", type=int, default=3)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    wrapper, state = _load(args.scene_state.resolve())
    ids, active = _ids_and_active(state)
    count = len(ids)
    _model_ready(args.vllm_url, args.model)
    endpoint = args.vllm_url.rstrip("/") + "/embeddings"
    missing: list[int] = []
    dimensions: set[int] = set()
    mask_index = None

    if args.mode == "text":
        captions = list(state.get("object_caption") or [])
        categories = list(state.get("object_category") or [])
        attributes = list(state.get("object_key_attributes") or [])
        if any(len(value) != count for value in (captions, categories, attributes)):
            raise ValueError("semantic fields are not aligned with object count")
        indices: list[int] = []
        texts: list[str] = []
        for index, enabled in enumerate(active):
            if not enabled:
                continue
            parts = [str(categories[index]).strip(), str(captions[index]).strip()]
            parts.extend(str(value).strip() for value in (attributes[index] or []))
            document = "; ".join(value for value in parts if value).lower()
            if not document:
                missing.append(ids[index])
                continue
            indices.append(index)
            texts.append(f"Instruct: {TEXT_TASK}\nDocument: {document}")
        vectors = _rows(
            _request(endpoint, {"model": args.model, "input": texts}),
            len(texts),
        )
        output = _ensure_list(state, "object_caption_embedding", count)
        history = _ensure_list(state, "object_caption_embedding_history", count)
        for index, vector in zip(indices, vectors):
            if not vector:
                missing.append(ids[index])
                continue
            output[index] = vector
            history[index] = [*list(history[index] or []), vector]
            dimensions.add(len(vector))
        state["object_caption_embedding"] = output
        state["object_caption_embedding_history"] = history
    else:
        if not args.mask_root:
            raise ValueError("--mode vl requires at least one --mask-root")
        mask_index = resolve_object_mask_observations(
            state, [root.expanduser().resolve() for root in args.mask_root]
        )
        output = _ensure_list(state, "object_qwen3_vl_embedding", count)
        history = _ensure_list(state, "object_qwen3_vl_embedding_history", count)
        for index, enabled in enumerate(active):
            if not enabled:
                continue
            crops = _crop_bytes(mask_index.for_index(index), int(args.crops_per_object))
            if not crops:
                missing.append(ids[index])
                continue
            crop_vectors: list[list[float]] = []
            for encoded in crops:
                uri = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
                payload = _vl_payload(args.model, uri)
                crop_vectors.extend(_rows(_request(endpoint, payload), 1))
            vector = _mean(crop_vectors)
            if not vector:
                missing.append(ids[index])
                continue
            output[index] = vector
            history[index] = [*list(history[index] or []), vector]
            dimensions.add(len(vector))
        state["object_qwen3_vl_embedding"] = output
        state["object_qwen3_vl_embedding_history"] = history

    report = {
        "schema": "farm.embedding-enrichment.v1",
        "status": "PASS" if not missing and len(dimensions) == 1 else "FAIL",
        "mode": args.mode,
        "model": args.model,
        "active_objects": int(sum(active)),
        "embedded_objects": int(sum(active) - len(set(missing))),
        "missing_object_ids": sorted(set(missing)),
        "dimensions": sorted(dimensions),
        "mask_observation_contract": (
            mask_index.diagnostics if mask_index is not None else {"mode": "not_applicable"}
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        print(json.dumps(report, indent=2))
        return 1
    wrapper.setdefault("meta", {})["embedding_enrichment"] = report
    _save_atomic(wrapper, args.output_state.resolve())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
