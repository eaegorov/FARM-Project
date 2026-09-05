#!/usr/bin/env python3
"""Build immutable frozen-heldout QC evidence, never a prediction surrogate."""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from farm_runtime.frozen_heldout_evidence import (  # noqa: E402
    materialize_evidence_payload,
    prepare_frozen_heldout_evidence,
)
from farm_runtime.frozen_heldout_qc import sha256_file  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _public_preflight(prepared: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in prepared.items()
        if key
        not in {
            "targets",
            "competing_unions",
            "projections",
        }
    } | {
        "targets": [
            {key: value for key, value in target.items() if key != "target_array"}
            for target in prepared.get("targets") or []
        ],
        "projection_count": len(prepared.get("projections") or {}),
    }


def _artifact(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": os.path.relpath(path, base),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _revalidate_inputs(prepared: Mapping[str, Any]) -> None:
    consumed = prepared.get("consumed_input_hashes")
    if not isinstance(consumed, Mapping) or not consumed:
        raise ValueError("evidence preflight lacks consumed input hashes")
    changed = [
        str(path)
        for path, expected in consumed.items()
        if sha256_file(Path(str(path)).resolve(strict=True)) != str(expected)
    ]
    if changed:
        raise RuntimeError(f"frozen inputs changed before publish: {changed[:8]}")


def _publish(output: Path, prepared: Mapping[str, Any]) -> Path | None:
    destination = output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    evidence_path: Path | None = None
    try:
        preflight_path = temporary / "preflight.json"
        _write_json(preflight_path, _public_preflight(prepared))
        result: dict[str, Any] = {
            "schema": "farm.frozen-heldout-evidence-build-result.v1",
            "status": prepared["status"],
            "preflight": "preflight.json",
            "preflight_sha256": sha256_file(preflight_path),
            "evidence": None,
            "evidence_sha256": None,
        }
        if prepared.get("status") == "PASS":
            union_root = temporary / "competing_masks"
            union_root.mkdir()
            competing_specs: dict[tuple[int, str], dict[str, Any]] = {}
            for index, target in enumerate(prepared.get("targets") or []):
                key = (int(target["object_id"]), str(target["source_image"]))
                mask = np.asarray(prepared["competing_unions"][key], dtype=np.uint8)
                path = union_root / f"object_{key[0]:06d}_view_{index:04d}.npy"
                with path.open("wb") as stream:
                    np.save(stream, mask, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                competing_specs[key] = _artifact(path, temporary)
            evidence = materialize_evidence_payload(prepared, competing_specs)
            evidence_path = temporary / "evidence.json"
            _write_json(evidence_path, evidence)
            result["evidence"] = "evidence.json"
            result["evidence_sha256"] = sha256_file(evidence_path)
        _revalidate_inputs(prepared)
        _write_json(temporary / "_RESULT.json", result)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination / "evidence.json" if evidence_path is not None else None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--source-ply", type=Path, required=True)
    parser.add_argument("--source-labels", type=Path, required=True)
    parser.add_argument("--candidate-ply", type=Path, required=True)
    parser.add_argument("--candidate-labels", type=Path, required=True)
    parser.add_argument("--heldout-reference-result", type=Path, required=True)
    parser.add_argument("--heldout-yoloe-state", type=Path, required=True)
    parser.add_argument("--heldout-yoloe-mask-root", type=Path, required=True)
    parser.add_argument("--projection-manifest", type=Path)
    parser.add_argument("--meters-per-scene-unit", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prepared = prepare_frozen_heldout_evidence(
        args.fold_manifest,
        args.candidate_manifest,
        args.source_ply,
        args.source_labels,
        args.candidate_ply,
        args.candidate_labels,
        args.heldout_reference_result,
        args.heldout_yoloe_state,
        args.heldout_yoloe_mask_root,
        meters_per_scene_unit=args.meters_per_scene_unit,
        projection_manifest_path=args.projection_manifest,
    )
    evidence = _publish(args.output_dir, prepared)
    print(
        json.dumps(
            {
                "status": prepared["status"],
                "reason": prepared["reason"],
                "output": str(args.output_dir.expanduser().resolve()),
                "evidence": str(evidence) if evidence is not None else None,
                "targets": prepared["target_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if prepared["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
