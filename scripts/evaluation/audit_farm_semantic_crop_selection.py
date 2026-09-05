#!/usr/bin/env python3
"""Audit a no-VLM crop-selection run and emit the safe Qwen ID allowlist."""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import hashlib
import json
from pathlib import Path

from farm_runtime.semantic_crop_selection_audit import audit_selection_payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _ids(path: Path) -> list[int]:
    return [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-catalog", type=Path, action="append", required=True
    )
    parser.add_argument("--requested-object-id-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-angle-degrees", type=float, default=10.0)
    parser.add_argument("--minimum-normalized-baseline", type=float, default=0.1)
    args = parser.parse_args()
    requested = args.requested_object_id_file.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    requested_ids = _ids(requested)
    sources = [path.expanduser().resolve(strict=True) for path in args.selection_catalog]
    source_audits = []
    candidates_by_id: dict[int, list[dict]] = {
        object_id: [] for object_id in requested_ids
    }
    for source_index, source in enumerate(sources):
        payload = json.loads(source.read_text(encoding="utf-8"))
        source_audit = audit_selection_payload(
            payload,
            requested_ids,
            minimum_angle_degrees=args.minimum_angle_degrees,
            minimum_normalized_baseline=args.minimum_normalized_baseline,
        )
        source_record = {
            "source_index": source_index,
            "path": str(source),
            "sha256": _sha256(source),
            "crops_per_object": int(payload.get("crops_per_object") or 0),
            "ready_object_ids": source_audit["ready_object_ids"],
        }
        source_audits.append(source_record)
        for row in source_audit["objects"]:
            if row["status"] == "missing_input":
                continue
            candidate = dict(row)
            candidate["source"] = source_record
            candidates_by_id[int(row["object_id"])].append(candidate)

    objects = []
    for object_id in sorted(set(requested_ids)):
        candidates = candidates_by_id.get(object_id) or []
        ready = [row for row in candidates if row["qwen_ready"]]
        if ready:
            # Prefer more independently valid views. For an equal crop count,
            # prefer the larger worst pose margin; all choices already passed.
            def rank(row: dict) -> tuple:
                metrics = row.get("metrics") or {}
                values = [
                    float(value)
                    for metric in metrics.values()
                    for value in metric.values()
                    if value is not None
                ]
                return (
                    min(
                        len(row.get("initial_image_ids") or []),
                        len(row.get("verification_image_ids") or []),
                    ),
                    min(values, default=0.0),
                    -int(row["source"]["source_index"]),
                )
            selected = max(ready, key=rank)
        elif candidates:
            selected = max(
                candidates,
                key=lambda row: (
                    min(
                        len(row.get("initial_image_ids") or []),
                        len(row.get("verification_image_ids") or []),
                    ),
                    -len(row.get("reason_codes") or []),
                ),
            )
        else:
            selected = {
                "object_id": object_id,
                "status": "missing_input",
                "qwen_ready": False,
                "reason_codes": ["object_absent_from_all_selection_catalogs"],
                "initial_image_ids": [],
                "verification_image_ids": [],
                "source": None,
            }
        selected = dict(selected)
        selected["evaluated_source_count"] = len(candidates)
        objects.append(selected)

    ready_ids = [row["object_id"] for row in objects if row["qwen_ready"]]
    groups: dict[int, list[int]] = {}
    for row in objects:
        if not row["qwen_ready"]:
            continue
        crop_count = int((row.get("source") or {}).get("crops_per_object") or 0)
        groups.setdefault(crop_count, []).append(int(row["object_id"]))
    audit = {
        "schema": "farm.semantic-crop-selection-audit.v1",
        "requested_object_ids": sorted(set(requested_ids)),
        "object_count": len(objects),
        "ready_count": len(ready_ids),
        "ready_object_ids": ready_ids,
        "blocked_object_ids": [
            row["object_id"] for row in objects if not row["qwen_ready"]
        ],
        "qwen_run_groups": [
            {"crops_per_object": count, "object_ids": sorted(ids)}
            for count, ids in sorted(groups.items(), reverse=True)
        ],
        "selection_sources": source_audits,
        "policy": {
            "no_vlm_requests_performed": True,
            "exact_disjoint_image_ids_required": True,
            "within_fold_and_cross_fold_pose_gates_required": True,
            "candidate_conditioned_prompts_forbidden": True,
            "largest_strictly_valid_crop_fold_preferred": True,
        },
        "objects": objects,
    }
    audit["requested_object_id_file"] = str(requested)
    audit["requested_object_id_file_sha256"] = _sha256(requested)
    _atomic_json(output / "selection_audit.json", audit)
    (output / "qwen_ready_object_ids.txt").write_text(
        "".join(f"{value}\n" for value in audit["ready_object_ids"]),
        encoding="utf-8",
    )
    for crop_count, values in sorted(groups.items(), reverse=True):
        (output / f"qwen_ready_crops_{crop_count}.txt").write_text(
            "".join(f"{value}\n" for value in sorted(values)),
            encoding="utf-8",
        )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
