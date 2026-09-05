#!/usr/bin/env python3
"""Sample existing Gaussian ownership for independent F1 3D inspection.

Samples are questions for a human, not gold labels. No new membership, OBB,
normal, graph growth, or threshold tuning is performed.
"""
from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import json
from pathlib import Path

import numpy as np

from farm_runtime.quality_baseline import describe_file, json_digest, write_json


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--ownership", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ownership-result", type=Path, required=True)
    p.add_argument("--samples-per-object", type=int, default=64)
    args = p.parse_args()
    if args.output.exists() or args.samples_per_object < 1:
        raise ValueError("output must be new and sample count positive")
    packet = json.loads(args.packet.read_text())
    release = json.loads(args.ownership_result.read_text())
    owner_descriptor = describe_file(args.ownership)
    expected_owner = release.get("artifacts", {}).get("final", {}).get("object_id", {}).get("sha256")
    if expected_owner != owner_descriptor["sha256"]:
        raise ValueError("ownership not bound to its original lift manifest")
    if release.get("inputs", {}).get("source_ply_sha256") != packet["source_ply"]["sha256"]:
        raise ValueError("ownership belongs to a different source Gaussian scene/order")
    source = np.load(args.ownership, mmap_mode="r", allow_pickle=False)
    if source.ndim != 1 or source.dtype.kind not in "iu":
        raise ValueError("ownership must be an integer vector in source PLY row order")
    # Header proof binds row range. The packet already hashes the original PLY.
    vertex_count = None
    with Path(packet["source_ply"]["path"]).open("rb") as stream:
        for _ in range(256):
            line = stream.readline().decode("ascii").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                break
    if vertex_count != len(source):
        raise ValueError("ownership row count does not match source PLY")
    objects = []
    for row in packet["objects"]:
        oid = row["object_id"]
        indexes = np.flatnonzero(source == oid)
        if indexes.size:
            rng = np.random.default_rng(int(json_digest([packet["source_ply"]["sha256"], oid])[:16], 16))
            selected = np.sort(rng.choice(indexes, size=min(args.samples_per_object, indexes.size), replace=False))
        else:
            selected = np.empty(0, dtype=np.int64)
        objects.append({"object_id": oid, "source_claim_count": int(indexes.size),
                        "sampling": "uniform_among_legacy_claims; additional unclaimed/protected samples required",
                        "gaussians": [{"gaussian_id": int(i), "label": "unknown", "reviewed": False,
                                       "reviewer_id": None, "evidence_views": [], "notes": ""} for i in selected],
                        "additional_gaussians": [], "scope_notes": ""})
    result = {"schema": "farm.gaussian-inspection-template.v1", "packet_sha256": json_digest(packet),
              "source_ply": packet["source_ply"], "source_gaussian_count": len(source),
              "ownership": owner_descriptor, "ownership_result": describe_file(args.ownership_result),
              "objects": objects, "gold": False,
              "allowed_labels": ["object_core", "object_boundary", "thin_part", "attachment", "protected_neighbour", "carrier", "content", "background", "unknown"],
              "limitation": "Samples from existing claims assess contamination, not recall. Human must add missed thin parts and protected/unclaimed Gaussians independently."}
    write_json(args.output, result)
    print(json.dumps({"objects": len(objects), "samples": sum(len(o["gaussians"]) for o in objects), "gold": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
