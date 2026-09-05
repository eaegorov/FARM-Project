"""F0 reproducibility helpers. No inference, fitting, or release promotion."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    # Normalize JSON object keys before sorting: integer keys become strings on
    # disk, so sorting integers before serialization is not round-trip stable.
    def unique_object(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key after normalization: {key}")
            result[key] = item
        return result
    normalized = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False), object_pairs_hook=unique_object)
    return hashlib.sha256(json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def describe_file(path: Path) -> dict[str, Any]:
    stat = path.stat()
    result = {"path": str(path.absolute()), "bytes": stat.st_size, "sha256": sha256_file(path)}
    if path.is_symlink():
        result["symlink"] = os.readlink(path)
    # A concurrently replaced input must never silently become a frozen input.
    after = path.stat()
    if (stat.st_ino, stat.st_size, stat.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"input changed while hashing: {path}")
    return result


def verify_source_snapshot(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = []
    expected = set()
    for row in rows:
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts or rel.as_posix() in expected:
            raise ValueError(f"invalid or duplicate source path: {rel}")
        expected.add(rel.as_posix())
        path = root / rel
        if not path.is_file():
            errors.append(f"missing:{rel}")
            continue
        if row.get("symlink") != (os.readlink(path) if path.is_symlink() else None):
            errors.append(f"symlink:{rel}")
        if sha256_file(path) != row["sha256"]:
            errors.append(f"content:{rel}")
        if row.get("mode") and oct(path.lstat().st_mode & 0o777) != row["mode"]:
            errors.append(f"mode:{rel}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    errors.extend(f"unexpected:{rel}" for rel in sorted(actual - expected))
    return {"status": "PASS" if not errors else "FAIL", "file_count": len(rows), "errors": errors}


def v11_inventory(experiment: Path) -> dict[str, Any]:
    """Read actual ownership and independent QC; a legacy bank name is not a gate."""
    import numpy as np

    candidate = experiment / "07_gaussian_lift/area_cap_planar_v5_dedup_upright_candidate"
    lift = candidate / "lift"
    qc = json.loads((candidate / "frozen_heldout_v1/qc.json/report.json").read_text())
    allow = json.loads((candidate / "post_lift_obb_consensus_yaw_full_queue_v1/post_lift_release_audit/verified_allowlists_manifest.json").read_text())
    semantic = json.loads((experiment / "06_full_colmap_refinement/semantic_scene_wide_resolved_v3/whole_object_audit.json").read_text())
    owners = np.load(lift / "per_gaussian_object_id.npy", mmap_mode="r", allow_pickle=False)
    ids, counts = np.unique(owners, return_counts=True)
    owned = {str(int(i)): int(n) for i, n in zip(ids, counts) if i >= 0}
    accepted = sorted(int(r["object_id"]) for r in qc["objects"] if r["status"] == "verified")
    rejected = sorted(int(r["object_id"]) for r in qc["objects"] if r["status"] == "rejected")
    exact = sorted(int(r["object_id"]) for r in semantic["objects"]
                   if r.get("semantic_identity_re_adjudication", {}).get("accepted") is True)
    return {
        "schema": "farm.quality-baseline-inventory.v1",
        "source_gaussians": int(owners.size), "owned_counts_before_external_qc": owned,
        "mask_verified": accepted, "mask_rejected": rejected,
        "obb_verified": allow["obb_verified"]["object_ids"], "exact_identity": exact,
        "rejected_ids_still_in_legacy_ownership": sorted(set(rejected) & set(map(int, owned))),
        "accepted_gaussians": sum(owned.get(str(i), 0) for i in accepted),
        "semantic_status_counts": semantic["status_counts"],
        "scope_readiness_counts": semantic["inpainting_tier_counts"],
        "release_eligible": allow["release_eligible"],
        "external_qc_status": qc["status"],
        "warning": "Legacy ownership is frozen evidence, not a post-QC release bank.",
    }
