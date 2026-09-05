#!/usr/bin/env python3
"""Freeze a train-only consensus-plane pruning ablation for held-out QC.

The stage never reads held-out masks or held-out QC.  It demotes only existing
Gaussian ownership to ``-1``, keeps every physical OBB unchanged, and publishes
a standard frozen-candidate manifest that can be consumed by FARM's isolated
projection/evaluation tools.  The result remains non-release until a new
frozen held-out projection passes.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from farm_runtime.frozen_heldout_evidence import (  # noqa: E402
    _resolve_exact_gaussian_artifacts,
    _verify_gaussian_shapes,
)
from farm_runtime.frozen_heldout_qc import (  # noqa: E402
    _remember_hash,
    _validate_candidate,
    _validate_fold_manifest,
    _verify_artifact,
    sha256_file,
)
from farm_runtime.post_lift_planar_support import (  # noqa: E402
    PlanarSupportPolicy,
    apply_pruning_to_labels,
    evaluate_planar_support_candidate,
)
from farm_runtime.post_lift_release_gate import (  # noqa: E402
    quaternion_wxyz_to_matrix,
)
from tools.farm_shaper_bridge.common import open_graphdeco_ply  # noqa: E402


REPORT_SCHEMA = "farm.post-lift-planar-support-refinement.v1"
TRAIN_REFINEMENT_SCHEMA = "farm.full-colmap-mask-refinement.v1"
GEOMETRY_AUDIT_SCHEMA = "farm.object-geometry-audit.v3"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _artifact(path: Path, *, published_path: Path | None = None) -> dict[str, Any]:
    return {
        "path": str((published_path or path).resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _atomic_json(path: Path, payload: object) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
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
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _object_ids(path: Path | None, available: set[int]) -> list[int]:
    if path is None:
        return sorted(available)
    values: list[int] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            values.append(int(text))
        except ValueError as exc:
            raise ValueError(f"invalid object ID at line {line_number}: {text!r}") from exc
    selected = sorted(set(values))
    if not selected:
        raise ValueError("object ID file is empty")
    missing = sorted(set(selected).difference(available))
    if missing:
        raise ValueError(f"requested IDs are absent from base candidate: {missing}")
    return selected


def _state_numpy(state: Mapping[str, Any], name: str) -> np.ndarray:
    value = state.get(name)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _verify_train_refinement(
    result_path: Path,
    *,
    train: Mapping[str, Mapping[str, Any]],
    heldout: Mapping[str, Mapping[str, Any]],
    hashes: dict[Path, str],
) -> dict[str, Any]:
    result = _load_json(result_path)
    _remember_hash(result_path, hashes)
    contract = result.get("input_view_contract")
    if (
        result.get("schema") != TRAIN_REFINEMENT_SCHEMA
        or result.get("status") != "PASS"
        or result.get("refinement_role") != "train_fit"
        or not isinstance(contract, Mapping)
        or contract.get("split") != "train"
        or contract.get("active_split") != "train"
        or contract.get("state_fit_authorized") is not True
    ):
        raise ValueError("train refinement result lacks an authorised train-fit contract")
    frames_path = Path(str(result.get("rescue_frames_json") or "")).expanduser().resolve(
        strict=True
    )
    frames = _load_json(frames_path)
    _remember_hash(frames_path, hashes)
    fold = frames.get("full_colmap_fold")
    if (
        frames.get("schema_version") != "farm_frames_json_v1"
        or not isinstance(fold, Mapping)
        or fold.get("role") != "train"
        or fold.get("state_fit_authorized") is not True
    ):
        raise ValueError("train refinement frames lack a train fold contract")
    sources = [str(row.get("source_image") or "") for row in frames.get("frames") or []]
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("train refinement sources are empty or duplicated")
    if not set(sources).issubset(train) or set(sources).intersection(heldout):
        raise ValueError("train refinement sources leak outside the frozen train fold")
    timestamps = {
        str(train[source].get("physical_timestamp") or "") for source in sources
    }
    heldout_timestamps = {
        str(row.get("physical_timestamp") or "") for row in heldout.values()
    }
    if timestamps.intersection(heldout_timestamps):
        raise ValueError("train refinement timestamps leak into heldout")
    return {
        "result": result,
        "frames_path": frames_path,
        "source_images": sources,
        "physical_timestamps": sorted(timestamps),
    }


def _candidate_paths(
    candidate: Mapping[str, Any], candidate_path: Path, hashes: dict[Path, str]
) -> dict[str, Path]:
    artifacts = candidate.get("gaussian_artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("base candidate lacks gaussian_artifacts")
    paths: dict[str, Path] = {}
    for name in ("source_ply", "source_labels", "candidate_ply", "candidate_labels"):
        path, _ = _verify_artifact(
            artifacts.get(name),
            base=candidate_path.parent,
            field=f"base candidate {name}",
            hashes=hashes,
        )
        paths[name] = path
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--base-candidate", type=Path, required=True)
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--train-refinement-result", type=Path, required=True)
    parser.add_argument("--object-id-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    )
    hashes: dict[Path, str] = {}
    try:
        fold_path = args.fold_manifest.expanduser().resolve(strict=True)
        base_path = args.base_candidate.expanduser().resolve(strict=True)
        state_path = args.scene_state.expanduser().resolve(strict=True)
        geometry_path = args.geometry_audit.expanduser().resolve(strict=True)
        train_result_path = args.train_refinement_result.expanduser().resolve(strict=True)
        base = _load_json(base_path)
        _remember_hash(base_path, hashes)
        scale = float(base.get("meters_per_scene_unit") or 0.0)
        (
            _fold,
            train,
            heldout,
            membership,
            _depth,
            train_frames_path,
            _heldout_frames_path,
        ) = _validate_fold_manifest(fold_path, meters_per_scene_unit=scale, hashes=hashes)
        base, base_ids = _validate_candidate(
            base_path,
            fold_sha256=hashes[fold_path],
            train_sha256=hashes[train_frames_path],
            scale=scale,
            train=train,
            heldout=heldout,
            membership=membership,
            hashes=hashes,
        )
        paths = _candidate_paths(base, base_path, hashes)
        _resolve_exact_gaussian_artifacts(base, base_path, paths, hashes)
        source_labels, labels = _verify_gaussian_shapes(
            paths["source_ply"],
            paths["source_labels"],
            paths["candidate_ply"],
            paths["candidate_labels"],
        )
        del source_labels
        upstream_path, upstream_spec = _verify_artifact(
            base.get("upstream_lift_result"),
            base=base_path.parent,
            field="base candidate upstream_lift_result",
            hashes=hashes,
        )
        upstream = _load_json(upstream_path)
        if (
            upstream.get("schema_version") != "farm.gaussian-lift.result.v1"
            or upstream.get("status") != "PASS"
        ):
            raise ValueError("base candidate upstream lift is not a PASS v1 result")

        train_refinement = _verify_train_refinement(
            train_result_path, train=train, heldout=heldout, hashes=hashes
        )
        geometry = _load_json(geometry_path)
        _remember_hash(geometry_path, hashes)
        if geometry.get("schema") != GEOMETRY_AUDIT_SCHEMA:
            raise ValueError("unsupported geometry audit schema")
        geometry_rows = {
            int(row["object_id"]): row
            for row in geometry.get("objects") or []
            if isinstance(row, Mapping) and "object_id" in row
        }
        if len(geometry_rows) != len(geometry.get("objects") or []):
            raise ValueError("geometry audit contains duplicate/invalid object rows")

        import torch

        wrapper = torch.load(state_path, map_location="cpu", weights_only=False)
        state = wrapper.get("state", wrapper) if isinstance(wrapper, Mapping) else wrapper
        if not isinstance(state, Mapping):
            raise TypeError("scene state is not a mapping")
        _remember_hash(state_path, hashes)
        state_ids = _state_numpy(state, "object_id").astype(np.int64).reshape(-1)
        if len(state_ids) != len(set(state_ids.tolist())):
            raise ValueError("scene state contains duplicate object IDs")
        index_by_id = {int(value): index for index, value in enumerate(state_ids.tolist())}
        centers = _state_numpy(state, "object_box_centers_m").astype(np.float64)
        dimensions = _state_numpy(state, "object_box_dimensions_m").astype(np.float64)
        quaternions = _state_numpy(state, "object_box_wxyz").astype(np.float64)
        geometry_status = list(state.get("object_geometry_status") or [])

        requested = _object_ids(
            args.object_id_file.expanduser().resolve(strict=True)
            if args.object_id_file is not None
            else None,
            set(base_ids),
        )
        table = open_graphdeco_ply(paths["source_ply"])
        means_m = table.xyz * scale
        refined = labels.astype(np.int32, copy=True)
        policy = PlanarSupportPolicy()
        rows: list[dict[str, Any]] = []
        applied: list[int] = []
        followup_obb: list[int] = []
        for object_id in requested:
            if object_id not in index_by_id:
                raise ValueError(f"object {object_id} is absent from scene state")
            index = index_by_id[object_id]
            audit_row = geometry_rows.get(object_id)
            if audit_row is None:
                rows.append(
                    {
                        "object_id": object_id,
                        "status": "not_applicable",
                        "reasons": ["object_missing_from_geometry_audit"],
                        "recommended_route": "leave",
                    }
                )
                continue
            selected = np.flatnonzero(refined == object_id)
            row_center = np.asarray(audit_row.get("box_center_m"), dtype=np.float64)
            row_dimensions = np.asarray(audit_row.get("box_dimensions_m"), dtype=np.float64)
            row_wxyz = np.asarray(audit_row.get("box_wxyz"), dtype=np.float64)
            if (
                row_center.shape != (3,)
                or row_dimensions.shape != (3,)
                or row_wxyz.shape != (4,)
                or not np.allclose(row_center, centers[index], rtol=1e-5, atol=1e-5)
                or not np.allclose(row_dimensions, dimensions[index], rtol=1e-5, atol=1e-5)
                or not np.allclose(row_wxyz, quaternions[index], rtol=1e-5, atol=1e-5)
            ):
                raise ValueError(f"geometry audit/state OBB mismatch for object {object_id}")
            selection = audit_row.get("observation_selection")
            train_geometry_ok = bool(
                audit_row.get("status") == "geometry_pass"
                and index < len(geometry_status)
                and str(geometry_status[index]) == "geometry_pass"
                and isinstance(selection, Mapping)
                and selection.get("mode") == "preferred_source"
                and selection.get("preferred_source") == "full_colmap_sam3_refinement"
                and selection.get("fallback_used") is False
            )
            if not train_geometry_ok:
                rows.append(
                    {
                        "object_id": object_id,
                        "status": "not_applicable",
                        "reasons": ["train_geometry_contract_not_satisfied"],
                        "recommended_route": "leave",
                    }
                )
                continue
            keep, decision = evaluate_planar_support_candidate(
                means_m[selected],
                center_m=centers[index],
                dimensions_m=dimensions[index],
                rotation_matrix=quaternion_wxyz_to_matrix(quaternions[index]),
                planar_evidence=audit_row.get("planar_evidence") or {},
                policy=policy,
            )
            decision = {"object_id": object_id, **decision}
            rows.append(decision)
            if decision["status"] == "candidate":
                refined = apply_pruning_to_labels(
                    refined, object_id=object_id, object_keep_mask=keep
                )
                applied.append(object_id)
                if decision.get("followup_route") == "obb_refit_after_mask_qc":
                    followup_obb.append(object_id)

        labels_path = temporary / "per_gaussian_object_id.npy"
        with labels_path.open("wb") as stream:
            np.save(stream, refined.astype(np.int32), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        published_labels = output / labels_path.name
        audit_path = temporary / "audit.json"
        published_audit = output / audit_path.name
        report = {
            "schema": REPORT_SCHEMA,
            "status": "PASS" if applied else "BLOCKED",
            "reason": None if applied else "no_safe_planar_pruning_candidate",
            "publication_eligible": False,
            "publication_blockers": [
                "new_unseen_holdout_fold_required",
                "fresh_frozen_heldout_projection_required",
                "fresh_frozen_heldout_qc_required",
            ],
            "policy": policy.to_dict(),
            "contracts": {
                "fit_uses_train_fold_only": True,
                "heldout_masks_read": False,
                "heldout_qc_read": False,
                "physical_obb_mutated": False,
                "gaussian_geometry_mutated": False,
                "ownership_promotion_forbidden": True,
                "only_object_to_unknown_demotion": True,
                "in_plane_support_must_be_preserved": True,
            },
            "counts": {
                "source_gaussians": int(labels.size),
                "requested_objects": len(requested),
                "candidate_objects": len(applied),
                "pruned_gaussians": int(np.count_nonzero((labels >= 0) & (refined < 0))),
            },
            "requested_object_ids": requested,
            "candidate_object_ids": applied,
            "obb_refit_after_mask_qc_ids": followup_obb,
            "objects": rows,
            "inputs": {
                "fold_manifest": str(fold_path),
                "base_candidate": str(base_path),
                "scene_state": str(state_path),
                "geometry_audit": str(geometry_path),
                "train_refinement_result": str(train_result_path),
                "source_ply": str(paths["source_ply"]),
                "base_candidate_labels": str(paths["candidate_labels"]),
            },
            "train_evidence": {
                "source_images": len(train_refinement["source_images"]),
                "physical_timestamps": len(train_refinement["physical_timestamps"]),
            },
            "timing_seconds": {"total": time.perf_counter() - started},
        }
        _atomic_json(audit_path, report)

        for name, ids in (
            ("mask_geometry_refinement_ids.txt", applied),
            ("obb_refit_after_mask_qc_ids.txt", followup_obb),
        ):
            (temporary / name).write_text(
                "".join(f"{object_id}\n" for object_id in ids), encoding="utf-8"
            )

        candidate_path: Path | None = None
        if applied:
            candidate_ply_spec = dict(base["gaussian_artifacts"]["candidate_ply"])
            labels_spec = _artifact(labels_path, published_path=published_labels)
            candidate = {
                "schema": "farm.frozen-heldout-candidate.v1",
                "status": "frozen",
                "meters_per_scene_unit": scale,
                "fold_manifest_sha256": hashes[fold_path],
                "train_frames_sha256": hashes[train_frames_path],
                "fit_contract": dict(base["fit_contract"]),
                "gaussian_contract": {
                    "source_candidate_gaussian_order_aligned": True,
                    "source_gaussian_count": int(labels.size),
                    "candidate_gaussian_count": int(labels.size),
                    "candidate_frozen": True,
                    "labels_frozen": True,
                    "heldout_used_for_fit": False,
                },
                "gaussian_artifacts": {
                    "source_ply": dict(base["gaussian_artifacts"]["source_ply"]),
                    "source_labels": dict(base["gaussian_artifacts"]["candidate_labels"]),
                    "candidate_ply": candidate_ply_spec,
                    "candidate_labels": labels_spec,
                },
                "upstream_lift_result": dict(upstream_spec),
                "publication_eligible": False,
                "publication_blockers": [
                    "new_unseen_holdout_fold_required",
                    "fresh_frozen_heldout_projection_required",
                    "fresh_frozen_heldout_qc_required",
                ],
                "post_lift_refinement": {
                    "schema": REPORT_SCHEMA,
                    "kind": "fixed_train_rgbd_consensus_normal_slab_pruning",
                    "base_candidate": _artifact(base_path),
                    "scene_state": _artifact(state_path),
                    "geometry_audit": _artifact(geometry_path),
                    "train_refinement_result": _artifact(train_result_path),
                    "audit": _artifact(audit_path, published_path=published_audit),
                    "candidate_object_ids": applied,
                    "physical_obb_mutated": False,
                    "heldout_consumed": False,
                },
                "provenance": {
                    "consumed_input_hashes": {
                        str(path): digest
                        for path, digest in sorted(hashes.items(), key=lambda item: str(item[0]))
                    }
                },
                "objects": [
                    {
                        "object_id": object_id,
                        "artifacts": [candidate_ply_spec, labels_spec],
                    }
                    for object_id in applied
                ],
            }
            candidate_path = temporary / "candidate.json"
            _atomic_json(candidate_path, candidate)

        changed = [
            str(path) for path, digest in hashes.items() if sha256_file(path) != digest
        ]
        if changed:
            raise RuntimeError(f"input changed during planar candidate build: {changed[:8]}")
        marker = {
            "schema": "farm.post-lift-planar-support-refinement.result.v1",
            "status": "success" if applied else "blocked",
            "audit": "audit.json",
            "audit_sha256": sha256_file(audit_path),
            "candidate": "candidate.json" if candidate_path is not None else None,
            "candidate_sha256": (
                sha256_file(candidate_path) if candidate_path is not None else None
            ),
            "candidate_object_ids": applied,
        }
        _atomic_json(
            temporary / ("_SUCCESS.json" if applied else "_BLOCKED.json"), marker
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "status": "PASS" if applied else "BLOCKED",
                "output": str(output),
                "candidate_object_ids": applied,
                "obb_refit_after_mask_qc_ids": followup_obb,
            },
            sort_keys=True,
        )
    )
    return 0 if applied else 2


if __name__ == "__main__":
    raise SystemExit(main())
