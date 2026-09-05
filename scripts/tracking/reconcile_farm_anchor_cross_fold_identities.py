#!/usr/bin/env python3
"""Audit paired shadow tracker IDs against canonical metric object anchors.

This CPU-only stage is intentionally separate from raw tracker and global
association outputs.  It can emit an exact bounded Gaussian Grouping allowlist,
but it cannot publish FARM identities or authorize production/GPU training.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import json
import os
import platform
import re
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from farm_runtime.anchor_cross_fold_reconciliation import (  # noqa: E402
    ALLOWLIST_SCHEMA,
    SCHEMA,
    AnchorReconciliationPolicy,
    policy_payload,
    reconcile_object_pair,
    robust_anchor_geometry,
)
from farm_runtime.tracker_3d_signature import (  # noqa: E402
    SCHEMA as SIGNATURE_SCHEMA,
    SHADOW_SCHEMA as SIGNATURE_SHADOW_SCHEMA,
)
from farm_runtime.tracker_episode_suite import (  # noqa: E402
    PeakRssSampler,
    atomic_output_directory,
    load_materialized_suite,
    sha256_file,
)
from farm_runtime.tracker_global_association import (  # noqa: E402
    SHADOW_ABLATION_SCHEMA,
    load_tracker_measurements,
)
from scripts.geometry.plan_farm_full_colmap_rescue import (  # noqa: E402
    load_state,
    object_support_points,
)


_IDENTITY_RE = re.compile(r"^(?P<episode>[^/]+)::local:(?P<local>[1-9][0-9]*)$")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    source = path.expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {source}")
    return payload


def _provenance(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve(strict=True)
    return {
        "path": str(source),
        "bytes": int(source.stat().st_size),
        "sha256": sha256_file(source),
    }


def _verify_declared_file(row: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(row.get("path") or "")).expanduser().resolve(strict=True)
    actual = _provenance(path)
    if (
        int(row.get("bytes", -1)) != actual["bytes"]
        or str(row.get("sha256") or "") != actual["sha256"]
    ):
        raise ValueError(f"{label} provenance mismatch: {path}")
    return path


def _same_file(left: Path, right: Path, *, label: str) -> None:
    if left.expanduser().resolve(strict=True) != right.expanduser().resolve(strict=True):
        raise ValueError(f"{label} path mismatch: {left} != {right}")


def _parse_identity_key(value: object) -> tuple[str, int]:
    key = str(value or "")
    match = _IDENTITY_RE.fullmatch(key)
    if match is None:
        raise ValueError(f"invalid identity key: {key!r}")
    return str(match.group("episode")), int(match.group("local"))


def _current_rss_mib() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _source_plan_contract(plan: Mapping[str, Any], scene_state: Path) -> tuple[int, float, int]:
    if plan.get("schema") != "farm.tracker-subset-plan.v2":
        raise ValueError("anchor reconciliation requires tracker subset plan v2")
    if plan.get("planner_revision") != "object-anchor-paired-support-tracks.v2":
        raise ValueError("source plan is not the object-anchor-paired planner")
    policy = plan.get("policy")
    selection = plan.get("selection")
    validation = plan.get("validation")
    inputs = plan.get("inputs")
    if not all(isinstance(row, Mapping) for row in (policy, selection, validation, inputs)):
        raise ValueError("paired source plan lacks contract sections")
    if (
        policy.get("obb_read_or_fallback") is not False
        or policy.get("detector_bbox_used") is not False
        or policy.get("global_physical_timestamp_train_heldout_disjoint") is not True
        or int(validation.get("global_timestamp_overlap", -1)) != 0
    ):
        raise ValueError("paired source plan does not guarantee independent metric anchors")
    declared_state = inputs.get("scene_state")
    if not isinstance(declared_state, Mapping):
        raise ValueError("paired source plan lacks scene-state provenance")
    declared_path = _verify_declared_file(declared_state, label="scene state")
    _same_file(declared_path, scene_state, label="scene state")
    gates = policy.get("source_support_track_gates")
    if not isinstance(gates, Mapping):
        raise ValueError("paired source plan lacks support-track gates")
    return (
        int(gates.get("maximum_object_support_points", -1)),
        float(gates.get("support_trim_quantile", -1.0)),
        int(gates.get("minimum_object_support_points", -1)),
    )


def _validate_association(
    association_path: Path,
    *,
    episodes_by_id: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    association = _load_json(association_path, label="shadow association")
    if association.get("schema") != SHADOW_ABLATION_SCHEMA:
        raise ValueError("unsupported shadow association schema")
    if (
        association.get("mode") != "diagnostic-only-no-publish"
        or association.get("automatic_promotion_allowed") is not False
        or association.get("gg_training_authorized") is not False
        or association.get("fit_splits") != ["train"]
    ):
        raise ValueError("shadow association publication/fit contract mismatch")
    inputs = association.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("shadow association lacks inputs")
    if (
        inputs.get("artifact_hashes_verified") is not True
        or inputs.get("artifact_verification_requested") is not True
        or inputs.get("colmap_hashes_verified") is not True
        or inputs.get("local_gate_source") != "v2_shadow"
    ):
        raise ValueError("shadow association lacks verified v2 inputs")

    signature_rows = inputs.get("signature_reports")
    if not isinstance(signature_rows, list) or not signature_rows:
        raise ValueError("shadow association has no signature reports")
    declared_by_episode: dict[str, Mapping[str, Any]] = {}
    signature_payloads: dict[str, dict[str, Any]] = {}
    all_identity_rows: dict[str, dict[str, Any]] = {}
    all_keys_by_episode: dict[str, list[dict[str, Any]]] = {}
    signature_provenance: dict[str, dict[str, Any]] = {}
    for declared in signature_rows:
        if not isinstance(declared, Mapping):
            raise ValueError("signature provenance row must be an object")
        episode_id = str(declared.get("episode_id") or "")
        if episode_id not in episodes_by_id or episode_id in declared_by_episode:
            raise ValueError(f"unknown or duplicate signature episode: {episode_id}")
        path = _verify_declared_file(declared, label="3D signature")
        signature = _load_json(path, label="3D signature")
        if signature.get("schema") != SIGNATURE_SCHEMA or signature.get("status") != "pass":
            raise ValueError(f"signature is not a passing {SIGNATURE_SCHEMA}: {path}")
        signature_inputs = signature.get("inputs")
        local_rows = signature.get("local_identities")
        if not isinstance(signature_inputs, Mapping) or not isinstance(local_rows, list):
            raise ValueError(f"malformed 3D signature: {path}")
        episode = episodes_by_id[episode_id]
        if (
            signature_inputs.get("episode_id") != episode_id
            or signature_inputs.get("split") != episode.split
            or declared.get("split") != episode.split
        ):
            raise ValueError(f"signature episode/split mismatch: {episode_id}")
        episode_source = signature_inputs.get("episode")
        if not isinstance(episode_source, Mapping):
            raise ValueError(f"signature lacks episode provenance: {episode_id}")
        exact_episode_path = _verify_declared_file(episode_source, label="episode JSON")
        _same_file(exact_episode_path, episode.episode_json, label="episode JSON")
        if str(declared.get("local_gate_source") or "") != "v2_shadow":
            raise ValueError(f"signature was not declared for v2 shadow: {episode_id}")
        per_episode: list[dict[str, Any]] = []
        for raw in local_rows:
            if not isinstance(raw, Mapping):
                raise ValueError("signature local identity must be an object")
            key = str(raw.get("identity_key") or "")
            key_episode, local_id = _parse_identity_key(key)
            if key_episode != episode_id or int(raw.get("local_id", -1)) != local_id:
                raise ValueError(f"signature identity mismatch: {key}")
            if key in all_identity_rows:
                raise ValueError(f"duplicate identity across signatures: {key}")
            shadow = raw.get("decision_v2_shadow")
            if not isinstance(shadow, Mapping):
                raise ValueError(f"identity lacks v2 shadow decision: {key}")
            passed = shadow.get("passed") is True
            expected_status = "would_pass_shadow_gate" if passed else "shadow_rejected"
            if (
                shadow.get("schema") != SIGNATURE_SHADOW_SCHEMA
                or shadow.get("status") != expected_status
                or shadow.get("mode") != "diagnostic-only-no-publish"
                or shadow.get("global_association_authorized") is not False
            ):
                raise ValueError(f"invalid shadow decision contract: {key}")
            compact = {
                "identity_key": key,
                "episode_id": episode_id,
                "split": episode.split,
                "local_id": local_id,
                "passed_shadow_gate": passed,
            }
            if passed:
                geometry = shadow.get("qualified_geometry")
                if not isinstance(geometry, Mapping) or geometry.get("available") is not True:
                    raise ValueError(f"accepted shadow identity lacks geometry: {key}")
                compact.update(
                    {
                        "center_m": geometry.get("coordinate_median_m"),
                        "shape_extents_m": geometry.get("robust_pca_extents_q02_q98_m"),
                    }
                )
            all_identity_rows[key] = compact
            per_episode.append(compact)
        declared_by_episode[episode_id] = declared
        signature_payloads[episode_id] = signature
        all_keys_by_episode[episode_id] = per_episode
        signature_provenance[episode_id] = _provenance(path)
    if set(declared_by_episode) != set(episodes_by_id):
        raise ValueError("shadow association signature episode set is incomplete")

    assignments = association.get("candidate_assignments")
    local_to_global = association.get("candidate_local_to_global")
    if not isinstance(assignments, list) or not isinstance(local_to_global, Mapping):
        raise ValueError("shadow association lacks candidate assignments")
    assignment_by_key: dict[str, Mapping[str, Any]] = {}
    for row in assignments:
        if not isinstance(row, Mapping):
            raise ValueError("candidate assignment must be an object")
        key = str(row.get("identity_key") or "")
        if key not in all_identity_rows or key in assignment_by_key:
            raise ValueError(f"unknown or duplicate candidate assignment: {key}")
        episode_id, local_id = _parse_identity_key(key)
        if (
            row.get("episode_id") != episode_id
            or int(row.get("local_id", -1)) != local_id
            or row.get("split") != episodes_by_id[episode_id].split
        ):
            raise ValueError(f"candidate assignment identity mismatch: {key}")
        global_id = int(row.get("global_id", -1))
        if global_id < 0 or int(local_to_global.get(key, -1)) != global_id:
            raise ValueError(f"candidate local/global mapping mismatch: {key}")
        assignment_by_key[key] = row
        all_identity_rows[key]["global_id"] = global_id
    if set(assignment_by_key) != set(all_identity_rows) or set(local_to_global) != set(all_identity_rows):
        raise ValueError("candidate assignments do not cover every signature identity exactly")

    measurement_rows = inputs.get("tracker_measurements")
    if not isinstance(measurement_rows, list) or len(measurement_rows) != len(episodes_by_id):
        raise ValueError("shadow association tracker measurement set is incomplete")
    measurement_by_episode: dict[str, dict[str, Any]] = {}
    measurement_paths: list[Path] = []
    for declared in measurement_rows:
        if not isinstance(declared, Mapping):
            raise ValueError("tracker measurement provenance must be an object")
        path = _verify_declared_file(declared, label="tracker measurement")
        payload = _load_json(path, label="tracker measurement")
        if payload.get("schema") != "farm.sam3-concept-episode.v1" or payload.get("status") != "pass":
            raise ValueError(f"unsupported tracker measurement: {path}")
        input_row = payload.get("input")
        if not isinstance(input_row, Mapping):
            raise ValueError(f"tracker measurement lacks input: {path}")
        episode_id = str(input_row.get("episode_id") or "")
        if episode_id not in episodes_by_id or episode_id in measurement_by_episode:
            raise ValueError(f"unknown or duplicate tracker measurement: {episode_id}")
        measurement_by_episode[episode_id] = _provenance(path)
        measurement_paths.append(path)
    if set(measurement_by_episode) != set(episodes_by_id):
        raise ValueError("tracker measurements do not cover every episode")
    prompt_evidence, _ = load_tracker_measurements(
        measurement_paths, allowed_identity_keys=set(all_identity_rows)
    )
    for key, row in all_identity_rows.items():
        if not row["passed_shadow_gate"]:
            continue
        evidence = prompt_evidence.get(key)
        prompts = evidence.get("prompts") if isinstance(evidence, Mapping) else None
        if not isinstance(prompts, list) or not prompts:
            raise ValueError(f"shadow-passed SAM3 identity lacks a stable prompt: {key}")
        row["prompts"] = prompts
    return association, all_identity_rows, all_keys_by_episode, {
        "signature": signature_provenance,
        "measurement": measurement_by_episode,
    }


def _build_allowlist(
    *,
    accepted_rows: list[dict[str, Any]],
    all_keys_by_episode: Mapping[str, list[dict[str, Any]]],
    episodes_by_id: Mapping[str, Any],
    artifact_provenance: Mapping[str, Mapping[str, dict[str, Any]]],
    reconciliation_provenance: dict[str, Any],
    source_gaussians: Path,
    sparse_points_ply: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    accepted_rows = sorted(accepted_rows, key=lambda row: int(row["object_id"]))
    selected_episodes: list[dict[str, Any]] = []
    mapping: dict[str, int] = {}
    selected_keys: set[str] = set()
    object_rows: list[dict[str, Any]] = []
    for compact_id, row in enumerate(accepted_rows, start=1):
        accepted = row["accepted_mapping"]
        keys = {accepted["train_identity_key"], accepted["heldout_identity_key"]}
        selected_keys.update(keys)
        episodes = [row["train_episode_id"], row["heldout_episode_id"]]
        object_rows.append(
            {
                "gg_identity_id": compact_id,
                "canonical_object_id": int(row["object_id"]),
                "raw_shadow_global_id": int(accepted["raw_global_id"]),
                "identity_keys": sorted(keys),
                "episode_ids": episodes,
            }
        )
        for episode_id in episodes:
            episode = episodes_by_id[episode_id]
            selected_episodes.append(
                {
                    "episode_id": episode_id,
                    "canonical_object_id": int(row["object_id"]),
                    "split": episode.split,
                    "frame_count": episode.frame_count,
                    "physical_timestamps": list(episode.physical_timestamps),
                    "episode_json": _provenance(episode.episode_json),
                    "signature": artifact_provenance["signature"][episode_id],
                    "tracker_measurement": artifact_provenance["measurement"][episode_id],
                }
            )
            for identity in all_keys_by_episode[episode_id]:
                key = str(identity["identity_key"])
                mapping[key] = compact_id if key in keys else 0
    if len(selected_keys) != 2 * len(accepted_rows):
        raise ValueError("accepted anchor identities are not unique")
    if sum(value > 0 for value in mapping.values()) != len(selected_keys):
        raise ValueError("allowlist did not map every accepted identity exactly")
    plan_inputs = plan.get("inputs")
    if not isinstance(plan_inputs, Mapping):
        raise ValueError("source plan lacks inputs for allowlist")
    return {
        "schema": ALLOWLIST_SCHEMA,
        "status": "pass",
        "scope": "bounded-diagnostic-gaussian-grouping-identity-dataset",
        "publication": {
            "production_farm_identity_authorized": False,
            "production_gaussian_grouping_training_authorized": False,
            "bounded_dataset_materialization_authorized": True,
            "gpu_training_executed": False,
            "native_v1_builder_compatible": False,
            "required_adapter": "consume this exact shadow allowlist without treating v2 shadow as v1 publication",
        },
        "inputs": {
            "reconciliation": reconciliation_provenance,
            "source_gaussians": _provenance(source_gaussians),
            "sparse_points_ply": _provenance(sparse_points_ply),
            "colmap_model": plan_inputs.get("colmap_model"),
            "colmap_contract_files": plan_inputs.get("colmap_contract_files"),
        },
        "summary": {
            "canonical_object_count": len(object_rows),
            "episode_count": len(selected_episodes),
            "foreground_identity_key_count": len(selected_keys),
            "explicitly_dropped_identity_key_count": sum(value == 0 for value in mapping.values()),
            "mapping_key_count": len(mapping),
        },
        "objects": object_rows,
        "episodes": sorted(selected_episodes, key=lambda row: row["episode_id"]),
        "identity_key_to_compact_gg_id": dict(sorted(mapping.items())),
        "selected_identity_keys": sorted(selected_keys),
        "dropped_identity_keys": sorted(key for key, value in mapping.items() if value == 0),
        "guarantees": [
            "Only the unique anchor-supported train/heldout identity pair for each accepted object maps to foreground.",
            "Every other observed local identity in each selected episode maps explicitly to zero/drop.",
            "Train and heldout physical timestamp sets are disjoint.",
            "The source tracker masks, signatures and shadow association are immutable and unchanged.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--episodes-root", required=True, type=Path)
    parser.add_argument("--association", required=True, type=Path)
    parser.add_argument("--scene-state", required=True, type=Path)
    parser.add_argument("--source-gaussians", required=True, type=Path)
    parser.add_argument("--sparse-points-ply", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    policy = AnchorReconciliationPolicy()
    policy.validate()
    with PeakRssSampler() as rss:
        episodes = load_materialized_suite(args.episodes_root, verify_frame_hashes=True)
        plan = _load_json(episodes.source_plan, label="paired source plan")
        state_path = args.scene_state.expanduser().resolve(strict=True)
        max_points, trim_quantile, minimum_points = _source_plan_contract(
            plan, state_path
        )
        if minimum_points != policy.minimum_anchor_points:
            raise ValueError("reconciler minimum anchor points differs from source planner")
        episodes_by_id = {episode.episode_id: episode for episode in episodes.episodes}
        association_path = args.association.expanduser().resolve(strict=True)
        association, identities_by_key, all_keys_by_episode, artifact_provenance = _validate_association(
            association_path, episodes_by_id=episodes_by_id
        )

        plan_episodes = plan.get("episodes")
        plan_objects = plan.get("objects")
        if not isinstance(plan_episodes, list) or not isinstance(plan_objects, list):
            raise ValueError("paired source plan lacks episodes/objects")
        plan_episode_ids = {str(row.get("episode_id") or "") for row in plan_episodes if isinstance(row, Mapping)}
        if plan_episode_ids != set(episodes_by_id):
            raise ValueError("materialized and planned episode sets differ")
        pairs: dict[int, dict[str, str]] = {}
        for row in plan_episodes:
            if not isinstance(row, Mapping):
                raise ValueError("paired plan episode row must be an object")
            object_id = int(row.get("anchor_object_id", -1))
            split = str(row.get("split") or "")
            episode_id = str(row.get("episode_id") or "")
            if split not in {"train", "heldout"} or split in pairs.setdefault(object_id, {}):
                raise ValueError(f"duplicate or invalid pair split for object {object_id}")
            pairs[object_id][split] = episode_id
        if not pairs or any(set(pair) != {"train", "heldout"} for pair in pairs.values()):
            raise ValueError("each target object requires exactly one train/heldout episode")
        global_train_timestamps = {
            value for episode in episodes.episodes if episode.split == "train" for value in episode.physical_timestamps
        }
        global_heldout_timestamps = {
            value for episode in episodes.episodes if episode.split == "heldout" for value in episode.physical_timestamps
        }
        if global_train_timestamps.intersection(global_heldout_timestamps):
            raise ValueError("materialized train/heldout timestamps overlap globally")

        state = load_state(state_path)
        raw_ids = state.get("object_id")
        if raw_ids is None:
            raise ValueError("scene state lacks object_id")
        if hasattr(raw_ids, "detach"):
            raw_ids = raw_ids.detach().cpu().numpy()
        state_ids = np.asarray(raw_ids, dtype=np.int64).reshape(-1)
        object_plan_by_id = {
            int(row["object_id"]): row
            for row in plan_objects
            if isinstance(row, Mapping) and "object_id" in row
        }
        results: list[dict[str, Any]] = []
        for object_id in sorted(pairs):
            object_plan = object_plan_by_id.get(object_id)
            if not isinstance(object_plan, Mapping):
                raise ValueError(f"source plan lacks object row {object_id}")
            state_index = int(object_plan.get("state_index", -1))
            if state_index < 0 or state_index >= state_ids.size or int(state_ids[state_index]) != object_id:
                raise ValueError(f"canonical state index mismatch for object {object_id}")
            support, diagnostics = object_support_points(
                state,
                state_index,
                trim_quantile=trim_quantile,
                max_points=max_points,
            )
            anchor = robust_anchor_geometry(support)
            anchor["support_source"] = diagnostics
            pair = pairs[object_id]
            train_episode = episodes_by_id[pair["train"]]
            heldout_episode = episodes_by_id[pair["heldout"]]
            pair_keys = {
                row["identity_key"]
                for episode_id in pair.values()
                for row in all_keys_by_episode[episode_id]
                if row["passed_shadow_gate"]
            }
            identities = [identities_by_key[key] for key in sorted(pair_keys)]
            results.append(
                reconcile_object_pair(
                    object_id=object_id,
                    train_episode_id=train_episode.episode_id,
                    heldout_episode_id=heldout_episode.episode_id,
                    train_physical_timestamps=train_episode.physical_timestamps,
                    heldout_physical_timestamps=heldout_episode.physical_timestamps,
                    anchor=anchor,
                    identities=identities,
                    policy=policy,
                )
            )

        accepted = [row for row in results if row["status"] == "accepted"]
        summary = {
            "target_object_count": len(results),
            "accepted_object_count": len(accepted),
            "accepted_object_ids": [int(row["object_id"]) for row in accepted],
            "ambiguous_object_count": sum(row["status"] == "ambiguous" for row in results),
            "unresolved_object_count": sum(row["status"] == "unresolved" for row in results),
            "rejected_object_count": sum(row["status"] == "rejected" for row in results),
        }
        output_root = args.output_root.expanduser().resolve()
        with atomic_output_directory(output_root) as staging:
            reconciliation = {
                "schema": SCHEMA,
                "status": "pass",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "diagnostic-bounded-fail-closed",
                "inputs": {
                    "episodes_manifest": _provenance(episodes.manifest_path),
                    "paired_source_plan": _provenance(episodes.source_plan),
                    "scene_state": _provenance(state_path),
                    "shadow_association": _provenance(association_path),
                    "shadow_association_schema": association["schema"],
                },
                "policy": policy_payload(policy),
                "summary": summary,
                "objects": results,
                "publication": {
                    "raw_tracker_outputs_changed": False,
                    "raw_association_changed": False,
                    "authoritative_v1_changed": False,
                    "shadow_gate_promoted": False,
                    "farm_production_publication_authorized": False,
                    "gaussian_grouping_production_training_authorized": False,
                },
                "guarantees": [
                    "A success requires one shared raw shadow ID inside the same target object's paired episodes.",
                    "Both folds must independently pass the metric-anchor, distance, shape and exact prompt gates.",
                    "Train and heldout physical timestamps are content-verified and disjoint.",
                    "Shared IDs on distant distractors are rejected and never counted as anchor successes.",
                    "Plausible near-anchor identities with different raw IDs remain unresolved.",
                ],
                "runtime": {
                    "implementation_files": [
                        _provenance(Path(__file__)),
                        _provenance(SRC / "farm_runtime" / "anchor_cross_fold_reconciliation.py"),
                    ],
                    "python": sys.version,
                    "platform": platform.platform(),
                    "gpu_used": False,
                    "measurement_scope": "compute through pre-publication assembly",
                    "wall_seconds": time.perf_counter() - started,
                    "current_cpu_rss_mib": _current_rss_mib(),
                    "peak_cpu_rss_mib": rss.peak_rss_mib,
                },
            }
            reconciliation_path = staging / "reconciliation.json"
            reconciliation_path.write_text(
                json.dumps(reconciliation, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            reconciliation_provenance = _provenance(reconciliation_path)
            reconciliation_provenance["path"] = str(
                output_root / "reconciliation.json"
            )
            allowlist = _build_allowlist(
                accepted_rows=accepted,
                all_keys_by_episode=all_keys_by_episode,
                episodes_by_id=episodes_by_id,
                artifact_provenance=artifact_provenance,
                reconciliation_provenance=reconciliation_provenance,
                source_gaussians=args.source_gaussians,
                sparse_points_ply=args.sparse_points_ply,
                plan=plan,
            )
            (staging / "gaussian_grouping_allowlist.json").write_text(
                json.dumps(allowlist, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    print(json.dumps({"output": str(output_root), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
