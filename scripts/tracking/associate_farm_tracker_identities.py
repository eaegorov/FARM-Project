#!/usr/bin/env python3
"""Associate episode-local tracker IDs into train-fit global FARM IDs.

The stage is CPU-only and fail-closed. It verifies tracker 3D reports,
reconstructs metric voxel support from their exact COLMAP provenance, derives
optional masked-color appearance evidence, and never uses an OBB as a gate.
Heldout identities are application-only queries against frozen train clusters.
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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from farm_runtime.tracker_global_association import (  # noqa: E402
    LOCAL_GATE_V1,
    LOCAL_GATE_V2_SHADOW,
    MAX_GLOBAL_ID,
    SCHEMA,
    SHADOW_ABLATION_SCHEMA,
    AssociationPolicy,
    associate_identities,
    derive_color_appearance_from_reports,
    load_identity_evidence_files,
    load_signature_reports,
    load_synonym_groups,
    load_tracker_measurements,
    merge_identity_evidence,
    policy_dict,
    sha256_file,
    signature_identity_keys,
)
from farm_runtime.tracker_episode_suite import PeakRssSampler  # noqa: E402


def _atomic_json(path: Path, payload: Any, *, force: bool) -> None:
    destination = path.expanduser().resolve()
    if destination.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing association without --force: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _policy(args: argparse.Namespace) -> AssociationPolicy:
    return AssociationPolicy(
        voxel_size_m=args.voxel_size_m,
        voxel_neighbor_radius=args.voxel_neighbor_radius,
        minimum_shared_sparse_tracks=args.minimum_shared_sparse_tracks,
        minimum_sparse_track_containment=args.minimum_sparse_track_containment,
        minimum_sparse_track_jaccard=args.minimum_sparse_track_jaccard,
        minimum_shared_metric_voxels=args.minimum_shared_metric_voxels,
        minimum_metric_voxel_containment=args.minimum_metric_voxel_containment,
        minimum_metric_voxel_jaccard=args.minimum_metric_voxel_jaccard,
        maximum_centroid_distance_m=args.maximum_centroid_distance_m,
        maximum_centroid_distance_scale=args.maximum_centroid_distance_scale,
        maximum_scale_ratio=args.maximum_scale_ratio,
        minimum_appearance_similarity=args.minimum_appearance_similarity,
        minimum_pair_score=args.minimum_pair_score,
        minimum_assignment_margin=args.minimum_assignment_margin,
        maximum_global_ids=args.maximum_global_ids,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signature-report",
        action="append",
        required=True,
        type=Path,
        help="Repeat for every farm.tracker-episode-3d-signatures.v1 report.",
    )
    parser.add_argument(
        "--tracker-measurement",
        action="append",
        default=[],
        type=Path,
        help="Optional SAM3/DEVA measurement; SAM3 contributes proposal prompts.",
    )
    parser.add_argument(
        "--identity-evidence",
        action="append",
        default=[],
        type=Path,
        help="Optional farm.tracker-identity-evidence.v1 prompt/embedding file.",
    )
    parser.add_argument("--synonym-groups", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--local-gate-source",
        choices=(LOCAL_GATE_V1, LOCAL_GATE_V2_SHADOW),
        default=LOCAL_GATE_V1,
        help=(
            "v1_publish emits the builder contract; v2_shadow emits a "
            "diagnostic-only ablation that cannot authorize publication or training."
        ),
    )
    parser.add_argument(
        "--derive-color-appearance", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--verify-artifact-hashes", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--verify-colmap-hashes", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--voxel-size-m", type=float, default=0.10)
    parser.add_argument("--voxel-neighbor-radius", type=int, default=0)
    parser.add_argument("--minimum-shared-sparse-tracks", type=int, default=3)
    parser.add_argument("--minimum-sparse-track-containment", type=float, default=0.30)
    parser.add_argument("--minimum-sparse-track-jaccard", type=float, default=0.10)
    parser.add_argument("--minimum-shared-metric-voxels", type=int, default=2)
    parser.add_argument("--minimum-metric-voxel-containment", type=float, default=0.30)
    parser.add_argument("--minimum-metric-voxel-jaccard", type=float, default=0.10)
    parser.add_argument("--maximum-centroid-distance-m", type=float, default=0.75)
    parser.add_argument("--maximum-centroid-distance-scale", type=float, default=1.50)
    parser.add_argument("--maximum-scale-ratio", type=float, default=3.0)
    parser.add_argument("--minimum-appearance-similarity", type=float, default=0.78)
    parser.add_argument("--minimum-pair-score", type=float, default=0.58)
    parser.add_argument("--minimum-assignment-margin", type=float, default=0.08)
    parser.add_argument("--maximum-global-ids", type=int, default=MAX_GLOBAL_ID)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    policy = _policy(args)
    policy.validate()

    with PeakRssSampler(interval_seconds=0.02) as rss_sampler:
        custom, custom_provenance = load_identity_evidence_files(args.identity_evidence)
        report_identity_keys = signature_identity_keys(args.signature_report)
        measurements, measurement_provenance = load_tracker_measurements(
            args.tracker_measurement, allowed_identity_keys=report_identity_keys
        )
        appearance: dict[str, dict[str, Any]] = {}
        appearance_provenance: list[dict[str, Any]] = []
        if args.derive_color_appearance:
            appearance, appearance_provenance = derive_color_appearance_from_reports(
                args.signature_report,
                verify_artifact_hashes=args.verify_artifact_hashes,
                local_gate_source=args.local_gate_source,
            )
            # Explicit embeddings are authoritative only when deliberately supplied.
            # Avoid a silent conflict with the deterministic fallback histogram.
            for key, value in custom.items():
                if value.get("appearance_embedding") is not None and key in appearance:
                    appearance[key] = {
                        "prompts": [],
                        "appearance_embedding": value["appearance_embedding"],
                    }
        combined = merge_identity_evidence(appearance, measurements, custom)
        synonyms, synonym_provenance = load_synonym_groups(args.synonym_groups)
        identities, report_provenance = load_signature_reports(
            args.signature_report,
            policy=policy,
            identity_evidence=combined,
            verify_colmap_hashes=args.verify_colmap_hashes,
            local_gate_source=args.local_gate_source,
        )
        result = associate_identities(identities, policy=policy, synonyms=synonyms)
    rss_measurement = rss_sampler.report()
    implementation = Path(__file__).resolve()
    module = ROOT / "src" / "farm_runtime" / "tracker_global_association.py"
    shadow_mode = args.local_gate_source == LOCAL_GATE_V2_SHADOW
    output_schema = SHADOW_ABLATION_SCHEMA if shadow_mode else SCHEMA
    payload = {
        "schema": output_schema,
        "mode": ("diagnostic-only-no-publish" if shadow_mode else "publish-contract"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit_splits": result["fit_splits"],
        "inputs": {
            "signature_reports": report_provenance,
            "tracker_measurements": measurement_provenance,
            "identity_evidence": custom_provenance,
            "derived_appearance": appearance_provenance,
            "synonym_groups": synonym_provenance,
            "colmap_hashes_verified": bool(args.verify_colmap_hashes),
            "artifact_hashes_verified": bool(
                args.derive_color_appearance and args.verify_artifact_hashes
            ),
            "artifact_verification_requested": bool(args.verify_artifact_hashes),
            "local_gate_source": args.local_gate_source,
        },
        "policy": {
            "name": (
                "farm-tracker-global-association-v2-shadow-ablation"
                if shadow_mode
                else "farm-tracker-global-association-fail-closed-v1"
            ),
            "thresholds": policy_dict(policy),
            "obb_used_as_gate": False,
            "heldout_updates_fit": False,
            "semantic_prompts_are_proposal_compatibility_only": True,
        },
        **(
            {
                "candidate_local_to_global": result["local_to_global"],
                "candidate_assignments": result["assignments"],
                "candidate_global_objects": result["global_objects"],
            }
            if shadow_mode
            else {
                "local_to_global": result["local_to_global"],
                "assignments": result["assignments"],
                "global_objects": result["global_objects"],
            }
        ),
        "pair_decisions": result["pair_decisions"],
        "ambiguities": result["ambiguities"],
        "summary": result["summary"],
        "measurement": {
            "cpu_only": True,
            "gpu_used": False,
            "wall_seconds": time.perf_counter() - started,
            "start_cpu_rss_mib": rss_measurement["start_rss_mib"],
            "peak_cpu_rss_mib": rss_measurement["peak_rss_mib"],
            "end_cpu_rss_mib": rss_measurement["end_rss_mib"],
            "rss_sampler_interval_seconds": 0.02,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "implementation_files": [
                {"path": str(implementation), "sha256": sha256_file(implementation)},
                {"path": str(module), "sha256": sha256_file(module)},
            ],
        },
        "automatic_promotion_allowed": not shadow_mode,
        "gg_training_authorized": not shadow_mode,
        "guarantees": [
            "Every episode-local identity in every input report is mapped exactly once.",
            "Global ID 0 means rejected, capacity-dropped, or unresolved/unknown.",
            "Only train identities fit clusters; heldout identities are application-only queries.",
            "Simultaneously visible distinct IDs in one episode are a hard non-merge constraint.",
            "No OBB is read or used as an association gate.",
            "Nonzero global IDs are deterministic, compact, and limited to 254.",
            *(
                [
                    "Shadow candidate IDs are ablation-only and must not be consumed by the GG builder.",
                    "The authoritative v1 decision and association remain unchanged.",
                ]
                if shadow_mode
                else []
            ),
        ],
    }
    payload["measurement"]["wall_seconds"] = time.perf_counter() - started
    _atomic_json(args.output, payload, force=args.force)
    print(
        json.dumps(
            {
                "schema": output_schema,
                "output": str(args.output.expanduser().resolve()),
                **result["summary"],
                "measurement": payload["measurement"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
