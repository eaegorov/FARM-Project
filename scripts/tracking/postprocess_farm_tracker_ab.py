#!/usr/bin/env python3
"""Build CPU-only 3D signatures and global tracker associations.

The command consumes one or both already-complete SAM1+DEVA and SAM3 suite
outputs.  It never launches a tracker or uses CUDA.  Every supplied backend
must cover the exact same materialized train/heldout episode set.  Publication
is atomic: partial signatures, associations, and logs disappear if any
contract fails.
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
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from farm_runtime.tracker_3d_signature import (  # noqa: E402
    SCHEMA as SIGNATURE_SCHEMA,
    SHADOW_SCHEMA as SIGNATURE_SHADOW_SCHEMA,
)
from farm_runtime.tracker_episode_suite import (  # noqa: E402
    EpisodeInput,
    PeakRssSampler,
    SuiteInput,
    atomic_output_directory,
    audit_8bit_masks,
    load_materialized_suite,
    sha256_file,
)
from farm_runtime.tracker_global_association import (  # noqa: E402
    LOCAL_GATE_V1,
    LOCAL_GATE_V2_SHADOW,
    MAX_GLOBAL_ID,
    SCHEMA as ASSOCIATION_SCHEMA,
    SHADOW_ABLATION_SCHEMA as ASSOCIATION_SHADOW_SCHEMA,
)

SCHEMA = "farm.tracker-ab-3d-postprocess.v2-shadow"
SUITE_SCHEMA = "farm.tracker-episode-suite.v1"
BACKENDS = {
    "deva": {
        "suite_backend": "sam1_deva_automatic",
        "episode_schema": "farm.deva-automatic-episode.v1",
    },
    "sam3": {
        "suite_backend": "sam3_concept_video",
        "episode_schema": "farm.sam3-concept-episode.v1",
    },
}


@dataclass(frozen=True)
class SuiteEpisodeOutput:
    episode: EpisodeInput
    root: Path
    mask_root: Path
    measurement_path: Path
    measurement_sha256: str
    local_identity_count: int


@dataclass(frozen=True)
class ValidatedBackendSuite:
    key: str
    backend: str
    root: Path
    suite_measurement_path: Path
    suite_measurement_sha256: str
    episodes: tuple[SuiteEpisodeOutput, ...]


@dataclass(frozen=True)
class SignatureJob:
    backend_key: str
    episode: SuiteEpisodeOutput
    staged_report: Path
    final_report: Path
    log_path: Path


def _load_json(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def _file_provenance(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve(strict=True)
    return {
        "path": str(source),
        "bytes": int(source.stat().st_size),
        "sha256": sha256_file(source),
    }


def _require_exact_path(value: object, expected: Path, *, label: str) -> None:
    actual = Path(str(value or "")).expanduser().resolve(strict=True)
    if actual != expected.resolve(strict=True):
        raise ValueError(f"{label} path mismatch: {actual} != {expected}")


def validate_backend_suite(
    root: Path,
    *,
    key: str,
    materialized: SuiteInput,
) -> ValidatedBackendSuite:
    """Validate one immutable tracker suite against the materialized episodes."""

    if key not in BACKENDS:
        raise ValueError(f"unknown backend key: {key}")
    source = Path(root).expanduser().resolve(strict=True)
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"tracker suite root must be a real directory: {source}")
    suite_path = source / "suite_measurement.json"
    suite = _load_json(suite_path)
    expected_backend = str(BACKENDS[key]["suite_backend"])
    if suite.get("schema") != SUITE_SCHEMA or suite.get("status") != "pass":
        raise ValueError(f"tracker suite is not a passing {SUITE_SCHEMA}: {source}")
    if suite.get("backend") != expected_backend:
        raise ValueError(
            f"tracker suite backend mismatch: {suite.get('backend')!r} != {expected_backend!r}"
        )
    input_row = suite.get("input")
    output_row = suite.get("output")
    if not isinstance(input_row, Mapping) or not isinstance(output_row, Mapping):
        raise ValueError(f"tracker suite lacks input/output contracts: {source}")
    _require_exact_path(
        input_row.get("materialized_manifest"),
        materialized.manifest_path,
        label="materialized manifest",
    )
    if input_row.get("materialized_manifest_sha256") != materialized.manifest_sha256:
        raise ValueError("tracker suite materialized manifest checksum mismatch")
    if input_row.get("source_plan_sha256") != materialized.source_plan_sha256:
        raise ValueError("tracker suite source-plan checksum mismatch")
    if input_row.get("exact_frame_hashes_verified") is not True:
        raise ValueError("tracker suite did not verify exact frame hashes")
    if int(input_row.get("episode_count", -1)) != len(materialized.episodes):
        raise ValueError("tracker suite input episode count mismatch")
    _require_exact_path(output_row.get("directory"), source, label="suite output")
    if int(output_row.get("episode_count", -1)) != len(materialized.episodes):
        raise ValueError("tracker suite output episode count mismatch")
    report_rows = output_row.get("episode_reports")
    if not isinstance(report_rows, list) or len(report_rows) != len(
        materialized.episodes
    ):
        raise ValueError("tracker suite episode report list mismatch")
    report_by_id: dict[str, Mapping[str, Any]] = {}
    for row in report_rows:
        if not isinstance(row, Mapping):
            raise ValueError("suite episode report row must be an object")
        episode_id = str(row.get("episode_id") or "")
        if episode_id in report_by_id:
            raise ValueError(f"duplicate suite episode report: {episode_id}")
        report_by_id[episode_id] = row
    expected_ids = {episode.episode_id for episode in materialized.episodes}
    if set(report_by_id) != expected_ids:
        raise ValueError("tracker suite episode set differs from materialized suite")
    episode_root = source / "episodes"
    if episode_root.is_symlink() or not episode_root.is_dir():
        raise ValueError("tracker suite episodes root must be a real directory")
    if {path.name for path in episode_root.iterdir()} != expected_ids:
        raise ValueError("tracker suite episodes directory contains unexpected entries")

    expected_episode_schema = str(BACKENDS[key]["episode_schema"])
    validated: list[SuiteEpisodeOutput] = []
    for episode in sorted(materialized.episodes, key=lambda value: value.episode_id):
        row = report_by_id[episode.episode_id]
        root_path = episode_root / episode.episode_id
        measurement_path = root_path / "measurement.json"
        _require_exact_path(
            row.get("path"), measurement_path, label="episode measurement"
        )
        actual_hash = sha256_file(measurement_path)
        if str(row.get("sha256") or "") != actual_hash:
            raise ValueError(
                f"episode measurement checksum mismatch: {episode.episode_id}"
            )
        measurement = _load_json(measurement_path)
        if (
            measurement.get("schema") != expected_episode_schema
            or measurement.get("status") != "pass"
        ):
            raise ValueError(
                f"episode measurement is not passing: {episode.episode_id}"
            )
        if measurement.get("input", {}).get("episode_id") != episode.episode_id:
            raise ValueError(
                f"episode measurement identity mismatch: {episode.episode_id}"
            )
        if (
            int(measurement.get("output", {}).get("mask_count", -1))
            != episode.frame_count
        ):
            raise ValueError(f"episode mask count mismatch: {episode.episode_id}")
        mask_root = root_path / "Annotations"
        audit = audit_8bit_masks(
            mask_root,
            episode.frame_names,
            reference_frame_root=episode.frame_root,
        )
        validated.append(
            SuiteEpisodeOutput(
                episode=episode,
                root=root_path,
                mask_root=mask_root,
                measurement_path=measurement_path,
                measurement_sha256=actual_hash,
                local_identity_count=int(audit["local_identity_count"]),
            )
        )
    return ValidatedBackendSuite(
        key=key,
        backend=expected_backend,
        root=source,
        suite_measurement_path=suite_path,
        suite_measurement_sha256=sha256_file(suite_path),
        episodes=tuple(validated),
    )


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    return environment


def _run_command(command: Sequence[str], log_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        env=_subprocess_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        tail = "\n".join(completed.stdout.splitlines()[-30:])
        raise RuntimeError(
            f"CPU postprocess command failed ({completed.returncode}): "
            f"{' '.join(command[:3])}\n{tail}"
        )
    return {
        "wall_seconds": elapsed,
        "returncode": completed.returncode,
        "log_bytes": int(log_path.stat().st_size),
    }


def _validate_signature(
    path: Path, *, episode: EpisodeInput, expected_mask_root: Path
) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema") != SIGNATURE_SCHEMA or payload.get("status") != "pass":
        raise ValueError(f"invalid 3D signature output: {path}")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"3D signature has no inputs: {path}")
    if (
        inputs.get("episode_id") != episode.episode_id
        or inputs.get("split") != episode.split
    ):
        raise ValueError(f"3D signature episode/split mismatch: {episode.episode_id}")
    _require_exact_path(
        inputs.get("mask_directory"), expected_mask_root, label="mask directory"
    )
    rows = payload.get("local_identities")
    summary = payload.get("summary")
    if not isinstance(rows, list) or not isinstance(summary, Mapping):
        raise ValueError(f"3D signature lacks identities/summary: {path}")
    if int(summary.get("local_identity_count", -1)) != len(rows):
        raise ValueError(f"3D signature identity count mismatch: {path}")
    shadow = payload.get("v2_shadow")
    if (
        not isinstance(shadow, Mapping)
        or shadow.get("schema") != SIGNATURE_SHADOW_SCHEMA
        or shadow.get("mode") != "diagnostic-only-no-publish"
        or shadow.get("automatic_promotion_allowed") is not False
        or shadow.get("global_association_authorized") is not False
    ):
        raise ValueError(f"3D signature lacks fail-closed v2 shadow: {path}")
    shadow_summary = shadow.get("summary")
    if not isinstance(shadow_summary, Mapping):
        raise ValueError(f"3D signature lacks v2 shadow summary: {path}")
    shadow_passed = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"3D signature identity is not an object: {path}")
        decision = row.get("decision_v2_shadow")
        evidence = row.get("shadow_v2_evidence")
        if (
            not isinstance(decision, Mapping)
            or decision.get("schema") != SIGNATURE_SHADOW_SCHEMA
            or decision.get("global_association_authorized") is not False
            or not isinstance(evidence, Mapping)
            or evidence.get("schema") != SIGNATURE_SHADOW_SCHEMA
        ):
            raise ValueError(f"invalid identity v2 shadow contract: {path}")
        shadow_passed += decision.get("passed") is True
    if int(shadow_summary.get("shadow_would_pass_count", -1)) != shadow_passed:
        raise ValueError(f"3D signature shadow count mismatch: {path}")
    return payload


def _signature_worker(job: SignatureJob, colmap_model: Path) -> dict[str, Any]:
    job.staged_report.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts/tracking/evaluate_farm_tracker_episode_3d.py"),
        "--episode-json",
        str(job.episode.episode.episode_json),
        "--mask-dir",
        str(job.episode.mask_root),
        "--colmap-model",
        str(colmap_model),
        "--output",
        str(job.staged_report),
    ]
    execution = _run_command(command, job.log_path)
    payload = _validate_signature(
        job.staged_report,
        episode=job.episode.episode,
        expected_mask_root=job.episode.mask_root,
    )
    return {
        "backend": job.backend_key,
        "episode_id": job.episode.episode.episode_id,
        "split": job.episode.episode.split,
        "path": str(job.final_report),
        "sha256": sha256_file(job.staged_report),
        "local_identity_count": int(payload["summary"]["local_identity_count"]),
        "accepted_identity_count": int(
            payload["summary"]["accepted_for_cross_episode_association_count"]
        ),
        "shadow_accepted_identity_count": int(
            payload["v2_shadow"]["summary"]["shadow_would_pass_count"]
        ),
        "shadow_recovered_identity_count": int(
            payload["v2_shadow"]["summary"]["shadow_recovered_from_v1_rejection_count"]
        ),
        "shadow_regressed_identity_count": int(
            payload["v2_shadow"]["summary"]["shadow_regressed_from_v1_acceptance_count"]
        ),
        "internal_measurement": payload.get("measurement"),
        "subprocess": execution,
    }


def _rebase_paths(value: Any, *, staging: Path, destination: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _rebase_paths(item, staging=staging, destination=destination)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rebase_paths(item, staging=staging, destination=destination)
            for item in value
        ]
    if isinstance(value, str):
        prefix = str(staging) + os.sep
        if value.startswith(prefix):
            return str(destination) + os.sep + value[len(prefix) :]
    return value


def _validate_association(
    payload: Mapping[str, Any],
    signature_paths: Sequence[Path],
    *,
    shadow: bool = False,
) -> None:
    expected_schema = ASSOCIATION_SHADOW_SCHEMA if shadow else ASSOCIATION_SCHEMA
    if payload.get("schema") != expected_schema:
        raise ValueError("global association schema mismatch")
    if shadow:
        if (
            payload.get("mode") != "diagnostic-only-no-publish"
            or payload.get("automatic_promotion_allowed") is not False
            or payload.get("gg_training_authorized") is not False
            or any(
                key in payload
                for key in ("local_to_global", "assignments", "global_objects")
            )
        ):
            raise ValueError("shadow association exposes a publishable contract")
    elif payload.get("mode") != "publish-contract":
        raise ValueError("v1 association lacks publish contract marker")
    if payload.get("fit_splits") != ["train"]:
        raise ValueError("global association must fit train only")
    expected_keys: set[str] = set()
    for path in signature_paths:
        report = _load_json(path)
        expected_keys.update(
            str(row["identity_key"]) for row in report["local_identities"]
        )
    mapping = payload.get("candidate_local_to_global" if shadow else "local_to_global")
    assignments = payload.get("candidate_assignments" if shadow else "assignments")
    if not isinstance(mapping, Mapping) or set(mapping) != expected_keys:
        raise ValueError("global association does not cover every local identity")
    if not isinstance(assignments, list) or len(assignments) != len(expected_keys):
        raise ValueError("global association assignment coverage mismatch")
    values = list(mapping.values())
    if any(
        not isinstance(value, int) or not 0 <= value <= MAX_GLOBAL_ID
        for value in values
    ):
        raise ValueError("global association ID is outside 0..254")
    nonzero = sorted({int(value) for value in values if value})
    if nonzero != list(range(1, len(nonzero) + 1)):
        raise ValueError("global association IDs are not compact")
    by_key = {str(row.get("identity_key") or ""): row for row in assignments}
    if set(by_key) != expected_keys:
        raise ValueError(
            "global association assignments contain duplicate/missing keys"
        )
    for key, global_id in mapping.items():
        if int(by_key[key].get("global_id", -1)) != int(global_id):
            raise ValueError(f"association map/assignment mismatch: {key}")


def _association_command(
    *,
    signatures: Sequence[Path],
    measurements: Sequence[Path],
    output: Path,
    synonym_groups: Path | None,
    local_gate_source: str = LOCAL_GATE_V1,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/tracking/associate_farm_tracker_identities.py"),
    ]
    for path in signatures:
        command.extend(["--signature-report", str(path)])
    for path in measurements:
        command.extend(["--tracker-measurement", str(path)])
    if synonym_groups is not None:
        command.extend(["--synonym-groups", str(synonym_groups)])
    command.extend(["--local-gate-source", local_gate_source])
    command.extend(["--output", str(output)])
    return command


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", required=True, type=Path)
    parser.add_argument("--deva-suite", type=Path)
    parser.add_argument("--sam3-suite", type=Path)
    parser.add_argument("--colmap-model", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--synonym-groups", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    requested_suites = {
        key: path
        for key, path in (("deva", args.deva_suite), ("sam3", args.sam3_suite))
        if path is not None
    }
    if not requested_suites:
        raise ValueError("at least one of --deva-suite or --sam3-suite is required")
    if int(args.expected_episodes) <= 0:
        raise ValueError("--expected-episodes must be positive")
    if not 1 <= int(args.jobs) <= 8:
        raise ValueError("--jobs must be in [1,8]")
    started = time.perf_counter()
    validation_started = time.perf_counter()
    materialized = load_materialized_suite(args.episodes_root, verify_frame_hashes=True)
    if len(materialized.episodes) != int(args.expected_episodes):
        raise ValueError(
            f"expected {args.expected_episodes} materialized episodes, "
            f"found {len(materialized.episodes)}"
        )
    splits = {episode.split for episode in materialized.episodes}
    if splits != {"train", "heldout"}:
        raise ValueError("postprocess requires both train and heldout episodes")
    colmap_model = args.colmap_model.expanduser().resolve(strict=True)
    if colmap_model.is_symlink() or not colmap_model.is_dir():
        raise ValueError("--colmap-model must be a real directory")
    synonym_groups = (
        args.synonym_groups.expanduser().resolve(strict=True)
        if args.synonym_groups is not None
        else None
    )
    suites = {
        key: validate_backend_suite(path, key=key, materialized=materialized)
        for key, path in requested_suites.items()
    }
    backend_count = len(suites)
    backend_mode = "ab" if backend_count == 2 else "single"
    validation_seconds = time.perf_counter() - validation_started
    destination = args.output_root.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite postprocess output: {destination}"
        )

    with PeakRssSampler(interval_seconds=0.02) as rss_sampler:
        with atomic_output_directory(destination) as staging:
            signature_jobs: list[SignatureJob] = []
            for key in sorted(suites):
                for episode in suites[key].episodes:
                    relative = (
                        Path("backends")
                        / suites[key].backend
                        / "signatures"
                        / f"{episode.episode.episode_id}.json"
                    )
                    signature_jobs.append(
                        SignatureJob(
                            backend_key=key,
                            episode=episode,
                            staged_report=staging / relative,
                            final_report=destination / relative,
                            log_path=(
                                staging
                                / "backends"
                                / suites[key].backend
                                / "logs"
                                / "signatures"
                                / f"{episode.episode.episode_id}.log"
                            ),
                        )
                    )
            signature_started = time.perf_counter()
            signature_rows: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=int(args.jobs)) as executor:
                future_by_job = {
                    executor.submit(_signature_worker, job, colmap_model): job
                    for job in signature_jobs
                }
                try:
                    for future in as_completed(future_by_job):
                        signature_rows.append(future.result())
                except BaseException:
                    for future in future_by_job:
                        future.cancel()
                    raise
            signature_rows.sort(key=lambda row: (row["backend"], row["episode_id"]))
            signature_seconds = time.perf_counter() - signature_started
            expected_signature_count = backend_count * int(args.expected_episodes)
            if len(signature_rows) != expected_signature_count:
                raise RuntimeError("signature output count is incomplete")

            association_started = time.perf_counter()
            association_rows: list[dict[str, Any]] = []
            shadow_association_rows: list[dict[str, Any]] = []
            for key in sorted(suites):
                jobs = sorted(
                    (job for job in signature_jobs if job.backend_key == key),
                    key=lambda value: value.episode.episode.episode_id,
                )
                staged_signatures = [job.staged_report for job in jobs]
                measurements = [job.episode.measurement_path for job in jobs]

                relative = (
                    Path("backends")
                    / suites[key].backend
                    / "global_association_v1.json"
                )
                staged_association = staging / relative
                command = _association_command(
                    signatures=staged_signatures,
                    measurements=measurements,
                    output=staged_association,
                    synonym_groups=synonym_groups,
                    local_gate_source=LOCAL_GATE_V1,
                )
                execution = _run_command(
                    command,
                    staging
                    / "backends"
                    / suites[key].backend
                    / "logs"
                    / "global_association_v1.log",
                )
                association = _load_json(staged_association)
                _validate_association(association, staged_signatures, shadow=False)
                association = _rebase_paths(
                    association, staging=staging, destination=destination
                )
                _write_json(staged_association, association)
                association_rows.append(
                    {
                        "backend": key,
                        "path": str(destination / relative),
                        "sha256": sha256_file(staged_association),
                        "summary": association.get("summary"),
                        "internal_measurement": association.get("measurement"),
                        "subprocess": execution,
                    }
                )

                shadow_relative = (
                    Path("backends")
                    / suites[key].backend
                    / "global_association_v2_shadow_ablation.json"
                )
                staged_shadow = staging / shadow_relative
                shadow_command = _association_command(
                    signatures=staged_signatures,
                    measurements=measurements,
                    output=staged_shadow,
                    synonym_groups=synonym_groups,
                    local_gate_source=LOCAL_GATE_V2_SHADOW,
                )
                shadow_execution = _run_command(
                    shadow_command,
                    staging
                    / "backends"
                    / suites[key].backend
                    / "logs"
                    / "global_association_v2_shadow_ablation.log",
                )
                shadow_association = _load_json(staged_shadow)
                _validate_association(
                    shadow_association, staged_signatures, shadow=True
                )
                shadow_association = _rebase_paths(
                    shadow_association, staging=staging, destination=destination
                )
                _write_json(staged_shadow, shadow_association)
                shadow_association_rows.append(
                    {
                        "backend": key,
                        "path": str(destination / shadow_relative),
                        "sha256": sha256_file(staged_shadow),
                        "summary": shadow_association.get("summary"),
                        "internal_measurement": shadow_association.get("measurement"),
                        "subprocess": shadow_execution,
                        "automatic_promotion_allowed": False,
                        "gg_training_authorized": False,
                    }
                )
            association_seconds = time.perf_counter() - association_started

            ablation_by_backend: list[dict[str, Any]] = []
            for key in sorted(suites):
                signature_subset = [
                    row for row in signature_rows if row["backend"] == key
                ]
                by_split = {}
                for split in ("train", "heldout"):
                    selected = [
                        row for row in signature_subset if row["split"] == split
                    ]
                    by_split[split] = {
                        "episode_count": len(selected),
                        "local_identity_count": sum(
                            int(row["local_identity_count"]) for row in selected
                        ),
                        "v1_publish_accepted_count": sum(
                            int(row["accepted_identity_count"]) for row in selected
                        ),
                        "v2_shadow_would_pass_count": sum(
                            int(row["shadow_accepted_identity_count"])
                            for row in selected
                        ),
                        "v2_shadow_recovered_count": sum(
                            int(row["shadow_recovered_identity_count"])
                            for row in selected
                        ),
                        "v2_shadow_regressed_count": sum(
                            int(row["shadow_regressed_identity_count"])
                            for row in selected
                        ),
                    }
                v1_summary = next(
                    row["summary"] for row in association_rows if row["backend"] == key
                )
                shadow_summary = next(
                    row["summary"]
                    for row in shadow_association_rows
                    if row["backend"] == key
                )
                ablation_by_backend.append(
                    {
                        "backend": key,
                        "tracker_backend": suites[key].backend,
                        "identity_counts_by_split": by_split,
                        "v1_association": v1_summary,
                        "v2_shadow_association": shadow_summary,
                        "potential_pairs": {
                            "train_fit_pairs_evaluated": shadow_summary[
                                "train_pair_count_evaluated"
                            ],
                            "train_fit_eligible_pairs": shadow_summary[
                                "train_eligible_pair_count"
                            ],
                            "heldout_application_pairs_evaluated": shadow_summary[
                                "heldout_pair_count_evaluated"
                            ],
                            "heldout_application_eligible_pairs": shadow_summary[
                                "heldout_eligible_pair_count"
                            ],
                        },
                        "heldout_updates_fit": False,
                        "automatic_promotion_allowed": False,
                        "gg_training_authorized": False,
                    }
                )

            rss = rss_sampler.report()
            report = {
                "schema": SCHEMA,
                "status": "pass",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "execution": {
                    "backend_mode": backend_mode,
                    "backend_count": backend_count,
                    "backend_keys": sorted(suites),
                    "cpu_only": True,
                    "gpu_used": False,
                    "cuda_visible_devices": "",
                    "network_required": False,
                    "publication": "atomic whole-postprocess rename",
                    "signature_workers": int(args.jobs),
                    "signature_count": len(signature_rows),
                    "association_count": len(association_rows),
                    "shadow_association_ablation_count": len(shadow_association_rows),
                    "association_subprocess_count": (
                        len(association_rows) + len(shadow_association_rows)
                    ),
                    "fit_splits": ["train"],
                    "heldout_mode": "application-only; never updates train clusters",
                    "shadow_mode": "diagnostic-only-no-publish",
                    "automatic_shadow_promotion_allowed": False,
                    "shadow_gg_training_authorized": False,
                },
                "input": {
                    "materialized_suite": {
                        "root": str(materialized.root),
                        "manifest": _file_provenance(materialized.manifest_path),
                        "source_plan": _file_provenance(materialized.source_plan),
                        "episode_count": len(materialized.episodes),
                        "split_counts": {
                            split: sum(
                                episode.split == split
                                for episode in materialized.episodes
                            )
                            for split in sorted(splits)
                        },
                    },
                    "tracker_suites": {
                        key: {
                            "backend": suite.backend,
                            "root": str(suite.root),
                            "suite_measurement": {
                                "path": str(suite.suite_measurement_path),
                                "sha256": suite.suite_measurement_sha256,
                            },
                        }
                        for key, suite in suites.items()
                    },
                    "colmap_model": str(colmap_model),
                    "synonym_groups": (
                        _file_provenance(synonym_groups)
                        if synonym_groups is not None
                        else None
                    ),
                },
                "outputs": {
                    "root": str(destination),
                    "signatures": signature_rows,
                    "associations": association_rows,
                    "shadow_association_ablations": shadow_association_rows,
                },
                "v2_shadow_ablation": {
                    "signature_schema": SIGNATURE_SHADOW_SCHEMA,
                    "association_schema": ASSOCIATION_SHADOW_SCHEMA,
                    "mode": "diagnostic-only-no-publish",
                    "authoritative_v1_outputs_unchanged": True,
                    "automatic_promotion_allowed": False,
                    "gg_training_authorized": False,
                    "fit_splits": ["train"],
                    "heldout_mode": "application-only; never updates train clusters",
                    "by_backend": ablation_by_backend,
                },
                "measurement": {
                    "input_validation_seconds": validation_seconds,
                    "signature_subprocess_wall_seconds": signature_seconds,
                    "association_subprocess_wall_seconds": association_seconds,
                    "wall_seconds_before_atomic_publish": time.perf_counter() - started,
                    "orchestrator_cpu_rss": rss,
                },
                "runtime": {
                    "python": platform.python_version(),
                    "implementation_files": [
                        _file_provenance(Path(__file__).resolve()),
                        _file_provenance(
                            ROOT / "scripts/tracking/evaluate_farm_tracker_episode_3d.py"
                        ),
                        _file_provenance(
                            ROOT / "scripts/tracking/associate_farm_tracker_identities.py"
                        ),
                        _file_provenance(
                            ROOT
                            / "src"
                            / "farm_runtime"
                            / "tracker_global_association.py"
                        ),
                        _file_provenance(
                            ROOT / "src" / "farm_runtime" / "tracker_3d_signature.py"
                        ),
                    ],
                },
                "guarantees": [
                    (
                        "Both tracker backends cover the exact same materialized episode set."
                        if backend_mode == "ab"
                        else "The supplied tracker backend covers the exact materialized episode set."
                    ),
                    "All 3D signatures are independent episode-local measurements.",
                    "Association fits train only and applies frozen clusters to heldout.",
                    "Every nonzero local mask ID maps to global 0..254 exactly once.",
                    "CUDA is hidden from all subprocesses and no tracker is launched.",
                    "No partial output survives a failed stage.",
                    "The authoritative v1 decision and builder-compatible mappings remain separate from v2 shadow ablations.",
                    "Shadow candidate mappings cannot authorize automatic promotion or GG training.",
                    "Heldout identities never fit either v1 or shadow train clusters.",
                ],
            }
            _write_json(staging / "postprocess_measurement.json", report)
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(destination),
                "backend_mode": backend_mode,
                "backend_keys": sorted(suites),
                "signature_count": backend_count * int(args.expected_episodes),
                "association_count": backend_count,
                "shadow_association_ablation_count": backend_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
