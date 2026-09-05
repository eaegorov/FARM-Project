from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from farm_runtime.tracker_global_association import (
    LOCAL_GATE_V2_SHADOW,
    SCHEMA,
    SHADOW_ABLATION_SCHEMA,
    AssociationPolicy,
    IdentityEvidence,
    associate_identities,
    compare_pair,
    derive_color_appearance_from_reports,
    load_signature_reports,
    load_tracker_measurements,
    sha256_file,
)


def _identity(
    key: str,
    *,
    split: str = "train",
    points: set[int] | None = None,
    voxels: set[tuple[int, int, int]] | None = None,
    centroid: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
    scale: float | None = 1.0,
    frames: set[int] | None = None,
    source_names: set[str] | None = None,
    prompts: set[str] | None = None,
    appearance: tuple[float, ...] | None = None,
    passed: bool = True,
) -> IdentityEvidence:
    episode, local = key.rsplit("::local:", 1)
    return IdentityEvidence(
        identity_key=key,
        episode_id=episode,
        local_id=int(local),
        split=split,
        passed_local_gate=passed,
        local_failed_checks=() if passed else ("synthetic_rejection",),
        core_point_ids=frozenset(points or set()),
        metric_voxels=frozenset(voxels or set()),
        centroid_m=centroid,
        scale_diagonal_m=scale,
        visible_frame_indices=frozenset(frames or set()),
        physical_timestamps=frozenset(str(value) for value in (frames or set())),
        visible_source_names=frozenset(source_names or set()),
        prompts=frozenset(prompts or set()),
        appearance_embedding=appearance,
        report_index=0,
    )


def _strong_pair(prefix: str = "object") -> tuple[IdentityEvidence, IdentityEvidence]:
    points = set(range(1, 11))
    voxels = {(value, 0, 0) for value in range(5)}
    return (
        _identity("episode-a::local:1", points=points, voxels=voxels, prompts={prefix}),
        _identity("episode-b::local:7", points=points, voxels=voxels, prompts={prefix}),
    )


def test_pair_requires_geometry_centroid_scale_and_semantic_or_appearance() -> None:
    left, right = _strong_pair("fire extinguisher")
    decision = compare_pair(left, right, policy=AssociationPolicy())
    assert decision["eligible"] is True
    assert decision["sparse_track_overlap"]["shared_count"] == 10
    assert decision["metric_voxel_overlap"]["shared_count"] == 5

    no_semantics = _identity(
        "episode-c::local:2",
        points=set(range(1, 11)),
        voxels={(value, 0, 0) for value in range(5)},
    )
    failed = compare_pair(left, no_semantics, policy=AssociationPolicy())
    assert failed["eligible"] is False
    assert "appearance_or_prompt_compatibility" in failed["failed_checks"]

    with_appearance = _identity(
        "episode-c::local:2",
        points=set(range(1, 11)),
        voxels={(value, 0, 0) for value in range(5)},
        appearance=(1.0, 0.0),
    )
    appearance_left = _identity(
        "episode-a::local:1",
        points=set(range(1, 11)),
        voxels={(value, 0, 0) for value in range(5)},
        appearance=(0.99, 0.01),
    )
    assert (
        compare_pair(appearance_left, with_appearance, policy=AssociationPolicy())[
            "eligible"
        ]
        is True
    )


def test_simultaneous_visibility_and_prompt_conflict_are_hard_vetoes() -> None:
    common = dict(
        points=set(range(1, 11)),
        voxels={(value, 0, 0) for value in range(5)},
        prompts={"cabinet"},
    )
    left = _identity("same-episode::local:1", frames={1, 2}, **common)
    right = _identity("same-episode::local:2", frames={2, 3}, **common)
    simultaneous = compare_pair(left, right, policy=AssociationPolicy())
    assert simultaneous["eligible"] is False
    assert simultaneous["hard_conflict"] is True
    assert simultaneous["simultaneous_visibility"]["shared_frame_indices"] == [2]

    reused_left = _identity("episode-x::local:1", source_names={"same.png"}, **common)
    reused_right = _identity("episode-y::local:2", source_names={"same.png"}, **common)
    reused = compare_pair(reused_left, reused_right, policy=AssociationPolicy())
    assert reused["eligible"] is False
    assert reused["simultaneous_visibility"]["shared_source_names"] == ["same.png"]

    other = _identity(
        "different-episode::local:2",
        points=set(range(1, 11)),
        voxels={(value, 0, 0) for value in range(5)},
        prompts={"monitor"},
        appearance=(1.0, 0.0),
    )
    left_with_appearance = _identity(
        "same-episode::local:1",
        points=set(range(1, 11)),
        voxels={(value, 0, 0) for value in range(5)},
        prompts={"cabinet"},
        appearance=(1.0, 0.0),
    )
    conflict = compare_pair(left_with_appearance, other, policy=AssociationPolicy())
    assert conflict["eligible"] is False
    assert "prompt_conflict" in conflict["failed_checks"]


def test_explicit_synonym_mapping_can_unlock_prompt_pair() -> None:
    left, _ = _strong_pair("vacuum cleaner")
    right = _identity(
        "episode-b::local:7",
        points=set(range(1, 11)),
        voxels={(value, 0, 0) for value in range(5)},
        prompts={"floor cleaner"},
    )
    blocked = compare_pair(left, right, policy=AssociationPolicy())
    assert blocked["eligible"] is False
    allowed = compare_pair(
        left,
        right,
        policy=AssociationPolicy(),
        synonyms={"vacuum cleaner": "cleaner", "floor cleaner": "cleaner"},
    )
    assert allowed["eligible"] is True


def test_weak_margin_becomes_unknown_and_complete_link_prevents_bridge_merge() -> None:
    points = set(range(20))
    voxels = {(value, 0, 0) for value in range(8)}
    query = _identity(
        "episode-a::local:1", points=points, voxels=voxels, prompts={"cabinet"}
    )
    first = _identity(
        "episode-b::local:1",
        points=points,
        voxels=voxels,
        prompts={"cabinet"},
        frames={0},
    )
    second = _identity(
        "episode-b::local:2",
        points=points,
        voxels=voxels,
        prompts={"cabinet"},
        frames={0},
    )
    result = associate_identities(
        [second, query, first], policy=AssociationPolicy(minimum_assignment_margin=0.08)
    )
    assert result["local_to_global"][query.identity_key] == 0
    assignment = next(
        row
        for row in result["assignments"]
        if row["identity_key"] == query.identity_key
    )
    assert assignment["reason"] == "train_unknown_weak_margin"
    assert (
        result["local_to_global"][first.identity_key]
        != result["local_to_global"][second.identity_key]
    )


def test_heldout_is_application_only_and_never_creates_global_object() -> None:
    train_a, train_b = _strong_pair("extinguisher")
    heldout = _identity(
        "episode-heldout::local:3",
        split="heldout",
        points=set(range(1, 11)),
        voxels={(value, 0, 0) for value in range(5)},
        prompts={"extinguisher"},
    )
    unmatched = _identity(
        "episode-heldout-2::local:4",
        split="heldout",
        points=set(range(100, 110)),
        voxels={(100 + value, 0, 0) for value in range(5)},
        centroid=(10.0, 0.0, 0.0),
        prompts={"extinguisher"},
    )
    result = associate_identities(
        [heldout, unmatched, train_b, train_a], policy=AssociationPolicy()
    )
    assert result["fit_splits"] == ["train"]
    assert len(result["global_objects"]) == 1
    assert result["local_to_global"][heldout.identity_key] == 1
    assert result["local_to_global"][unmatched.identity_key] == 0
    assert result["global_objects"][0]["train_member_count"] == 2
    assert result["global_objects"][0]["heldout_application_identity_keys"] == [
        heldout.identity_key
    ]


def test_deterministic_compact_capacity_and_full_zero_mapping() -> None:
    identities = [
        _identity(
            f"episode-{index}::local:1",
            points={index * 10 + 1, index * 10 + 2},
            voxels={(index * 10, 0, 0)},
            centroid=(float(index * 10), 0.0, 0.0),
            prompts={"object"},
        )
        for index in range(3)
    ]
    identities.append(_identity("rejected::local:1", passed=False))
    policy = AssociationPolicy(maximum_global_ids=2)
    forward = associate_identities(identities, policy=policy)
    reverse = associate_identities(list(reversed(identities)), policy=policy)
    assert forward["local_to_global"] == reverse["local_to_global"]
    assert sorted(
        {value for value in forward["local_to_global"].values() if value}
    ) == [1, 2]
    assert forward["local_to_global"]["rejected::local:1"] == 0
    assert sum(value == 0 for value in forward["local_to_global"].values()) == 2


def _signature_report(
    root: Path,
    *,
    episode: str,
    split: str,
    local_id: int,
    point_ids: list[int],
    episode_path: Path | None = None,
    mask_rows: list[dict] | None = None,
) -> dict:
    return {
        "schema": "farm.tracker-episode-3d-signatures.v1",
        "inputs": {
            "episode_id": episode,
            "split": split,
            "meters_per_scene_unit": 1.0,
            "colmap_format": "text",
            "colmap_model": str(root),
            "colmap_contract_files": [],
            "episode": (
                {
                    "path": str(episode_path),
                    "bytes": episode_path.stat().st_size,
                    "sha256": sha256_file(episode_path),
                }
                if episode_path is not None
                else {}
            ),
            "mask_files": mask_rows or [],
        },
        "summary": {
            "local_identity_count": 1,
            "accepted_for_cross_episode_association_count": 1,
            "accepted_identity_keys": [f"{episode}::local:{local_id}"],
        },
        "v2_shadow": {
            "schema": "farm.tracker-episode-3d-signatures.v2-shadow",
            "mode": "diagnostic-only-no-publish",
            "automatic_promotion_allowed": False,
            "global_association_authorized": False,
            "summary": {
                "shadow_would_pass_count": 1,
                "shadow_would_pass_identity_keys": [f"{episode}::local:{local_id}"],
            },
        },
        "local_identities": [
            {
                "identity_key": f"{episode}::local:{local_id}",
                "episode_id": episode,
                "local_id": local_id,
                "decision": {
                    "passed": True,
                    "status": "accepted_for_cross_episode_association",
                    "failed_checks": [],
                    "qualified_geometry": {
                        "available": True,
                        "point3d_ids": point_ids,
                        "connectivity": {
                            "largest_component_point3d_ids": point_ids,
                        },
                    },
                },
                "decision_v2_shadow": {
                    "schema": "farm.tracker-episode-3d-signatures.v2-shadow",
                    "mode": "diagnostic-only-no-publish",
                    "passed": True,
                    "status": "would_pass_shadow_gate",
                    "failed_checks": [],
                    "global_association_authorized": False,
                    "qualified_geometry": {
                        "available": True,
                        "point3d_ids": point_ids,
                        "connectivity": {
                            "largest_component_point3d_ids": point_ids,
                        },
                    },
                },
                "raw": {
                    "frame_observations": [
                        {"episode_index": 0, "physical_timestamp": "000001"}
                    ]
                },
            }
        ],
    }


def test_report_loader_reconstructs_true_metric_voxels(tmp_path: Path) -> None:
    (tmp_path / "points3D.txt").write_text(
        "1 0.01 0.01 0.01 255 0 0 0.1\n"
        "2 0.09 0.02 0.01 255 0 0 0.1\n"
        "3 0.11 0.02 0.01 255 0 0 0.1\n",
        encoding="utf-8",
    )
    report = _signature_report(
        tmp_path,
        episode="episode-a",
        split="train",
        local_id=1,
        point_ids=[1, 2, 3],
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    rows, provenance = load_signature_reports(
        [path],
        policy=AssociationPolicy(),
        identity_evidence={
            "episode-a::local:1": {
                "prompts": ["cabinet"],
                "appearance_embedding": None,
            }
        },
        verify_colmap_hashes=False,
    )
    assert rows[0].metric_voxels == frozenset({(0, 0, 0), (1, 0, 0)})
    assert rows[0].centroid_m == pytest.approx((0.09, 0.02, 0.01))
    assert provenance[0]["episode_id"] == "episode-a"


def test_color_appearance_is_masked_hash_verified_and_normalized(
    tmp_path: Path,
) -> None:
    from PIL import Image

    source = tmp_path / "frame.png"
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :2] = [255, 0, 0]
    rgb[:, 2:] = [0, 0, 255]
    Image.fromarray(rgb).save(source)
    mask = tmp_path / "mask.png"
    labels = np.zeros((4, 4), dtype=np.uint8)
    labels[:, :2] = 1
    Image.fromarray(labels).save(mask)
    episode_path = tmp_path / "episode.json"
    episode = {
        "frames": [
            {
                "source_path": str(source),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        ]
    }
    episode_path.write_text(json.dumps(episode), encoding="utf-8")
    report = _signature_report(
        tmp_path,
        episode="episode-a",
        split="train",
        local_id=1,
        point_ids=[],
        episode_path=episode_path,
        mask_rows=[
            {
                "path": str(mask),
                "bytes": mask.stat().st_size,
                "sha256": sha256_file(mask),
            }
        ],
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    evidence, provenance = derive_color_appearance_from_reports([report_path])
    vector = np.asarray(evidence["episode-a::local:1"]["appearance_embedding"])
    assert vector.shape == (256,)
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert np.count_nonzero(vector) == 1
    assert provenance[0]["artifact_hashes_verified"] is True


def test_suite_measurement_prefers_identity_table_and_skips_unrasterized_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "measurement.json"
    path.write_text(
        json.dumps(
            {
                "schema": "farm.sam3-concept-episode.v1",
                "status": "pass",
                "input": {"episode_id": "episode-a"},
                "output": {
                    "identities": [
                        {"local_id": 1, "prompt": "cabinet"},
                        {"local_id": None, "prompt": "server"},
                    ],
                    "frames": [{"objects": [{"local_id": 1, "prompt": "wrong"}]}],
                },
            }
        ),
        encoding="utf-8",
    )
    evidence, provenance = load_tracker_measurements(
        [path], allowed_identity_keys={"episode-a::local:1"}
    )
    assert evidence == {
        "episode-a::local:1": {
            "prompts": ["cabinet"],
            "appearance_embedding": None,
        }
    }
    assert provenance[0]["prompt_observation_count"] == 1


def test_cli_writes_builder_compatible_contract(tmp_path: Path) -> None:
    (tmp_path / "points3D.txt").write_text(
        "1 0 0 0 255 0 0 0.1\n2 0.1 0 0 255 0 0 0.1\n3 0.2 0 0 255 0 0 0.1\n",
        encoding="utf-8",
    )
    report_paths = []
    identities = []
    for episode in ("episode-a", "episode-b"):
        report = _signature_report(
            tmp_path,
            episode=episode,
            split="train",
            local_id=1,
            point_ids=[1, 2, 3],
        )
        path = tmp_path / f"{episode}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        report_paths.append(path)
        identities.append(
            {
                "identity_key": f"{episode}::local:1",
                "prompts": ["cabinet"],
            }
        )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {"schema": "farm.tracker-identity-evidence.v1", "identities": identities}
        ),
        encoding="utf-8",
    )
    output = tmp_path / "association.json"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/tracking/associate_farm_tracker_identities.py"
    )
    command = [sys.executable, str(script)]
    for path in report_paths:
        command.extend(["--signature-report", str(path)])
    command.extend(
        [
            "--identity-evidence",
            str(evidence),
            "--output",
            str(output),
            "--no-derive-color-appearance",
            "--no-verify-colmap-hashes",
        ]
    )
    subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA
    assert payload["fit_splits"] == ["train"]
    assert payload["local_to_global"] == {
        "episode-a::local:1": 1,
        "episode-b::local:1": 1,
    }
    assert [row["global_id"] for row in payload["assignments"]] == [1, 1]
    assert payload["policy"]["obb_used_as_gate"] is False


def test_cli_shadow_ablation_is_not_builder_compatible(tmp_path: Path) -> None:
    (tmp_path / "points3D.txt").write_text(
        "1 0 0 0 255 0 0 0.1\n2 0.1 0 0 255 0 0 0.1\n3 0.2 0 0 255 0 0 0.1\n",
        encoding="utf-8",
    )
    report_paths = []
    identities = []
    for episode in ("episode-a", "episode-b"):
        report = _signature_report(
            tmp_path,
            episode=episode,
            split="train",
            local_id=1,
            point_ids=[1, 2, 3],
        )
        path = tmp_path / f"{episode}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        report_paths.append(path)
        identities.append(
            {
                "identity_key": f"{episode}::local:1",
                "prompts": ["cabinet"],
            }
        )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {"schema": "farm.tracker-identity-evidence.v1", "identities": identities}
        ),
        encoding="utf-8",
    )
    output = tmp_path / "shadow.json"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/tracking/associate_farm_tracker_identities.py"
    )
    command = [sys.executable, str(script)]
    for path in report_paths:
        command.extend(["--signature-report", str(path)])
    command.extend(
        [
            "--identity-evidence",
            str(evidence),
            "--output",
            str(output),
            "--local-gate-source",
            LOCAL_GATE_V2_SHADOW,
            "--no-derive-color-appearance",
            "--no-verify-colmap-hashes",
        ]
    )
    subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == SHADOW_ABLATION_SCHEMA
    assert payload["fit_splits"] == ["train"]
    assert payload["automatic_promotion_allowed"] is False
    assert payload["gg_training_authorized"] is False
    assert "local_to_global" not in payload
    assert "assignments" not in payload
    assert "global_objects" not in payload
    assert payload["candidate_local_to_global"] == {
        "episode-a::local:1": 1,
        "episode-b::local:1": 1,
    }
