from __future__ import annotations

import json
import random
import math
from pathlib import Path

from scene_graph.captioning.evidence import (
    crop_evidence_manifest,
    evidence_camera_poses,
    evidence_view_overlap,
    partition_crop_candidates,
    select_independent_evidence,
    select_pose_diverse_crop_candidates,
)


def _crop(image_id: int, det: int = 0, *, member: int | None = None):
    member_token = f"_member_{member:06d}" if member is not None else ""
    return (Path(f"/run/masks/object_000007/img_{image_id:06d}{member_token}_det_{det:04d}.npz"), b"jpeg")


def _detail_crop(image_id: int, detail: int, det: int = 0):
    return (
        Path(
            f"/run/masks/object_000007/"
            f"img_{image_id:06d}_det_{det:04d}.npz"
        ),
        b"x" * detail,
    )


def _poses_by_angle(angles: dict[int, float]) -> dict:
    result = {}
    for image_id, angle_degrees in angles.items():
        angle = math.radians(angle_degrees)
        center = [10.0 * math.cos(angle), 10.0 * math.sin(angle), 0.0]
        result[image_id] = {
            "camera_center_world_m": center,
            "camera_forward_world": [
                -center[0] / 10.0,
                -center[1] / 10.0,
                0.0,
            ],
        }
    return result


def _event(
    source: str,
    image_ids: list[int],
    *,
    fingerprint: str | None = None,
    complete: bool = True,
    confirmation_eligible: bool = True,
) -> dict:
    pose_index = {}
    for image_id in image_ids:
        angle = math.radians(float(image_id) * 20.0)
        center = [10.0 * math.cos(angle), 10.0 * math.sin(angle), 0.0]
        pose_index[image_id] = {
            "camera_center_world_m": center,
            "camera_forward_world": [-center[0] / 10.0, -center[1] / 10.0, 0.0],
        }
    manifest = crop_evidence_manifest(
        7,
        [_crop(image_id) for image_id in image_ids],
        frame_pose_index=pose_index,
        object_position_world_m=[0.0, 0.0, 0.0],
    )
    return {
        "source": source,
        "event_id": source,
        "category": "cabinet",
        "confidence": 0.95,
        "decision": "keep",
        **manifest,
        "crop_image_ids_complete": complete,
        "evidence_fingerprint_sha256": fingerprint or manifest["evidence_fingerprint_sha256"],
        "confirmation_eligible": confirmation_eligible,
    }


def _ids(rows) -> set[int]:
    return {
        int(row[0].name.split("img_", 1)[1].split("_", 1)[0])
        for row in rows
    }


def test_partitions_are_deterministic_and_keep_one_image_in_one_fold() -> None:
    candidates = [
        _crop(10, 0, member=1),
        _crop(10, 1, member=2),
        *[_crop(value) for value in (11, 12, 13, 14, 15, 16)],
    ]
    shuffled = list(candidates)
    random.Random(7).shuffle(shuffled)

    first, first_meta = partition_crop_candidates(candidates, "blind_pass_b")
    second, second_meta = partition_crop_candidates(shuffled, "blind_pass_b")

    assert [row[0].name for row in first] == [row[0].name for row in second]
    assert first_meta == second_meta
    assert sum(row[0].name.startswith("img_000010") for row in first) in {0, 2}


def test_blind_b_and_c_are_disjoint_with_six_to_eight_views() -> None:
    for count in (6, 7, 8):
        candidates = [_crop(index) for index in range(count)]
        blind_b, meta_b = partition_crop_candidates(candidates, "blind_pass_b")
        blind_c, meta_c = partition_crop_candidates(candidates, "blind_pass_c")

        assert meta_b["partition_count"] == 2
        assert meta_c["partition_count"] == 2
        assert _ids(blind_b).isdisjoint(_ids(blind_c))
        assert _ids(blind_b) | _ids(blind_c) == set(range(count))


def test_physical_recovery_and_verification_use_disjoint_folds() -> None:
    candidates = [_crop(index) for index in range(6)]
    recovery, recovery_meta = partition_crop_candidates(
        candidates, "physical_form_recovery"
    )
    verification, verification_meta = partition_crop_candidates(
        candidates, "physical_form_recovery_verification"
    )

    assert recovery_meta["partition_count"] == 2
    assert verification_meta["partition_count"] == 2
    assert _ids(recovery).isdisjoint(_ids(verification))
    assert _ids(recovery) | _ids(verification) == set(range(6))


def test_blind_b_c_and_paired_are_pairwise_disjoint_from_six_views() -> None:
    candidates = [_crop(index) for index in range(9)]
    groups = [
        partition_crop_candidates(candidates, source)[0]
        for source in (
            "blind_pass_b",
            "blind_pass_c",
            "paired_physical_form_guard",
        )
    ]

    assert all(
        _ids(left).isdisjoint(_ids(right))
        for index, left in enumerate(groups)
        for right in groups[index + 1 :]
    )
    assert set().union(*(_ids(group) for group in groups)) == set(range(9))


def test_tracks_below_six_views_reuse_one_fold() -> None:
    candidates = [_crop(index) for index in range(5)]
    blind_b, meta_b = partition_crop_candidates(candidates, "blind_pass_b")
    blind_c, meta_c = partition_crop_candidates(candidates, "blind_pass_c")

    assert meta_b["partition_count"] == meta_c["partition_count"] == 1
    assert _ids(blind_b) == _ids(blind_c) == {0, 1, 2, 3, 4}


def test_manifest_fingerprint_is_relocatable_and_source_independent() -> None:
    first = [
        (Path("/archive/run-a/object_000007/img_000010_det_0001.npz"), b"a"),
        (Path("/archive/run-a/object_000007/img_000012_det_0002.npz"), b"b"),
    ]
    second = [
        (Path("/new/root/object_000007/img_000012_det_0002.npz"), b"different"),
        (Path("/new/root/object_000007/img_000010_det_0001.npz"), b"bytes"),
    ]

    left = crop_evidence_manifest(7, first)
    right = crop_evidence_manifest(7, second)

    assert left["crop_image_ids"] == [10, 12]
    assert left["crop_image_ids_complete"] is True
    assert left["evidence_fingerprint_sha256"] == right["evidence_fingerprint_sha256"]
    assert len(left["evidence_fingerprint_sha256"]) == 64


def test_manifest_marks_unparseable_legacy_path_incomplete() -> None:
    manifest = crop_evidence_manifest(7, [(Path("legacy_crop.npz"), b"jpeg")])

    assert manifest["crop_image_ids"] == []
    assert manifest["crop_image_ids_complete"] is False


def test_overlap_coefficient_treats_subset_as_fully_correlated() -> None:
    left = _event("blind_pass_b", [1, 2, 3, 4, 5], fingerprint="a" * 64)
    right = _event("blind_pass_c", [2, 3, 4], fingerprint="b" * 64)

    overlap = evidence_view_overlap(left, right)

    assert overlap["overlap_coefficient"] == 1.0
    assert overlap["jaccard"] == 0.6
    assert overlap["independent"] is False
    assert overlap["reason"] == "view_overlap_above_threshold"


def test_low_quantified_overlap_is_accepted_at_contract_threshold() -> None:
    left = _event("blind_pass_b", [1, 2, 3, 4, 5], fingerprint="a" * 64)
    right = _event("blind_pass_c", [5, 6, 7, 8, 9], fingerprint="b" * 64)

    overlap = evidence_view_overlap(left, right)

    assert overlap["overlap_coefficient"] == 0.2
    assert overlap["independent"] is True
    assert overlap["reason"] == "independent_image_ids_and_camera_poses"


def test_different_image_ids_with_near_duplicate_camera_poses_are_rejected() -> None:
    def event(source: str, image_ids: list[int], angles: list[float]) -> dict:
        poses = {}
        for image_id, angle_degrees in zip(image_ids, angles):
            angle = math.radians(angle_degrees)
            center = [10.0 * math.cos(angle), 10.0 * math.sin(angle), 0.0]
            poses[image_id] = {
                "camera_center_world_m": center,
                "camera_forward_world": [-center[0] / 10.0, -center[1] / 10.0, 0.0],
            }
        return {
            "source": source,
            "event_id": source,
            "confirmation_eligible": True,
            **crop_evidence_manifest(
                7,
                [_crop(image_id) for image_id in image_ids],
                frame_pose_index=poses,
                object_position_world_m=[0.0, 0.0, 0.0],
            ),
        }

    left = event("blind_pass_b", [10, 11, 12], [0.0, 0.5, 1.0])
    right = event("blind_pass_c", [20, 21, 22], [1.5, 2.0, 2.5])
    overlap = evidence_view_overlap(left, right)

    assert overlap["intersection_count"] == 0
    assert overlap["independent"] is False
    assert overlap["reason"] == "viewpoint_separation_below_threshold"
    assert overlap["pose_diversity"][
        "symmetric_median_nearest_viewpoint_angle_degrees"
    ] < 2.0


def test_same_fingerprint_is_never_independent() -> None:
    left = _event("blind_pass_b", [1, 2, 3], fingerprint="a" * 64)
    right = _event("blind_pass_c", [4, 5, 6], fingerprint="a" * 64)

    overlap = evidence_view_overlap(left, right)

    assert overlap["independent"] is False
    assert overlap["reason"] == "same_evidence_fingerprint"


def test_independent_selector_finds_largest_pairwise_view_group() -> None:
    events = [
        _event("blind_pass_b", [1, 4, 7], fingerprint="a" * 64),
        _event("blind_pass_c", [2, 5, 8], fingerprint="b" * 64),
        _event("paired_crop_blind_adjudicator", [3, 6, 9], fingerprint="c" * 64),
        _event("correlated_copy", [1, 4, 7], fingerprint="d" * 64),
    ]

    selected, diagnostics = select_independent_evidence(events)

    assert {row["source"] for row in selected} == {
        "blind_pass_b",
        "blind_pass_c",
        "paired_crop_blind_adjudicator",
    }
    assert diagnostics["confirmation_ready"] is True
    assert diagnostics["selected_event_count"] == 3


def test_legacy_and_one_view_events_fail_closed_for_confirmation() -> None:
    legacy = {
        "source": "legacy_pass",
        "category": "cabinet",
        "confidence": 0.95,
        "decision": "keep",
    }
    one_view = _event("blind_pass_b", [1], fingerprint="a" * 64)
    good = _event("blind_pass_c", [2, 3, 4], fingerprint="b" * 64)

    selected, diagnostics = select_independent_evidence([legacy, one_view, good])

    assert selected == []
    assert diagnostics["confirmation_ready"] is False
    assert diagnostics["incomplete_provenance_event_ids"] == ["legacy_pass"]
    assert diagnostics["under_observed_event_ids"] == ["blind_pass_b"]


def test_aliases_and_explicitly_ineligible_guard_do_not_add_confirmation() -> None:
    initial = _event("initial_blind_review", [1, 2, 3], fingerprint="a" * 64)
    alias = _event("blind_pass_a", [4, 5, 6], fingerprint="b" * 64)
    guard = _event(
        "paired_physical_form_guard",
        [7, 8, 9],
        fingerprint="c" * 64,
        confirmation_eligible=False,
    )

    selected, diagnostics = select_independent_evidence([initial, alias, guard])

    assert selected == []
    assert diagnostics["candidate_event_count"] == 2
    assert diagnostics["confirmation_ineligible_event_ids"] == [
        "paired_physical_form_guard"
    ]


def test_confirmation_requires_six_total_unique_views() -> None:
    left = _event("blind_pass_b", [1, 2, 3], fingerprint="a" * 64)
    right = _event("blind_pass_c", [3, 4, 5], fingerprint="b" * 64)

    selected, diagnostics = select_independent_evidence(
        [left, right], max_overlap_coefficient=0.5
    )

    assert selected == []
    assert diagnostics["confirmation_ready"] is False
    assert diagnostics["min_total_unique_views"] == 6


def test_pose_crop_selector_is_deterministic_and_prefers_diverse_views() -> None:
    candidates = [
        _detail_crop(0, 100),
        _detail_crop(1, 100),
        _detail_crop(2, 100),
        _detail_crop(3, 80),
        _detail_crop(4, 80),
    ]
    fallback = candidates[:3]
    poses = _poses_by_angle({0: 0.0, 1: 1.0, 2: 2.0, 3: 30.0, 4: 60.0})
    shuffled = list(candidates)
    random.Random(11).shuffle(shuffled)

    selected, diagnostics = select_pose_diverse_crop_candidates(
        candidates,
        fallback,
        3,
        frame_pose_index=poses,
        object_position_world_m=[0.0, 0.0, 0.0],
    )
    repeated, repeated_diagnostics = select_pose_diverse_crop_candidates(
        shuffled,
        fallback,
        3,
        frame_pose_index=poses,
        object_position_world_m=[0.0, 0.0, 0.0],
    )

    assert _ids(selected) == _ids(repeated) == {0, 3, 4}
    assert diagnostics == repeated_diagnostics
    assert diagnostics["method"] == "pose_aware_exact"
    assert diagnostics["reason"] == "diversity_gate_met"
    assert diagnostics["diversity_gate_met"] is True
    assert diagnostics["detail_ratio_to_temporal_fallback"] >= 0.8


def test_pose_crop_selector_detail_floor_blocks_low_information_views() -> None:
    candidates = [
        _detail_crop(0, 100),
        _detail_crop(1, 100),
        _detail_crop(2, 100),
        _detail_crop(3, 5),
        _detail_crop(4, 5),
    ]
    fallback = candidates[:3]
    selected, diagnostics = select_pose_diverse_crop_candidates(
        candidates,
        fallback,
        3,
        frame_pose_index=_poses_by_angle(
            {0: 0.0, 1: 1.0, 2: 2.0, 3: 30.0, 4: 60.0}
        ),
        object_position_world_m=[0.0, 0.0, 0.0],
    )

    assert selected == fallback
    assert diagnostics["diversity_gate_met"] is False
    assert diagnostics["reason"] == "best_available_below_diversity_gate"
    assert diagnostics["detail_ratio_to_temporal_fallback"] == 1.0


def test_pose_crop_selector_missing_pose_returns_exact_fallback() -> None:
    candidates = [_detail_crop(value, 100) for value in range(5)]
    fallback = [candidates[2], candidates[0], candidates[1]]

    selected, diagnostics = select_pose_diverse_crop_candidates(
        candidates,
        fallback,
        3,
        frame_pose_index=_poses_by_angle({0: 0.0, 1: 20.0}),
        object_position_world_m=[0.0, 0.0, 0.0],
    )

    assert selected == fallback
    assert diagnostics["method"] == "temporal_fallback"
    assert diagnostics["reason"] == "insufficient_unique_posed_views"
    assert diagnostics["posed_unique_image_count"] == 2


def test_pose_crop_selector_collapses_duplicate_image_ids_by_detail() -> None:
    candidates = [
        _detail_crop(0, 50),
        _detail_crop(0, 200, det=1),
        _detail_crop(1, 100),
        _detail_crop(2, 100),
    ]
    fallback = [candidates[0], candidates[2], candidates[3]]

    selected, diagnostics = select_pose_diverse_crop_candidates(
        candidates,
        fallback,
        3,
        frame_pose_index=_poses_by_angle({0: 0.0, 1: 30.0, 2: 60.0}),
        object_position_world_m=[0.0, 0.0, 0.0],
    )

    selected_zero = [row for row in selected if row[0].name.startswith("img_000000")]
    assert len(selected_zero) == 1
    assert selected_zero[0][0].name.endswith("det_0001.npz")
    assert diagnostics["posed_unique_image_count"] == 3
    assert diagnostics["candidate_combination_count"] == 1


def test_pose_crop_selector_beam_search_is_bounded_and_deterministic() -> None:
    candidates = [_detail_crop(value, 100) for value in range(7)]
    fallback = candidates[:3]
    poses = _poses_by_angle({value: float(value * 20) for value in range(7)})
    shuffled = list(candidates)
    random.Random(29).shuffle(shuffled)

    selected, diagnostics = select_pose_diverse_crop_candidates(
        candidates,
        fallback,
        3,
        frame_pose_index=poses,
        object_position_world_m=[0.0, 0.0, 0.0],
        max_combinations=4,
    )
    repeated, repeated_diagnostics = select_pose_diverse_crop_candidates(
        shuffled,
        fallback,
        3,
        frame_pose_index=poses,
        object_position_world_m=[0.0, 0.0, 0.0],
        max_combinations=4,
    )

    assert _ids(selected) == _ids(repeated)
    assert diagnostics == repeated_diagnostics
    assert diagnostics["method"] == "pose_aware_beam"
    assert diagnostics["candidate_combination_count"] == 35
    assert diagnostics["evaluated_combination_count"] <= 4


def test_manifest_preserves_additive_crop_selection_diagnostics() -> None:
    diagnostics = {
        "schema": "farm.pose-aware-crop-selection.v1",
        "method": "pose_aware_exact",
        "diversity_gate_met": True,
    }
    manifest = crop_evidence_manifest(
        7,
        [_crop(1), _crop(2), _crop(3)],
        partition={
            "partition_id": "view-fold-0-of-1",
            "partition_index": 0,
            "partition_count": 1,
            "selection_diagnostics": diagnostics,
        },
    )

    assert manifest["crop_selection_diagnostics"] == diagnostics


def test_persisted_pose_fingerprint_is_not_changed_by_re_normalization() -> None:
    # This non-axis-aligned direction normalizes to a vector whose serialized
    # norm is not bit-exactly 1.0.  The verifier must validate, not normalize a
    # second time, because the producer fingerprint covers the stored floats.
    manifest = crop_evidence_manifest(
        7,
        [_crop(34), _crop(74), _crop(197)],
        frame_pose_index={
            34: {
                "camera_center_world_m": [1.25, -2.5, 0.75],
                "camera_forward_world": [-0.314159265, 0.271828182, 0.577215664],
            },
            74: {
                "camera_center_world_m": [-2.0, 1.5, 1.25],
                "camera_forward_world": [0.707106781, -0.4, 0.23],
            },
            197: {
                "camera_center_world_m": [0.5, 3.0, -0.25],
                "camera_forward_world": [-0.13, -0.91, 0.37],
            },
        },
        object_position_world_m=[0.0, 0.0, 0.0],
    )

    persisted = json.loads(json.dumps(manifest))
    verified = evidence_camera_poses(persisted)

    assert verified is not None
    assert [row["image_id"] for row in verified[1]] == [34, 74, 197]


def test_persisted_pose_fingerprints_retain_event_independence() -> None:
    def persisted_event(source: str, image_ids: list[int], offset: float) -> dict:
        poses = {}
        for image_id, base_angle in zip(image_ids, (0.0, 120.0, 240.0)):
            angle = math.radians(base_angle + offset)
            center = [10.0 * math.cos(angle), 10.0 * math.sin(angle), 0.0]
            poses[image_id] = {
                "camera_center_world_m": center,
                "camera_forward_world": [
                    -center[0] / 10.0,
                    -center[1] / 10.0,
                    0.0,
                ],
            }
        event = {
            "source": source,
            "event_id": source,
            "confirmation_eligible": True,
            **crop_evidence_manifest(
                7,
                [_crop(image_id) for image_id in image_ids],
                frame_pose_index=poses,
                object_position_world_m=[0.0, 0.0, 0.0],
            ),
        }
        return json.loads(json.dumps(event))

    overlap = evidence_view_overlap(
        persisted_event("initial_blind_review", [10, 11, 12], 0.0),
        persisted_event("independent_verification", [20, 21, 22], 25.0),
    )

    assert overlap["independent"] is True
    assert overlap["reason"] == "independent_image_ids_and_camera_poses"
    assert overlap["pose_diversity"]["pose_independent"] is True


def test_pose_fingerprint_still_rejects_non_unit_or_tampered_direction() -> None:
    manifest = crop_evidence_manifest(
        7,
        [_crop(1)],
        frame_pose_index={
            1: {
                "camera_center_world_m": [1.0, 0.0, 0.0],
                "camera_forward_world": [-1.0, 0.0, 0.0],
            }
        },
        object_position_world_m=[0.0, 0.0, 0.0],
    )
    manifest["crop_camera_poses"][0]["camera_forward_world"] = [-2.0, 0.0, 0.0]

    assert evidence_camera_poses(manifest) is None
