from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tools.farm_shaper_bridge.common import UNKNOWN_ID
from tools.farm_shaper_bridge.lift_refinement import (
    apply_provisional_geometry_gate,
    refine_connected_claims,
)


def _config() -> dict:
    return {
        "refinement": {
            "enabled": True,
            "minimum_positive_timestamps": 1,
            "maximum_negative_timestamps": 0,
            "minimum_timestamp_margin": 1,
            "minimum_positive_weight": 0.1,
            "minimum_global_purity": 0.7,
            "minimum_visible_share": 0.1,
            "minimum_core_neighbors": 2,
            "maximum_growth_steps": 2,
            "minimum_connection_radius_m": 0.016,
            "maximum_connection_radius_m": 0.016,
            "connection_radius_multiplier": 1.0,
            "maximum_growth_ratio": 2.0,
            "score_log_weight": 0.08,
            "minimum_winner_margin": 0.2,
            "minimum_winner_ratio": 1.1,
        },
        "build": {
            "minimum_object_gaussians": 2,
            "maximum_object_fraction": 0.8,
        },
    }


def _evidence(object_id: int = 1):
    return SimpleNamespace(
        indices=np.arange(5, dtype=np.int64),
        positive_weight=np.ones(5, dtype=np.float32),
        negative_weight=np.zeros(5, dtype=np.float32),
        visible_weight=np.ones(5, dtype=np.float32),
        positive_timestamps=np.ones(5, dtype=np.uint16),
        negative_timestamps=np.zeros(5, dtype=np.uint16),
        build_timestamp_count=4,
        obj=SimpleNamespace(object_id=object_id, category="fixture"),
    )


def _run_refinement(labels: np.ndarray, *, blocked=()):
    means = np.asarray([
        [0.000, 0, 0],
        [0.010, 0, 0],
        [0.014, 0, 0],
        [0.025, 0, 0],
        [0.200, 0, 0],
    ], dtype=np.float32)
    confidence = np.where(labels >= 0, 0.9, 0).astype(np.float32)
    support = np.where(labels >= 0, 2, 0).astype(np.uint16)
    return refine_connected_claims(
        means,
        np.full(5, 0.001, dtype=np.float32),
        {1: _evidence()},
        labels.copy(),
        confidence,
        support,
        np.asarray(blocked, dtype=np.int64),
        _config(),
    )


def test_refinement_grows_two_hops_but_not_a_disconnected_island() -> None:
    labels, confidence, support, audit = _run_refinement(
        np.asarray([1, 1, UNKNOWN_ID, UNKNOWN_ID, UNKNOWN_ID], dtype=np.int32)
    )
    assert labels.tolist() == [1, 1, 1, 1, UNKNOWN_ID]
    assert confidence[2] > 0 and confidence[3] > 0 and confidence[4] == 0
    assert support.tolist() == [2, 2, 1, 1, 0]
    assert audit["added"] == 2
    assert audit["objects_grown"] == 1
    assert audit["heldout_masks_opened"] is False
    assert audit["strong_labels_overwritten"] == 0
    assert audit["objects"][0]["maximum_growth_step"] == 2


def test_refinement_never_promotes_a_strong_unresolved_conflict() -> None:
    labels, _, _, audit = _run_refinement(
        np.asarray([1, 1, UNKNOWN_ID, UNKNOWN_ID, UNKNOWN_ID], dtype=np.int32),
        blocked=[2],
    )
    assert labels[2] == UNKNOWN_ID
    assert labels[3] == UNKNOWN_ID
    assert audit["blocked_strong_conflicts"] == 1


def test_refinement_never_overwrites_another_object() -> None:
    labels, _, _, audit = _run_refinement(
        np.asarray([1, 1, 2, UNKNOWN_ID, UNKNOWN_ID], dtype=np.int32)
    )
    assert labels[2] == 2
    assert labels[3] == UNKNOWN_ID
    assert audit["strong_labels_overwritten"] == 0


def test_refinement_rejects_negative_or_low_purity_growth() -> None:
    evidence = _evidence()
    evidence.negative_timestamps[2] = 1
    evidence.negative_weight[3] = 2.0
    labels = np.asarray([1, 1, UNKNOWN_ID, UNKNOWN_ID, UNKNOWN_ID], dtype=np.int32)
    means = np.asarray([[0, 0, 0], [.01, 0, 0], [.014, 0, 0], [.015, 0, 0], [.2, 0, 0]])
    refined, _, _, audit = refine_connected_claims(
        means,
        np.full(5, .001),
        {1: evidence},
        labels,
        np.asarray([.9, .9, 0, 0, 0], dtype=np.float32),
        np.asarray([2, 2, 0, 0, 0], dtype=np.uint16),
        np.zeros(0, dtype=np.int64),
        _config(),
    )
    assert refined.tolist() == [1, 1, UNKNOWN_ID, UNKNOWN_ID, UNKNOWN_ID]
    assert audit["eligible"] == 1  # only the distant row survives evidence gates
    assert audit["added"] == 0


def test_geometry_gate_runs_after_growth_and_zeros_rejected_rows() -> None:
    labels = np.asarray([1, 1, 1, 2, UNKNOWN_ID], dtype=np.int32)
    confidence = np.asarray([.9, .8, .7, .6, 0], dtype=np.float32)
    support = np.asarray([2, 2, 1, 1, 0], dtype=np.uint16)
    rows = [
        {"object_id": 1, "seed_gaussians": 2},
        {"object_id": 2, "seed_gaussians": 1},
    ]
    labels, confidence, support, audit = apply_provisional_geometry_gate(
        labels, confidence, support, rows, len(labels), _config()
    )
    assert labels.tolist() == [1, 1, 1, UNKNOWN_ID, UNKNOWN_ID]
    assert confidence[3] == 0 and support[3] == 0
    assert rows[0]["geometry_gate"] == "PASS"
    assert rows[1]["geometry_gate"] == "REJECT"
    assert audit["rejected_object_ids"] == [2]


def test_refinement_is_deterministic_in_source_row_order() -> None:
    initial = np.asarray([1, 1, UNKNOWN_ID, UNKNOWN_ID, UNKNOWN_ID], dtype=np.int32)
    first = _run_refinement(initial)
    second = _run_refinement(initial)
    for left, right in zip(first[:3], second[:3], strict=True):
        assert np.array_equal(left, right)
    assert first[3] == second[3]


def test_growth_does_not_reintroduce_rejected_build_visible_share():
    item = _evidence()
    item.visible_weight[2:] = 5.0  # Share 0.2: passes weak 0.1, fails build 0.3.
    labels = np.asarray([1, 1, UNKNOWN_ID, UNKNOWN_ID, UNKNOWN_ID], dtype=np.int32)
    means = np.asarray([[0, 0, 0], [.01, 0, 0], [.014, 0, 0], [.015, 0, 0], [.2, 0, 0]])
    config = _config()
    config["build"]["minimum_visible_share"] = 0.3
    result, _, _, audit = refine_connected_claims(
        means, np.full(5, .001), {1: item}, labels.copy(),
        np.where(labels >= 0, .9, 0).astype(np.float32),
        np.where(labels >= 0, 2, 0).astype(np.uint16),
        np.zeros(0, np.int64), config,
    )
    np.testing.assert_array_equal(result, labels)
    assert audit["eligible"] == 0

def _remote_policy():
    return dict(enabled=True, neighbors=16, minimum_dominant_fraction=0.6,
                maximum_component_fraction=0.05, maximum_removed_fraction=0.15,
                minimum_relative_separation=0.25)


def _remote_fixture():
    x, y = np.meshgrid(np.arange(10) * .01, np.arange(10) * .01)
    core = np.column_stack([x.ravel(), y.ravel(), np.zeros(100)])
    def island(x, count):
        return np.column_stack([x + np.arange(count) * .005,
                                np.zeros(count), np.zeros(count)])
    # A nearby part and a substantial remote part must remain.
    points = np.concatenate([core, island(.25, 5), island(2, 4), island(4, 10)])
    connection = dict(connection_radius_multiplier=2.,
                      minimum_connection_radius_m=.025,
                      maximum_connection_radius_m=.06)
    return points, np.full(len(points), .012), connection


def test_remote_components_preserve_nearby_and_substantial_separate_parts():
    from tools.farm_shaper_bridge.lift_refinement import remote_component_mask
    points, radii, connection = _remote_fixture()
    removed, audit = remote_component_mask(points, radii, connection, _remote_policy())
    assert np.flatnonzero(removed).tolist() == list(range(105, 109))
    assert audit["removed"] == 4
    assert audit["reason"] == "remote_small_components"


def test_remote_components_abstain_when_fragmented_or_over_budget():
    from tools.farm_shaper_bridge.lift_refinement import remote_component_mask
    points, radii, connection = _remote_fixture()
    policy = _remote_policy()
    policy["maximum_removed_fraction"] = .02
    removed, audit = remote_component_mask(points, radii, connection, policy)
    assert not removed.any()
    assert audit["reason"] == "removal_budget_exceeded"
    separated = np.column_stack([np.arange(50), np.zeros(50), np.zeros(50)])
    removed, audit = remote_component_mask(separated, np.full(50, .01), connection, _remote_policy())
    assert not removed.any()
    assert audit["reason"] == "no_dominant_surface"


def test_remote_components_are_rotation_translation_and_scale_equivariant():
    from tools.farm_shaper_bridge.lift_refinement import remote_component_mask
    points, radii, connection = _remote_fixture()
    expected, _ = remote_component_mask(points, radii, connection, _remote_policy())
    angle = .71
    rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    moved = (points @ rot.T) * 17 + [23, -61, 14]
    connection = {k: v * 17 if k.endswith("_m") else v for k, v in connection.items()}
    actual, _ = remote_component_mask(moved, radii * 17, connection, _remote_policy())
    np.testing.assert_array_equal(actual, expected)


def test_remote_component_removal_clears_evidence_without_reassigning_other_objects():
    points, radii, connection = _remote_fixture()
    n = len(points)
    item = SimpleNamespace(
        indices=np.arange(n), positive_weight=np.ones(n), negative_weight=np.zeros(n),
        visible_weight=np.ones(n), positive_timestamps=np.full(n, 2, np.uint16),
        negative_timestamps=np.zeros(n, np.uint16), build_timestamp_count=3,
    )
    config = _config()
    config["refinement"].update(connection, remote_components=_remote_policy())
    labels = np.full(n, 41, np.int32)
    labels[-1] = 73
    conf = np.full(n, .9, np.float32)
    support = np.full(n, 2, np.uint16)
    result, conf, support, audit = refine_connected_claims(
        points, radii, {41: item}, labels, conf, support,
        np.array([], np.int64), config,
    )
    assert result[-1] == 73
    assert np.flatnonzero(result == UNKNOWN_ID).tolist() == list(range(105, 109))
    assert not conf[105:109].any() and not support[105:109].any()
    assert audit["objects"][0]["remote_components"]["removed"] == 4
    assert audit["strong_labels_overwritten"] == 0


def test_remote_component_disabled_preserves_membership_exactly():
    points, radii, connection = _remote_fixture()
    n = len(points)
    item = SimpleNamespace(
        indices=np.arange(n), positive_weight=np.ones(n), negative_weight=np.zeros(n),
        visible_weight=np.ones(n), positive_timestamps=np.full(n, 2, np.uint16),
        negative_timestamps=np.zeros(n, np.uint16), build_timestamp_count=3,
    )
    initial = np.full(n, 41, np.int32)
    config = _config()
    config["refinement"].update(connection)
    result, conf, support, audit = refine_connected_claims(
        points, radii, {41: item}, initial.copy(), np.full(n, .9),
        np.full(n, 2, np.uint16), np.array([], np.int64), config,
    )
    np.testing.assert_array_equal(result, initial)
    assert "remote_components" not in audit["objects"][0]
