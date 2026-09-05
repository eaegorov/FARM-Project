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
