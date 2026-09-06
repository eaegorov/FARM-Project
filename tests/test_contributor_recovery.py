from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from farm_runtime.quality import contributor_recovery as recovery
from tools.farm_shaper_bridge import gaussian_lift as lift
from tools.farm_shaper_bridge.common import MaskObservation


def fixture():
    config, _ = lift.load_config(Path("configs/quality/native_rendered_v1.json"))
    config = deepcopy(config)
    config["candidate"]["rendered_recovery_budget"] = 4
    config["build"]["minimum_object_gaussians"] = 1
    config["build"]["maximum_object_fraction"] = 0.9
    config["refinement"]["enabled"] = False
    frames = [
        SimpleNamespace(
            image_id=i,
            physical_timestamp=str(i),
            depth_size=(3, 3),
            T_world_cam=np.eye(4),
            K=np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]], float),
        )
        for i in range(3)
    ]
    obj = SimpleNamespace(
        object_id=1,
        category="arbitrary category",
        observations=tuple(MaskObservation(1, i, "", {}) for i in [0, 1]),
    )
    g = SimpleNamespace(
        count=10,
        means_m=np.tile([0.0, 0.0, 8.0], (10, 1)),
        radius_m=np.full(10, 0.01),
        opacity_cpu=np.ones(10),
    )
    item = lift.ObjectEvidence(
        obj,
        np.zeros(0, np.int64),
        np.zeros(0),
        2,
        *[np.zeros(0, np.float32) for _ in range(3)],
        *[np.zeros(0, np.uint16) for _ in range(2)],
    )
    run = SimpleNamespace(objects=(obj,), frame=lambda i: frames[i])
    split = dict(
        build_timestamps=["0", "1"],
        heldout_timestamps=["2"],
        objects=[dict(object_id=1, build_timestamps=["0", "1"])],
    )
    rows = [
        dict(object_id=1, supported_claims_before_conflicts=0, build_timestamp_count=2)
    ]
    return run, g, split, {1: item}, config, rows, frames


def measurements(calls, negative_frame=None, two_candidates=False):
    def measure(g, run, frame, oid, config):
        calls.append(frame.image_id)
        values = np.zeros((g.count, 3), np.float32)
        values[: 2 if two_candidates else 1, [0, 2]] = 1
        if frame.image_id == negative_frame:
            values[0] = [0, 1, 1]
        return (
            values,
            np.full((3, 3), 2.0),
            np.full((3, 3), 6.0),
            dict(image_id=frame.image_id),
        )

    return measure


def test_rendered_recovery_finds_candidate_outside_spatial_search(monkeypatch):
    run, g, split, evidence, config, rows, _ = fixture()
    calls = []
    monkeypatch.setattr(recovery, "_measure_view", measurements(calls))
    result = recovery.recover_contributor_candidates(
        run, g, split, evidence, config, rows
    )
    assert result["updated_object_ids"] == [1]
    assert evidence[1].indices.tolist() == [0]
    assert evidence[1].positive_timestamps.tolist() == [2]
    assert result["additional_vjp_calls"] == 2
    assert calls == [0, 1]


def test_supported_objects_do_not_render_or_change(monkeypatch):
    run, g, split, evidence, config, rows, _ = fixture()
    original = evidence[1]
    rows[0]["supported_claims_before_conflicts"] = 1
    monkeypatch.setattr(
        recovery, "_measure_view", lambda *a: pytest.fail("unexpected render")
    )
    result = recovery.recover_contributor_candidates(
        run, g, split, evidence, config, rows
    )
    assert result["additional_vjp_calls"] == 0
    assert evidence[1] is original


def test_recovery_excludes_heldout_observations_before_render(monkeypatch):
    run, g, split, evidence, config, rows, _ = fixture()
    evidence[1].obj.observations += (MaskObservation(1, 2, "", {}),)
    calls = []
    monkeypatch.setattr(recovery, "_measure_view", measurements(calls))
    result = recovery.recover_contributor_candidates(
        run, g, split, evidence, config, rows
    )
    assert calls == [0, 1]
    assert result["updated_object_ids"] == [1]
    assert evidence[1].build_timestamp_count == 2


def test_sibling_view_conflict_does_not_become_independent_support(monkeypatch):
    run, g, split, evidence, config, rows, frames = fixture()
    original = evidence[1]
    frames[1].physical_timestamp = "0"
    frames[2].physical_timestamp = "1"
    original.obj.observations += (MaskObservation(1, 2, "", {}),)
    calls = []
    monkeypatch.setattr(
        recovery, "_measure_view", measurements(calls, negative_frame=1)
    )
    result = recovery.recover_contributor_candidates(
        run, g, split, evidence, config, rows
    )
    assert calls == [0, 1, 2]
    assert result["updated_object_ids"] == []
    assert evidence[1] is original


def test_recovery_candidate_budget_defers_without_truncating(monkeypatch):
    run, g, split, evidence, config, rows, _ = fixture()
    original = evidence[1]
    calls = []
    monkeypatch.setattr(recovery, "MAXIMUM_CANDIDATES", 1)
    monkeypatch.setattr(
        recovery, "_measure_view", measurements(calls, two_candidates=True)
    )
    result = recovery.recover_contributor_candidates(
        run, g, split, evidence, config, rows
    )
    assert calls == [0]
    assert result["objects"][0]["status"] == "deferred_candidate_budget"
    assert evidence[1] is original


def test_object_budget_zero_and_single_timestamp_are_no_work(monkeypatch):
    run, g, split, evidence, config, rows, frames = fixture()
    monkeypatch.setattr(
        recovery, "_measure_view", lambda *a: pytest.fail("unexpected render")
    )
    config["candidate"]["rendered_recovery_budget"] = 0
    assert (
        recovery.recover_contributor_candidates(run, g, split, evidence, config, rows)[
            "additional_vjp_calls"
        ]
        == 0
    )
    config["candidate"]["rendered_recovery_budget"] = 4
    frames[1].physical_timestamp = "0"
    result = recovery.recover_contributor_candidates(
        run, g, split, evidence, config, rows
    )
    assert (
        result["objects"][0]["status"]
        == "deferred_view_budget_or_insufficient_timestamps"
    )


def test_recovery_uses_global_competition_after_proposing_new_evidence(monkeypatch):
    run, g, split, evidence, config, rows, _ = fixture()
    monkeypatch.setattr(recovery, "_measure_view", measurements([]))
    recovery.recover_contributor_candidates(run, g, split, evidence, config, rows)
    competitor = deepcopy(evidence[1])
    competitor.obj = SimpleNamespace(object_id=2, category="unrelated")
    competitor.positive_timestamps[:] = 3
    competitor.positive_weight[:] = 3
    competitor.visible_weight[:] = 3
    evidence[2] = competitor
    owner = lift.make_claims(evidence, g.count, config)[0]
    assert owner[0] == 2


def test_depth_spread_preserves_solid_occlusion_and_handles_mixture():
    run, g, _, evidence, config, _, frames = fixture()
    item = evidence[1]
    item.indices = np.array([0, 1, 2])
    item.radius_m = np.full(3, 0.01)
    g.means_m[1] = [0, 0, 4]
    g.means_m[2] = [100, 0, 8]
    depth = np.full((3, 3), 6.0)
    zero = np.zeros((3, 3))
    actual = recovery.depth_spread_gate(g, item, frames[0], depth, zero, config)
    assert actual.tolist() == [False, True, False]
    np.testing.assert_array_equal(
        actual, lift._depth_gate(g, item, frames[0], depth, config)
    )
    mixed = recovery.depth_spread_gate(g, item, frames[0], depth, zero + 2, config)
    assert mixed.tolist() == [True, True, False]
    assert not recovery.depth_spread_gate(
        g, item, frames[0], depth, zero + np.nan, config
    ).any()
    assert not recovery.depth_spread_gate(
        g, item, frames[0], depth * 0, zero + 2, config
    ).any()


@pytest.mark.parametrize("budget", [True, -1, 9, 1.5])
def test_invalid_budget_rejected_before_render(budget):
    run, g, split, evidence, config, rows, _ = fixture()
    config["candidate"]["rendered_recovery_budget"] = budget
    with pytest.raises(ValueError, match="budget"):
        recovery.recover_contributor_candidates(run, g, split, evidence, config, rows)


def test_recovery_requires_background_evidence_and_global_split():
    run, g, split, evidence, config, rows, _ = fixture()
    config["mask"]["negative_domain"] = "ring"
    with pytest.raises(ValueError, match="visible background"):
        recovery.recover_contributor_candidates(run, g, split, evidence, config, rows)
    config["mask"]["negative_domain"] = "visible_background"
    split["heldout_timestamps"] = ["0"]
    with pytest.raises(ValueError, match="disjoint"):
        recovery.recover_contributor_candidates(run, g, split, evidence, config, rows)
