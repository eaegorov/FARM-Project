import math
import pytest

from farm_runtime.quality.discovery_coverage import choose_views


def test_coverage_prefers_weak_groups_and_does_not_spend_budget_on_siblings():
    visibility = {
        "a": {1: 1.0},
        "a_stereo": {1: 1.0},
        "b": {2: 1.0},
        "c": {1: 0.5, 2: 0.5},
    }
    selected, covered = choose_views(
        visibility,
        {1: {"old"}, 2: {"old", "other"}},
        {"a": "100", "a_stereo": "100", "b": "200", "c": "300"},
        3,
    )
    assert [r["name"] for r in selected] == ["a_stereo", "b"]
    assert covered == {1: 1.0, 2: 1.0}
    assert selected[0]["marginal_score"] == 1 and selected[1]["marginal_score"] == 0.5


def test_already_observed_timestamp_cannot_confirm_a_group():
    selected, covered = choose_views(
        {"same": {1: 1.0, 2: 0.8}, "new": {1: 0.7}},
        {1: {"100"}, 2: {"200"}},
        {"same": "100", "new": "300"},
        1,
    )
    assert selected[0]["name"] == "same"
    assert selected[0]["visibility"] == {2: 0.8}
    assert covered[1] == 0
    selected, _ = choose_views({"same": {1: 1.0}}, {1: {"100"}}, {"same": "100"}, 8)
    assert selected == []


@pytest.mark.parametrize("fraction", [-0.1, 1.1, math.nan, math.inf])
def test_invalid_visibility_is_not_camera_evidence(fraction):
    with pytest.raises(ValueError, match="visibility"):
        choose_views({"a": {1: fraction}}, {1: {"old"}}, {"a": "new"}, 1)


def test_budget_limits_work_and_empty_candidates_stop():
    with pytest.raises(ValueError, match="budget"):
        choose_views({}, {}, {}, 25)
    with pytest.raises(ValueError, match="observed timestamp"):
        choose_views({}, {1: set()}, {}, 1)
    assert choose_views({}, {1: {"old"}}, {}, 1) == ([], {1: 0.0})
    selected, _ = choose_views(
        {"a": {1: 1}, "b": {2: 1}}, {1: {"old"}, 2: {"old"}}, {"a": "a", "b": "b"}, 1
    )
    assert len(selected) == 1


def test_exploration_can_find_new_objects_without_fabricating_foreground_votes():
    import numpy as np
    from farm_runtime.quality.discovery_coverage import explore_views

    def frame(ts, angle):
        c, s = np.cos(angle), np.sin(angle)
        t = np.eye(4)
        t[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        return dict(frame_id=ts, T_world_cam=t.tolist())

    old = {"old": frame("0", 0)}
    frames = {
        "opposite": frame("0", np.pi),
        "new": frame("1", np.pi / 2),
        "new_sibling": frame("1", np.pi / 2),
        "duplicate": frame("2", 0),
    }
    extra = explore_views(frames, old, [], 3)
    assert [r["name"] for r in extra] == ["opposite", "new_sibling"]
    assert all(
        r["visibility"] == {}
        and r["improved_group_ids"] == []
        and r["marginal_score"] == 0
        for r in extra
    )
    # An already-targeted observation remains unchanged and consumes budget.
    targeted = [dict(name="new", timestamp="1", visibility={9: 0.9})]
    original = dict(targeted[0])
    extra = explore_views(frames, old, targeted, 2)
    assert [r["name"] for r in extra] == ["opposite"]
    assert targeted == [original]
    assert explore_views(frames, old, targeted, 1) == []
    assert explore_views({}, old, [], 3) == []


def test_exploration_is_scale_invariant_and_rejects_invalid_poses():
    import copy
    import numpy as np
    from farm_runtime.quality.discovery_coverage import explore_views

    def frame(ts, x):
        t = np.eye(4)
        t[0, 3] = x
        return dict(frame_id=ts, T_world_cam=t.tolist())

    old = {"old": frame("0", 0)}
    frames = {"near": frame("1", 1), "far": frame("2", 10)}
    a = explore_views(frames, old, [], 2)
    scaled = copy.deepcopy(frames)
    for f in scaled.values():
        f["T_world_cam"][0][3] *= 100
    b = explore_views(scaled, old, [], 2)
    assert [r["name"] for r in a] == [r["name"] for r in b]
    assert [r["pose_novelty"] for r in a] == pytest.approx([r["pose_novelty"] for r in b], abs=1e-12)
    assert a[0]["name"] == "far"
    frames["far"]["T_world_cam"][0][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        explore_views(frames, old, [], 2)



def test_confirmation_target_schedules_third_capture_without_counting_stereo_twice():
    visibility = {"a": {1: .9}, "a_stereo": {1: 1.0},
                  "b": {1: .8}, "c": {1: .7}, "old": {1: 1.0}}
    timestamps = {"a": "1", "a_stereo": "1", "b": "2", "c": "3", "old": "0"}
    base, _ = choose_views(visibility, {1: {"0"}}, timestamps, 8)
    confirmed, covered = choose_views(
        visibility, {1: {"0"}}, timestamps, 8, target_timestamps=3
    )
    assert [r["name"] for r in base] == ["a_stereo"]
    assert [r["name"] for r in confirmed] == ["a_stereo", "b"]
    assert confirmed[1]["improved_group_ids"] == [1]
    assert confirmed[1]["marginal_score"] == pytest.approx(.8)
    assert covered == {1: 1.0}


def test_confirmation_obeys_budget_and_never_treats_existing_time_as_new():
    vis = {"same": {1: 1.0, 2: .9}, "next": {1: .8, 2: .8}, "other": {1: .7}}
    groups = {1: {"0", "1"}, 2: {"0"}}
    timestamps = {"same": "1", "next": "2", "other": "3"}
    selected, _ = choose_views(vis, groups, timestamps, 2, target_timestamps=3)
    assert [r["name"] for r in selected] == ["next", "same"]
    assert selected[1]["visibility"] == {2: .9}
    assert all(1 not in r["improved_group_ids"] for r in selected[1:])
    assert len({r["timestamp"] for r in selected}) == 2


def test_confirmation_target_does_not_invent_visibility_or_require_all_targets():
    selected, covered = choose_views(
        {"empty": {1: 0.0}}, {1: {"old"}}, {"empty": "new"}, 8, target_timestamps=4
    )
    assert selected == [] and covered == {1: 0.0}


@pytest.mark.parametrize("target", [True, 1, 5, 3.0, "3"])
def test_invalid_confirmation_target_is_rejected(target):
    with pytest.raises(ValueError, match="target timestamps"):
        choose_views({}, {}, {}, 8, target_timestamps=target)
