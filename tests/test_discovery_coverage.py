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
