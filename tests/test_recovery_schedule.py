import copy
from types import SimpleNamespace

import numpy as np
import pytest

from farm_runtime.quality.recovery_schedule import recovery_candidates, budgeted_views
from farm_runtime.quality.surface_evidence import rank_views


def population():
    nodes, groups = [], []
    for source in (0, 1):
        for i in range(30):
            oid = len(nodes)
            nodes.append(
                dict(
                    id=oid,
                    frame="build",
                    representative_detection=0,
                    mask_pixels=100 * (i + 1),
                    eligible_surface_pixels=80 * (i + 1),
                    sampled_points=80,
                    source_priority=source,
                )
            )
            groups.append(
                dict(
                    id=oid,
                    members=[oid],
                    independent_timestamps=1,
                    candidate_labels=["arbitrary"],
                )
            )
    return dict(nodes=nodes, groups=groups)


def test_bounded_pool_spans_sources_and_sizes_without_semantic_scores():
    geometry = population()
    rows, total = recovery_candidates(geometry, 6, {"build"})
    assert len(rows) == 6 and total == 60
    assert {r["source_priority"] for r in rows} == {0, 1}
    assert {r["size_band"] for r in rows} == {0, 1, 2}
    other = copy.deepcopy(geometry)
    for group in other["groups"]:
        group["candidate_labels"] = ["wrong hallucinated label"]
    for node in other["nodes"]:
        node["score"] = 0.0001
    assert recovery_candidates(other, 6, {"build"}) == (rows, total)
    assert recovery_candidates(geometry, 6, set()) == ([], 0)


def test_pool_excludes_confirmed_and_unsupported_groups_and_accepts_old_source_schema():
    geometry = population()
    geometry["groups"][0]["independent_timestamps"] = 2
    geometry["nodes"][1]["sampled_points"] = 19
    geometry["nodes"][2]["eligible_surface_pixels"] = 19
    geometry["nodes"][3]["representative_detection"] = None
    geometry["nodes"][4]["frame"] = "unapproved"
    for node in geometry["nodes"]:
        node.pop("source_priority")
    rows, total = recovery_candidates(geometry, 128, {"build"})
    assert total == 55 and len(rows) == 55
    assert not set(range(5)) & {r["group_id"] for r in rows}
    assert {r["source_priority"] for r in rows} == {0}


@pytest.mark.parametrize("budget", [0, 129, True, 3.5])
def test_candidate_budget_is_bounded(budget):
    with pytest.raises(ValueError):
        recovery_candidates(population(), budget, {"build"})


def test_shared_rgb_budget_can_reuse_views_without_spending_a_new_slot():
    selected = [dict(name="new", timestamp="2"), dict(name="shared", timestamp="3")]
    assert budgeted_views(selected, {"shared"}, 1) == [selected[1]]
    assert budgeted_views(selected, {"shared"}, 2) == selected


def test_ranking_never_reads_unapproved_depth_or_counts_sibling_as_new_timestamp():
    xy = np.stack(
        np.meshgrid(np.linspace(-0.3, 0.3, 6), np.linspace(-0.3, 0.3, 6)), -1
    ).reshape(-1, 2)
    points = np.column_stack([xy, np.full(len(xy), 2.0)])
    reads = []

    def frame(name):
        reads.append(name)
        assert name == "new"
        return dict(
            depth=np.full((20, 20), 2.0),
            K=np.array([[10.0, 0, 10], [0, 10, 10], [0, 0, 1]]),
            T_world_cam=np.eye(4),
            excluded=np.zeros((20, 20), bool),
        )

    inputs = SimpleNamespace(
        frames={
            "restricted": dict(frame_id="3"),
            "sibling": dict(frame_id="1"),
            "new": dict(frame_id="2"),
        },
        frame=frame,
        transients={},
    )
    candidates, selected = rank_views(
        points,
        np.ones(len(points), bool),
        {"1"},
        inputs,
        2,
        allowed_names={"sibling", "new"},
    )
    assert reads == ["new"]
    assert [r["name"] for r in selected] == ["new"]
    assert candidates[0]["core_visible_fraction"] == 1


def test_group_bound_tracker_cannot_win_for_another_object():
    from farm_runtime.proposal_geometry import surface_points
    from farm_runtime.quality.surface_validation import select_observation

    f = dict(
        depth=np.full((20, 20), 2.0),
        K=np.array([[10.0, 0, 10], [0, 10, 10], [0, 0, 1]]),
        T_world_cam=np.eye(4),
        excluded=np.zeros((20, 20), bool),
    )
    mask = np.zeros((20, 20), bool)
    mask[3:17, 3:17] = True
    points, _ = surface_points(mask, **f)
    detections = [
        dict(label="projected target", score=0.99, source_group_id=9),
        dict(label="generic hypothesis", score=0.8),
    ]
    index, rows, _ = select_observation(
        points,
        np.ones(len(points), bool),
        [mask, mask],
        detections,
        f,
        None,
        0.05,
        group_id=7,
    )
    assert index == 1 and [r["detection_index"] for r in rows] == [1]
    index, rows, _ = select_observation(
        points,
        np.ones(len(points), bool),
        [mask],
        detections[:1],
        f,
        None,
        0.05,
        group_id=7,
    )
    assert index is None and rows == []
