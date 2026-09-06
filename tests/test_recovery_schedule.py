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


def novelty_population():
    geometry = dict(nodes=[], groups=[])
    masks = {}
    for oid in range(6):
        geometry["nodes"].append(
            dict(
                id=oid,
                frame="build",
                representative_detection=oid,
                mask_pixels=100,
                eligible_surface_pixels=80 if oid == 1 else 90,
                sampled_points=80,
                source_priority=0,
            )
        )
        geometry["groups"].append(
            dict(
                id=oid,
                members=[oid],
                independent_timestamps=1,
                candidate_labels=[],
            )
        )
        masks[oid] = np.zeros((20, 20), bool)
        masks[oid][:10, 10:20] = oid == 1
        masks[oid][:10, :10] = oid != 1
    for nid, name in ((99, "build"), (100, "other_build")):
        geometry["nodes"].append(
            dict(
                id=nid,
                frame=name,
                representative_detection=0,
                mask_pixels=100,
                eligible_surface_pixels=90,
                sampled_points=80,
            )
        )
        masks[nid] = masks[0].copy()
    geometry["groups"].append(
        dict(
            id=99,
            members=[99, 100],
            independent_timestamps=2,
            candidate_labels=[],
        )
    )
    return geometry, masks


def test_new_region_precedes_duplicate_without_losing_strata_or_groups():
    geometry, masks = novelty_population()
    allowed = {"build", "other_build"}
    baseline, _ = recovery_candidates(geometry, 128, allowed)
    rows, total = recovery_candidates(
        geometry,
        128,
        allowed,
        mask_reader=lambda n: masks[n["id"]],
    )
    assert baseline[0]["group_id"] == 0 and rows[0]["group_id"] == 1
    assert total == 6
    assert {r["group_id"] for r in rows} == set(range(6))
    assert [(r["source_priority"], r["size_band"]) for r in rows] == [
        (r["source_priority"], r["size_band"]) for r in baseline
    ]
    duplicate = next(r for r in rows if r["group_id"] == 0)
    assert duplicate["multiview_overlap"] == dict(
        iou=1.0,
        group_id=99,
        node_id=99,
        candidate_coverage=1.0,
        reference_coverage=1.0,
    )
    assert duplicate["novel_support_score"] == 0
    assert (
        len(
            recovery_candidates(
                geometry,
                1,
                allowed,
                mask_reader=lambda n: masks[n["id"]],
            )[0]
        )
        == 1
    )


def test_small_nested_object_keeps_novelty_despite_complete_containment():
    geometry, masks = novelty_population()
    masks[0][:] = False
    masks[0][2:4, 2:4] = True
    rows, _ = recovery_candidates(
        geometry,
        128,
        {"build", "other_build"},
        mask_reader=lambda n: masks[n["id"]],
    )
    row = next(r for r in rows if r["group_id"] == 0)
    assert row["multiview_overlap"]["candidate_coverage"] == 1
    assert row["multiview_overlap"]["iou"] == pytest.approx(0.04)
    assert row["novel_support_score"] == pytest.approx(0.9 * 0.96)


def test_overlap_never_reads_forbidden_reference_or_candidate_views():
    geometry, masks = novelty_population()
    geometry["nodes"][5]["frame"] = "forbidden"
    reads = []

    def read(node):
        reads.append(node["id"])
        assert node["frame"] != "forbidden"
        return masks[node["id"]]

    # A partly disallowed multiview group is not coverage evidence.
    rows, total = recovery_candidates(geometry, 128, {"build"}, mask_reader=read)
    assert total == 5 and reads == []
    assert all(r["novel_support_score"] == r["eligible_surface_fraction"] for r in rows)


def test_same_pixels_in_another_frame_do_not_penalize_a_candidate():
    geometry, masks = novelty_population()
    geometry["nodes"][-2]["frame"] = "other_build"
    rows, _ = recovery_candidates(
        geometry,
        128,
        {"build", "other_build"},
        mask_reader=lambda _: pytest.fail("no same-frame reference"),
    )
    assert rows[0]["group_id"] == 0
    assert all(r["multiview_overlap"]["iou"] == 0 for r in rows)


def test_overlap_order_is_deterministic_and_decodes_each_used_mask_once():
    geometry, masks = novelty_population()
    reads = []

    def read(node):
        reads.append(node["id"])
        return masks[node["id"]]

    first = recovery_candidates(
        geometry, 128, {"build", "other_build"}, mask_reader=read
    )
    assert len(reads) == len(set(reads)) and 100 not in reads
    shuffled = copy.deepcopy(geometry)
    shuffled["nodes"].reverse()
    shuffled["groups"].reverse()
    assert first == recovery_candidates(
        shuffled,
        128,
        {"build", "other_build"},
        mask_reader=lambda n: masks[n["id"]],
    )


@pytest.mark.parametrize("invalid", ["shape", "dtype"])
def test_novelty_rejects_misaligned_or_nonbinary_masks(invalid):
    geometry, masks = novelty_population()
    masks[99] = (
        np.ones((19, 20), bool) if invalid == "shape" else masks[99].astype(float)
    )
    with pytest.raises(ValueError):
        recovery_candidates(
            geometry,
            128,
            {"build", "other_build"},
            mask_reader=lambda n: masks[n["id"]],
        )


def test_empty_multiview_mask_does_not_invent_coverage():
    geometry, masks = novelty_population()
    masks[99][:] = False
    rows, _ = recovery_candidates(
        geometry,
        128,
        {"build", "other_build"},
        mask_reader=lambda n: masks[n["id"]],
    )
    assert all(r["multiview_overlap"]["iou"] == 0 for r in rows)
