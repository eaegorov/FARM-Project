import copy
from types import SimpleNamespace

import numpy as np
import pytest

from farm_runtime.quality.recovery_schedule import (
    coverage_candidates,
    candidate_cohorts,
)
from farm_runtime.quality.surface_evidence import main


def coverage_inputs(missing=((80, 0),), counts=(100, 100), stereo=False):
    nodes, groups, clouds, masks, frames = [], [], {}, {}, {}
    reads = []
    for gid, fractions in enumerate(missing):
        members = []
        for index, timestamp in enumerate(("a", "b")):
            count = counts[index]
            absent = round(count * fractions[index] / 100)
            points = np.concatenate(
                [
                    np.tile([0.0, 0, 2], (count - absent, 1)),
                    np.tile([-2.0 if index == 0 else 2.0, 0, 2], (absent, 1)),
                ]
            )
            for sibling in range(2 if stereo and index == 0 else 1):
                nid = len(nodes)
                name = f"group{gid}_{timestamp}_{sibling}"
                members.append(nid)
                nodes.append(
                    dict(
                        id=nid,
                        frame=name,
                        timestamp=timestamp,
                        representative_detection=0,
                        mask_pixels=2500,
                        source_priority=0,
                    )
                )
                clouds[nid] = points
                depth = np.full((50, 50), 2.0)
                # The other timestamp's missing surface is outside valid depth.
                depth[:, 32:39] = 0 if index == 0 else 2
                depth[:, 12:19] = 0 if index == 1 else 2
                frames[name] = dict(
                    depth=depth,
                    K=np.array([[10.0, 0, 25], [0, 10.0, 25], [0, 0, 1]]),
                    T_world_cam=np.eye(4),
                    excluded=np.zeros((50, 50), bool),
                )
                masks[nid] = np.ones((50, 50), bool)
        groups.append(
            dict(
                id=gid,
                members=members,
                independent_timestamps=2,
                candidate_labels=["untrusted label"],
            )
        )
    by_id = {n["id"]: n for n in nodes}

    def support(group):
        reads.append(("support", group["id"]))
        members = [by_id[i] for i in group["members"]]
        points = np.concatenate([clouds[n["id"]] for n in members])
        timestamps = np.concatenate(
            [np.full(len(clouds[n["id"]]), n["timestamp"]) for n in members]
        )
        return members, points, timestamps, np.full(len(points), 0.025)

    def mask(node):
        reads.append(("mask", node["frame"]))
        return masks[node["id"]]

    return SimpleNamespace(
        geometry=dict(groups=groups, nodes=nodes),
        nodes=by_id,
        support=support,
        mask=mask,
        frame=lambda name: frames[name],
        allowed=set(frames),
        reads=reads,
    )


def test_sparse_distant_evidence_is_not_drowned_by_dense_close_view_or_stereo():
    expected = None
    for counts, stereo in [
        ((100, 100), False),
        ((100, 1000), False),
        ((100, 1000), True),
    ]:
        inputs = coverage_inputs(counts=counts, stereo=stereo)
        rows, total = coverage_candidates(inputs, 8, inputs.allowed)
        assert total == 1
        row = rows[0]
        values = {
            k: row[k]
            for k in (
                "worst_timestamp_unconfirmed_fraction",
                "balanced_unconfirmed_fraction",
                "corroborated_anchor_fraction",
                "coverage_gain_score",
            )
        }
        assert values == pytest.approx(
            dict(
                worst_timestamp_unconfirmed_fraction=0.8,
                balanced_unconfirmed_fraction=0.4,
                corroborated_anchor_fraction=0.6,
                coverage_gain_score=0.48,
            )
        )
        if expected is not None:
            assert values == expected
        expected = values


def test_supported_missing_region_precedes_nearly_unanchored_group():
    inputs = coverage_inputs(missing=((80, 0), *((99, 99),) * 5))
    rows, total = coverage_candidates(inputs, 128, inputs.allowed)
    assert total == 6 and rows[0]["group_id"] == 0
    assert (
        rows[0]["coverage_gain_score"]
        > next(r for r in rows if r["group_id"] == 1)["coverage_gain_score"]
    )
    # No hard exclusion, class preference or detector score dependency.
    assert {r["group_id"] for r in rows} == set(range(6))
    for group in inputs.geometry["groups"]:
        group["candidate_labels"] = ["another hallucination"]
    for node in inputs.nodes.values():
        node["score"] = 1e-6
    assert coverage_candidates(inputs, 128, inputs.allowed) == (rows, total)


@pytest.mark.parametrize(
    "missing,count,eligible",
    [
        (24, 100, False),
        (25, 100, True),
        (50, 38, False),
        (50, 40, True),
        (0, 100, False),
    ],
)
def test_missing_support_needs_both_fraction_and_sample_count(missing, count, eligible):
    inputs = coverage_inputs(((missing, 0),), counts=(count, 100))
    rows, total = coverage_candidates(inputs, 8, inputs.allowed)
    assert bool(rows) is eligible and total == int(eligible)


def test_no_group_evidence_is_read_if_any_member_is_outside_allowlist():
    inputs = coverage_inputs()
    inputs.allowed.remove(inputs.nodes[1]["frame"])
    assert coverage_candidates(inputs, 8, inputs.allowed) == ([], 0)
    assert inputs.reads == []


def test_single_timestamp_and_unusable_groups_do_not_enter_coverage():
    inputs = coverage_inputs(missing=((80, 0), (80, 0), (80, 0)))
    inputs.geometry["groups"][0]["independent_timestamps"] = 1
    inputs.nodes[inputs.geometry["groups"][1]["members"][0]][
        "representative_detection"
    ] = None
    rows, total = coverage_candidates(inputs, 8, inputs.allowed)
    assert total == 1 and rows[0]["group_id"] == 2
    assert [v for kind, v in inputs.reads if kind == "support"] == [2]


def test_candidate_cohorts_alternate_and_cannot_overlap():
    singles = [dict(group_id=i) for i in (1, 2, 3)]
    coverage = [dict(group_id=i) for i in (4, 5)]
    rows = candidate_cohorts(singles, coverage)
    assert [r["group_id"] for r in rows] == [1, 4, 2, 5, 3]
    assert [r["candidate_kind"] for r in rows] == [
        "single_timestamp",
        "incomplete_multiview",
        "single_timestamp",
        "incomplete_multiview",
        "single_timestamp",
    ]
    assert candidate_cohorts([], []) == []
    with pytest.raises(ValueError, match="disjoint"):
        candidate_cohorts(singles, [dict(group_id=2)])


@pytest.mark.parametrize("single,coverage", [(0, 0), (-1, 4), (8, 9), (0, -1)])
def test_invalid_combined_budget_is_rejected_before_reading_inputs(
    tmp_path, single, coverage
):
    with pytest.raises(ValueError, match="budgets"):
        main(
            [
                "--geometry",
                str(tmp_path / "absent.json"),
                "--plan",
                str(tmp_path / "absent_plan.json"),
                "--auto-groups",
                str(single),
                "--coverage-groups",
                str(coverage),
                "--output",
                str(tmp_path / "out"),
            ]
        )


def test_manual_group_mode_cannot_silently_ignore_coverage_quota(tmp_path):
    with pytest.raises(ValueError, match="requires --auto-groups"):
        main(
            [
                "--geometry",
                str(tmp_path / "absent.json"),
                "--plan",
                str(tmp_path / "absent_plan.json"),
                "--group-id",
                "3",
                "--coverage-groups",
                "1",
                "--output",
                str(tmp_path / "out"),
            ]
        )
