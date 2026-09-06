"""Completion priority changes scope ranking, never geometric eligibility."""

import json

import numpy as np
import pytest

from farm_runtime.proposal_geometry import surface_points
from farm_runtime.quality.completion_policy import load_completion_preferences
from farm_runtime.quality.surface_validation import select_observation
from farm_runtime.quality_baseline import describe_file


def selection_case():
    frame = dict(
        depth=np.full((20, 20), 2.0),
        K=np.array([[10.0, 0, 10], [0, 10.0, 10], [0, 0, 1]]),
        T_world_cam=np.eye(4),
        excluded=np.zeros((20, 20), bool),
    )
    partial = np.zeros((20, 20), bool)
    partial[5:15, 5:15] = True
    full = np.zeros_like(partial)
    full[2:18, 2:18] = True
    points, _ = surface_points(partial, **frame)
    return frame, partial, full, points


def test_lower_similarity_complete_surface_can_win_only_after_identity_pass():
    frame, partial, full, points = selection_case()
    args = (
        points,
        np.ones(len(points), bool),
        [partial, full],
        [dict(label="object", score=0.9)] * 2,
        frame,
        None,
        0.05,
    )
    selected, rows, _ = select_observation(*args)
    assert selected == 0 and all(r["eligible"] for r in rows)
    assert rows[0]["score"] > rows[1]["score"]
    selected, prioritized, _ = select_observation(
        *args,
        completion_preference=dict(candidate_indices=[1], fallback_detection=0),
    )
    assert selected == 1
    assert prioritized == rows  # Identity evidence and thresholds are unchanged.


@pytest.mark.parametrize("kind", ["ineligible", "ambiguous", "invalid_fallback"])
def test_failed_completion_keeps_only_an_eligible_prior(kind):
    frame, partial, full, points = selection_case()
    bad = np.zeros_like(partial)
    bad[:3, :3] = True
    if kind == "ineligible":
        masks, preference = [partial, bad], [1]
    else:
        first, second = np.zeros_like(partial), np.zeros_like(partial)
        first[2:17, 2:17] = True
        second[3:18, 3:18] = True
        masks, preference = [partial if kind == "ambiguous" else bad, first, second], [
            1,
            2,
        ]
    selected, rows, _ = select_observation(
        points,
        np.ones(len(points), bool),
        masks,
        [dict(label="object", score=0.9)] * len(masks),
        frame,
        None,
        0.05,
        completion_preference=dict(candidate_indices=preference, fallback_detection=0),
    )
    assert selected == (None if kind == "invalid_fallback" else 0)
    if selected is not None:
        assert next(r for r in rows if r["detection_index"] == selected)["eligible"]


def policy_fixture(root):
    def write(name, doc):
        path = root / name
        path.write_text(json.dumps(doc))
        return path

    artifact = write("pixels.json", {})
    record = describe_file(artifact)
    audit = write("audit.json", {})

    def proposals(name, detections):
        return write(
            name,
            dict(
                test_opened=False,
                observations=[dict(name="view.png", detections=detections)],
            ),
        )

    base = proposals("base.json", [dict(label="object")] * 3)
    tracker = proposals(
        "tracker.json",
        [
            dict(label="object", source_group_id=7),
            dict(label="object", source_group_id=8),
        ],
    )
    extra = proposals("extra.json", [dict(label="object")])
    parent = write(
        "parent.json",
        dict(
            closed_test_opened=False,
            release_eligible=False,
            partial_view_association=True,
            source_audit=describe_file(audit),
            source_proposals=describe_file(base),
            supplements=[describe_file(tracker), describe_file(extra)],
            groups=[
                dict(
                    group_id=gid,
                    extra_matches=[
                        dict(
                            name="view.png",
                            selected_detection=index,
                            decision="matched_static_surface",
                            candidates=[dict(detection_index=index, eligible=True)],
                        )
                    ],
                )
                for gid, index in [(7, 3), (8, 4)]
            ],
        ),
    )
    parsed = dict(scope_complete=False, confidence="high")
    completion = dict(
        schema="farm.visible-scope-completion.v1",
        test_opened=False,
        closed_test_opened=False,
        release_eligible=False,
        source_validation=describe_file(parent),
        source_tracker=describe_file(tracker),
        observations=[
            dict(
                name="view.png",
                detections=[
                    dict(label="object", source_group_id=7),
                    dict(label="object", source_group_id=8),
                    dict(label="object", source_group_id=7),
                ],
            )
        ],
        outputs=[
            dict(
                group_id=gid,
                name="view.png",
                selected_detection=index,
                candidates=count,
                source_crop=record,
                source_image=record,
                source_mask=record,
                response=dict(
                    raw=json.dumps(parsed), parsed=parsed.copy(), validation_error=None
                ),
            )
            for gid, index, count in [(7, 3, 2), (8, 4, 1)]
        ],
    )
    path = write("completion.json", completion)
    return path, audit, base, [tracker, extra], parent


def test_completion_binds_each_object_to_its_own_flattened_candidate_indices(tmp_path):
    path, audit, base, supplements, parent = policy_fixture(tmp_path)
    assert load_completion_preferences(path, audit, base, supplements, True) == {
        (7, "view.png"): dict(candidate_indices=[6, 8], fallback_detection=3),
        (8, "view.png"): dict(candidate_indices=[7], fallback_detection=4),
    }


@pytest.mark.parametrize(
    "fault",
    [
        "parent_changed",
        "order",
        "policy",
        "raw",
        "other_object",
        "wrong_prior",
        "closed_test",
        "uncertain",
    ],
)
def test_completion_preference_rejects_changed_evidence_or_abstains(tmp_path, fault):
    path, audit, base, supplements, parent = policy_fixture(tmp_path)
    data = json.loads(path.read_text())
    policy = True
    if fault == "parent_changed":
        parent.write_text("{}")
    elif fault == "order":
        supplements = list(reversed(supplements))
    elif fault == "policy":
        policy = False
    elif fault == "raw":
        data["outputs"][0]["response"]["parsed"]["scope_complete"] = True
    elif fault == "other_object":
        data["observations"][0]["detections"][0]["source_group_id"] = 8
    elif fault == "wrong_prior":
        data["outputs"][0]["selected_detection"] = 4
    elif fault == "closed_test":
        data["closed_test_opened"] = True
    elif fault == "uncertain":
        parsed = dict(scope_complete=False, confidence="medium")
        data["outputs"][0]["response"].update(raw=json.dumps(parsed), parsed=parsed)
    path.write_text(json.dumps(data))
    if fault == "uncertain":
        result = load_completion_preferences(path, audit, base, supplements, policy)
        assert set(result) == {(8, "view.png")}
    else:
        with pytest.raises(ValueError):
            load_completion_preferences(path, audit, base, supplements, policy)


def test_completion_priority_does_not_retract_the_prior_visible_mask():
    frame, partial, full, points = selection_case()
    smaller = partial.copy()
    smaller[5:7] = False
    selected, rows, _ = select_observation(
        points,
        np.ones(len(points), bool),
        [partial, smaller],
        [dict(label="object", score=0.9)] * 2,
        frame,
        None,
        0.05,
        completion_preference=dict(candidate_indices=[1], fallback_detection=0),
    )
    assert all(r["eligible"] for r in rows)
    assert selected == 0


def test_ambiguous_seed_completion_has_no_accepted_prior_fallback():
    frame, partial, full, points = selection_case()
    args = (
        points,
        np.ones(len(points), bool),
        [partial, full],
        [dict(label="object", score=0.9)] * 2,
        frame,
        None,
        0.05,
    )
    preference = dict(candidate_indices=[1], seed_detection=0, fallback_detection=None)
    assert select_observation(*args, completion_preference=preference)[0] == 1
    # If the generated completion fails geometry, the eligible seed stays unresolved.
    bad = np.zeros_like(partial)
    bad[:3, :3] = True
    changed = (*args[:2], [partial, bad], *args[3:])
    chosen, rows, reason = select_observation(
        *changed, completion_preference=preference
    )
    assert chosen is None and reason == "ambiguous_competing_scope"
    assert rows[0]["eligible"]


@pytest.mark.parametrize(
    "fault", [None, "policy_missing", "wrong_seed", "prior_accepted"]
)
def test_ambiguous_completion_binds_policy_and_geometric_best(tmp_path, fault):
    path, audit, base, supplements, parent = policy_fixture(tmp_path)
    data = json.loads(parent.read_text())
    match = data["groups"][0]["extra_matches"][0]
    match.update(
        selected_detection=None,
        decision="ambiguous_competing_scope",
        candidates=[
            dict(detection_index=3, eligible=True, score=0.9),
            dict(detection_index=0, eligible=True, score=0.8),
        ],
    )
    if fault == "prior_accepted":
        match.update(selected_detection=3, decision="matched_static_surface")
    parent.write_text(json.dumps(data))
    completion = json.loads(path.read_text())
    completion["source_validation"] = describe_file(parent)
    completion["ambiguous_scope_proposals"] = True
    completion["outputs"][0]["ambiguous_seed"] = True
    if fault == "policy_missing":
        del completion["ambiguous_scope_proposals"]
    elif fault == "wrong_seed":
        completion["outputs"][0]["selected_detection"] = 0
    path.write_text(json.dumps(completion))
    if fault:
        with pytest.raises(ValueError):
            load_completion_preferences(path, audit, base, supplements, True)
    else:
        result = load_completion_preferences(path, audit, base, supplements, True)
        assert result[7, "view.png"] == dict(
            candidate_indices=[6, 8], seed_detection=3, fallback_detection=None
        )
        assert result[8, "view.png"]["fallback_detection"] == 4
