import copy
import json

import numpy as np
import pytest

from farm_runtime.quality.scope_review import (
    consensus_choice,
    load_scope_choices,
    validate_response,
)
from farm_runtime.quality_baseline import describe_file


def responses(choice=1):
    parsed = dict(candidate_id=choice, confidence="high", reason="same whole object")
    return [
        dict(
            order=order,
            raw=json.dumps(parsed),
            parsed=parsed.copy(),
            validation_error=None,
        )
        for order in ([1, 3], [3, 1])
    ]


def test_consensus_requires_independent_orders_and_valid_high_choices():
    assert consensus_choice(responses(), [1, 3]) == 1
    assert consensus_choice(responses(None), [1, 3]) is None
    for alteration in ("disagree", "medium", "same_order", "error"):
        rows = responses()
        if alteration == "disagree":
            rows[1]["parsed"]["candidate_id"] = 3
        if alteration == "medium":
            rows[1]["parsed"]["confidence"] = "medium"
        if alteration == "same_order":
            rows[1]["order"] = [1, 3]
        if alteration == "error":
            rows[1]["validation_error"] = "invalid JSON"
        rows[1]["raw"] = json.dumps(rows[1]["parsed"])
        assert consensus_choice(rows, [1, 3]) is None
    rows = responses()
    rows[1]["parsed"]["candidate_id"] = 3
    with pytest.raises(ValueError, match="raw"):
        consensus_choice(rows, [1, 3])
    rows = responses()
    rows[1]["order"] = [1, 4]
    with pytest.raises(ValueError, match="candidate mismatch"):
        consensus_choice(rows, [1, 3])


@pytest.mark.parametrize("choice", [True, 2, "1"])
def test_invalid_candidate_cannot_be_vlm_identity(choice):
    with pytest.raises(ValueError, match="geometrically eligible"):
        validate_response(
            json.dumps(dict(candidate_id=choice, confidence="high", reason="x")), [1, 3]
        )


def test_scope_review_binds_sources_policy_order_and_candidates(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path

    audit = write("audit.json", {})
    proposals = write("proposals.json", {})
    first = write("first.json", {})
    second = write("second.json", {})
    evidence = write("image.json", {})
    descriptor = describe_file(evidence)
    candidates = [dict(detection_index=i, eligible=True) for i in [1, 3]]
    baseline = dict(
        closed_test_opened=False,
        release_eligible=False,
        partial_view_association=True,
        source_audit=describe_file(audit),
        source_proposals=describe_file(proposals),
        supplements=[describe_file(first), describe_file(second)],
        groups=[
            dict(
                group_id=0,
                extra_matches=[
                    dict(
                        name="view.png",
                        decision="ambiguous_competing_scope",
                        candidates=candidates,
                    )
                ],
            )
        ],
    )
    baseline_path = write("validation.json", baseline)
    rows = responses()
    for row in rows:
        row["canvas"] = descriptor
    review = dict(
        schema="farm.cross-view-scope-review.v1",
        closed_test_opened=False,
        release_eligible=False,
        validation=describe_file(baseline_path),
        prompt=descriptor,
        model_config=descriptor,
        reviews=[
            dict(
                group_id=0,
                name="view.png",
                eligible_candidates=[1, 3],
                reference_image=descriptor,
                reference_canvas=descriptor,
                responses=rows,
            )
        ],
    )
    path = write("review.json", review)

    def load(**kwargs):
        return load_scope_choices(
            path,
            kwargs.get("audit", audit),
            proposals,
            kwargs.get("supplements", [first, second]),
            kwargs.get("partial_view", True),
        )

    assert load()[0, "view.png"]["choice"] == 1
    with pytest.raises(ValueError, match="source mismatch"):
        load(audit=first)
    with pytest.raises(ValueError, match="order mismatch"):
        load(supplements=[second, first])
    with pytest.raises(ValueError, match="policy"):
        load(partial_view=False)
    changed = copy.deepcopy(review)
    changed["reviews"][0]["eligible_candidates"] = [1, 4]
    write("review.json", changed)
    with pytest.raises(ValueError, match="candidates differ"):
        load()
    write("review.json", review)
    baseline_path.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        load()


def test_scope_choice_can_only_resolve_the_existing_geometric_best():
    from farm_runtime.proposal_geometry import surface_points
    from farm_runtime.quality.surface_validation import select_observation

    frame = dict(
        depth=np.full((20, 20), 2.0),
        K=np.array([[10.0, 0, 10], [0, 10.0, 10], [0, 0, 1]]),
        T_world_cam=np.eye(4),
        excluded=np.zeros((20, 20), bool),
    )
    full = np.zeros((20, 20), bool)
    full[1:19, 1:19] = True
    points, _ = surface_points(full, **frame)
    core = (np.abs(points[:, :2]) <= 0.5).all(axis=1)
    first = np.zeros_like(full)
    first[2:16, 2:16] = True
    second = np.zeros_like(full)
    second[4:18, 4:18] = True
    empty = np.zeros_like(full)
    args = (
        points,
        core,
        [first, second, empty],
        [dict(label="object", score=0.9)] * 3,
        frame,
        None,
        0.05,
    )
    selected, rows, decision = select_observation(*args)
    assert selected is None and decision == "ambiguous_competing_scope"
    best = sorted(
        (r for r in rows if r["eligible"]),
        key=lambda r: (-r["score"], r["detection_index"]),
    )[0]["detection_index"]
    assert (
        select_observation(*args, scope_review=dict(choice=best, candidates=rows))[0]
        == best
    )
    for choice in (None, 1 - best, 2):
        assert (
            select_observation(
                *args, scope_review=dict(choice=choice, candidates=rows)
            )[0]
            is None
        )
    changed = copy.deepcopy(rows)
    changed[0]["score"] += 0.01
    with pytest.raises(ValueError, match="evidence changed"):
        select_observation(*args, scope_review=dict(choice=best, candidates=changed))


def test_empty_scope_schedule_does_not_load_a_model(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from farm_runtime.quality import scope_review

    def write(name, data):
        path = tmp_path / name
        path.write_text(json.dumps(data))
        return path

    geometry = write("geometry.json", {})
    audit = write("audit.json", dict(source_geometry=describe_file(geometry)))
    proposals = write("proposals.json", {})
    validation = write(
        "validation.json",
        dict(
            closed_test_opened=False,
            release_eligible=False,
            source_audit=describe_file(audit),
            source_proposals=describe_file(proposals),
            groups=[],
        ),
    )
    monkeypatch.setattr(
        scope_review,
        "SurfaceInputs",
        lambda p: SimpleNamespace(geometry=dict(groups=[])),
    )
    monkeypatch.setattr(scope_review, "read_observations", lambda p: ({}, []))
    output = tmp_path / "output"
    assert (
        scope_review.main(
            [
                "--validation",
                str(validation),
                "--model",
                str(tmp_path / "absent-model"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads((output / "manifest.json").read_text())
    assert result["reviews"] == [] and result["model_config"] is None
    assert result["model_load_seconds"] == 0
