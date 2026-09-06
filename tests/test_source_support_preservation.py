"""Recovery may complete a surface without discarding visible source support."""

import numpy as np
import pytest

from farm_runtime.proposal_geometry import surface_points
from farm_runtime.quality import surface_validation as validation
from farm_runtime.surface_evidence import point_observation


def surface_case():
    frame = dict(
        depth=np.full((80, 80), 2.0),
        K=np.array([[40.0, 0, 40], [0, 40.0, 40], [0, 0, 1]]),
        T_world_cam=np.eye(4),
        excluded=np.zeros((80, 80), bool),
    )
    left = np.zeros((80, 80), bool)
    left[10:65, 8:35] = True
    right = np.zeros_like(left)
    right[15:45, 55:70] = True
    a, _ = surface_points(left, **frame)
    b, _ = surface_points(right, **frame)
    points = np.concatenate([a, b])
    core = np.arange(len(points)) < len(a)
    counts = [("large-source", len(a)), ("small-source", len(b))]
    args = (points, core, [left], [dict(label="surface", score=0.9)], frame, None, 0.05)
    return args, counts, left | right


@pytest.mark.parametrize("mode", ["ordinary", "partial", "completion"])
def test_supported_core_cannot_hide_loss_from_another_source(monkeypatch, mode):
    args, counts, _ = surface_case()
    kwargs = {}
    if mode == "partial":
        args = (args[0], np.ones(len(args[0]), bool), *args[2:])
        kwargs["partial_context"] = dict(
            target_name="target", references=[(dict(frame="source"), args[4])]
        )
        monkeypatch.setattr(
            validation,
            "compare_surfaces",
            lambda *a: dict(decision="match", reason="partial_view_surface_support"),
        )
    elif mode == "completion":
        kwargs["completion_preference"] = dict(
            candidate_indices=[0], fallback_detection=0
        )
    assert validation.select_observation(*args, **kwargs)[0] == 0
    evidence = point_observation(args[0], args[2][0], **args[4])
    # A global average would miss the loss of this less-sampled source.
    assert evidence["negative"].mean() < 0.35
    selected, rows, _ = validation.select_observation(
        *args, source_counts=counts, **kwargs
    )
    assert selected is None
    preservation = rows[0]["source_support_preservation"]
    assert not preservation[0]["source_foreground_contradicted"]
    assert preservation[1]["source_foreground_contradicted"]


@pytest.mark.parametrize(
    "unknown", ["occlusion", "excluded", "missing_depth", "outside"]
)
def test_unknown_source_support_never_vetoes_a_visible_match(unknown):
    args, counts, _ = surface_case()
    points, core, masks, detections, frame, labels, radius = args
    if unknown == "occlusion":
        frame["depth"][:, 50:] = 1
    elif unknown == "excluded":
        frame["excluded"][:, 50:] = True
    elif unknown == "missing_depth":
        frame["depth"][:, 50:] = np.nan
    else:
        points[len(points) - counts[1][1] :, 0] = 100
    selected, rows, _ = validation.select_observation(
        points, core, masks, detections, frame, labels, radius, source_counts=counts
    )
    assert selected == 0
    assert rows[0]["source_support_preservation"][1]["visible_points"] == 0
    assert not rows[0]["source_support_preservation"][1][
        "source_foreground_contradicted"
    ]


def test_valid_expansion_beyond_incomplete_sources_is_allowed():
    args, counts, full = surface_case()
    args = (*args[:2], [full], *args[3:])
    selected, rows, _ = validation.select_observation(*args, source_counts=counts)
    assert selected == 0
    assert not any(
        r["source_foreground_contradicted"]
        for r in rows[0]["source_support_preservation"]
    )


@pytest.mark.parametrize(
    "count,visible,negative,reject",
    [
        (100, 19, 19, False),
        (200, 20, 20, False),
        (100, 20, 7, False),
        (100, 20, 8, True),
        (200, 30, 30, True),
        (0, 0, 0, False),
    ],
)
def test_existing_visibility_and_background_thresholds(
    count, visible, negative, reject
):
    evidence = dict(
        visible=np.arange(count) < visible,
        negative=np.arange(count) < negative,
    )
    result = validation.source_support_conflicts(evidence, [("source", count)])
    assert result[0]["source_foreground_contradicted"] is reject


@pytest.mark.parametrize(
    "counts", [[("x", -1)], [("x", True)], [("x", 1.0)], [("x", 2)]]
)
def test_changed_source_partition_is_rejected(counts):
    evidence = dict(visible=np.ones(1, bool), negative=np.zeros(1, bool))
    with pytest.raises(ValueError):
        validation.source_support_conflicts(evidence, counts)
