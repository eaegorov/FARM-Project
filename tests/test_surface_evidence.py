import numpy as np

from farm_runtime.proposal_geometry import project_evidence
from farm_runtime.surface_evidence import (
    aggregate_timestamps,
    point_observation,
    surface_components,
)


def frame():
    return dict(
        depth=np.full((20, 20), 2.0),
        K=np.array([[10.0, 0, 10], [0, 10.0, 10], [0, 0, 1]]),
        T_world_cam=np.eye(4),
        excluded=np.zeros((20, 20), bool),
    )


def test_visible_background_guard_and_missing_instance():
    f = frame()
    mask = np.zeros((20, 20), bool)
    mask[8:13, 8:13] = True
    points = np.array([[0, 0, 2], [0.8, 0, 2], [1.4, 0, 2], [0, 0, 3], [0, 0, 1]])
    e = point_observation(points, mask, **f)
    assert e["positive"].tolist() == [True, False, False, False, False]
    assert e["negative"].tolist() == [False, False, True, False, False]
    assert e["occluded"][3] and e["in_front_of_surface"][4]
    for missing in (None, np.zeros_like(mask)):
        e = point_observation(points, missing, **f)
        assert e["unknown"].all() and not e["negative"].any()


def test_unknown_depth_exclusion_outside_never_vote():
    f = frame()
    mask = np.ones((20, 20), bool)
    p = np.array([[0, 0, 2], [20, 0, 2]])
    for kind in ("depth", "excluded"):
        changed = {k: v.copy() for k, v in f.items()}
        changed[kind][:] = 0 if kind == "depth" else True
        e = point_observation(p, mask, **changed)
        assert e["unknown"].all() and not e["positive"].any()


def test_optional_indices_preserve_default_projection_counts():
    f = frame()
    mask = np.ones((20, 20), bool)
    p = np.array([[0, 0, 2], [0, 0, 3], [0, 0, 1], [20, 0, 2]])
    old = project_evidence(p, mask, **f)
    new = project_evidence(p, mask, **f, include_point_indices=True)
    assert old == {k: new[k] for k in old}
    assert new["surface_indices"].tolist() == [0]
    assert new["surface_pixels_xy"].tolist() == [[10, 10]]
    assert new["occluded_indices"].tolist() == [1]
    assert new["in_front_of_surface_indices"].tolist() == [2]


def test_stereo_votes_cannot_multiply_and_conflicts_are_explicit():
    positive = dict(positive=np.array([1, 1], bool), negative=np.array([0, 0], bool))
    negative = dict(positive=np.array([0, 0], bool), negative=np.array([1, 1], bool))
    e = aggregate_timestamps(
        [("a", positive), ("a", positive), ("b", negative), ("b", negative)], 2
    )
    assert e["positive"].tolist() == [1, 1] and e["negative"].tolist() == [1, 1]
    assert not e["contradicted"].any()
    e = aggregate_timestamps([("a", positive), ("b", negative), ("c", negative)], 2)
    assert e["contradicted"].all()
    e = aggregate_timestamps(
        [("a", positive), ("b", negative), ("b", positive), ("c", negative)], 2
    )
    assert e["conflicted"].tolist() == [1, 1] and not e["contradicted"].any()


def test_source_timestamp_counts_once_and_unobserved_parts_survive():
    obs = dict(positive=np.array([1, 0], bool), negative=np.array([0, 0], bool))
    e = aggregate_timestamps([("a", obs)], 2, ["a", "a"])
    assert e["positive"].tolist() == [1, 1]
    assert not e["contradicted"].any() and not e["corroborated"].any()


def test_disconnected_handle_is_unverified_never_automatically_removed():
    p = np.array([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [1, 0, 0], [1.01, 0, 0]])
    labels, rows = surface_components(p, ["a", "b", "a", "a", "a"], np.full(5, 0.025))
    assert len(rows) == 2 and labels[0] != labels[3]
    assert rows[0]["points"] == 3 and rows[1]["role"] == "disconnected_unverified"
    assert all(not r["automatic_removal_allowed"] for r in rows)


def test_new_view_identity_requires_geometry_not_matching_label():
    from farm_runtime.proposal_geometry import surface_points
    from farm_runtime.quality.surface_validation import select_observation

    f = frame()
    mask = np.zeros((20, 20), bool)
    mask[3:17, 3:17] = True
    points, _ = surface_points(mask, **f)
    core = np.ones(len(points), bool)
    detections = [dict(label="cart", score=0.9)]
    selected, _, decision = select_observation(
        points, core, [mask], detections, f, ["cart"], 0.05
    )
    assert selected == 0 and decision == "matched_static_surface"
    selected, _, _ = select_observation(
        points + np.array([10, 0, 0]), core, [mask], detections, f, ["cart"], 0.05
    )
    assert selected is None
    selected, _, _ = select_observation(points, core, [], [], f, ["cart"], 0.05)
    assert selected is None


def test_review_does_not_change_evidence_arrays(tmp_path):
    from PIL import Image
    from farm_runtime.quality.surface_validation import review_panel
    from farm_runtime.quality_baseline import describe_file

    path = tmp_path / "source.png"
    Image.new("RGB", (20, 20)).save(path)
    points = np.array([[0, 0, 2], [20, 0, 2.0]])
    counts = dict(
        corroborated=np.array([True, True]), contradicted=np.array([False, False])
    )
    expected = {k: v.copy() for k, v in counts.items()}
    review_panel(
        points,
        counts,
        np.ones((20, 20), bool),
        dict(source_image=describe_file(path), applied_quarter_turns=0),
        frame(),
        tmp_path / "review.jpg",
        "audit",
    )
    for k in counts:
        np.testing.assert_array_equal(counts[k], expected[k])


def test_equally_supported_competing_scopes_stay_ambiguous():
    from farm_runtime.proposal_geometry import surface_points
    from farm_runtime.quality.surface_validation import select_observation

    f = frame()
    full = np.zeros((20, 20), bool)
    full[1:19, 1:19] = True
    points, _ = surface_points(full, **f)
    core = (np.abs(points[:, :2]) <= 0.5).all(axis=1)
    first = np.zeros_like(full)
    first[2:16, 2:16] = True
    second = np.zeros_like(full)
    second[4:18, 4:18] = True
    selected, _, decision = select_observation(
        points,
        core,
        [first, second],
        [dict(label="cabinet", score=0.9), dict(label="machine", score=0.9)],
        f,
        None,
        0.05,
    )
    assert selected is None and decision == "ambiguous_competing_scope"


def test_adaptive_views_are_independent_timestamps_and_visible():
    from types import SimpleNamespace
    from farm_runtime.proposal_geometry import surface_points
    from farm_runtime.quality.surface_evidence import rank_views

    f = frame()
    mask = np.zeros((20, 20), bool)
    mask[2:18, 2:18] = True
    points, _ = surface_points(mask, **f)
    core = np.ones(len(points), bool)
    core[-10:] = False
    frames = {
        n: dict(frame_id=t)
        for n, t in [
            ("old", "old"),
            ("a_left", "a"),
            ("a_right", "a"),
            ("b", "b"),
            ("occluded", "c"),
        ]
    }

    def get_frame(name):
        result = {k: v.copy() for k, v in f.items()}
        if name == "occluded":
            result["depth"][:] = 1
        return result

    inputs = SimpleNamespace(frames=frames, frame=get_frame, transients={})
    candidates, selected = rank_views(points, core, {"old"}, inputs, 3)
    assert {r["timestamp"] for r in selected} == {"a", "b"}
    assert len(selected) == 2 and "occluded" not in {r["name"] for r in candidates}


def test_existing_surface_depth_cannot_change_after_sampling(tmp_path):
    import pytest
    from farm_runtime.quality.surface_evidence import SurfaceInputs
    from farm_runtime.quality_baseline import describe_file

    path = tmp_path / "depth.npy"
    np.save(path, np.full((20, 20), 2.0, np.float32))
    descriptor = describe_file(path)
    np.save(path, np.full((20, 20), 3.0, np.float32))
    inputs = SurfaceInputs.__new__(SurfaceInputs)
    inputs._frames = {}
    inputs.frames = {"source": dict(depth_path="depth.npy")}
    inputs.frames_path = tmp_path / "frames.json"
    inputs.recorded_depths = {"source": descriptor}
    with pytest.raises(ValueError):
        inputs.frame("source")


def partial_validation_case():
    from farm_runtime.proposal_geometry import surface_points
    from farm_runtime.quality.surface_validation import prepared_surface

    whole_frame = dict(
        depth=np.full((40, 40), 2.0, np.float32),
        K=np.array([[40.0, 0, 20], [0, 40.0, 20], [0, 0, 1.0]]),
        T_world_cam=np.eye(4),
        excluded=np.zeros((40, 40), bool),
    )
    clipped_frame = {k: v.copy() for k, v in whole_frame.items()}
    clipped_frame["T_world_cam"][0, 3] = 1.6
    whole = np.zeros((40, 40), bool)
    whole[5:35, 5:35] = True
    clipped = np.zeros_like(whole)
    clipped[5:35, :3] = True
    points, metrics = surface_points(clipped, **clipped_frame)
    reference = prepared_surface(
        "clipped", points, clipped, metrics["pixel_footprint_m"]
    )
    context = dict(target_name="whole", references=[(reference, clipped_frame)])
    return points, whole, whole_frame, context


def test_additional_full_view_can_confirm_an_edge_clipped_source():
    from farm_runtime.quality.surface_validation import select_observation

    points, mask, frame, context = partial_validation_case()
    args = (
        points,
        np.ones(len(points), bool),
        [mask],
        [dict(label="cabinet", score=0.8)],
        frame,
        None,
        0.1,
    )
    assert select_observation(*args)[0] is None
    selected, rows, reason = select_observation(*args, partial_context=context)
    assert selected == 0 and reason == "matched_static_surface"
    assert rows[0]["reciprocal_surface_agreement"] < 0.3
    assert (
        rows[0]["partial_view_evidence"][0]["reason"] == "partial_view_surface_support"
    )


def test_partial_additional_view_preserves_nested_scope_guard():
    from farm_runtime.quality.surface_validation import select_observation

    points, mask, frame, context = partial_validation_case()
    context["references"][0][0]["scope_containers"] = [99]
    assert (
        select_observation(
            points,
            np.ones(len(points), bool),
            [mask],
            [dict(label="cabinet", score=0.8)],
            frame,
            None,
            0.1,
            partial_context=context,
        )[0]
        is None
    )


def test_partial_additional_view_keeps_visible_background_and_exclusions():
    from farm_runtime.quality.surface_validation import select_observation
    from scipy.ndimage import binary_dilation

    for mode in ("background", "excluded", "missing"):
        points, mask, frame, context = partial_validation_case()
        reference, source_frame = context["references"][0]
        if mode == "background":
            reference["mask"][20:] = False
            reference["projection_mask"] = binary_dilation(
                reference["mask"], iterations=1
            )
        elif mode == "excluded":
            source_frame["excluded"][:] = True
        else:
            source_frame["depth"][:] = 0
        assert (
            select_observation(
                points,
                np.ones(len(points), bool),
                [mask],
                [dict(label="cabinet", score=0.8)],
                frame,
                None,
                0.1,
                partial_context=context,
            )[0]
            is None
        )
