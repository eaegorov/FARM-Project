import numpy as np

from farm_runtime.proposal_geometry import (
    GeometryPolicy,
    associate,
    equivalent_masks,
    project_evidence,
    stable_depth,
    surface_points,
)


def scene():
    depth = np.full((40, 40), 2.0, dtype=np.float32)
    K = np.array([[40.0, 0, 20], [0, 40.0, 20], [0, 0, 1.0]])
    return dict(
        depth=depth, K=K, T_world_cam=np.eye(4), excluded=np.zeros((40, 40), bool)
    )


def node(frame, mask, data, label="object"):
    points, metrics = surface_points(mask, **data)
    return dict(
        frame=frame,
        timestamp=frame,
        mask=mask,
        points=points,
        labels=[label],
        **metrics,
    )


def test_metric_backprojection_and_exclusion_are_not_background():
    f = scene()
    f["T_world_cam"][:3, 3] = [1.0, 2.0, 3.0]
    mask = np.zeros((40, 40), bool)
    mask[15:26, 15:26] = True
    points, metrics = surface_points(mask, **f)
    np.testing.assert_allclose(np.median(points, axis=0), [1.0, 2.0, 5.0])
    f["excluded"][:] = True
    points, metrics = surface_points(mask, **f)
    assert len(points) == 0
    assert metrics["excluded_pixels"] == 121


def test_depth_discontinuity_is_not_a_surface_between_objects():
    depth = np.full((20, 20), 2.0, np.float32)
    depth[:, 10:] = 4.0
    stable = stable_depth(depth)
    assert not stable[:, 9:11].any()
    assert stable[:, :9].all() and stable[:, 11:].all()


def test_occlusion_missing_depth_and_exclusion_are_unknown():
    f = scene()
    points = np.array([[0.0, 0.0, 3.0]])
    mask = np.zeros((40, 40), bool)
    evidence = project_evidence(points, mask, **f)
    assert evidence["occluded"] == 1
    assert evidence["mask_opposed"] == 0
    f["depth"][:] = 0
    assert project_evidence(points, mask, **f)["known_depth"] == 0
    f["depth"][:] = 3
    f["excluded"][:] = True
    assert project_evidence(points, mask, **f)["known_depth"] == 0


def test_visible_surface_outside_mask_is_negative():
    f = scene()
    mask = np.zeros((40, 40), bool)
    evidence = project_evidence(np.array([[0.0, 0.0, 2.0]]), mask, **f)
    assert evidence["mask_opposed"] == 1
    assert evidence["surface_visible"] == 1


def test_duplicate_chain_cannot_bridge_different_masks():
    masks = []
    for shift in (0, 5, 10):
        mask = np.zeros((1, 120), bool)
        mask[:, shift : shift + 100] = True
        masks.append(mask)
    assert equivalent_masks(masks, [1.0, 0.9, 0.8], 0.90) == [[0, 1], [2]]


def test_same_surface_matches_despite_conflicting_category_names():
    f = scene()
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    nodes = [node("a", mask, f, "extinguisher"), node("b", mask, f, "motor")]
    groups, evidence = associate(nodes, {"a": f, "b": f})
    assert groups == [[0, 1]]
    assert evidence["mutual_edges"][0]["component_merge_allowed"]


def test_same_label_and_pixels_at_remote_world_location_do_not_merge():
    f, g = scene(), scene()
    g["T_world_cam"][0, 3] = 10
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    groups, _ = associate([node("a", mask, f), node("b", mask, g)], {"a": f, "b": g})
    assert groups == [[0], [1]]


def test_same_view_scope_alternatives_do_not_union():
    f = scene()
    mask = np.zeros((40, 40), bool)
    mask[8:32, 8:32] = True
    part = np.zeros_like(mask)
    part[8:20, 8:32] = True
    groups, evidence = associate([node("a", mask, f), node("a", part, f)], {"a": f})
    assert groups == [[0], [1]]
    assert len(evidence["scope_alternatives"]) == 1


def test_component_cannot_link_survives_transitive_bridge(monkeypatch):
    import farm_runtime.proposal_geometry as module

    f = scene()
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    nodes = [node("a", mask, f), node("b", mask, f), node("c", mask, f)]

    def compare(a, b, frames, policy):
        conflict = (a["frame"], b["frame"]) == ("a", "c")
        return dict(
            decision="separate" if conflict else "match",
            reason="visible_mask_conflict" if conflict else "mutual_surface_support",
            score=0.9 if a["frame"] == "a" else 0.8,
        )

    monkeypatch.setattr(module, "compare_surfaces", compare)
    groups, evidence = associate(nodes, {name: f for name in "abc"})
    assert groups == [[0, 1], [2]]
    assert [e["component_merge_allowed"] for e in evidence["mutual_edges"]] == [
        True,
        False,
    ]


def cropped_plane():
    f, g = scene(), scene()
    g["T_world_cam"][0, 3] = 1.6
    whole = np.zeros((40, 40), bool)
    whole[5:35, 5:35] = True
    clipped = np.zeros_like(whole)
    clipped[5:35, :3] = True
    return [node("a", whole, f), node("b", clipped, g)], {"a": f, "b": g}


def test_partial_fov_recovers_clipped_plane_without_changing_default():
    nodes, frames = cropped_plane()
    groups, baseline = associate(nodes, frames)
    assert groups == [[0], [1]]
    groups, evidence = associate(
        nodes, frames, GeometryPolicy(partial_view_association=True)
    )
    assert groups == [[0, 1]]
    edge = evidence["mutual_edges"][0]
    assert edge["reason"] == "partial_view_surface_support"
    assert edge["directions"][0]["visible_fraction"] < 0.15
    assert edge["partial_view_directions"] == [True, False]
    assert min(edge["partial_surface_agreement"]) > 0.9


def test_partial_fov_keeps_visible_background_conflict():
    nodes, frames = cropped_plane()
    # The crop now observes only the top half of the candidate. The rest is
    # reconstructed visible background, not missing support outside the image.
    mask = nodes[1]["mask"].copy()
    mask[20:] = False
    nodes[1] = node("b", mask, frames["b"])
    groups, evidence = associate(
        nodes, frames, GeometryPolicy(partial_view_association=True, min_points=10)
    )
    assert groups == [[0], [1]]
    assert evidence["candidate_pairs"][0]["reason"] == "visible_mask_conflict"


def test_partial_fov_does_not_treat_occlusion_or_exclusion_as_support():
    for mode in ("occluded", "missing", "excluded"):
        nodes, frames = cropped_plane()
        if mode == "occluded":
            frames["b"]["depth"][:] = 1.0
        elif mode == "missing":
            frames["b"]["depth"][:] = 0.0
        else:
            frames["b"]["excluded"][:] = True
        groups, evidence = associate(
            nodes, frames, GeometryPolicy(partial_view_association=True)
        )
        assert groups == [[0], [1]]
        assert evidence["candidate_pairs"][0]["decision"] == "unknown"


def test_partial_fov_requires_mask_at_crossed_image_edge():
    nodes, frames = cropped_plane()
    # Preserve the surface points to isolate the image-edge eligibility gate.
    nodes[1]["mask"][:, 0] = False
    groups, evidence = associate(
        nodes, frames, GeometryPolicy(partial_view_association=True)
    )
    assert groups == [[0], [1]]
    assert "partial_view_directions" not in evidence["candidate_pairs"][0]


def test_partial_fov_preserves_same_frame_scope_constraint():
    nodes, frames = cropped_plane()
    nodes.append(dict(nodes[1]))
    groups, evidence = associate(
        nodes, frames, GeometryPolicy(partial_view_association=True)
    )
    assert not any(1 in group and 2 in group for group in groups)
    assert (1, 2) in evidence["cannot_link_pairs"]


def test_partial_fov_does_not_promote_clipped_part_over_larger_scope():
    nodes, frames = cropped_plane()
    larger = nodes[1]["mask"].copy()
    larger[:, :8] = True
    nodes.append(node("b", larger, frames["b"]))
    groups, evidence = associate(
        nodes, frames, GeometryPolicy(partial_view_association=True)
    )
    assert not any(0 in group and 1 in group for group in groups)
    pair = next(e for e in evidence["candidate_pairs"] if (e["a"], e["b"]) == (0, 1))
    assert pair["decision"] == "unknown"
    assert pair["directions"][0]["partial_scope_ambiguous_with"] == [2]


def test_detector_priority_keeps_primary_pixels_without_cross_model_score_comparison():
    primary = np.zeros((10, 10), bool)
    primary[1:9, 1:9] = True
    supplement = primary.copy()
    supplement[1, 1] = False
    assert equivalent_masks([primary, supplement], [0.55, 0.99])[0][0] == 1
    assert equivalent_masks([primary, supplement], [0.55, 0.99], priorities=[0, 1]) == [
        [0, 1]
    ]


def test_complementary_mask_cannot_steal_an_existing_primary_correspondence():
    f = scene()
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    larger = np.zeros_like(mask)
    larger[8:32, 8:32] = True
    nodes = [node("a", mask, f), node("b", mask, f), node("b", larger, f)]
    nodes[2]["source_priority"] = 1
    groups, _ = associate(nodes, {"a": f, "b": f})
    assert groups == [[0, 1], [2]]


def test_lower_priority_container_does_not_disable_primary_partial_evidence():
    nodes, frames = cropped_plane()
    larger = nodes[1]["mask"].copy()
    larger[:, :8] = True
    nodes.append(dict(node("b", larger, frames["b"]), source_priority=1))
    groups, evidence = associate(
        nodes, frames, GeometryPolicy(partial_view_association=True)
    )
    assert groups == [[0, 1], [2]]
    pair = next(e for e in evidence["candidate_pairs"] if (e["a"], e["b"]) == (0, 1))
    assert pair["reason"] == "partial_view_surface_support"


def test_complementary_edges_cannot_block_an_existing_primary_component(monkeypatch):
    f = scene()
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    nodes = [node(name, mask, f) for name in ("a", "b", "c", "c")]
    for i, n in enumerate(nodes):
        n["fixture_id"] = i
        n["source_priority"] = int(i >= 2)
    scores = {(0, 1): 0.6, (0, 2): 0.9, (1, 3): 0.8}

    def compare(a, b, frames, policy):
        score = scores.get((a["fixture_id"], b["fixture_id"]))
        return dict(
            decision="match" if score else "unknown",
            reason="mutual_surface_support" if score else "disjoint_surface_bounds",
            score=score or 0,
        )

    monkeypatch.setattr("farm_runtime.proposal_geometry.compare_surfaces", compare)
    groups, _ = associate(nodes, {name: f for name in ("a", "b", "c")})
    assert any(0 in g and 1 in g and 2 in g for g in groups)
    assert not any(2 in g and 3 in g for g in groups)


def depth_offset_pair(offset=0.032):
    f, g = scene(), scene()
    f["K"][:2, :2] *= 10
    g["K"][:2, :2] *= 10
    g["depth"] += offset
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    return [node("a", mask, f), node("b", mask, g)], {"a": f, "b": g}


def test_depth_consistent_proximity_accepts_only_projection_supported_offset():
    nodes, frames = depth_offset_pair()
    assert associate(nodes, frames)[0] == [[0], [1]]
    groups, evidence = associate(
        nodes, frames, GeometryPolicy(depth_consistent_proximity=True)
    )
    assert groups == [[0, 1]]
    pair = evidence["candidate_pairs"][0]
    assert pair["radius_m"] == 0.04
    assert min(d["mask_agreement"] for d in pair["directions"]) > 0.9
    # Unchanged default must still be independent of opt-in metadata on nodes.
    assert associate(nodes, frames)[0] == [[0], [1]]


def test_depth_consistent_proximity_preserves_unknown_and_scope_conflicts():
    policy = GeometryPolicy(depth_consistent_proximity=True)
    for condition in ("missing", "excluded", "background", "remote", "same_frame"):
        nodes, frames = depth_offset_pair(0.15 if condition == "remote" else 0.032)
        if condition == "missing":
            frames["b"]["depth"][:] = 0
        elif condition == "excluded":
            frames["b"]["excluded"][:] = True
        elif condition == "background":
            nodes[1]["mask"][:] = False
            nodes[1]["mask"][:8, :8] = True
        elif condition == "same_frame":
            nodes[1]["frame"] = "a"
        assert associate(nodes, frames, policy)[0] == [[0], [1]]
