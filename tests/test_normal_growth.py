import numpy as np
import pytest

from farm_runtime.quality.normal_growth import (
    NormalGrowthPolicy, graph_surface_candidates,
)


def run(points, seeds, **kwargs):
    count = len(points)
    kwargs.setdefault("positive_mask", np.ones(count, bool))
    kwargs.setdefault("unknown_mask", np.zeros(count, bool))
    kwargs.setdefault("normals", np.tile([0.0, 0.0, 1.0], (count, 1)))
    kwargs.setdefault("voxel_size", 0.9)
    kwargs.setdefault("policy", NormalGrowthPolicy(max_growth_ratio=30.0))
    return graph_surface_candidates(points, np.asarray(seeds, dtype=int), **kwargs)


def plane():
    return np.array([[x, y, 0.0] for x in range(7) for y in range(5)], float)


def test_surface_growth_is_bounded_and_leaves_inputs_unchanged():
    points = plane()
    seeds = np.flatnonzero(points[:, 0] == 0)
    originals = points.copy(), seeds.copy()
    found, audit = run(points, seeds, policy=NormalGrowthPolicy(
        max_steps=2, max_growth_ratio=10.0,
        connection_spacing_multiplier=1.1, max_connection_voxels=1.3))
    assert len(found) > 0
    assert set(points[found, 0]) == {1.0, 2.0}
    assert not set(found) & set(seeds)
    assert audit["labels_assigned"] is False
    assert audit["candidate_hops"] == sorted(audit["candidate_hops"])
    np.testing.assert_array_equal(points, originals[0])
    np.testing.assert_array_equal(seeds, originals[1])


@pytest.mark.parametrize("kind", ["negative_mask", "protected_mask"])
def test_occupied_barrier_stops_long_edges_across_a_surface(kind):
    points = plane()
    seeds = np.flatnonzero(points[:, 0] <= 1)
    barrier = points[:, 0] == 3
    # A two-cell jump could reach x=4 without entering the x=3 nodes.
    found, audit = run(
        points, seeds, **{kind: barrier},
        policy=NormalGrowthPolicy(max_growth_ratio=30.0,
                                  connection_spacing_multiplier=2.5,
                                  max_connection_voxels=3.0),
    )
    assert 2.0 in points[found, 0]
    assert not np.any(points[found, 0] >= 3.0)
    assert audit["rejected_edges"]["barrier"] > 0


def test_unknown_domain_does_not_propagate_without_renewed_positive_evidence():
    points = np.column_stack((np.arange(8), np.zeros((8, 2))))
    positive = np.zeros(8, bool)
    positive[[0, 2]] = True
    found, audit = run(
        points, [0], positive_mask=positive, unknown_mask=~positive,
        policy=NormalGrowthPolicy(max_steps=7, max_growth_ratio=20,
                                  connection_spacing_multiplier=1.1,
                                  max_connection_voxels=1.3,
                                  max_unknown_hops=1),
    )
    assert found.tolist() == [1, 2, 3]
    assert audit["candidate_unknown_hops"] == [1, 0, 1]


def test_normals_separate_parallel_layers_with_nearby_centers():
    lower = np.array([[x, y, 0.0] for x in range(4) for y in range(4)])
    upper = lower + [0, 0, 0.4]
    points = np.concatenate((lower, upper))
    seeds = np.flatnonzero((points[:, 0] == 0) & (points[:, 2] == 0))
    found, audit = run(
        points, seeds, voxel_size=0.3,
        policy=NormalGrowthPolicy(max_growth_ratio=30.0,
                                  connection_spacing_multiplier=3.0,
                                  max_connection_voxels=4.0),
    )
    assert len(found) > 0
    assert np.all(points[found, 2] == 0)
    assert audit["rejected_edges"]["normal_offset"] > 0


def test_sign_of_supplied_normals_does_not_change_membership():
    points = plane()
    seeds = np.flatnonzero(points[:, 0] == 0)
    expected, _ = run(points, seeds)
    normals = np.tile([0.0, 0, 1], (len(points), 1))
    normals[::2] *= -1
    result, _ = run(points, seeds, normals=normals)
    np.testing.assert_array_equal(result, expected)


def test_local_pca_grows_planar_surface_but_rejects_line_and_volume():
    points = plane()
    seeds = np.flatnonzero(points[:, 0] == 0)
    found, audit = run(points, seeds, normals=None)
    assert len(found) > 0
    assert audit["normals_source"] == "local_voxel_pca"
    line = np.column_stack((np.arange(15), np.zeros((15, 2))))
    found, audit = run(line, [0], normals=None)
    assert not len(found)
    assert audit["reliable_voxels"] == 0
    volume = np.array([[x, y, z] for x in range(3)
                       for y in range(3) for z in range(3)], float)
    found, audit = run(
        volume, [0], normals=None, voxel_size=0.8,
        policy=NormalGrowthPolicy(normal_neighbors=27,
                                  normal_radius_voxels=10.0))
    assert not len(found)
    assert audit["reliable_voxels"] == 0


def test_curved_pipe_surface_grows_without_crossing_to_a_disconnected_pipe():
    angles = np.arange(32) * (2 * np.pi / 32)
    cylinder = np.array([[x, np.cos(t), np.sin(t)]
                         for x in np.arange(0, 2.1, 0.3) for t in angles])
    normals = cylinder.copy()
    normals[:, 0] = 0
    neighbor = cylinder + [0, 3.2, 0]
    points = np.concatenate((cylinder, neighbor))
    normals = np.concatenate((normals, normals))
    seeds = np.flatnonzero(np.arange(len(points)) < 64)
    found, audit = run(
        points, seeds, normals=normals, voxel_size=0.15,
        gaussian_radii=np.full(len(points), 100.0),
        policy=NormalGrowthPolicy(max_steps=3, max_growth_ratio=10,
                                  max_connection_voxels=3.0))
    assert len(found) > 0
    assert np.all(found < len(cylinder))
    assert points[found, 0].max() <= 1.2 + 1e-8
    assert audit["candidates_only"] is True


def test_scaling_units_preserves_candidates_and_budgets_are_deterministic():
    points = plane()
    seeds = np.flatnonzero(points[:, 0] == 0)
    policy = NormalGrowthPolicy(max_candidates=4, max_growth_ratio=1.0)
    first, audit = run(points, seeds, policy=policy)
    second, _ = run(points * 7, seeds, voxel_size=0.9 * 7, policy=policy)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 4
    assert audit["stop_reason"] == "candidate_budget"


def test_coplanar_neighbor_cannot_be_identified_from_normals_alone():
    points = plane()
    seeds = np.flatnonzero(points[:, 0] == 0)
    # Demonstrates the actual limit, not an unsupported safety claim.
    found, _ = run(points, seeds)
    assert np.any(points[found, 0] >= 3)
    unknown_only = np.zeros(len(points), bool)
    unknown_only[points[:, 0] <= 1] = True
    positive = np.zeros(len(points), bool)
    positive[seeds] = True
    unknown_only[seeds] = False
    bounded, _ = run(points, seeds, positive_mask=positive,
                     unknown_mask=unknown_only)
    assert np.all(points[bounded, 0] <= 1)


def test_supplied_unreliable_and_zero_normals_cannot_be_frontier():
    points = plane()
    seeds = np.flatnonzero(points[:, 0] == 0)
    reliable = np.ones(len(points), bool)
    reliable[seeds] = False
    found, _ = run(points, seeds, normal_reliable=reliable)
    assert not len(found)
    found, _ = run(points, seeds, normals=np.zeros_like(points))
    assert not len(found)


def test_invalid_domains_and_protected_seeds_fail_explicitly():
    points = plane()
    both = np.ones(len(points), bool)
    with pytest.raises(ValueError, match="disjoint"):
        run(points, [0], unknown_mask=both)
    with pytest.raises(ValueError, match="seeds cannot"):
        run(points, [0], protected_mask=both)
    with pytest.raises(ValueError, match="voxel budget"):
        run(points, [0], policy=NormalGrowthPolicy(max_voxels=1))
    with pytest.raises(ValueError, match="integer"):
        graph_surface_candidates(points, [0.5], positive_mask=both,
                                 unknown_mask=~both)


def test_empty_seed_and_disabled_growth_produce_no_candidates():
    points = plane()
    result, _ = run(points, [])
    assert not len(result)
    result, _ = run(points, [0], policy=NormalGrowthPolicy(max_steps=0))
    assert not len(result)
