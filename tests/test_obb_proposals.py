import numpy as np
import pytest

from farm_runtime.obb_proposals import (
    z_up_rotation,
    depth_world_points,
    project_world,
    metric_box_corners,
)


@pytest.mark.parametrize("up", [[0, -1, 0], [0, 0, 1], [1, 2, 3]])
def test_world_up_transform_preserves_projection_and_metric_extent(up):
    G = z_up_rotation(up)
    np.testing.assert_allclose(
        G @ (np.array(up) / np.linalg.norm(up)), [0, 0, 1], atol=1e-12
    )
    assert np.linalg.det(G) == pytest.approx(1)
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    K = np.array([[100, 0, 50], [0, 120, 60], [0, 0, 1]])
    points = np.array([[1.2, 2.3, 5.0], [0.8, 1.7, 6.0]])
    p2 = pose.copy()
    p2[:3] = G @ pose[:3]
    uv, z = project_world(points, K, pose)
    uv2, z2 = project_world(points @ G.T, K, p2)
    np.testing.assert_allclose(uv, uv2, atol=1e-10)
    np.testing.assert_allclose(z, z2)
    corners = metric_box_corners(np.zeros(3), G, [1, 2, 3])
    np.testing.assert_allclose(np.ptp(corners @ G, axis=0), [1, 2, 3])


def test_depth_sampling_never_invents_excluded_or_missing_surfaces():
    depth = np.array([[1.0, 0], [np.nan, 3.0]])
    K = np.eye(3)
    pose = np.eye(4)
    valid = np.ones((2, 2), bool)
    valid[1, 1] = False
    pts = depth_world_points(depth, K, pose, valid)
    np.testing.assert_allclose(pts, [[0, 0, 1]])
    assert depth_world_points(depth, K, pose, np.zeros((2, 2), bool)).shape == (0, 3)


@pytest.mark.parametrize("turns", [0, 1, 2, 3, 4])
def test_upright_camera_matches_exact_rot90_pixels(turns):
    from farm_runtime.obb_proposals import rotate_pinhole_camera

    K = np.array([[120.0, 0, 60.0], [0, 100.0, 40.0], [0, 0, 1.0]])
    pose = np.eye(4)
    points = np.array([[0.2, -0.3, 2], [0.6, 0.5, 3]])
    original, _ = project_world(points, K, pose)
    expected = original.copy()
    h, w = 80, 120
    for _ in range(turns % 4):
        expected = np.column_stack((expected[:, 1], w - 1 - expected[:, 0]))
        h, w = w, h
    K2, pose2, shape = rotate_pinhole_camera(K, pose, (80, 120), turns)
    actual, _ = project_world(points, K2, pose2)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert shape == (h, w)
    assert np.linalg.det(pose2[:3, :3]) == pytest.approx(1)


def test_surface_envelope_never_turns_a_plane_into_a_padded_physical_volume():
    from farm_runtime.obb_proposals import fit_surface_envelope, surface_coverage

    xx, yy = np.meshgrid(np.linspace(-1, 1, 12), np.linspace(-0.5, 0.5, 12))
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    box = fit_surface_envelope(points, np.ones(len(points)), np.eye(3), "fixture")
    assert box["dimensions_m"][2] == 0.0
    assert box["physical_hidden_extent_known"] is False
    assert box["minimum_size_padding_applied"] is False
    assert (
        surface_coverage(points, box["center_m"], np.eye(3), box["dimensions_m"])[
            "fraction_within_tolerance"
        ]
        == 1.0
    )


def test_surface_support_weights_do_not_double_count_simultaneous_cameras():
    from farm_runtime.quality.surface_obb import support

    nodes = [
        dict(id=0, timestamp="a"),
        dict(id=1, timestamp="a"),
        dict(id=2, timestamp="b"),
    ]
    clouds = {
        "node_0000": np.zeros((10, 3)),
        "node_0001": np.zeros((20, 3)),
        "node_0002": np.ones((5, 3)),
    }
    _, weights = support(nodes, clouds)
    assert weights[:30].sum() == pytest.approx(weights[30:].sum())
