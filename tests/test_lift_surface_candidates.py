import numpy as np
import pytest

from tools.farm_shaper_bridge.lift_candidates import (
    backproject_mask_surface,
    surface_candidate_indices,
)
from tools.farm_shaper_bridge.gaussian_lift import mask_surface_candidates


def test_mask_surface_preserves_thin_foreground_and_metric_camera_transform():
    mask = np.zeros((8, 8), bool)
    mask[1:7, 1] = True
    depth = np.full((8, 8), 2.0)
    depth[1, 1] = np.nan
    K = np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 1.0]])
    pose = np.eye(4)
    pose[0, 3] = 5.0
    points, far, spacing = backproject_mask_surface(
        mask, depth, K, pose, stride=4, depth_min=0.1, depth_max=10.0
    )
    assert points.shape == (2, 3)
    assert np.allclose(points[:, 0], 5.2)
    assert np.allclose(points[:, 2], 2.0)
    assert far == 2.0 and spacing > 0


def test_candidates_include_missing_surface_but_exclude_remote_and_transparent_rows():
    # The supplied surface is deliberately outside the old box around the origin.
    means = np.array(
        [[0.0, 0, 0], [1.0, 0, 2.0], [1.08, 0, 2.0], [1.0, 0, 4.0], [1.0, 0, 2.001]]
    )
    ids, _ = surface_candidate_indices(
        means,
        np.full(5, 0.02),
        np.array([1.0, 1, 1, 1, 0.001]),
        np.array([[1.0, 0, 2.0]]),
        minimum_opacity=0.005,
        maximum_radius=0.3,
        radius_multiplier=2.5,
        surface_tolerance=0.04,
        chunk_size=2,
    )
    assert ids.tolist() == [1, 2]  # Native row IDs, not newly sampled geometry.


def test_candidate_builder_never_opens_heldout_masks(monkeypatch):
    from types import SimpleNamespace
    from tools.farm_shaper_bridge import gaussian_lift

    heldout = SimpleNamespace(image_id=0)
    run = SimpleNamespace(
        frame=lambda _: SimpleNamespace(image_id=0, physical_timestamp="20")
    )
    obj = SimpleNamespace(object_id=1, observations=[heldout])
    gaussian = SimpleNamespace(
        means_m=np.zeros((2, 3)), radius_m=np.ones(2), opacity_cpu=np.ones(2)
    )

    def forbidden(*a, **k):
        raise AssertionError("heldout data was opened")

    monkeypatch.setattr(gaussian_lift, "_load_view_masks", forbidden)
    config = {
        "candidate": {
            "depth_absolute_tolerance_m": 0.04,
            "depth_relative_tolerance": 0.015,
            "minimum_opacity": 0.005,
            "maximum_radius_m": 0.3,
            "radius_tolerance_multiplier": 2.5,
            "surface_voxel_m": 0.01,
        }
    }
    ids, audit = mask_surface_candidates(
        run, obj, gaussian, {"build_timestamps": ["10"]}, config
    )
    assert len(ids) == 0 and audit["heldout_consumed"] is False


def test_surface_candidates_empty_and_bad_alignment():
    common = dict(
        minimum_opacity=0.005,
        maximum_radius=0.3,
        radius_multiplier=2.5,
        surface_tolerance=0.04,
    )
    ids, _ = surface_candidate_indices(
        np.zeros((2, 3)), np.ones(2), np.ones(2), np.empty((0, 3)), **common
    )
    assert len(ids) == 0
    with pytest.raises(ValueError, match="row aligned"):
        surface_candidate_indices(
            np.zeros((2, 3)), np.ones(1), np.ones(2), np.ones((1, 3)), **common
        )
