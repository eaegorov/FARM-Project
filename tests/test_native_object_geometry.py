import numpy as np
import pytest
from farm_runtime.native_object_geometry import fit_native_obb, weighted_quantile


def test_native_obb_rejects_low_weight_outlier_without_inventing_poster_thickness():
    rng = np.random.default_rng(8)
    plane = rng.uniform([-1.0, -0.5, -0.0005], [1.0, 0.5, 0.0005], (5000, 3))
    points = np.concatenate([plane, [[20.0, 20.0, 20.0]]])
    weights = np.concatenate([np.ones(len(plane)), [1e-5]])
    obb = fit_native_obb(points, weights, np.eye(3))
    assert obb["dimensions_m"][0] == pytest.approx(2.0, abs=0.04)
    assert obb["dimensions_m"][2] < 0.0011
    assert obb["weighted_center_containment"] > 0.96
    assert obb["minimum_size_padding_applied"] is False


def test_native_obb_preserves_metric_pose_and_rejects_bad_rotation():
    rng = np.random.default_rng(11)
    points = rng.uniform(-1, 1, (1000, 3)) + [3, 4, 5]
    R = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    result = fit_native_obb(points, np.ones(1000), R)
    assert np.allclose(result["center_m"], [3, 4, 5], atol=0.04)
    with pytest.raises(ValueError, match="orientation"):
        fit_native_obb(points, np.ones(1000), np.ones((3, 3)))


def test_zero_weight_outliers_cannot_expand_obb_even_at_extreme_quantiles():
    assert np.array_equal(
        weighted_quantile([-999.0, 1.0, 2.0, 999.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0]),
        [1.0, 2.0],
    )
    with pytest.raises(ValueError, match="positive support"):
        weighted_quantile([1.0, 2.0], [0.0, 0.0], [0.5])
