import math

import pytest

from scripts.geometry.audit_farm_gaussian_support import gaussian_support_gate


def test_supported_raw_and_scale_aware_voxels_pass() -> None:
    passed, diagnostics = gaussian_support_gate(
        raw_median_m=0.011,
        voxel_median_m=0.09,
        voxel_p90_m=0.23,
        voxel_spacing_m=0.08,
    )

    assert passed is True
    assert diagnostics["raw_supported"] is True
    assert diagnostics["voxel_supported"] is True
    assert diagnostics["surface_tier"] == "surface_pass"
    assert diagnostics["support_signal_count"] == 2


def test_coarse_crane_like_voxels_fail_independently_of_raw_depth() -> None:
    passed, diagnostics = gaussian_support_gate(
        raw_median_m=0.011,
        voxel_median_m=0.217,
        voxel_p90_m=0.649,
        voxel_spacing_m=0.16,
    )

    assert passed is False
    assert diagnostics["raw_supported"] is True
    assert diagnostics["voxel_supported"] is False
    assert diagnostics["voxel_median_limit_m"] == pytest.approx(0.16)
    assert diagnostics["voxel_p90_limit_m"] == pytest.approx(0.48)
    assert diagnostics["surface_tier"] == "surface_borderline"
    assert diagnostics["support_signal_count"] == 1


def test_two_failed_signals_are_rejected() -> None:
    passed, diagnostics = gaussian_support_gate(
        raw_median_m=0.2,
        voxel_median_m=0.3,
        voxel_p90_m=0.7,
        voxel_spacing_m=0.08,
    )

    assert passed is False
    assert diagnostics["surface_tier"] == "surface_rejected"
    assert diagnostics["support_signal_count"] == 0


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_surface_metrics_fail_closed(value: float) -> None:
    passed, _ = gaussian_support_gate(
        raw_median_m=value,
        voxel_median_m=0.01,
        voxel_p90_m=0.02,
        voxel_spacing_m=0.04,
    )

    assert passed is False
