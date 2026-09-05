"""Metric OBB estimates from native Gaussian membership with explicit support.

An observed surface envelope is not a measurement of hidden physical thickness.
No class-specific minimum size or invented thickness is added to a thin object.
"""

from __future__ import annotations

import numpy as np


def weighted_quantile(values, weights, probabilities):
    values, weights = np.asarray(values), np.asarray(weights)
    positive = weights > 0
    values, weights = values[positive], weights[positive]
    if not len(values):
        raise ValueError("weighted quantiles require positive support")
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= np.sum(weights)
    return np.interp(probabilities, cumulative, values)


def fit_native_obb(points, weights, prior_rotation, *, tail_fraction=0.005):
    """Refit extents in a supplied, independently established orientation.

    Keeping orientation provenance avoids unstable PCA yaw for symmetric or
    incomplete objects. Mask completeness and orientation are reviewed separately.
    """
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    rotation = np.asarray(prior_rotation, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or weights.shape != (len(points),):
        raise ValueError("point and weight arrays must be aligned")
    if (
        len(points) < 8
        or not np.isfinite(points).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError("insufficient finite positive-weight geometry")
    if (
        rotation.shape != (3, 3)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or np.linalg.det(rotation) < 0.999
    ):
        raise ValueError("orientation must be a proper orthonormal rotation")
    if not 0 <= tail_fraction < 0.1:
        raise ValueError("tail fraction must be within [0, .1)")
    origin = np.average(points, axis=0, weights=weights)
    local = (points - origin) @ rotation
    extents = np.array(
        [
            weighted_quantile(
                local[:, axis], weights, [tail_fraction, 1 - tail_fraction]
            )
            for axis in range(3)
        ]
    )
    lo, hi = extents[:, 0], extents[:, 1]
    center = origin + ((lo + hi) * 0.5) @ rotation.T
    dimensions = np.maximum(hi - lo, 0.0)
    inside = np.all((local >= lo) & (local <= hi), axis=1)
    probe_tail = max(tail_fraction, 0.025)
    probe_extents = np.array(
        [
            weighted_quantile(local[:, axis], weights, [probe_tail, 1 - probe_tail])
            for axis in range(3)
        ]
    )
    probe_dimensions = np.maximum(probe_extents[:, 1] - probe_extents[:, 0], 0)
    sensitivity = np.maximum(dimensions - probe_dimensions, 0)
    relative = sensitivity / np.maximum(dimensions, 1e-12)
    return {
        "center_m": center.tolist(),
        "dimensions_m": dimensions.tolist(),
        "rotation_world_from_box": rotation.tolist(),
        "units": "metres",
        "source_points": len(points),
        "weighted_center_containment": float(weights[inside].sum() / weights.sum()),
        "center_containment": float(inside.mean()),
        "tail_fraction": tail_fraction,
        "orientation_source": "prior geometry; orientation not silently refitted",
        "size_semantics": "robust observed Gaussian-center envelope",
        "physical_hidden_extent_known": False,
        "minimum_size_padding_applied": False,
        "extent_stability": {
            "probe_tail_fraction": probe_tail,
            "probe_dimensions_m": probe_dimensions.tolist(),
            "sensitivity_m": sensitivity.tolist(),
            "relative_sensitivity": relative.tolist(),
            "sensitive_axes": np.flatnonzero(
                (relative >= 0.20) & (sensitivity >= 0.01)
            ).tolist(),
            "interpretation": "tail-sensitivity diagnostic, not a physical confidence interval",
            "physical_size_validated": False,
        },
    }
