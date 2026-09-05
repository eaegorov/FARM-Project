"""Metric transforms and sampling for independent learned OBB proposals."""

import numpy as np


def z_up_rotation(world_up):
    """Return a proper rotation mapping the declared world up to positive Z."""
    up = np.asarray(world_up, dtype=np.float64)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8:
        raise ValueError("finite nonzero world-up vector required")
    up = up / np.linalg.norm(up)
    seed = np.eye(3)[np.argmin(np.abs(up))]
    x = seed - up * (seed @ up)
    x /= np.linalg.norm(x)
    return np.stack((x, np.cross(up, x), up))


def depth_world_points(depth, K, pose, valid, maximum=20000):
    """Deterministic spatial sampling; missing pixels are never reconstructed."""
    depth = np.asarray(depth)
    valid = np.asarray(valid, bool) & np.isfinite(depth) & (depth > 0)
    yy, xx = np.where(valid)
    if len(xx) > maximum:
        chosen = np.linspace(0, len(xx) - 1, maximum).round().astype(int)
        yy, xx = yy[chosen], xx[chosen]
    camera = np.column_stack((xx, yy, np.ones(len(xx)))) @ np.linalg.inv(K).T
    camera *= depth[yy, xx, None]
    pose = np.asarray(pose)
    return (camera @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)


def metric_box_corners(center, rotation, dimensions):
    signs = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
    return (signs * np.asarray(dimensions) / 2) @ np.asarray(rotation).T + center


def project_world(points, K, pose):
    camera = (np.asarray(points) - pose[:3, 3]) @ pose[:3, :3]
    projected = camera @ K.T
    return projected[:, :2] / projected[:, 2, None], camera[:, 2]


def rotate_pinhole_camera(K, pose, shape_hw, quarter_turns):
    """Exact np.rot90 pixel/camera transform, including the principal point."""
    K, pose = np.asarray(K, float).copy(), np.asarray(pose, float).copy()
    h, w = shape_hw
    roll = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    for _ in range(int(quarter_turns) % 4):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        K = np.array([[fy, 0, cy], [0, fx, w - 1 - cx], [0, 0, 1]])
        pose[:3, :3] = pose[:3, :3] @ roll
        h, w = w, h
    return K, pose, (h, w)


def surface_coverage(points, center, rotation, dimensions, tolerance_m=0.04):
    """Distance to a box is a surface diagnostic, not an object mask metric."""
    local = (np.asarray(points) - center) @ np.asarray(rotation)
    outside = np.maximum(np.abs(local) - np.asarray(dimensions) / 2, 0)
    distance = np.linalg.norm(outside, axis=1)
    return {
        "points": len(distance),
        "fraction_within_tolerance": float(np.mean(distance <= tolerance_m)),
        "outside_distance_median_m": float(np.median(distance)),
        "outside_distance_q90_m": float(np.quantile(distance, 0.9)),
        "tolerance_m": tolerance_m,
    }


def fit_surface_envelope(points, weights, rotation, orientation_source):
    """Reuse FARM's robust extent fitter, without inventing hidden thickness."""
    from farm_runtime.native_object_geometry import fit_native_obb

    box = fit_native_obb(points, weights, rotation)
    box.update(
        size_semantics="robust observed registered RGBD surface envelope",
        orientation_source=orientation_source,
    )
    return box
