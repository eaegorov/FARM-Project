"""Conservative RGB-to-render correspondence and metric PnP diagnostics.

Camera proposals never mutate input COLMAP data. Feature correspondences are
split spatially before fitting; validation pixels do not optimize the pose.
"""

from __future__ import annotations

import cv2
import numpy as np


def match_render_to_source(rendered, source, depth, alpha, K, T_world_cam):
    if (
        rendered.shape != source.shape
        or depth.shape != rendered.shape[:2]
        or alpha.shape != depth.shape
    ):
        raise ValueError("RGB, alpha and depth grids must agree")
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.02)
    kr, dr = sift.detectAndCompute(cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY), None)
    ks, ds = sift.detectAndCompute(cv2.cvtColor(source, cv2.COLOR_RGB2GRAY), None)
    if dr is None or ds is None or min(len(dr), len(ds)) < 2:
        return np.empty((0, 3)), np.empty((0, 2)), {"matches": 0}
    bf = cv2.BFMatcher(cv2.NORM_L2)
    forward = bf.knnMatch(dr, ds, k=2)
    reverse = bf.knnMatch(ds, dr, k=2)
    backward = {
        m.queryIdx: m.trainIdx
        for pair in reverse
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.75 * n.distance
    }
    matches = [
        m
        for pair in forward
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.75 * n.distance and backward.get(m.trainIdx) == m.queryIdx
    ]
    pr = np.array([kr[m.queryIdx].pt for m in matches], np.float64).reshape(-1, 2)
    ps = np.array([ks[m.trainIdx].pt for m in matches], np.float64).reshape(-1, 2)
    x, y = np.rint(pr).astype(int).T
    z = depth[y, x]
    max_depth = cv2.dilate(depth, np.ones((3, 3), np.uint8))
    min_depth = cv2.erode(depth, np.ones((3, 3), np.uint8))
    valid = (
        np.isfinite(z)
        & (z > 0.1)
        & (alpha[y, x] > 0.95)
        & ((max_depth - min_depth)[y, x] < 0.10)
    )
    pr, ps, z = pr[valid], ps[valid], z[valid]
    camera = np.column_stack(
        ((pr[:, 0] - K[0, 2]) * z / K[0, 0], (pr[:, 1] - K[1, 2]) * z / K[1, 1], z)
    )
    world = camera @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
    return (
        world,
        ps,
        {
            "matches": len(matches),
            "valid_surface_matches": len(world),
            "raw_displacement_median_px": (
                float(np.median(np.linalg.norm(pr - ps, axis=1)))
                if len(world)
                else None
            ),
        },
    )


def propose_camera_pose(
    points_world,
    pixels,
    K,
    initial_pose,
    image_shape,
    *,
    max_translation_m=0.5,
    max_rotation_degrees=5.0,
):
    points_world, pixels = np.asarray(points_world, np.float64), np.asarray(
        pixels, np.float64
    )
    if points_world.shape != (len(pixels), 3) or pixels.shape != (len(points_world), 2):
        raise ValueError("3D/2D correspondence arrays must be aligned")
    if len(points_world) < 40:
        return initial_pose.copy(), {
            "accepted": False,
            "reason": "insufficient_correspondences",
            "count": len(points_world),
        }
    h, w = image_shape
    cell_x = np.clip((pixels[:, 0] / w * 8).astype(int), 0, 7)
    cell_y = np.clip((pixels[:, 1] / h * 8).astype(int), 0, 7)
    validation = (cell_x + 2 * cell_y) % 4 == 0
    train = ~validation
    if train.sum() < 24 or validation.sum() < 12:
        return initial_pose.copy(), {
            "accepted": False,
            "reason": "insufficient_spatial_validation",
        }
    cv2.setRNGSeed(20260905)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_world[train],
        pixels[train],
        K,
        None,
        iterationsCount=500,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 24:
        return initial_pose.copy(), {
            "accepted": False,
            "reason": "insufficient_pnp_inliers",
        }
    selected = np.flatnonzero(train)[inliers[:, 0]]
    rvec, tvec = cv2.solvePnPRefineLM(
        points_world[selected], pixels[selected], K, None, rvec, tvec
    )
    R = cv2.Rodrigues(rvec)[0]
    candidate = np.eye(4)
    candidate[:3, :3] = R.T
    candidate[:3, 3] = -R.T @ tvec[:, 0]

    def errors(pose):
        camera = (points_world - pose[:3, 3]) @ pose[:3, :3]
        uv = camera @ K.T
        uv = uv[:, :2] / uv[:, 2:3]
        error = np.linalg.norm(uv - pixels, axis=1)
        error[camera[:, 2] <= 0] = 1e6
        return error

    old, new = errors(initial_pose), errors(candidate)
    before, after = float(np.median(old[validation])), float(np.median(new[validation]))
    translation = float(np.linalg.norm(candidate[:3, 3] - initial_pose[:3, 3]))
    angle = float(
        np.degrees(
            np.arccos(
                np.clip(
                    (np.trace(candidate[:3, :3].T @ initial_pose[:3, :3]) - 1) / 2,
                    -1,
                    1,
                )
            )
        )
    )
    coverage = len(set(zip(cell_x[selected], cell_y[selected])))
    accepted = (
        before > 1.5
        and after < before * 0.7
        and after < 4.0
        and translation <= max_translation_m
        and angle <= max_rotation_degrees
        and coverage >= 6
    )
    report = {
        "accepted": bool(accepted),
        "reason": "validated_pose_proposal" if accepted else "pose_or_validation_guard",
        "correspondences": len(points_world),
        "fit_inliers": len(selected),
        "fit_grid_cells": coverage,
        "validation_correspondences": int(validation.sum()),
        "validation_before_px": before,
        "validation_after_px": after,
        "translation_change_m": translation,
        "rotation_change_degrees": angle,
        "candidate_T_world_cam": candidate.tolist(),
        "rig_consistency_checked": False,
    }
    return (candidate if accepted else initial_pose.copy()), report


def refine_camera_rig(observations, *, max_translation_m=0.5, max_rotation_degrees=5.0):
    """Fit one rigid correction shared by all cameras at a physical timestamp.

    Each observation provides points_world_m, pixels, K, T_world_cam and
    shape_hw. Intrinsics and all relative camera transforms remain fixed.
    No object labels, masks, instance IDs or heldout annotations enter this fit.
    """
    from scipy.optimize import least_squares

    if not observations:
        raise ValueError("a rig needs at least one observation")
    anchor = np.asarray(observations[0]["T_world_cam"], float)
    relative = [
        np.linalg.inv(anchor) @ np.asarray(o["T_world_cam"], float)
        for o in observations
    ]
    partitions = []
    for o in observations:
        xy = np.asarray(o["pixels"], float)
        h, w = o["shape_hw"]
        cx = np.clip((xy[:, 0] / w * 8).astype(int), 0, 7)
        cy = np.clip((xy[:, 1] / h * 8).astype(int), 0, 7)
        validation = (cx + 2 * cy) % 4 == 0
        fit = np.flatnonzero(~validation)
        if len(fit) > 512:
            fit = fit[np.linspace(0, len(fit) - 1, 512).round().astype(int)]
        partitions.append((fit, np.flatnonzero(validation), cx, cy))
    usable = [
        i
        for i, (fit, val, _, _) in enumerate(partitions)
        if len(fit) >= 24 and len(val) >= 12
    ]
    original = [np.asarray(o["T_world_cam"], float).copy() for o in observations]
    if not usable:
        return original, {
            "accepted": False,
            "reason": "insufficient_spatial_validation",
            "rig_preserved": True,
        }

    def poses(x):
        delta = np.eye(4)
        delta[:3, :3] = cv2.Rodrigues(x[:3])[0]
        delta[:3, 3] = x[3:]
        return [anchor @ delta @ E for E in relative]

    def residual(o, pose, indices):
        points = np.asarray(o["points_world_m"])[indices]
        camera = (points - pose[:3, 3]) @ pose[:3, :3]
        projected = camera @ np.asarray(o["K"]).T
        uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-6)
        return uv - np.asarray(o["pixels"])[indices]

    def fit_residual(x):
        candidate = poses(x)
        return np.concatenate(
            [
                residual(observations[i], candidate[i], partitions[i][0]).ravel()
                for i in usable
            ]
        )

    bound = np.array([np.radians(max_rotation_degrees)] * 3 + [max_translation_m] * 3)
    fitted = least_squares(
        fit_residual,
        np.zeros(6),
        loss="huber",
        f_scale=3.0,
        bounds=(-bound, bound),
        x_scale=np.array([0.02] * 3 + [0.05] * 3),
        max_nfev=80,
    )
    corrected = poses(fitted.x)
    rows = []
    for i in usable:
        fit, val, cx, cy = partitions[i]
        before = np.linalg.norm(residual(observations[i], original[i], val), axis=1)
        after = np.linalg.norm(residual(observations[i], corrected[i], val), axis=1)
        fit_errors = np.linalg.norm(
            residual(observations[i], corrected[i], fit), axis=1
        )
        inliers = fit[fit_errors < 3.0]
        rows.append(
            {
                "view_index": i,
                "fit_inliers": len(inliers),
                "fit_grid_cells": len(set(zip(cx[inliers], cy[inliers]))),
                "validation_before_px": float(np.median(before)),
                "validation_after_px": float(np.median(after)),
                "validation_correspondences": len(val),
            }
        )
    before = float(np.median([r["validation_before_px"] for r in rows]))
    after = float(np.median([r["validation_after_px"] for r in rows]))
    translation = float(np.linalg.norm(fitted.x[3:]))
    angle = float(np.degrees(np.linalg.norm(fitted.x[:3])))
    enough = all(r["fit_inliers"] >= 24 and r["fit_grid_cells"] >= 6 for r in rows)
    no_regression = all(
        r["validation_after_px"] <= max(2.0, r["validation_before_px"] * 1.1)
        for r in rows
    )
    accepted = (
        before > 1.5
        and after < min(4.0, before * 0.7)
        and enough
        and no_regression
        and translation <= max_translation_m
        and angle <= max_rotation_degrees
    )
    report = {
        "accepted": bool(accepted),
        "reason": "validated_rig_correction" if accepted else "rig_validation_guard",
        "rig_preserved": True,
        "used_views": len(usable),
        "views": rows,
        "validation_before_px": before,
        "validation_after_px": after,
        "translation_change_m": translation,
        "rotation_change_degrees": angle,
        "optimizer_success": bool(fitted.success),
        "delta_anchor": (np.linalg.inv(anchor) @ corrected[0]).tolist(),
    }
    return (corrected if accepted else original), report
