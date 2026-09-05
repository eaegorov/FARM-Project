"""Shared rig registration before RGB-D generation and object discovery."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np

from farm_runtime.camera_alignment import match_render_to_source, refine_camera_rig


def register_preparation_views(
    views, identities, geometries, source_paths, *, scale, render, read_source
):
    """Return trusted views in original order; never change intrinsics or rig scale.

    RGB-D is rendered again after accepting a correction. A rejected timestamp
    is omitted as a complete group and explicitly reported, rather than entering
    the mapper with mismatched RGB and depth. Callers enforce retained coverage.
    """
    if not (len(views) == len(identities) == len(geometries) == len(source_paths)):
        raise ValueError("registration inputs must describe the same selected views")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("registration requires a positive finite metric scale")
    groups = defaultdict(list)
    for index, identity in enumerate(identities):
        groups[identity.timestamp].append(index)
    corrected = list(views)
    retained = []
    audit = []
    for timestamp, indices in groups.items():
        observations = []
        for index in indices:
            view = views[index]
            width, height, K = geometries[index]
            rgb = np.ascontiguousarray(
                read_source(source_paths[index], view, width, height)[..., ::-1]
            )
            rendered, depth, alpha = render(view, K, width, height)
            pose = view.camera_to_world.copy()
            pose[:3, 3] *= scale
            points, pixels, matches = match_render_to_source(
                np.clip(rendered * 255, 0, 255).astype(np.uint8),
                rgb,
                depth,
                alpha,
                K,
                pose,
            )
            observations.append(
                dict(
                    points_world_m=points,
                    pixels=pixels,
                    K=K,
                    T_world_cam=pose,
                    shape_hw=(height, width),
                )
            )
        poses, report = refine_camera_rig(observations)
        already_aligned = report.get(
            "validation_before_px", float("inf")
        ) <= 1.5 and all(r["fit_grid_cells"] >= 6 for r in report.get("views", []))
        if len(poses) != len(indices):
            raise ValueError("registration returned an incomplete camera group")
        trusted = bool(report["accepted"] or already_aligned)
        if trusted:
            for index, pose in zip(indices, poses):
                view = views[index]
                scene_pose = pose.copy()
                scene_pose[:3, 3] /= scale
                w2c = np.linalg.inv(scene_pose)
                # Keep sparse geometry QA in the corrected camera coordinate
                # system instead of comparing depths at stale projected pixels.
                xy, z = view.sparse_xy, view.sparse_depth_z
                camera = np.column_stack(
                    (
                        (xy[:, 0] - view.cx) * z / view.fx,
                        (xy[:, 1] - view.cy) * z / view.fy,
                        z,
                    )
                )
                world = (
                    camera @ view.camera_to_world[:3, :3].T
                    + view.camera_to_world[:3, 3]
                )
                transformed = world @ w2c[:3, :3].T + w2c[:3, 3]
                valid = transformed[:, 2] > 1e-6
                new_xy = np.column_stack(
                    (
                        view.fx * transformed[valid, 0] / transformed[valid, 2]
                        + view.cx,
                        view.fy * transformed[valid, 1] / transformed[valid, 2]
                        + view.cy,
                    )
                )
                corrected[index] = replace(
                    view,
                    camera_to_world=scene_pose,
                    world_to_camera=w2c,
                    sparse_xy=new_xy,
                    sparse_depth_z=transformed[valid, 2],
                )
            retained.extend(indices)
        audit.append(
            {
                "timestamp": timestamp,
                "source_images": [views[i].name for i in indices],
                **report,
                "trusted": trusted,
                "already_aligned": bool(already_aligned),
            }
        )
    retained.sort()
    return (
        [corrected[i] for i in retained],
        retained,
        {
            "schema": "farm.input-rig-registration.v1",
            "groups": audit,
            "source_views": len(views),
            "retained_views": len(retained),
            "omitted_timestamps": [r["timestamp"] for r in audit if not r["trusted"]],
            "rig_baseline_preserved": True,
            "semantic_masks_used": False,
            "registration_before_discovery": True,
        },
    )
