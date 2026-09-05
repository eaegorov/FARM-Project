"""Angular coverage within a fixed physical-timestamp and source-image budget.

This stage consumes registered camera metadata only. Geometry anchors remain
separate from emitted discovery views; source pixels and model pixels must be
accounted independently. No categories, object IDs or masks enter selection.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from scene_graph.captioning.evidence import gravity_upright_orientation


def balanced_view_selection(
    by_timestamp: Mapping[str, Sequence[Any]],
    timestamps: Sequence[str],
    sensors: Sequence[str],
    families: Sequence[str],
    *,
    views_per_sensor: int = 1,
) -> tuple[list[Any], dict[str, Any]]:
    """Balance families per sensor; use new COLMAP tracks only to break ties.

    Every selected timestamp retains the same sensor/view count. Missing
    families are diagnosed, never replaced by a duplicate or another time.
    Over a complete rig, family counts differ by at most one.
    """
    if not timestamps or len(timestamps) != len(set(timestamps)):
        raise ValueError("nonempty unique timestamps are required")
    sensors = list(sensors)
    families = list(families)
    if not sensors or len(sensors) != len(set(sensors)):
        raise ValueError("nonempty unique sensors are required")
    if not families or len(families) != len(set(families)):
        raise ValueError("nonempty unique families are required")
    if not 1 <= views_per_sensor <= len(families):
        raise ValueError("views_per_sensor must be within the requested family count")
    counts = {sensor: Counter({f: 0 for f in families}) for sensor in sensors}
    covered: dict[str, set[int]] = defaultdict(set)
    selected = []
    deficits = []
    decisions = []
    for ts in timestamps:
        for sensor in sensors:
            candidates = [
                v
                for v in by_timestamp.get(ts, ())
                if v.sensor == sensor and v.family in families
            ]
            by_family = {}
            for view in candidates:
                if view.family in by_family:
                    raise ValueError(
                        f"duplicate camera/family at physical timestamp {ts}: {sensor}/{view.family}"
                    )
                by_family[view.family] = view
            missing = sorted(set(families) - set(by_family))
            if missing:
                deficits.append(
                    {"timestamp": ts, "sensor": sensor, "missing_families": missing}
                )
            if len(by_family) < views_per_sensor:
                raise ValueError(f"insufficient angular views at {ts}/{sensor}")
            for _ in range(views_per_sensor):
                view = min(
                    by_family.values(),
                    key=lambda v: (
                        counts[sensor][v.family],
                        -len(set(v.tracks) - covered[sensor]),
                        -len(v.tracks),
                        families.index(v.family),
                        v.name,
                    ),
                )
                new_tracks = len(set(view.tracks) - covered[sensor])
                counts[sensor][view.family] += 1
                covered[sensor].update(view.tracks)
                selected.append(view)
                decisions.append(
                    {
                        "name": view.name,
                        "timestamp": ts,
                        "sensor": sensor,
                        "family": view.family,
                        "new_sparse_tracks": new_tracks,
                    }
                )
                del by_family[view.family]
    return selected, {
        "policy": "balanced",
        "views_per_sensor": views_per_sensor,
        "family_counts_by_sensor": {s: dict(c) for s, c in counts.items()},
        "unique_sparse_tracks_by_sensor": {s: len(covered[s]) for s in sensors},
        "missing_family_observations": deficits,
        "decisions": decisions,
        "anchor_connectivity_is_not_emitted_view_connectivity": True,
        "uses_rgb_or_object_evidence": False,
    }


def upright_quarter_turns(
    rotation_world_to_camera: np.ndarray,
    world_up: Sequence[float],
    *,
    image_point=None,
    intrinsics=None,
) -> dict[str, Any]:
    """Reuse FARM's pose/gravity normalization without interpolation.

    The caller must record unavailable orientation and preserve pixels in
    that case. Predictions are inverse-rotated before any 3D operation.
    """
    r = np.asarray(rotation_world_to_camera, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("camera rotation must be a finite 3x3 matrix")
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(r), 1.0, atol=1e-5
    ):
        raise ValueError("camera rotation must be proper orthonormal")
    result = gravity_upright_orientation(
        0,
        frame_pose_index={
            0: {"camera_right_world": r[0].tolist(), "camera_down_world": r[1].tolist()}
        },
        world_up_vector=np.asarray(world_up, dtype=float).tolist(),
        world_up_source="declared_scene_gravity",
    )
    if (image_point is None) != (intrinsics is None):
        raise ValueError("image_point and intrinsics must be supplied together")
    if image_point is None:
        return result
    k = np.asarray(intrinsics, float)
    uv = np.asarray(image_point, float)
    if (
        k.shape != (3, 3)
        or uv.shape != (2,)
        or not np.isfinite(k).all()
        or not np.isfinite(uv).all()
        or k[0, 0] <= 0
        or k[1, 1] <= 0
    ):
        raise ValueError("valid pixel point and pinhole intrinsics required")
    up = np.asarray(world_up, float)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-12:
        raise ValueError("finite nonzero world up required")
    g = r @ (up / np.linalg.norm(up))
    # Derivative of pinhole projection at this object's viewing ray. Omitting
    # g_z is only valid at the principal point (or for orthographic cameras).
    projected = np.array(
        [
            k[0, 0] * g[0] - (uv[0] - k[0, 2]) * g[2],
            k[1, 1] * g[1] - (uv[1] - k[1, 2]) * g[2],
        ]
    )
    magnitude = np.linalg.norm(projected)
    if magnitude <= 1e-6:
        result.update(
            status="unavailable",
            reason="world_up_parallel_to_object_ray",
            applied_quarter_turns_ccw=0,
            applied_rotation_degrees_ccw=0,
        )
        return result
    x, y = projected / magnitude
    candidates = [(0, -y), (1, x), (2, y), (3, -x)]
    turns = max(candidates, key=lambda item: (item[1], -item[0]))[0]
    corrected = (x, y)
    for _ in range(turns):
        corrected = (corrected[1], -corrected[0])
    result.update(
        status="applied" if turns else "already_upright",
        applied_quarter_turns_ccw=turns,
        applied_rotation_degrees_ccw=turns * 90,
        projected_world_up_image_xy=[float(x), float(y)],
        source_roll_degrees_clockwise=float(np.degrees(np.arctan2(x, -y))),
        residual_roll_degrees=float(
            np.degrees(np.arctan2(corrected[0], -corrected[1]))
        ),
        source_orientation={
            0: "upright",
            1: "clockwise_90",
            2: "upside_down",
            3: "counterclockwise_90",
        }[turns],
        reason="pinhole_world_up_at_object_ray",
        image_point=uv.tolist(),
    )
    return result


def rotate_image(values: np.ndarray, quarter_turns: int) -> np.ndarray:
    """Rotate HxW or HxWxC data exactly. Floating logits retain all values."""
    array = np.asarray(values)
    if array.ndim not in (2, 3):
        raise ValueError("image/logits must be HxW or HxWxC")
    if isinstance(quarter_turns, bool) or int(quarter_turns) != quarter_turns:
        raise ValueError("quarter_turns must be an integer")
    return np.ascontiguousarray(np.rot90(array, int(quarter_turns) % 4, axes=(0, 1)))
