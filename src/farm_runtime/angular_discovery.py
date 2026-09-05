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
    rotation_world_to_camera: np.ndarray, world_up: Sequence[float]
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
    return gravity_upright_orientation(
        0,
        frame_pose_index={
            0: {"camera_right_world": r[0].tolist(), "camera_down_world": r[1].tolist()}
        },
        world_up_vector=world_up,
        world_up_source="declared_scene_gravity",
    )


def rotate_image(values: np.ndarray, quarter_turns: int) -> np.ndarray:
    """Rotate HxW or HxWxC data exactly. Floating logits retain all values."""
    array = np.asarray(values)
    if array.ndim not in (2, 3):
        raise ValueError("image/logits must be HxW or HxWxC")
    if isinstance(quarter_turns, bool) or int(quarter_turns) != quarter_turns:
        raise ValueError("quarter_turns must be an integer")
    return np.ascontiguousarray(np.rot90(array, int(quarter_turns) % 4, axes=(0, 1)))
