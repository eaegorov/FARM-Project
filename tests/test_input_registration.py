from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from farm_runtime import input_registration as registration


@dataclass
class View:
    name: str
    camera_to_world: np.ndarray
    world_to_camera: np.ndarray
    sparse_xy: np.ndarray
    sparse_depth_z: np.ndarray
    fx: float = 100.0
    fy: float = 100.0
    cx: float = 0.0
    cy: float = 0.0


def test_registration_omits_whole_bad_timestamp_and_reprojects_sparse_geometry(
    monkeypatch,
):
    views = [
        View(str(i), np.eye(4), np.eye(4), np.array([[10.0, 0.0]]), np.array([2.0]))
        for i in range(4)
    ]
    identities = [SimpleNamespace(timestamp=t) for t in ("good", "bad", "good", "bad")]

    def match(rendered, source, *args):
        assert source[0, 0].tolist() == [30, 20, 10]  # source reader is BGR
        return np.zeros((30, 3)), np.zeros((30, 2)), {}

    monkeypatch.setattr(registration, "match_render_to_source", match)
    calls = []

    def refine(observations):
        accepted = not calls
        calls.append(True)
        poses = [o["T_world_cam"].copy() for o in observations]
        if accepted:
            for pose in poses:
                pose[0, 3] += 0.02
        return poses, dict(accepted=accepted, validation_before_px=8.0)

    monkeypatch.setattr(registration, "refine_camera_rig", refine)
    result, indices, audit = registration.register_preparation_views(
        views,
        identities,
        [(2, 2, np.eye(3))] * 4,
        list(range(4)),
        scale=2.0,
        read_source=lambda *a: np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8),
        render=lambda *a: (np.zeros((2, 2, 3)), np.ones((2, 2)), np.ones((2, 2))),
    )
    assert indices == [0, 2]
    assert [v.name for v in result] == ["0", "2"]
    assert audit["omitted_timestamps"] == ["bad"]
    assert np.allclose(result[0].sparse_xy, [[9.5, 0.0]])
    assert np.allclose(result[0].camera_to_world[0, 3], 0.01)
    assert np.array_equal(views[0].camera_to_world, np.eye(4))
    assert np.allclose(result[0].world_to_camera @ result[0].camera_to_world, np.eye(4))
